"""Stopping a document that is still being processed.

A teacher who starts over must not leave the old file running: on the free
server it holds one of two worker slots and most of the memory until it
ends, so the next upload would wait behind work nobody wants.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time

import pytest

from conftest import corpus_file
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.jobs import runner as runner_mod
from ct_engine.jobs.runner import (
    CANCELLED_MESSAGE,
    JobCancelled,
    JobRunner,
    _StopWhenCancelled,
)
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import JobStage
from ct_engine.model.job import JobSettings
from ct_engine.pipeline.manifest_store import ManifestStore
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
)
from ct_engine.util.hashing import sha256_bytes

SETTINGS = JobSettings(target_language="es", translate_image_text=False)


class _Blocking(TranslationProvider):
    """Holds the first page until the test lets it go."""

    name = "blocking"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        self.calls += 1
        self.started.set()
        assert self.release.wait(20), "the test never released the page"
        return [
            TranslationResult(segment_id=s.segment_id, text=f"es {s.text}", provider=self.name)
            for s in segments
        ]


def _analyzed(store, document_id: str, name: str = "13_rotated_pages.pdf") -> ManifestStore:
    """A document read and ready to translate, saved as the runner finds it."""
    data = corpus_file(name)
    source_key = f"documents/{document_id}/source.pdf"
    store.put_bytes(source_key, data)
    manifest = DocumentManifest(
        document_id=document_id,
        filename=name,
        content_type="application/pdf",
        byte_size=len(data),
        sha256=sha256_bytes(data),
        source_key=source_key,
    )
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            manifest.pages.append(page)
    manifest.analysis.page_count = len(manifest.pages)
    manifests = ManifestStore(store)
    manifests.save_manifest(manifest)
    return manifests


def _wait_terminal(manifests, document_id, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while True:
        job = manifests.load_job(document_id, job_id)
        if job is not None and job.is_terminal:
            return job
        assert time.time() < deadline, f"job still {job.stage if job else None}"
        time.sleep(0.02)


@pytest.fixture
def blocking(monkeypatch):
    provider = _Blocking()
    monkeypatch.setattr(runner_mod, "get_translation_provider", lambda: provider)
    yield provider
    provider.release.set()


def test_nothing_running_is_not_cancelled(store):
    runner = JobRunner(store, ManifestStore(store))
    try:
        assert runner.cancel("no-such-document") is False
    finally:
        runner.shutdown(wait=False)


def test_a_running_translation_stops_at_the_next_page(store, blocking):
    manifests = _analyzed(store, "run")
    assert len(manifests.load_manifest("run").pages) >= 2
    runner = JobRunner(store, manifests)
    try:
        job = runner.submit_translation("run", SETTINGS)
        assert blocking.started.wait(20)
        assert runner.cancel("run") is True
        blocking.release.set()

        done = _wait_terminal(manifests, "run", job.job_id)
        assert done.stage is JobStage.CANCELLED
        assert done.error is None
        assert done.finished_at is not None
        assert done.events[-1].message == CANCELLED_MESSAGE
        # Stopped after the page under way: nothing more was translated,
        # rebuilt or exported.
        assert blocking.calls == 1
        assert done.export_key is None
        assert not any(p.stage is JobStage.RECONSTRUCTING for p in done.progress)

        # The slot is free and there is nothing left to stop.
        deadline = time.time() + 5
        while runner._active or runner._futures or runner._cancel_requested:
            assert time.time() < deadline
            time.sleep(0.01)
        assert runner.cancel("run") is False
    finally:
        blocking.release.set()
        runner.shutdown(wait=True)


def test_a_waiting_job_is_dropped_at_once(store, blocking):
    manifests = _analyzed(store, "busy")
    _analyzed(store, "waiting")
    runner = JobRunner(store, manifests)
    runner._pool.shutdown(wait=False)
    runner._pool = cf.ThreadPoolExecutor(max_workers=1)  # one free slot
    try:
        busy = runner.submit_translation("busy", SETTINGS)
        assert blocking.started.wait(20)
        waiting = runner.submit_translation("waiting", SETTINGS)
        assert manifests.load_job("waiting", waiting.job_id).stage is JobStage.QUEUED

        assert runner.cancel("waiting") is True
        # No worker had to reach it: it is already finished.
        dropped = manifests.load_job("waiting", waiting.job_id)
        assert dropped.stage is JobStage.CANCELLED
        assert "waiting" not in runner._active

        blocking.release.set()
        assert _wait_terminal(manifests, "busy", busy.job_id).stage is JobStage.COMPLETE
        # It never ran afterwards either.
        assert manifests.load_job("waiting", waiting.job_id).stage is JobStage.CANCELLED
        assert blocking.calls >= 1
    finally:
        blocking.release.set()
        runner.shutdown(wait=True)


def test_a_new_request_after_cancelling_is_not_handed_the_stopping_job(store, blocking):
    manifests = _analyzed(store, "again")
    runner = JobRunner(store, manifests)
    try:
        first = runner.submit_translation("again", SETTINGS)
        assert blocking.started.wait(20)
        assert runner.cancel("again")
        second = runner.submit_translation("again", SETTINGS)
        assert second.job_id != first.job_id
        blocking.release.set()
        assert _wait_terminal(manifests, "again", first.job_id).stage is JobStage.CANCELLED
        assert _wait_terminal(manifests, "again", second.job_id).stage is JobStage.COMPLETE
    finally:
        blocking.release.set()
        runner.shutdown(wait=True)


def test_cancelling_analysis_stops_reading_pages(store, monkeypatch):
    _analyzed(store, "reading")
    manifests = ManifestStore(store)
    reached = threading.Event()
    release = threading.Event()
    pages_read: list[int] = []

    def slow_analyze(doc, manifest, *, store, on_progress=None, **_kw):
        reached.set()
        assert release.wait(20)
        for i in range(doc.page_count):
            doc.parse_page(i)
            pages_read.append(i)
        return manifest

    monkeypatch.setattr(runner_mod, "analyze_document", slow_analyze)
    runner = JobRunner(store, manifests)
    try:
        job = runner.submit_analysis("reading")
        assert reached.wait(20)
        assert runner.cancel("reading") is True
        release.set()
        done = _wait_terminal(manifests, "reading", job.job_id)
        assert done.stage is JobStage.CANCELLED
        assert pages_read == []
    finally:
        release.set()
        runner.shutdown(wait=True)


def test_the_stopping_document_raises_on_use():
    class Doc:
        page_count = 3

    flag = {"stop": False}

    def check():
        if flag["stop"]:
            raise JobCancelled()

    doc = _StopWhenCancelled(Doc(), check)
    assert doc.page_count == 3
    assert getattr(doc, "embedded_media_count", None) is None
    flag["stop"] = True
    with pytest.raises(JobCancelled):
        doc.page_count  # noqa: B018
    # A cancellation is not an ordinary error: the page guards let it pass.
    assert not issubclass(JobCancelled, Exception)


# --- Repeated requests ------------------------------------------------------
class _Held(JobRunner):
    """Keeps every job running until released, without doing the work."""

    def __init__(self, store, manifests):
        super().__init__(store, manifests)
        self.release = threading.Event()

    def _run_translation(self, document_id, job_id):
        self.release.wait(20)
        self._job_ended(document_id, job_id)


def test_a_repeated_translate_request_saves_no_second_job(store):
    manifests = _analyzed(store, "twice")
    runner = _Held(store, manifests)
    try:
        first = runner.submit_translation("twice", SETTINGS)
        again = runner.submit_translation("twice", SETTINGS)
        assert again.job_id == first.job_id
        # Only the job that runs is on disk, so it is the one the
        # document's progress is read from.
        jobs = manifests.list_jobs("twice")
        assert [j.job_id for j in jobs] == [first.job_id]
    finally:
        runner.release.set()
        runner.shutdown(wait=True)


# -- a stop that has to survive a restart ------------------------------------
def test_a_stop_is_written_down_and_cleared_when_the_job_ends(store, blocking):
    manifests = _analyzed(store, "mark")
    runner = JobRunner(store, manifests)
    try:
        job = runner.submit_translation("mark", SETTINGS)
        assert blocking.started.wait(20)
        assert runner.cancel("mark") is True
        marker = runner._cancel_marker_key("mark", job.job_id)
        assert store.exists(marker)
        # Not mistaken for a job of its own.
        assert [j.job_id for j in manifests.list_jobs("mark")] == [job.job_id]

        blocking.release.set()
        assert _wait_terminal(manifests, "mark", job.job_id).stage is JobStage.CANCELLED
        deadline = time.time() + 5
        while store.exists(marker):
            assert time.time() < deadline
            time.sleep(0.01)
    finally:
        blocking.release.set()
        runner.shutdown(wait=True)


def test_a_job_stopped_before_a_restart_is_not_picked_up_again(store, monkeypatch):
    from ct_engine.model.job import DocumentJob

    manifests = _analyzed(store, "restart")
    # The process died while the job was stopping: on disk it is still
    # mid-translation, with only the stop marker to say otherwise.
    job = DocumentJob(
        job_id="job_stopping",
        document_id="restart",
        settings=SETTINGS,
        stage=JobStage.TRANSLATING,
    )
    manifests.save_job(job)
    store.put_text(JobRunner._cancel_marker_key("restart", job.job_id), "1")

    runner = JobRunner(store, manifests)
    submitted: list = []
    monkeypatch.setattr(runner._pool, "submit", lambda *a, **k: submitted.append(a))
    try:
        assert runner.resume_interrupted() == []
        assert submitted == []
        assert "restart" not in runner._active
        ended = manifests.load_job("restart", job.job_id)
        assert ended.stage is JobStage.CANCELLED
        assert ended.resume_count == 0
        assert ended.events[-1].message == CANCELLED_MESSAGE
        assert not store.exists(JobRunner._cancel_marker_key("restart", job.job_id))
    finally:
        runner.shutdown(wait=False)


def test_a_stop_asked_for_at_the_last_moment_still_ends_cancelled(store, monkeypatch):
    """Past every page and stage, a True from cancel() must still mean stopped."""

    class _Quick(_Blocking):
        def translate(self, segments, *, source_lang, target_lang):
            self.release.set()
            return super().translate(segments, source_lang=source_lang, target_lang=target_lang)

    monkeypatch.setattr(runner_mod, "get_translation_provider", lambda: _Quick())
    manifests = _analyzed(store, "late")
    runner = JobRunner(store, manifests)
    answers: list[bool] = []
    real_gate = runner_mod.check_completion

    def gate_then_cancel(*args, **kwargs):
        answers.append(runner.cancel("late"))
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "check_completion", gate_then_cancel)
    try:
        job = runner.submit_translation("late", SETTINGS)
        done = _wait_terminal(manifests, "late", job.job_id)
        assert answers == [True]
        assert done.stage is JobStage.CANCELLED
    finally:
        runner.shutdown(wait=True)


def test_the_next_job_starts_from_what_is_on_disk_once_it_has_the_document(
    store, monkeypatch
):
    """A job queued behind one being stopped must not work from a stale copy."""
    provider = _Blocking()
    second_reading = threading.Event()
    calls = {"n": 0}

    def get_provider():
        # Asked for after the manifest is first read and before the
        # document lock is taken, so the second call means the new job
        # holds its early copy and is about to wait for the lock.
        calls["n"] += 1
        if calls["n"] == 2:
            second_reading.set()
        return provider

    monkeypatch.setattr(runner_mod, "get_translation_provider", get_provider)
    manifests = _analyzed(store, "stale")
    runner = JobRunner(store, manifests)
    try:
        first = runner.submit_translation("stale", SETTINGS)
        assert provider.started.wait(20)
        assert runner.cancel("stale")
        second = runner.submit_translation("stale", SETTINGS)
        assert second_reading.wait(20)

        # Saved while the new job waits for the document.
        edited = manifests.load_manifest("stale")
        edited.filename = "edited while waiting.pdf"
        manifests.save_manifest(edited)

        provider.release.set()
        assert _wait_terminal(manifests, "stale", first.job_id).stage is JobStage.CANCELLED
        assert _wait_terminal(manifests, "stale", second.job_id).stage is JobStage.COMPLETE
        assert manifests.load_manifest("stale").filename == "edited while waiting.pdf"
    finally:
        provider.release.set()
        runner.shutdown(wait=True)
