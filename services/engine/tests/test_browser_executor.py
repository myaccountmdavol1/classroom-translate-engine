"""The browser engine's job pool and entry points (ct_browser), run natively.

In the browser, Python has one thread and the worker's JS queue decides what
runs next: one request, or one whole job, at a time. These tests drive
ct_browser exactly that way -- handle() for each request, run_next_job() for
each job -- so the ordering the browser relies on is checked here, where a
deadlock shows up as a failed test rather than a frozen tab.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading

import httpx
import pytest

from conftest import corpus_file
from ct_browser.boot import BrowserEngine
from ct_browser.relay import ACCESS_CODE, RelayDeepL
from ct_browser.executor import DeferredJobExecutor
from ct_engine.jobs.runner import JobRunner
from ct_engine.model.enums import JobStage
from ct_engine.pipeline.manifest_store import ManifestStore
from ct_engine.providers.registry import OCR_PROVIDERS, TRANSLATION_PROVIDERS
from ct_engine.providers.translation.base import TranslationProvider, TranslationResult

from test_cancel import _analyzed


class _Stub(TranslationProvider):
    """Stands in for DeepL: no network. Records the class code it ran under."""

    name = "local"
    sends_data_externally = False
    codes_seen: list = []

    def supports(self, source_lang, target_lang) -> bool:
        return True

    def translate(self, segments, *, source_lang, target_lang):
        _Stub.codes_seen.append(ACCESS_CODE.get())
        return [
            TranslationResult(s.segment_id, "[ES] " + s.text, provider=self.name)
            for s in segments
        ]


def _in_thread(fn, timeout=60.0):
    """Run `fn` on a thread and fail (instead of hanging) if it deadlocks."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported below
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "deadlocked: the call never returned"
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---- The executor on its own --------------------------------------------------
def test_submit_only_records_the_work():
    pool = DeferredJobExecutor()
    ran = []
    fut = pool.submit(ran.append, 1)
    assert ran == [] and not fut.done() and pool.pending() == 1
    assert pool.run_next() is fut
    assert ran == [1] and fut.result() is None and pool.pending() == 0
    assert pool.run_next() is None


def test_a_job_cancelled_before_it_starts_never_runs():
    pool = DeferredJobExecutor()
    ran = []
    first = pool.submit(ran.append, "first")
    second = pool.submit(ran.append, "second")
    assert first.cancel()
    assert pool.pending() == 1
    assert pool.peek() == ("second",)
    assert pool.run_next() is second
    assert ran == ["second"] and first.cancelled()


def test_shutdown_cancels_waiting_futures_and_refuses_new_ones():
    pool = DeferredJobExecutor()
    futs = [pool.submit(print, i) for i in range(3)]
    pool.shutdown(wait=True, cancel_futures=True)
    assert all(f.cancelled() for f in futs)
    assert pool.pending() == 0 and pool.run_next() is None
    with pytest.raises(RuntimeError):
        pool.submit(print, "late")


def test_shutdown_without_cancelling_keeps_waiting_work():
    pool = DeferredJobExecutor()
    fut = pool.submit(lambda: 7)
    pool.shutdown(wait=False)
    assert pool.run_next() is fut and fut.result() == 7


def test_errors_are_kept_on_the_future_not_raised():
    pool = DeferredJobExecutor()

    def boom():
        raise ValueError("bad page")

    fut = pool.submit(boom)
    assert pool.run_next() is fut
    assert isinstance(fut.exception(), ValueError)


def test_the_job_runs_in_the_context_it_was_submitted_from():
    var: contextvars.ContextVar = contextvars.ContextVar("v", default="none")
    pool = DeferredJobExecutor()

    def submit_with(value):
        var.set(value)
        return pool.submit(var.get)

    fut = contextvars.copy_context().run(submit_with, "the request's")
    assert var.get() == "none"
    pool.run_next()
    assert fut.result() == "the request's"


# ---- With the real JobRunner ----------------------------------------------------
def test_submit_under_the_runner_guard_returns_before_the_job_runs(store):
    manifests = _analyzed(store, "guard", "01_clean_worksheet.pdf")
    runner = JobRunner(store, manifests)
    runner._pool.shutdown(wait=False)
    runner._pool = DeferredJobExecutor()
    try:
        # _enqueue calls submit() while holding _guard; a pool that ran the
        # job inline here would deadlock in _job_ended.
        job = _in_thread(lambda: runner.submit_analysis("guard"))
        assert job.stage is JobStage.QUEUED
        assert not runner._guard.locked()
        assert runner._pool.pending() == 1
        assert runner._pool.peek() == ("guard", job.job_id)
        _in_thread(runner._pool.run_next)
        done = manifests.load_job("guard", job.job_id)
        assert done.stage is JobStage.REVIEW_READY, done.error
        assert "guard" not in runner._active
    finally:
        runner.shutdown(wait=False)


def test_cancel_drops_a_waiting_job_through_the_runner(store):
    manifests = _analyzed(store, "drop", "01_clean_worksheet.pdf")
    runner = JobRunner(store, manifests)
    runner._pool.shutdown(wait=False)
    runner._pool = DeferredJobExecutor()
    try:
        job = runner.submit_analysis("drop")
        assert runner.cancel("drop") is True
        assert runner._pool.pending() == 0
        assert runner._pool.run_next() is None
        assert manifests.load_job("drop", job.job_id).stage is JobStage.CANCELLED
    finally:
        runner.shutdown(wait=False)


# ---- The whole engine, request by request -------------------------------------------
@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setitem(TRANSLATION_PROVIDERS, "local", _Stub)
    _Stub.codes_seen = []
    posts: list[dict] = []
    eng = BrowserEngine(post=lambda raw: posts.append(json.loads(raw)))
    eng.start()
    eng.posts = posts
    try:
        yield eng
    finally:
        eng.stop()


def call(engine, method, path, *, json_body=None, files=None, headers=None):
    """One request, as the worker sends it: method, path, headers, raw body."""
    req = httpx.Request(method, "http://x" + path, json=json_body, files=files, headers=headers)
    body = req.read()
    sent = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
    status, resp_headers, content = asyncio.run(engine.handle(method, path, sent, body))
    ctype = dict((k.lower(), v) for k, v in resp_headers).get("content-type", "")
    return status, (json.loads(content) if "json" in ctype else content)


def upload(engine, name="01_clean_worksheet.pdf"):
    status, body = call(
        engine, "POST", "/api/documents",
        files={"file": (name, corpus_file(name), "application/pdf")},
    )
    assert status == 200, body
    return body["document_id"]


def drain(engine):
    ran = []
    while engine.has_pending_job():
        out = _in_thread(engine.run_next_job, timeout=180)
        assert "error" not in out, out
        ran.append(out)
    return ran


def test_start_installs_the_shims_and_stop_removes_them(monkeypatch):
    import anyio.to_thread

    from ct_engine.fonts import registry
    from ct_engine.jobs import runner as runner_mod

    before = anyio.to_thread.run_sync
    eng = BrowserEngine()
    eng.start()
    try:
        assert anyio.to_thread.run_sync is not before
        assert registry._SYSTEM_DIRS["Emscripten"][0].as_posix() == "/fonts"
        assert runner_mod.MAX_RESUME_ATTEMPTS >= 8
        assert isinstance(eng.runner._pool, DeferredJobExecutor)
        assert "save_job" in vars(eng.manifests)
        assert TRANSLATION_PROVIDERS["deepl"] is RelayDeepL
        assert "browser" in OCR_PROVIDERS
    finally:
        eng.stop()
    assert anyio.to_thread.run_sync is before
    assert TRANSLATION_PROVIDERS["deepl"] is not RelayDeepL
    assert "browser" not in OCR_PROVIDERS
    assert "Emscripten" not in registry._SYSTEM_DIRS
    assert runner_mod.MAX_RESUME_ATTEMPTS == 3


def test_inline_run_sync_passes_keyword_arguments():
    from ct_browser.boot import _run_sync_inline

    result = asyncio.run(_run_sync_inline(lambda a, *, b: (a, b), 1, b=2, abandon_on_cancel=True))
    assert result == (1, 2)


def test_upload_answers_before_analysis_and_a_manual_drain_finishes_it(engine):
    doc = upload(engine)
    # Answered with the job still waiting: the worker runs it next.
    assert engine.has_pending_job()
    info = engine.next_job()
    assert info["document_id"] == doc and info["translation"] is False
    assert info["ocr_language"] is None
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert status == 200 and job["stage"] == "QUEUED"
    ran = drain(engine)
    assert len(ran) == 1 and ran[0]["document_id"] == doc
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert job["stage"] == "REVIEW_READY"


def test_translation_job_carries_the_class_code_of_its_request(engine):
    doc = upload(engine)
    drain(engine)
    status, job = call(
        engine, "POST", f"/api/documents/{doc}/translate",
        json_body={"target_language": "es"}, headers={"X-CT-Access": "class-42"},
    )
    assert status == 200, job
    assert ACCESS_CODE.get() is None  # reset once the request answered
    info = engine.next_job()
    assert info["translation"] is True
    drain(engine)
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert job["stage"] == "COMPLETE", job
    assert _Stub.codes_seen and set(_Stub.codes_seen) == {"class-42"}


def test_an_edit_sent_while_a_job_waits_is_answered_first(engine):
    doc = upload(engine)
    drain(engine)
    call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    drain(engine)
    status, blocks = call(engine, "GET", f"/api/documents/{doc}/pages/1/blocks")
    assert status == 200 and blocks["blocks"]
    first = blocks["blocks"][0]["id"]

    # A second translation waits in the pool...
    status, queued = call(
        engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"}
    )
    assert status == 200 and queued["stage"] == "QUEUED"
    assert engine.has_pending_job()
    # ...and an edit made now is applied and answered before it starts.
    status, edited = _in_thread(lambda: call(
        engine, "PATCH", f"/api/documents/{doc}/pages/1/blocks/{first}",
        json_body={"text": "Texto del maestro"},
    ))
    assert status == 200, edited
    assert edited["block"]["translation"] == "Texto del maestro"
    assert engine.has_pending_job()
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert job["job_id"] == queued["job_id"] and job["stage"] == "QUEUED"
    drain(engine)


def test_stop_drops_a_waiting_translation(engine):
    doc = upload(engine)
    drain(engine)
    call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    status, body = call(engine, "POST", f"/api/documents/{doc}/cancel")
    assert status == 200 and body == {"cancelled": True}
    assert not engine.has_pending_job()
    assert engine.run_next_job()["ran"] is False
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert job["stage"] == "CANCELLED"
    assert _Stub.codes_seen == []


def _latest(posts, kind, doc):
    found = [p for p in posts if p["type"] == kind and p["document_id"] == doc]
    assert found, f"no {kind} snapshot posted for {doc}"
    return found[-1]


def test_snapshots_equal_what_the_routes_return(engine):
    doc = upload(engine)
    drain(engine)
    call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    drain(engine)
    for _ in range(2):  # after the job, and after an edit
        status, job = call(engine, "GET", f"/api/documents/{doc}/job")
        assert _latest(engine.posts, "job", doc)["job"] == job
        status, document = call(engine, "GET", f"/api/documents/{doc}")
        posted = dict(_latest(engine.posts, "document", doc)["document"])
        # The host merges in the newest job snapshot, as the route does.
        posted["job"] = _latest(engine.posts, "job", doc)["job"]
        assert posted == document
        status, blocks = call(engine, "GET", f"/api/documents/{doc}/pages/1/blocks")
        call(
            engine, "PATCH", f"/api/documents/{doc}/pages/1/blocks/{blocks['blocks'][0]['id']}",
            json_body={"text": "otra cosa"},
        )


def test_snapshots_show_progress_during_the_job(engine):
    doc = upload(engine, "13_rotated_pages.pdf")
    drain(engine)
    before = len(engine.posts)
    call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    drain(engine)
    stages = [p["job"]["stage"] for p in engine.posts[before:] if p["type"] == "job"]
    assert stages[0] == "QUEUED" and stages[-1] == "COMPLETE"
    assert {"TRANSLATING", "RECONSTRUCTING"} <= set(stages)


def test_a_document_with_a_scan_asks_for_the_text_reader(engine):
    doc = upload(engine, "02_scanned_worksheet.pdf")
    assert engine.next_job()["ocr_language"] is None  # analysis reads no text
    drain(engine)
    call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    info = engine.next_job()
    assert info["translation"] is True and info["ocr_language"] == "eng"
    status, _ = call(engine, "POST", f"/api/documents/{doc}/cancel")
    assert status == 200 and not engine.has_pending_job()


def test_an_unknown_route_is_answered_not_raised(engine):
    status, body = call(engine, "GET", "/api/no-such-thing")
    assert status == 404 and body["error"]
