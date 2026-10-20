"""Resuming jobs interrupted by a restart (spec sec. 40).

The runner's docstring promised that "a restart resumes instead of
restarting", but nothing picked up an interrupted job on boot -- so a deploy,
a crash, or a host suspending an idle machine abandoned the job forever.
That matters as soon as the engine is hosted rather than run by hand.
"""

from __future__ import annotations

import pytest

from ct_engine.jobs.runner import MAX_RESUME_ATTEMPTS, JobRunner
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import JobStage
from ct_engine.model.job import DocumentJob, JobSettings
from ct_engine.pipeline.manifest_store import ManifestStore


def seeded(
    store, stage: JobStage, *, resume_count: int = 0, target_language: str = "es"
):
    manifests = ManifestStore(store)
    manifest = DocumentManifest(
        document_id="resume",
        filename="x.pdf",
        content_type="application/pdf",
        byte_size=10,
        sha256="a" * 64,
        source_key="documents/resume/source.pdf",
    )
    manifests.save_manifest(manifest)
    store.put_bytes("documents/resume/source.pdf", b"%PDF-1.4 stub")
    job = DocumentJob(
        job_id="job_interrupted",
        document_id="resume",
        settings=JobSettings(target_language=target_language),
        stage=stage,
        resume_count=resume_count,
    )
    manifests.save_job(job)
    return manifests, job


class _Runner(JobRunner):
    """Captures what would be submitted instead of running it."""

    def __init__(self, store, manifests):
        super().__init__(store, manifests)
        self.submitted: list[tuple[str, str]] = []

    def _run_analysis(self, document_id, job_id):  # noqa: D401
        self.submitted.append(("analysis", job_id))

    def _run_translation(self, document_id, job_id):
        self.submitted.append(("translation", job_id))


@pytest.mark.parametrize(
    "stage,language,expected",
    [
        # An analysis job carries no target language.
        (JobStage.QUEUED, "und", "analysis"),
        (JobStage.ANALYZING, "und", "analysis"),
        # A translation still waiting for a worker is a translation: sent to
        # analysis it ended at "choose a language", her request dropped.
        (JobStage.QUEUED, "es", "translation"),
        (JobStage.OCR, "es", "translation"),
        (JobStage.TRANSLATING, "es", "translation"),
        (JobStage.RECONSTRUCTING, "es", "translation"),
        (JobStage.VERIFYING, "es", "translation"),
        (JobStage.EXPORTING, "es", "translation"),
    ],
)
def test_an_interrupted_job_resumes_at_the_right_stage(store, stage, language, expected):
    manifests, _job = seeded(store, stage, target_language=language)
    runner = _Runner(store, manifests)
    try:
        resumed = runner.resume_interrupted()
        assert resumed == ["job_interrupted"]
    finally:
        # Waiting lets the pool run what it was handed.
        runner.shutdown(wait=True)
    assert runner.submitted == [(expected, "job_interrupted")]

    reloaded = manifests.load_job("resume", "job_interrupted")
    assert reloaded.resume_count == 1


@pytest.mark.parametrize(
    "stage", [JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED]
)
def test_finished_jobs_are_left_alone(store, stage):
    manifests, _job = seeded(store, stage)
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)


def test_review_ready_is_not_resumed(store):
    """Analysis finished and is awaiting a person; nothing to continue."""
    manifests, _job = seeded(store, JobStage.REVIEW_READY)
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)


def test_a_job_that_keeps_crashing_is_failed_not_looped(store):
    """Otherwise a bad document restart-loops the worker forever."""
    manifests, _job = seeded(
        store, JobStage.TRANSLATING, resume_count=MAX_RESUME_ATTEMPTS
    )
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)

    failed = manifests.load_job("resume", "job_interrupted")
    assert failed.stage is JobStage.FAILED
    assert "restarted several times" in failed.error
    assert failed.finished_at is not None


def test_resume_count_climbs_across_restarts(store):
    manifests, _job = seeded(store, JobStage.TRANSLATING)
    for expected in (1, 2, 3):
        runner = _Runner(store, manifests)
        try:
            runner.resume_interrupted()
        finally:
            runner.shutdown(wait=False)
        assert manifests.load_job("resume", "job_interrupted").resume_count == expected

    # The next boot gives up rather than trying a fourth time.
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)
    assert manifests.load_job("resume", "job_interrupted").stage is JobStage.FAILED


def test_an_empty_store_resumes_nothing(store):
    manifests = ManifestStore(store)
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)


def test_resume_is_recorded_in_the_job_log(store):
    manifests, _job = seeded(store, JobStage.TRANSLATING)
    runner = _Runner(store, manifests)
    try:
        runner.resume_interrupted()
    finally:
        runner.shutdown(wait=False)
    reloaded = manifests.load_job("resume", "job_interrupted")
    assert any("restarted" in e.message for e in reloaded.events)


def test_an_analysis_that_keeps_crashing_asks_for_the_file_again(store):
    """Nothing has been translated yet, so 'try translating it again'
    pointed at a button that does not exist at that point."""
    manifests, _job = seeded(
        store,
        JobStage.ANALYZING,
        resume_count=MAX_RESUME_ATTEMPTS,
        target_language="und",
    )
    runner = _Runner(store, manifests)
    try:
        assert runner.resume_interrupted() == []
    finally:
        runner.shutdown(wait=False)
    failed = manifests.load_job("resume", "job_interrupted")
    assert failed.stage is JobStage.FAILED
    assert "restarted several times" in failed.error
    assert "translat" not in failed.error.lower()
    assert "upload it again" in failed.error
