"""Job model, resumability bookkeeping, and retention."""

from __future__ import annotations

import datetime as _dt

from ct_engine.jobs.retention import expired_document_ids, sweep_once
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import JobStage
from ct_engine.model.job import STAGE_LABELS, DocumentJob, JobSettings
from ct_engine.pipeline.manifest_store import ManifestStore


def _job() -> DocumentJob:
    return DocumentJob(
        job_id="job_test",
        document_id="doc_test",
        settings=JobSettings(target_language="es"),
    )


def test_stage_labels_are_plain_language():
    """Teachers must never see 'OCR' or 'raster inpainting' (spec sec. 57)."""
    for stage, label in STAGE_LABELS.items():
        assert label and label[0].isupper()
        lowered = label.lower()
        for jargon in ("ocr", "raster", "bbox", "segmentation", "inpaint"):
            assert jargon not in lowered, f"{stage} label leaks jargon: {label}"


def test_completed_pages_are_tracked_per_stage():
    job = _job()
    job.mark_page_done(JobStage.ANALYZING, 5)
    job.mark_page_done(JobStage.ANALYZING, 5)  # idempotent
    assert job.is_page_done(JobStage.ANALYZING, 5)
    assert not job.is_page_done(JobStage.TRANSLATING, 5)
    assert job.completed_pages["ANALYZING"] == [5]


def test_page_errors_accumulate_attempts_not_duplicates():
    job = _job()
    job.stage = JobStage.ANALYZING
    job.record_page_error(7, "Page 7 could not be read.")
    job.record_page_error(7, "Page 7 could not be read.")
    assert len(job.page_errors) == 1
    assert job.page_errors[0].attempts == 2
    job.clear_page_error(7)
    assert job.page_errors == []


def test_event_log_is_bounded():
    job = _job()
    for i in range(2500):
        job.log("info", f"event {i}")
    assert len(job.events) <= 2000
    assert job.events[-1].message == "event 2499"


def test_progress_fraction_is_safe_when_total_unknown():
    job = _job()
    p = job.stage_progress(JobStage.ANALYZING)
    assert p.fraction == 0.0
    p.total, p.completed = 10, 4
    assert p.fraction == 0.4


def test_terminal_states():
    job = _job()
    assert not job.is_terminal
    for stage in (JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED):
        job.stage = stage
        assert job.is_terminal


def test_job_round_trips_through_storage(store):
    ms = ManifestStore(store)
    job = _job()
    job.log("info", "hello")
    job.mark_page_done(JobStage.ANALYZING, 2)
    ms.save_job(job)
    loaded = ms.load_job("doc_test", "job_test")
    assert loaded is not None
    assert loaded.settings.target_language == "es"
    assert loaded.is_page_done(JobStage.ANALYZING, 2)
    assert ms.list_jobs("doc_test")[-1].job_id == "job_test"


def _manifest(doc_id: str, age_hours: float) -> DocumentManifest:
    created = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=age_hours)
    return DocumentManifest(
        document_id=doc_id,
        filename="x.pdf",
        content_type="application/pdf",
        byte_size=10,
        sha256="a" * 64,
        source_key=f"documents/{doc_id}/source.pdf",
        created_at=created,
    )


def test_retention_identifies_only_expired_documents(store):
    ms = ManifestStore(store)
    ms.save_manifest(_manifest("oldone", age_hours=48))
    ms.save_manifest(_manifest("newone", age_hours=1))
    store.put_bytes("documents/oldone/source.pdf", b"x")
    store.put_bytes("documents/newone/source.pdf", b"y")

    expired = expired_document_ids(store, ms)
    assert "oldone" in expired
    assert "newone" not in expired


def test_retention_sweep_deletes_expired_and_keeps_fresh(store):
    ms = ManifestStore(store)
    ms.save_manifest(_manifest("gone", age_hours=72))
    ms.save_manifest(_manifest("kept", age_hours=2))
    store.put_bytes("documents/gone/pages/0000/source.png", b"x")
    store.put_bytes("documents/kept/pages/0000/source.png", b"y")

    assert sweep_once(store, ms) == 1
    assert ms.load_manifest("gone") is None
    assert not store.exists("documents/gone/pages/0000/source.png")
    assert ms.load_manifest("kept") is not None
    assert store.exists("documents/kept/pages/0000/source.png")
