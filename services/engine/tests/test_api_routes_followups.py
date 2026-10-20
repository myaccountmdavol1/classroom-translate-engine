"""Route follow-ups: refusing a translation that cannot run, what an Office
download kept in English, the report's picture line, cancelling, and checking
an upload without reading it into memory."""

from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path

import pytest

from conftest import corpus_file

ORIGIN = "http://localhost:3000"


def _seed(*, analyzed: bool = True, job_stage=None, translate_image_text=True,
          findings=(), images: int = 0, finding_objects=None):
    """A stored document with one page, and optionally a job."""
    from ct_engine.model.document import DocumentManifest, Page, QAFinding
    from ct_engine.model.geometry import NormRect, Rect
    from ct_engine.model.job import DocumentJob, JobSettings
    from ct_engine.model.objects import ImageObject
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    doc_id = uuid.uuid4().hex
    manifests = ManifestStore(get_store())
    page = Page(index=0, number=1, width=612, height=792, analyzed=analyzed)
    for i in range(images):
        page.image_objects.append(
            ImageObject(
                id=f"img{i}",
                bbox=Rect(x0=10, y0=10, x1=100, y1=100),
                norm_bbox=NormRect(x0=0.01, y0=0.01, x1=0.1, y1=0.1),
            )
        )
    page.qa.findings.extend(
        QAFinding(severity="warning", code=c, message=c,
                  object_ids=list((finding_objects or {}).get(c, [])))
        for c in findings
    )
    manifest = DocumentManifest(
        document_id=doc_id,
        filename="x.pdf",
        content_type="application/pdf",
        byte_size=10,
        sha256="a" * 64,
        source_key=f"documents/{doc_id}/source.pdf",
        pages=[page],
    )
    manifests.save_manifest(manifest)
    if job_stage is not None:
        manifests.save_job(
            DocumentJob(
                job_id="job_" + doc_id[:8],
                document_id=doc_id,
                settings=JobSettings(
                    target_language="es", translate_image_text=translate_image_text
                ),
                stage=job_stage,
            )
        )
    return doc_id


# --- Translate refusals ------------------------------------------------------
def test_translate_refuses_a_file_that_could_not_be_read(client):
    from ct_engine.model.enums import JobStage

    doc_id = _seed(analyzed=False, job_stage=JobStage.FAILED)
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    assert r.status_code == 409
    assert r.json()["error"] == (
        "This file could not be read, so it can't be translated. "
        "Please upload a different copy."
    )


def test_translate_refuses_a_file_whose_reading_was_stopped(client):
    from ct_engine.model.enums import JobStage

    doc_id = _seed(analyzed=False, job_stage=JobStage.CANCELLED)
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    assert r.status_code == 409
    assert r.json()["error"] == (
        "Reading this file was stopped, so it can't be translated. "
        "Please upload it again."
    )


def test_translate_refuses_without_a_retryable_status_when_not_configured(
    client, monkeypatch
):
    from ct_engine.api import routes

    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(routes, "get_translation_provider", boom)
    doc_id = _seed()
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    assert r.status_code not in (0, 429, 502, 503, 504)
    assert "not configured" in r.json()["error"]


def test_translate_still_asks_to_wait_while_reading(client):
    from ct_engine.model.enums import JobStage

    doc_id = _seed(analyzed=False, job_stage=JobStage.ANALYZING)
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    # Removed, so a later client's start-up does not resume it as interrupted.
    client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 409
    assert "wait" in r.json()["error"]


class _FakeProvider:
    name = "fake-followup"

    def supports(self, source_lang, target_lang):
        return True


@pytest.fixture
def fake_provider(monkeypatch):
    from ct_engine.api import routes
    from ct_engine.providers.translation import chain

    monkeypatch.setattr(routes, "get_translation_provider", lambda: _FakeProvider())
    yield _FakeProvider
    chain.reset_exhausted()


def test_translate_refused_up_front_when_the_allowance_is_used_up(client, fake_provider):
    from ct_engine.providers.translation import chain

    chain.mark_exhausted(fake_provider.name, chain.next_month_start())
    doc_id = _seed()
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    # Not 503, which clients retry as a server that is still waking.
    assert r.status_code == 409
    assert r.json()["error"] == chain.ALLOWANCE_USED_UP


def test_daily_cap_says_tomorrow(client, fake_provider):
    from ct_engine.providers.translation import chain

    chain.mark_exhausted(
        fake_provider.name,
        _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=5),
    )
    doc_id = _seed()
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    assert r.status_code == 409
    assert r.json()["error"] == chain.ALLOWANCE_USED_UP_TODAY


def test_translate_is_queued_when_allowance_remains(client, fake_provider, monkeypatch):
    from ct_engine.api import routes
    from ct_engine.model.job import DocumentJob

    submitted = []

    class Runner:
        def submit_translation(self, document_id, settings):
            submitted.append(document_id)
            return DocumentJob(job_id="j", document_id=document_id, settings=settings)

    monkeypatch.setattr(routes, "get_job_runner", lambda: Runner())
    doc_id = _seed()
    r = client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    assert r.status_code == 200, r.text
    assert submitted == [doc_id]


# --- Office download: pictures kept in English -------------------------------
def _findings(doc_id: str, code: str) -> list:
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    m = ManifestStore(get_store()).load_manifest(doc_id)
    return [f for p in m.pages for f in p.qa.findings if f.code == code]


def test_kept_pictures_are_noted_once_per_picture_without_duplicates():
    from ct_engine.api.routes import PICTURE_NOT_REPLACED, _record_pictures_kept

    doc_id = _seed()
    _record_pictures_kept(doc_id, [1, 1])
    _record_pictures_kept(doc_id, [1, 1])  # a second download
    kept = _findings(doc_id, PICTURE_NOT_REPLACED)
    assert len(kept) == 2
    assert kept[0].message == "A picture on page 1 kept its original words in the saved file."
    assert kept[0].severity == "warning"

    # Once the picture comes through, the note goes.
    _record_pictures_kept(doc_id, [])
    assert _findings(doc_id, PICTURE_NOT_REPLACED) == []


def test_office_download_says_which_pictures_kept_their_words(client, monkeypatch):
    from types import SimpleNamespace

    from ct_engine.api import routes
    from ct_engine.model.job import DocumentJob, JobSettings
    from ct_engine.model.enums import JobStage
    from ct_engine.pipeline import word_export

    doc_id = _seed()
    manifest = routes._load_manifest(doc_id)
    manifest.office_source_key = f"documents/{doc_id}/source.pptx"
    manifest.office_format = "pptx"
    job = DocumentJob(
        job_id="j", document_id=doc_id, settings=JobSettings(target_language="es"),
        stage=JobStage.COMPLETE, export_key=f"documents/{doc_id}/export.pdf",
    )
    monkeypatch.setattr(routes, "_fresh_export", lambda _id: (manifest, job))

    class Store:
        def get_bytes(self, key):
            return b"%PDF"

    monkeypatch.setattr(routes, "get_object_store", lambda: Store())
    monkeypatch.setattr(
        word_export, "build_office_export",
        lambda *a, **k: SimpleNamespace(
            data=b"PK", pictures_kept_pages=[1, 1], paragraphs_kept_original=3
        ),
    )
    r = client.get(f"/api/documents/{doc_id}/download", headers={"Origin": ORIGIN})
    assert r.status_code == 200, r.text
    assert r.headers["x-ct-pictures-kept"] == "2;pages=1"
    assert r.headers["x-ct-paragraphs-kept"] == "3"
    exposed = r.headers.get("access-control-expose-headers", "").lower()
    assert "x-ct-pictures-kept" in exposed and "x-ct-paragraphs-kept" in exposed
    assert len(_findings(doc_id, routes.PICTURE_NOT_REPLACED)) == 2


# --- Report ------------------------------------------------------------------
def _report(client, doc_id):
    r = client.get(f"/api/documents/{doc_id}/report")
    assert r.status_code == 200, r.text
    return r.json()


def test_report_does_not_call_a_failed_read_no_text(client):
    from ct_engine.model.enums import JobStage

    doc_id = _seed(job_stage=JobStage.COMPLETE, images=1,
                   findings=["image_text_not_read"])
    report = _report(client, doc_id)
    assert report["embedded_image_text"] == "Partially processed"
    assert report["images_with_unreadable_text"] == [1]


def test_report_says_when_picture_reading_was_turned_off(client):
    from ct_engine.model.enums import JobStage

    doc_id = _seed(job_stage=JobStage.COMPLETE, images=1, translate_image_text=False)
    assert _report(client, doc_id)["embedded_image_text"] == "Not examined (turned off)"


def test_report_counts_text_the_service_did_not_translate(client):
    from ct_engine.model.enums import JobStage

    # One block the service did not translate: it is named by its own
    # finding and by the page's untranslated_text finding. That is one block.
    doc_id = _seed(job_stage=JobStage.COMPLETE,
                   findings=["translation_unavailable", "untranslated_text"],
                   finding_objects={"translation_unavailable": ["t1"],
                                    "untranslated_text": ["t1"]})
    report = _report(client, doc_id)
    assert report["blocks_left_untranslated"] == 1
    assert report["target_language_code"] == "es"
    assert report["target_language"] == "Spanish"


def test_report_counts_each_untranslated_block_once(client):
    from ct_engine.model.enums import JobStage

    blocks = [f"t{i}" for i in range(10)]
    doc_id = _seed(job_stage=JobStage.COMPLETE,
                   findings=["translation_unavailable", "segment_untranslated",
                             "untranslated_text"],
                   finding_objects={"translation_unavailable": blocks[:9],
                                    "segment_untranslated": blocks[9:],
                                    "untranslated_text": blocks})
    assert _report(client, doc_id)["blocks_left_untranslated"] == 10


# --- Cancel ------------------------------------------------------------------
def test_cancel_unknown_document_is_404(client):
    r = client.post("/api/documents/doesnotexist0000/cancel")
    assert r.status_code == 404


@pytest.mark.parametrize("running", [True, False])
def test_cancel_asks_the_runner(client, monkeypatch, running):
    from ct_engine.api import routes

    asked = []

    class Runner:
        def cancel(self, document_id):
            asked.append(document_id)
            return running

    monkeypatch.setattr(routes, "get_job_runner", lambda: Runner())
    doc_id = _seed()
    r = client.post(f"/api/documents/{doc_id}/cancel", headers={"Origin": ORIGIN})
    assert r.status_code == 200
    assert r.json() == {"cancelled": running}
    assert asked == [doc_id]
    assert r.headers.get("access-control-allow-origin") == ORIGIN


# --- Validating from disk ----------------------------------------------------
def test_pdf_is_validated_from_a_path_without_reading_it_whole(tmp_path, monkeypatch):
    from ct_engine.ingest.validate import validate_pdf, validate_upload

    data = corpus_file("01_clean_worksheet.pdf")
    path = tmp_path / "w.pdf"
    path.write_bytes(data)

    def no_whole_read(self):
        raise AssertionError("read the whole file")

    monkeypatch.setattr(Path, "read_bytes", no_whole_read)
    from_path = validate_upload(path, "w.pdf")
    monkeypatch.undo()
    from_bytes = validate_pdf(data, "w.pdf")
    assert from_path.ok and from_bytes.ok
    assert from_path.page_count == from_bytes.page_count
    assert from_path.pdf_version == from_bytes.pdf_version


def test_path_validation_refuses_what_bytes_validation_refuses(tmp_path):
    from ct_engine.ingest.validate import validate_upload

    empty = tmp_path / "e.pdf"
    empty.write_bytes(b"")
    assert validate_upload(empty, "e.pdf").message == "That file is empty."

    fake = tmp_path / "f.pdf"
    fake.write_bytes(corpus_file("20_worksheet.docx"))
    assert "does not look like a PDF" in validate_upload(fake, "f.pdf").message

    missing = tmp_path / "gone.pdf"
    assert validate_upload(missing, "gone.pdf").ok is False


def test_office_files_validate_from_a_path_too(tmp_path):
    from ct_engine.ingest.validate import validate_upload

    docx = tmp_path / "w.docx"
    docx.write_bytes(corpus_file("20_worksheet.docx"))
    assert validate_upload(docx, "w.docx").ok
    pptx = tmp_path / "noext"
    pptx.write_bytes(corpus_file("22_lesson.pptx"))
    assert validate_upload(pptx, None).ok
    pdf = tmp_path / "noext2"
    pdf.write_bytes(corpus_file("01_clean_worksheet.pdf"))
    assert validate_upload(pdf, None).ok
