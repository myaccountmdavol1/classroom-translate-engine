"""The API's edges: upload limits, error shapes, CORS, download names, and
rebuilding only the pages an edit touched."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from conftest import corpus_file

ORIGIN = "http://localhost:3000"


# --- Upload limit ------------------------------------------------------------
def test_health_reports_the_upload_limit_in_megabytes(client):
    from ct_engine.config import get_settings

    body = client.get("/api/health").json()
    assert body["max_upload_mb"] == get_settings().max_upload_bytes // (1024 * 1024)
    assert body["max_upload_mb"] == 100


def test_oversized_upload_is_refused_while_streaming(client, monkeypatch):
    from ct_engine.api import routes
    from ct_engine.config import get_settings
    from ct_engine.storage import get_store

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 3 * 1024 * 1024)
    # Small chunks, so the limit is crossed part-way through the copy rather
    # than on the first read.
    monkeypatch.setattr(routes, "UPLOAD_CHUNK", 64 * 1024)
    scratch = get_store().scratch_dir()
    before = set(scratch.iterdir())

    r = client.post(
        "/api/documents",
        files={"file": ("big.pdf", b"%PDF-1.7\n" + b"0" * (4 * 1024 * 1024), "application/pdf")},
    )
    assert r.status_code == 413
    body = r.json()
    assert "3 MB limit" in body["error"]
    assert body["detail"] == body["error"]
    # Nothing left behind half-copied.
    assert set(scratch.iterdir()) == before


def test_spool_stops_at_the_limit_without_reading_the_rest(tmp_path):
    from fastapi import HTTPException

    from ct_engine.api.routes import _spool_upload

    class Endless(io.RawIOBase):
        reads = 0

        def readable(self):
            return True

        def seek(self, *_):
            return 0

        def read(self, n=-1):
            Endless.reads += 1
            return b"x" * (n if n and n > 0 else 1024)

    with pytest.raises(HTTPException) as caught:
        _spool_upload(Endless(), tmp_path, 3 * 1024 * 1024)
    assert caught.value.status_code == 413
    assert Endless.reads <= 4  # three chunks fit, the fourth is one too many
    assert list(tmp_path.iterdir()) == []


def test_pdf_upload_is_stored_byte_for_byte(client):
    import hashlib

    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    data = corpus_file("01_clean_worksheet.pdf")
    r = client.post(
        "/api/documents",
        files={"file": ("01_clean_worksheet.pdf", data, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    doc_id = r.json()["document_id"]
    store = get_store()
    manifest = ManifestStore(store).load_manifest(doc_id)
    assert store.get_bytes(manifest.source_key) == data
    assert manifest.sha256 == hashlib.sha256(data).hexdigest()
    assert manifest.byte_size == len(data)


# --- Pictures in an Office file that could not be laid out ------------------
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _without_media(data: bytes) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in src.infolist():
            if not item.filename.startswith("word/media/"):
                z.writestr(item, src.read(item.filename))
    return out.getvalue()


def test_office_pictures_left_untranslated_are_reported(client, monkeypatch):
    from ct_engine.ingest import word_render

    monkeypatch.setattr(word_render, "can_render_word", lambda: False)
    # The corpus worksheet holds a picture.
    r = client.post(
        "/api/documents",
        files={"file": ("lesson.docx", corpus_file("20_worksheet.docx"), DOCX)},
    )
    assert r.status_code == 200, r.text
    warning = next(w for w in r.json()["warnings"] if "pictures" in w)
    assert "will not be translated" in warning

    # Still there after analysis has rebuilt the document's summary.
    doc_id = r.json()["document_id"]
    import time
    for _ in range(400):
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("REVIEW_READY", "COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert warning in doc["analysis"]["warnings"]


def test_picture_check_looks_for_media_in_the_package():
    from ct_engine.api.routes import _has_pictures
    from ct_engine.model.formats import SourceFormat

    data = corpus_file("20_worksheet.docx")
    assert _has_pictures(data, SourceFormat.DOCX)
    assert not _has_pictures(_without_media(data), SourceFormat.DOCX)
    assert not _has_pictures(b"not a zip", SourceFormat.PPTX)


# --- Error shapes and CORS ---------------------------------------------------
def test_unexpected_error_is_readable_cross_origin(monkeypatch):
    from fastapi.testclient import TestClient

    from ct_engine.api import routes
    from ct_engine.main import app

    def boom(_document_id):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(routes, "_latest_job", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/api/documents/abc/job", headers={"Origin": ORIGIN})
    assert r.status_code == 500
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    body = r.json()
    assert body["error"] == body["detail"] == "Something went wrong on our side. Please try again."


def test_validation_error_says_what_to_change(client):
    r = client.patch(
        "/api/documents/abc/pages/1/blocks/b1",
        json={"text": "a" * 8001},
        headers={"Origin": ORIGIN},
    )
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], str)
    assert "8,000 characters" in body["detail"]
    assert body["error"] == body["detail"]
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_unknown_address_has_the_same_shape(client):
    r = client.get("/api/no-such-thing")
    assert r.status_code == 404
    body = r.json()
    assert isinstance(body["detail"], str) and body["detail"] == body["error"]
    assert "Not Found" not in body["detail"]


def test_route_errors_carry_detail_too(client):
    r = client.get("/api/documents/doesnotexist0000/job")
    assert r.status_code == 404
    body = r.json()
    assert body["detail"] == body["error"]


def test_content_disposition_is_exposed_to_the_page(client):
    r = client.get("/api/health", headers={"Origin": ORIGIN})
    exposed = r.headers.get("access-control-expose-headers", "")
    assert "content-disposition" in exposed.lower()


# --- Download names ----------------------------------------------------------
def test_content_disposition_carries_any_file_name():
    from urllib.parse import unquote

    from ct_engine.api.routes import _content_disposition

    header = _content_disposition("attachment", 'Bài tập "số" 1 (Spanish).pdf')
    header.encode("latin-1")  # must be sendable at all
    assert header.endswith('filename="B_i t_p _s__ 1 (Spanish).pdf"')
    star = header.split("filename*=UTF-8''", 1)[1].split(";", 1)[0]
    assert unquote(star) == 'Bài tập "số" 1 (Spanish).pdf'


def test_download_name_adds_the_language():
    from types import SimpleNamespace

    from ct_engine.api.routes import _download_name

    manifest = SimpleNamespace(filename="Fractions worksheet.pdf")
    job = SimpleNamespace(settings=SimpleNamespace(target_language="es"))
    assert _download_name(manifest, job, "pdf") == "Fractions worksheet (Spanish).pdf"
    manifest.filename = "noext"
    assert _download_name(manifest, job, "docx") == "noext (Spanish).docx"


# --- A missing document says why ----------------------------------------------
def test_missing_document_after_a_restart_says_so(client, monkeypatch):
    import datetime as dt

    from ct_engine.storage import get_store

    store = get_store()
    monkeypatch.setattr(
        store, "created_at",
        lambda: dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=40),
    )
    body = client.get("/api/documents/doesnotexist0000").json()
    assert "could not be found" in body["error"]
    assert "restarted 40 minutes ago" in body["error"]
    assert "Save project to reopen later" in body["error"]


def test_missing_document_on_an_old_store_mentions_retention(client, monkeypatch):
    import datetime as dt

    from ct_engine.storage import get_store

    store = get_store()
    monkeypatch.setattr(
        store, "created_at",
        lambda: dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30),
    )
    body = client.get("/api/documents/doesnotexist0000").json()
    assert "deleted 24 hours after" in body["error"]
    assert "restarted" not in body["error"]


def test_store_dates_itself_only_when_it_starts_empty(tmp_path):
    from ct_engine.storage.local_fs import LocalFileSystemStore

    fresh = LocalFileSystemStore(tmp_path / "fresh")
    assert fresh.created_at() is not None

    old = tmp_path / "old"
    (old / "documents" / "x").mkdir(parents=True)
    (old / "documents" / "x" / "manifest.json").write_text("{}")
    assert LocalFileSystemStore(old).created_at() is None


def test_put_file_moves_the_file_into_place(store):
    src = store.scratch_dir() / "upload.tmp"
    src.write_bytes(b"hello")
    store.put_file("documents/abc/source.pdf", src)
    assert store.get_bytes("documents/abc/source.pdf") == b"hello"
    assert not src.exists()
    # Scratch is never listed or swept as a document.
    assert list(store.list_prefix("documents")) == ["documents/abc/source.pdf"]


# --- Page images without the manifest ----------------------------------------
def test_page_images_404_plainly_when_missing(client):
    r = client.get("/api/documents/doesnotexist0000/pages/1/thumb")
    assert r.status_code == 404
    assert "No thumbnail is available for page 1." == r.json()["error"]
    assert client.get("/api/documents/doesnotexist0000/pages/0/preview").status_code == 404
    assert client.get("/api/documents/bad..id/pages/1/plate").status_code == 404


# --- Download after an edit rebuilds only the edited page --------------------
def test_download_after_edit_rebuilds_only_the_edited_page(store, monkeypatch):
    import fitz

    from ct_engine.pipeline import reconstruct_stage as R
    from ct_engine.pipeline.edit import apply_edit
    from test_edit import first_editable, prepared

    manifest = prepared(store, "03_math_worksheet.pdf")
    if len(manifest.pages) < 2:
        manifest = prepared(store, "10_mixed_orientation.pdf")
    assert len(manifest.pages) >= 2
    target = first_editable(manifest, 1)
    apply_edit(manifest, 1, target.id, "ZEBRA QUOKKA", store=store, target_lang="es")

    calls: list[int] = []
    real = R.rebuild_page

    def counting(pdf_page, page, **kw):
        calls.append(page.index)
        return real(pdf_page, page, **kw)

    def no_full_rebuild(*_a, **_k):
        raise AssertionError("the whole document was rebuilt")

    monkeypatch.setattr(R, "rebuild_page", counting)
    monkeypatch.setattr(R, "rebuild_document", no_full_rebuild)
    export = R.rebuild_export_only(manifest, store=store, target_lang="es")

    assert calls == [0]
    assert export.output_page_count == len(manifest.pages)
    doc = fitz.open(str(store.local_path(export.export_key)))
    try:
        assert doc.page_count == len(manifest.pages)
        assert "ZEBRA" in doc[0].get_text()
    finally:
        doc.close()

    # Nothing changed since: the next stale download has nothing to patch,
    # which is treated as "rebuild it all" rather than trusted.
    assert R._stale_pages(manifest, store, "es") == []


def test_without_a_record_the_whole_file_is_rebuilt(store, monkeypatch):
    from ct_engine.pipeline import reconstruct_stage as R
    from ct_engine.pipeline.edit import apply_edit
    from ct_engine.storage import keys as K
    from test_edit import first_editable, prepared

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_edit(manifest, 1, target.id, "ZEBRA", store=store, target_lang="es")
    store.delete(K.export_state_key(manifest.document_id))

    used: list[bool] = []
    real = R.rebuild_document
    monkeypatch.setattr(
        R, "rebuild_document",
        lambda *a, **k: (used.append(True), real(*a, **k))[1],
    )
    R.rebuild_export_only(manifest, store=store, target_lang="es")
    assert used == [True]
    # ...and the record is written again for next time.
    state = json.loads(store.get_text(K.export_state_key(manifest.document_id)))
    assert state["target_lang"] == "es"
