"""A Word file with pictures, where nothing lays it out (the browser): its
pictures become pages, and it comes back a Word file with its paragraphs
translated."""

from __future__ import annotations

import io
import json
import re
import time
import zipfile

from docx import Document
from PIL import Image, ImageDraw, ImageFont

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ENGLISH = "Write your name at the top of the page."


def _docx() -> bytes:
    pic = io.BytesIO()
    img = Image.new("RGB", (1200, 240), "white")
    try:
        font = ImageFont.load_default(size=96)
    except TypeError:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 96)
    ImageDraw.Draw(img).text((40, 60), "HOME AGAIN", fill="black", font=font)
    img.save(pic, "PNG")
    pic.seek(0)
    d = Document()
    d.add_heading("Chapter one", level=1)
    d.add_paragraph(ENGLISH)
    d.add_picture(pic)
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


def _wait(client, doc_id, stages, seconds=180):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in stages:
            return job
        time.sleep(0.05)
    return job


def test_a_word_file_with_pictures_is_shown_through_its_pictures(client, monkeypatch):
    from ct_engine.ingest import word_render

    monkeypatch.setattr(word_render, "can_render_word", lambda: False)
    r = client.post("/api/documents", files={"file": ("lesson.docx", _docx(), DOCX)})
    assert r.status_code == 200, r.text
    # Its picture is translated now, so nothing says it will not be.
    assert not any("pictures" in w for w in r.json()["warnings"])
    doc_id = r.json()["document_id"]
    _wait(client, doc_id, ("REVIEW_READY", "COMPLETE", "FAILED"))
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    assert ManifestStore(get_store()).load_manifest(doc_id).office_pictures_only is True
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["analysis"]["page_count"] == 1  # one page: the one picture

    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    job = _wait(client, doc_id, ("COMPLETE", "FAILED"))
    assert job["stage"] == "COMPLETE", job

    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["output_format"] == "DOCX"
    assert report["pictures_shown"] == 1
    assert report["pictures_translated"] == 1
    # The picture was lettered in capitals, and so is its translation.
    m = ManifestStore(get_store()).load_manifest(doc_id)
    labels = [t.translated_text for t in m.pages[0].text_objects if t.translated_text]
    assert labels and all(t == t.upper() for t in labels), labels

    # Only pictures whose words were painted count: a label translated but
    # left as it was (as over a photo) is not in the file, so not counted.
    for t in m.pages[0].text_objects:
        t.keep_original = True
    ManifestStore(get_store()).save_manifest(m)
    after = client.get(f"/api/documents/{doc_id}/report").json()
    assert after["pictures_translated"] == 0

    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["content-type"].startswith(DOCX)
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        xml = z.read("word/document.xml").decode()
        assert any(n.startswith("word/media/") for n in z.namelist())
    words = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    # Translated straight from the Word file, not left in English.
    assert ENGLISH not in words
    assert words.strip()


def test_a_word_file_without_pictures_stays_a_list_of_its_text(client, monkeypatch):
    from ct_engine.ingest import word_render

    monkeypatch.setattr(word_render, "can_render_word", lambda: False)
    d = Document()
    d.add_paragraph(ENGLISH)
    buf = io.BytesIO()
    d.save(buf)
    r = client.post("/api/documents", files={"file": ("plain.docx", buf.getvalue(), DOCX)})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document_id"]
    _wait(client, doc_id, ("REVIEW_READY", "COMPLETE", "FAILED"))
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    _wait(client, doc_id, ("COMPLETE", "FAILED"))
    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["pictures_shown"] is None
    assert report["unit_noun"] == "section"


def test_a_plain_pdf_is_not_marked_pictures_only(client):
    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hello there")
    r = client.post("/api/documents", files={"file": ("a.pdf", pdf.tobytes(), "application/pdf")})
    assert r.status_code == 200, r.text
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    m = ManifestStore(get_store()).load_manifest(r.json()["document_id"])
    assert m.office_pictures_only is False


def _translated_word_file(client, monkeypatch) -> str:
    from ct_engine.ingest import word_render

    monkeypatch.setattr(word_render, "can_render_word", lambda: False)
    r = client.post("/api/documents", files={"file": ("lesson.docx", _docx(), DOCX)})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document_id"]
    _wait(client, doc_id, ("REVIEW_READY", "COMPLETE", "FAILED"))
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    job = _wait(client, doc_id, ("COMPLETE", "FAILED"))
    assert job["stage"] == "COMPLETE", job
    return doc_id


def test_a_word_files_paragraphs_are_translated_during_the_job(client, monkeypatch):
    """In the browser the translators are ready only during a job; a download
    after a reload must not need them."""
    doc_id = _translated_word_file(client, monkeypatch)
    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["paragraphs_kept_original"] == 0

    from ct_engine.translate import one_off

    def no_translators(*a, **kw):
        raise AssertionError("the download translated again")

    monkeypatch.setattr(one_off, "translate_many", no_translators)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert "X-CT-Paragraphs-Kept" not in dl.headers
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        xml = z.read("word/document.xml").decode()
    words = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    assert ENGLISH not in words
    assert words.strip()


def test_a_job_whose_paragraphs_cannot_be_translated_still_completes(client, monkeypatch):
    from ct_engine.translate import one_off
    from ct_engine.translate.one_off import OneOffError

    def unavailable(*a, **kw):
        raise OneOffError("No translation service is available right now.")

    monkeypatch.setattr(one_off, "translate_many", unavailable)
    doc_id = _translated_word_file(client, monkeypatch)
    report = client.get(f"/api/documents/{doc_id}/report").json()
    # Both paragraphs -- the heading and the sentence -- stay in English.
    assert report["paragraphs_kept_original"] == 2


def test_a_job_survives_paragraph_translation_breaking(client, monkeypatch):
    from ct_engine.translate import one_off

    def broken(*a, **kw):
        raise RuntimeError("memory access out of bounds")

    monkeypatch.setattr(one_off, "translate_many", broken)
    doc_id = _translated_word_file(client, monkeypatch)
    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["status"] == "COMPLETE"
    # Counted all the same: the page says both stay in English.
    assert report["paragraphs_kept_original"] == 2
    # And the download keeps their English and says so, rather than failing.
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["X-CT-Paragraphs-Kept"] == "2"


def test_a_pdf_reports_no_kept_paragraphs(client):
    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hello there")
    r = client.post("/api/documents", files={"file": ("a.pdf", pdf.tobytes(), "application/pdf")})
    doc_id = r.json()["document_id"]
    _wait(client, doc_id, ("REVIEW_READY", "COMPLETE", "FAILED"))
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    _wait(client, doc_id, ("COMPLETE", "FAILED"))
    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["paragraphs_kept_original"] is None


def _job_that_left_two_in_english(client, monkeypatch) -> str:
    from ct_engine.translate import one_off

    working = one_off.translate_many

    def broken(*a, **kw):
        raise RuntimeError("memory access out of bounds")

    monkeypatch.setattr(one_off, "translate_many", broken)
    doc_id = _translated_word_file(client, monkeypatch)
    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["paragraphs_kept_original"] == 2
    monkeypatch.setattr(one_off, "translate_many", working)
    return doc_id


def test_the_download_never_calls_a_translator_after_the_job_tried(client, monkeypatch):
    """In the browser, translating outside a job crashed the engine."""
    doc_id = _job_that_left_two_in_english(client, monkeypatch)
    from ct_engine.translate import one_off

    calls = []

    def must_not(*a, **kw):
        # Recorded as well as raised: the export survives a broken
        # translator, so the error alone would not show the call.
        calls.append(a)
        raise AssertionError("download must not translate")

    monkeypatch.setattr(one_off, "translate_many", must_not)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["X-CT-Paragraphs-Kept"] == "2"
    assert calls == []


def test_an_older_document_still_translates_at_download(client, monkeypatch):
    doc_id = _job_that_left_two_in_english(client, monkeypatch)
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    manifests = ManifestStore(get_store())
    m = manifests.load_manifest(doc_id)
    m.word_paragraphs_kept = None  # as before the job counted them
    manifests.save_manifest(m)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert "X-CT-Paragraphs-Kept" not in dl.headers
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        xml = z.read("word/document.xml").decode()
    words = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    assert ENGLISH not in words


def _stored_word_file(doc_id: str):
    from ct_engine.storage import get_store
    from ct_engine.storage import keys as K

    store = get_store()
    return (
        store.exists(K.word_final_key(doc_id, "es")),
        store.exists(K.word_final_meta_key(doc_id, "es")),
    )


def test_the_job_builds_the_word_file_and_the_download_hands_it_over(
    client, monkeypatch
):
    """In the browser, building the file at download after a reload crashed
    the engine on a long book; the job builds it, the download sends it."""
    doc_id = _translated_word_file(client, monkeypatch)
    assert _stored_word_file(doc_id) == (True, True)

    from ct_engine.pipeline import word_export

    def must_not(*a, **kw):
        raise AssertionError("the download built the file again")

    monkeypatch.setattr(word_export, "build_office_export", must_not)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["content-type"].startswith(DOCX)
    assert "lesson" in dl.headers["content-disposition"]
    assert "X-CT-Paragraphs-Kept" not in dl.headers
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        xml = z.read("word/document.xml").decode()
        assert any(n.startswith("word/media/") for n in z.namelist())
    words = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    assert ENGLISH not in words
    assert words.strip()


def test_the_stored_word_file_keeps_the_jobs_counts(client, monkeypatch):
    doc_id = _job_that_left_two_in_english(client, monkeypatch)
    from ct_engine.pipeline import word_export

    def must_not(*a, **kw):
        raise AssertionError("the download built the file again")

    monkeypatch.setattr(word_export, "build_office_export", must_not)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["X-CT-Paragraphs-Kept"] == "2"


def test_an_edit_in_the_editor_rebuilds_the_word_file(client, monkeypatch):
    """The stored file must never be handed over once she has changed
    anything it holds."""
    doc_id = _translated_word_file(client, monkeypatch)
    blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()["blocks"]
    target = next(b for b in blocks if b.get("translation"))
    patched = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{target['id']}",
        json={"text": "OTRA VEZ EN CASA"},
    )
    assert patched.status_code == 200, patched.text

    from ct_engine.pipeline import word_export

    real = word_export.build_office_export
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(word_export, "build_office_export", counted)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert calls == [1]
    with zipfile.ZipFile(io.BytesIO(dl.content)):
        pass


def test_a_new_job_replaces_the_stored_word_file(client, monkeypatch):
    from ct_engine.pipeline.word_export import stored_office_export
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store
    from ct_engine.storage import keys as K

    doc_id = _translated_word_file(client, monkeypatch)
    store = get_store()
    store.put_text(K.word_final_meta_key(doc_id, "es"), '{"stamp": "old"}')
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    job = _wait(client, doc_id, ("COMPLETE", "FAILED"))
    assert job["stage"] == "COMPLETE", job
    manifests = ManifestStore(store)
    m = manifests.load_manifest(doc_id)
    j = manifests.list_jobs(doc_id)[-1]
    assert stored_office_export(m, store=store, target_lang="es", export_key=j.export_key)


def test_a_job_finishes_when_the_word_file_cannot_be_stored(client, monkeypatch):
    from ct_engine.pipeline import word_export

    def broken(*a, **kw):
        raise RuntimeError("memory access out of bounds")

    monkeypatch.setattr(word_export, "build_office_export", broken)
    doc_id = _translated_word_file(client, monkeypatch)
    assert _stored_word_file(doc_id) == (False, False)
    monkeypatch.undo()
    from ct_engine.ingest import word_render

    monkeypatch.setattr(word_render, "can_render_word", lambda: False)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text


def test_a_pdf_stores_no_word_file(client):
    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hello there")
    r = client.post("/api/documents", files={"file": ("a.pdf", pdf.tobytes(), "application/pdf")})
    doc_id = r.json()["document_id"]
    _wait(client, doc_id, ("REVIEW_READY", "COMPLETE", "FAILED"))
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    _wait(client, doc_id, ("COMPLETE", "FAILED"))
    assert _stored_word_file(doc_id) == (False, False)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/pdf")


def test_a_box_drawn_on_the_picture_rebuilds_the_word_file(client, monkeypatch):
    doc_id = _translated_word_file(client, monkeypatch)
    page = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()
    r = client.post(
        f"/api/documents/{doc_id}/pages/1/blocks",
        json={
            "text": "NUEVO",
            "bbox": {"x0": 10, "y0": 10, "x1": page["width"] / 2, "y1": 40},
        },
    )
    assert r.status_code == 200, r.text

    from ct_engine.pipeline import word_export

    real = word_export.build_office_export
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(word_export, "build_office_export", counted)
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert calls == [1]


# -- final review fixes ----------------------------------------------------------


def test_the_document_says_whether_it_is_shown_through_its_pictures(client, monkeypatch):
    """The page tells a Word file shown through its pictures from one laid
    out as real pages (Render) by this, not by whether pages have previews."""
    doc_id = _translated_word_file(client, monkeypatch)
    assert client.get(f"/api/documents/{doc_id}").json()["office_pictures_only"] is True

    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hello there")
    r = client.post("/api/documents", files={"file": ("a.pdf", pdf.tobytes(), "application/pdf")})
    pdf_id = r.json()["document_id"]
    assert client.get(f"/api/documents/{pdf_id}").json()["office_pictures_only"] is False


def test_a_job_that_crashed_building_the_word_file_skips_it_when_resumed(
    client, monkeypatch
):
    """A WebAssembly crash while building cannot be caught: the worker dies
    and the job resumes into the same step. The mark left before building
    tells the resumed job to leave the file to the download."""
    from ct_engine.pipeline import word_export
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store
    from ct_engine.storage import keys as K

    store = get_store()
    real_paragraphs = word_export.translate_word_paragraphs

    def crashed_before(manifest, **kw):
        # As a previous attempt of this same job left it: the mark written,
        # the build never finished.
        job = ManifestStore(store).list_jobs(manifest.document_id)[-1]
        store.put_text(
            K.word_final_meta_key(manifest.document_id, "es"),
            json.dumps({"building": True, "job_id": job.job_id, "target_lang": "es"}),
        )
        return real_paragraphs(manifest, **kw)

    real_build = word_export.build_office_export
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return real_build(*a, **kw)

    monkeypatch.setattr(word_export, "translate_word_paragraphs", crashed_before)
    monkeypatch.setattr(word_export, "build_office_export", counted)
    doc_id = _translated_word_file(client, monkeypatch)
    assert calls == []  # the job did not try again
    assert _stored_word_file(doc_id)[0] is False
    m = ManifestStore(store).load_manifest(doc_id)
    j = ManifestStore(store).list_jobs(doc_id)[-1]
    assert word_export.stored_office_export(
        m, store=store, target_lang="es", export_key=j.export_key
    ) is None

    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200, dl.text
    assert calls == [1]  # built at download, as before
    with zipfile.ZipFile(io.BytesIO(dl.content)):
        pass


def test_a_mark_left_by_another_job_does_not_stop_the_build(client, monkeypatch):
    from ct_engine.pipeline import word_export
    from ct_engine.storage import get_store
    from ct_engine.storage import keys as K

    store = get_store()
    real_paragraphs = word_export.translate_word_paragraphs

    def old_mark(manifest, **kw):
        store.put_text(
            K.word_final_meta_key(manifest.document_id, "es"),
            json.dumps({"building": True, "job_id": "an-older-job", "target_lang": "es"}),
        )
        return real_paragraphs(manifest, **kw)

    monkeypatch.setattr(word_export, "translate_word_paragraphs", old_mark)
    doc_id = _translated_word_file(client, monkeypatch)
    assert _stored_word_file(doc_id) == (True, True)


def test_a_building_record_is_never_served(client, monkeypatch):
    from ct_engine.pipeline import word_export
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store
    from ct_engine.storage import keys as K

    doc_id = _translated_word_file(client, monkeypatch)
    store = get_store()
    m = ManifestStore(store).load_manifest(doc_id)
    j = ManifestStore(store).list_jobs(doc_id)[-1]
    meta_key = K.word_final_meta_key(doc_id, "es")
    meta = json.loads(store.get_text(meta_key))
    meta["building"] = True
    store.put_text(meta_key, json.dumps(meta))
    assert word_export.stored_office_export(
        m, store=store, target_lang="es", export_key=j.export_key
    ) is None


def test_a_word_file_stored_by_an_older_engine_is_rebuilt(client, monkeypatch):
    from ct_engine.pipeline import word_export
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.storage import get_store

    doc_id = _translated_word_file(client, monkeypatch)
    store = get_store()
    m = ManifestStore(store).load_manifest(doc_id)
    j = ManifestStore(store).list_jobs(doc_id)[-1]
    assert word_export.stored_office_export(
        m, store=store, target_lang="es", export_key=j.export_key
    )
    monkeypatch.setattr(
        word_export, "_WORD_EXPORT_VERSION", word_export._WORD_EXPORT_VERSION + 1
    )
    assert word_export.stored_office_export(
        m, store=store, target_lang="es", export_key=j.export_key
    ) is None


def test_a_saved_project_leaves_out_the_stored_word_file(client, monkeypatch):
    from ct_engine.pipeline.project_package import export_project
    from ct_engine.storage import get_store

    doc_id = _translated_word_file(client, monkeypatch)
    assert _stored_word_file(doc_id) == (True, True)
    package = export_project(doc_id, get_store())
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        names = z.namelist()
    assert names
    assert not [n for n in names if "word-final-" in n], names
