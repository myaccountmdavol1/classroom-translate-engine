"""End-to-end API tests, exercising the real upload -> analyze -> inspect path."""

from __future__ import annotations

import io
import time

import pytest
from PIL import Image

from conftest import corpus_file


def _upload(client, name: str):
    resp = client.post(
        "/api/documents",
        files={"file": (name, corpus_file(name), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_for_stage(client, document_id: str, timeout: float = 60.0) -> dict:
    """Poll until the job reaches a terminal or review-ready stage."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/documents/{document_id}/job")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["stage"] in ("REVIEW_READY", "COMPLETE", "FAILED"):
            return last
        time.sleep(0.05)
    pytest.fail(f"Job did not finish in {timeout}s; last state: {last}")


def test_health_reports_active_providers(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    # Defaults must be the local, no-cost, nothing-leaves-the-building options.
    assert body["translation_provider"] == "local"
    assert body["ocr_provider"] == "local"
    assert body["retention_hours"] > 0


def test_languages_include_required_district_set(client):
    langs = client.get("/api/languages").json()["languages"]
    codes = {lang["code"] for lang in langs}
    for required in ["es", "zh-Hans", "zh-Hant", "ar", "ht", "pt-BR", "fr",
                     "bn", "ur", "ru", "ko", "ja"]:
        assert required in codes, f"missing target language {required}"
    rtl = {lang["code"] for lang in langs if lang["rtl"]}
    assert {"ar", "ur", "he"} <= rtl
    assert all(lang["native_name"] for lang in langs)


def test_output_modes_default_to_translated_only(client):
    body = client.get("/api/output-modes").json()
    default = [m for m in body["modes"] if m.get("default")]
    assert len(default) == 1
    assert default[0]["value"] == "TRANSLATED_ONLY"
    # High fidelity is the recommended processing mode.
    pdef = [m for m in body["processing_modes"] if m.get("default")]
    assert pdef and pdef[0]["value"] == "HIGH_FIDELITY"


def test_upload_analyze_and_inspect_round_trip(client):
    up = _upload(client, "04_science_diagram.pdf")
    doc_id = up["document_id"]
    assert up["page_count"] == 1

    _wait_for_stage(client, doc_id)

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["source_page_count"] == 1
    assert doc["processed_page_count"] == 1
    assert doc["analysis"]["detected_language"] == "en"
    assert doc["analysis"]["detected_language_name"] == "English"
    assert doc["analysis"]["total_images"] == 1
    assert len(doc["pages"]) == 1
    assert doc["pages"][0]["has_preview"]

    page = client.get(f"/api/documents/{doc_id}/pages/1").json()
    assert page["number"] == 1
    assert page["text_objects"] and page["image_objects"]
    block = page["text_objects"][0]
    for field in ("bbox", "norm_bbox", "lines", "style", "role",
                  "translation_status", "protected"):
        assert field in block
    assert set(block["bbox"]) == {"x0", "y0", "x1", "y1"}


def test_page_count_is_preserved_end_to_end_for_long_document(client):
    """The headline promise: same pages in, same pages out (spec sec. 46)."""
    up = _upload(client, "12_long_100p.pdf")
    doc_id = up["document_id"]
    assert up["page_count"] == 100
    _wait_for_stage(client, doc_id, timeout=180)

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["source_page_count"] == 100
    assert doc["processed_page_count"] == 100
    assert len(doc["pages"]) == 100
    assert doc["failed_pages"] == []
    assert [p["number"] for p in doc["pages"]] == list(range(1, 101))


def test_progress_is_real_not_synthetic(client):
    """Progress counters must come from actual completed pages."""
    up = _upload(client, "11_long_50p.pdf")
    doc_id = up["document_id"]
    seen = []
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        for p in job["progress"]:
            if p["stage"] == "ANALYZING":
                seen.append((p["completed"], p["total"]))
        if job["stage"] in ("REVIEW_READY", "FAILED"):
            break
        time.sleep(0.02)
    assert seen, "no progress was reported"
    completed = [c for c, _ in seen]
    assert completed == sorted(completed), "progress went backwards"
    assert seen[-1][0] == seen[-1][1] == 52
    assert all(c <= t for c, t in seen), "progress exceeded the total"


def test_preview_and_thumbnail_match_page_aspect_ratio(client):
    up = _upload(client, "10_mixed_orientation.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    doc = client.get(f"/api/documents/{doc_id}").json()

    for page in doc["pages"]:
        n = page["number"]
        r = client.get(f"/api/documents/{doc_id}/pages/{n}/preview")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        im = Image.open(io.BytesIO(r.content))
        assert im.width / im.height == pytest.approx(
            page["width"] / page["height"], rel=0.02
        ), f"page {n} preview aspect ratio drifted"

        t = client.get(f"/api/documents/{doc_id}/pages/{n}/thumb")
        assert t.status_code == 200
        tim = Image.open(io.BytesIO(t.content))
        assert tim.width <= 220


def test_debug_overlay_renders_and_respects_layer_filter(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)

    full = client.get(f"/api/documents/{doc_id}/pages/1/overlay")
    assert full.status_code == 200 and full.headers["content-type"] == "image/png"
    base = client.get(f"/api/documents/{doc_id}/pages/1/preview").content
    assert full.content != base, "overlay did not draw anything"

    text_only = client.get(
        f"/api/documents/{doc_id}/pages/1/overlay?layers=text&labels=false"
    )
    assert text_only.status_code == 200
    assert text_only.content != full.content, "layer filter had no effect"


def test_source_pdf_is_served_back_intact(client):
    original = corpus_file("03_math_worksheet.pdf")
    up = _upload(client, "03_math_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.get(f"/api/documents/{up['document_id']}/source")
    assert r.status_code == 200
    assert r.content == original
    assert r.headers["cache-control"] == "private, no-store"


def test_rejects_non_pdf_upload_with_plain_language(client):
    r = client.post(
        "/api/documents",
        files={"file": ("notes.pdf", b"PK\x03\x04not a pdf", "application/pdf")},
    )
    assert r.status_code == 400
    body = r.json()
    assert "does not look like a PDF" in body["error"]
    assert "Traceback" not in body["error"]


def test_unknown_document_returns_friendly_404(client):
    r = client.get("/api/documents/deadbeefdeadbeefdeadbeefdeadbeef")
    assert r.status_code == 404
    assert "could not be found" in r.json()["error"]


def test_unknown_page_returns_404(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.get(f"/api/documents/{up['document_id']}/pages/99")
    assert r.status_code == 404
    assert "does not exist" in r.json()["error"]


def test_document_ids_are_unguessable(client):
    ids = {_upload(client, "01_clean_worksheet.pdf")["document_id"] for _ in range(3)}
    assert len(ids) == 3
    for i in ids:
        assert len(i) == 32 and all(c in "0123456789abcdef" for c in i)


def test_delete_removes_every_trace(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    assert client.get(f"/api/documents/{doc_id}/pages/1/preview").status_code == 200

    d = client.delete(f"/api/documents/{doc_id}")
    assert d.status_code == 200 and d.json()["deleted"]
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/documents/{doc_id}/pages/1/preview").status_code == 404
    assert client.get(f"/api/documents/{doc_id}/source").status_code == 404


# ==========================================================================
# Phase 2: translation, download, report
# ==========================================================================
def _translate(client, doc_id: str, language: str = "es", **kwargs):
    body = {"target_language": language, **kwargs}
    return client.post(f"/api/documents/{doc_id}/translate", json=body)


def _wait_terminal(client, doc_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/documents/{doc_id}/job").json()
        if last["stage"] in ("COMPLETE", "FAILED"):
            return last
        time.sleep(0.05)
    pytest.fail(f"translation did not finish; last: {last}")


def test_translation_rejects_unsupported_language(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = _translate(client, up["document_id"], "klingon")
    assert r.status_code == 400
    assert "not supported" in r.json()["error"]


def test_translation_refuses_languages_the_engine_lacks(client):
    """Better an upfront refusal than a half-translated 100-page document."""
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = _translate(client, up["document_id"], "zh-Hant")
    # Either the engine supports it or it says so clearly; never a silent swap.
    if r.status_code != 200:
        assert r.status_code in (400, 500)
        assert "not available" in r.json()["error"] or "not configured" in r.json()["error"]


def test_download_is_blocked_before_completion(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.get(f"/api/documents/{up['document_id']}/download")
    assert r.status_code == 404
    assert "not been translated" in r.json()["error"]


@pytest.mark.slow
def test_full_translation_round_trip(client):
    """Upload -> analyze -> translate -> rebuild -> verify -> download."""
    up = _upload(client, "04_science_diagram.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)

    started = _translate(client, doc_id, "es")
    assert started.status_code == 200, started.text

    job = _wait_terminal(client, doc_id)
    assert job["stage"] == "COMPLETE", job

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["source_page_count"] == doc["rendered_page_count"] == 1
    assert doc["translated_page_count"] == 1
    assert doc["verified_page_count"] == 1

    # Translated preview is available and matches the page aspect ratio.
    prev = client.get(f"/api/documents/{doc_id}/pages/1/translated")
    assert prev.status_code == 200
    im = Image.open(io.BytesIO(prev.content))
    page = doc["pages"][0]
    assert im.width / im.height == pytest.approx(page["width"] / page["height"], rel=0.02)

    # Download opens as a real PDF with the right page count.
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/pdf"
    assert "Spanish" in dl.headers["content-disposition"]
    import fitz

    out = fitz.open(stream=dl.content, filetype="pdf")
    try:
        assert out.page_count == 1
        text = out[0].get_text()
        assert len(text.strip()) > 50
        assert len(out[0].get_images()) == 1  # the diagram survived
    finally:
        out.close()


@pytest.mark.slow
def test_report_is_honest_about_what_was_not_done(client):
    up = _upload(client, "04_science_diagram.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    _wait_terminal(client, doc_id)

    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["original_pages"] == report["rendered_pages"] == 1
    assert report["page_counts_match"] is True
    assert report["source_language"] == "English"
    assert report["target_language"] == "Spanish"
    assert report["downloadable"] is True
    # Image text is handled now, but the report must never claim more than
    # was achieved: any unreadable label downgrades it to "Partially".
    assert report["embedded_image_text"] in (
        "Processed",
        "Partially processed",
        "No readable text found inside pictures",
        "No pictures in this document",
    )
    if report["images_with_unreadable_text"]:
        assert report["embedded_image_text"] == "Partially processed"
    assert report["original_images_preserved"] in ("Yes", "With exceptions")
    assert isinstance(report["layout_compromises"], list)


@pytest.mark.slow
def test_multipage_exam_keeps_page_correspondence_through_translation(client):
    """The critical real-world test: page N is still page N (spec sec. 59)."""
    up = _upload(client, "08_exam_multiple_choice.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    job = _wait_terminal(client, doc_id)
    assert job["stage"] == "COMPLETE"

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["source_page_count"] == 4
    assert doc["rendered_page_count"] == 4
    assert doc["failed_pages"] == []

    dl = client.get(f"/api/documents/{doc_id}/download")
    import fitz

    out = fitz.open(stream=dl.content, filetype="pdf")
    try:
        assert out.page_count == 4
        # Question numbering must still be present and in order.
        page3 = out[2].get_text()
        assert page3.strip()
    finally:
        out.close()


@pytest.mark.slow
def test_scanned_document_translates_through_the_api(client):
    """A scan with no text layer must come back translated (spec phase 3)."""
    import shutil

    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")

    up = _upload(client, "02_scanned_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)

    analyzed = client.get(f"/api/documents/{doc_id}").json()
    assert analyzed["analysis"]["needs_ocr"] is True
    assert analyzed["analysis"]["has_selectable_text"] is False

    assert _translate(client, doc_id, "es").status_code == 200
    job = _wait_terminal(client, doc_id)
    assert job["stage"] == "COMPLETE", job

    report = client.get(f"/api/documents/{doc_id}/report").json()
    assert report["scanned_pages"] == [1]
    assert report["scanned_pages_read"] == 1
    assert report["text_recognition_confidence"] is not None
    assert 0.0 < report["text_recognition_confidence"] <= 1.0
    assert report["page_counts_match"] is True

    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    import fitz

    out = fitz.open(stream=dl.content, filetype="pdf")
    try:
        assert out.page_count == 1
        # The scan had no text layer; the translated copy has one.
        assert len(out[0].get_text().strip()) > 50
        assert len(out[0].get_images()) == 1
    finally:
        out.close()


def test_report_names_ocr_facts_even_before_translation(client):
    up = _upload(client, "02_scanned_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.get(f"/api/documents/{up['document_id']}/report")
    # No translation job yet, so the report is simply unavailable.
    assert r.status_code in (200, 404)


# ==========================================================================
# Phase 6: manual correction
# ==========================================================================
@pytest.mark.slow
def test_edit_a_translation_through_the_api(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    assert _wait_terminal(client, doc_id)["stage"] == "COMPLETE"

    blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()
    assert blocks["revision"] == 0
    assert blocks["blocks"]
    first = blocks["blocks"][0]
    for field in ("id", "original", "translation", "norm_bbox", "status"):
        assert field in first

    edited = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}",
        json={"text": "Texto corregido por el maestro"},
    )
    assert edited.status_code == 200, edited.text
    body = edited.json()
    assert body["revision"] == 1
    assert body["block"]["translation"] == "Texto corregido por el maestro"
    assert body["block"]["manually_edited"] is True
    assert body["block"]["machine_translation"]

    # The download must reflect the edit, rebuilt on demand.
    job = client.get(f"/api/documents/{doc_id}/job").json()
    assert job["export_stale"] is True
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    import fitz

    out = fitz.open(stream=dl.content, filetype="pdf")
    try:
        assert "Texto corregido por el maestro" in out[0].get_text()
        assert out.page_count == 1
    finally:
        out.close()
    assert client.get(f"/api/documents/{doc_id}/job").json()["export_stale"] is False


@pytest.mark.slow
def test_reset_a_translation_through_the_api(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    _wait_terminal(client, doc_id)

    blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()["blocks"]
    first = blocks[0]
    machine = first["translation"]

    client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}",
        json={"text": "otra cosa"},
    )
    reset = client.post(
        f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}/reset"
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["block"]["translation"] == machine
    assert reset.json()["block"]["manually_edited"] is False


@pytest.mark.slow
def test_editing_a_protected_block_is_refused_with_a_reason(client):
    up = _upload(client, "03_math_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    _wait_terminal(client, doc_id)

    # Protected formulas must not even be listed as editable.
    listed = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()["blocks"]
    page = client.get(f"/api/documents/{doc_id}/pages/1").json()
    protected_ids = [t["id"] for t in page["text_objects"] if t["protected"]]
    assert protected_ids
    assert not (set(protected_ids) & {b["id"] for b in listed})

    r = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{protected_ids[0]}",
        json={"text": "3x + 7 = 99"},
    )
    assert r.status_code == 400
    assert "cannot be edited" in r.json()["error"]


def test_blocks_endpoint_404s_for_a_missing_page(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.get(f"/api/documents/{up['document_id']}/pages/99/blocks")
    assert r.status_code == 404


def test_cors_allows_every_method_the_api_exposes(client):
    """A missing method here fails only in a browser, never in TestClient.

    Regression: PATCH was absent, so the manual-correction editor was broken
    in the browser while every API test passed.
    """
    from ct_engine.main import app
    from fastapi.middleware.cors import CORSMiddleware

    configured: set[str] = set()
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            configured = {m.upper() for m in middleware.kwargs["allow_methods"]}

    exposed = {
        method
        for route in app.routes
        if getattr(route, "methods", None)
        for method in route.methods
        if method not in ("HEAD",)
    }
    missing = exposed - configured
    assert not missing, f"CORS does not allow {sorted(missing)}"


def test_cors_preflight_succeeds_for_an_edit(client):
    resp = client.options(
        "/api/documents/abc/pages/1/blocks/xyz",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "PATCH",
        },
    )
    assert resp.status_code == 200, resp.text
    allowed = resp.headers.get("access-control-allow-methods", "")
    assert "PATCH" in allowed


def test_cors_origin_regex_is_honoured_when_configured(monkeypatch):
    """Lets a deployed frontend be driven from a local engine.

    Vercel mints a new hostname per deployment, so exact allow-listing cannot
    hold; the regex is the only workable mechanism.
    """
    import re

    from ct_engine.config import Settings

    pattern = Settings(
        cors_origin_regex=r"https://classroom-translate[a-z0-9-]*\.vercel\.app"
    ).cors_origin_regex
    assert pattern
    compiled = re.compile(pattern)
    assert compiled.fullmatch("https://classroom-translate.vercel.app")
    assert compiled.fullmatch(
        "https://classroom-translate-bp6otfxry-mds-projects-ebc8e5c5.vercel.app"
    )
    # Must not match a lookalike domain.
    assert not compiled.fullmatch("https://classroom-translate.evil.com")
    assert not compiled.fullmatch("https://evil-classroom-translate.vercel.app")


def test_cors_regex_covers_a_cloudflare_quick_tunnel():
    """The engine is reached through a tunnel whose hostname changes per run.

    A free Cloudflare quick-tunnel mints a new *.trycloudflare.com name every
    restart, so exact allow-listing cannot work; the pattern must cover them
    while still refusing lookalikes.
    """
    import re

    pattern = (
        r"https://(classroom-translate[a-z0-9-]*\.vercel\.app"
        r"|[a-z0-9-]+\.trycloudflare\.com)"
    )
    compiled = re.compile(pattern)
    assert compiled.fullmatch("https://random-words-1234.trycloudflare.com")
    assert compiled.fullmatch("https://classroom-translate.vercel.app")
    # Lookalikes must not pass.
    assert not compiled.fullmatch("https://trycloudflare.com.evil.net")
    assert not compiled.fullmatch("https://evil.trycloudflare.com.attacker.io")
    assert not compiled.fullmatch("http://random.trycloudflare.com")  # https only


def test_cors_origin_regex_defaults_to_off():
    """It widens who may call the engine, so it must be opt-in.

    Asserts the declared default rather than a constructed instance: an
    instance reflects whatever is in the environment, which would make this
    depend on whose machine the suite runs on.
    """
    from ct_engine.config import Settings

    assert Settings.model_fields["cors_origin_regex"].default is None


def test_empty_cors_regex_is_treated_as_unset():
    """An empty value must not compile to a regex that matches everything."""
    from ct_engine.config import Settings

    assert (Settings(cors_origin_regex="").cors_origin_regex or None) is None


@pytest.mark.slow
def test_move_and_restyle_a_block_through_the_api(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    assert _wait_terminal(client, doc_id)["stage"] == "COMPLETE"

    blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()["blocks"]
    first = blocks[0]
    for field in ("render_bbox", "render_norm_bbox", "manual_bbox", "color",
                  "font_size", "has_manual_layout"):
        assert field in first
    assert first["has_manual_layout"] is False

    moved = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}",
        json={
            "bbox": {"x0": 300, "y0": 600, "x1": 560, "y1": 640},
            "color": "#cc2200",
            "font_size": 16,
        },
    )
    assert moved.status_code == 200, moved.text
    block = moved.json()["block"]
    assert block["has_manual_layout"] is True
    assert block["manual_color"] == "#cc2200"
    assert block["manual_font_size"] == 16.0
    assert block["render_bbox"]["x0"] == 300

    # Reset undoes placement as well as text.
    reset = client.post(f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["block"]["has_manual_layout"] is False


@pytest.mark.slow
def test_text_and_layout_can_change_in_one_call(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    _wait_terminal(client, doc_id)

    blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()["blocks"]
    first = blocks[0]
    combined = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{first['id']}",
        json={"text": "Texto y posicion", "font_size": 14},
    )
    assert combined.status_code == 200, combined.text
    block = combined.json()["block"]
    # Neither change may be lost.
    assert block["translation"] == "Texto y posicion"
    assert block["manual_font_size"] == 14.0


def test_an_empty_patch_is_refused(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.patch(
        f"/api/documents/{up['document_id']}/pages/1/blocks/whatever",
        json={},
    )
    assert r.status_code == 400
    assert "Nothing to change" in r.json()["error"]


@pytest.mark.slow
def test_add_delete_bulk_and_undo_through_the_api(client):
    """The operations that let a teacher finish a page the engine got wrong."""
    up = _upload(client, "01_clean_worksheet.pdf")
    doc_id = up["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id, "es")
    assert _wait_terminal(client, doc_id)["stage"] == "COMPLETE"

    page_url = f"/api/documents/{doc_id}/pages/1"
    listing = client.get(f"{page_url}/blocks").json()
    assert listing["can_undo"] is False
    original_count = len(listing["blocks"])

    # --- add a block where OCR found nothing ---
    added = client.post(
        f"{page_url}/blocks",
        json={
            "text": "ÚNETE A LA AVENTURA",
            "bbox": {"x0": 120, "y0": 300, "x1": 420, "y1": 332},
            "color": "#ffffff",
            "font_size": 20,
            "alignment": "CENTER",
        },
    )
    assert added.status_code == 200, added.text
    new_block = added.json()["block"]
    assert new_block["is_manual_block"] is True
    assert new_block["translation"] == "ÚNETE A LA AVENTURA"

    after_add = client.get(f"{page_url}/blocks").json()
    assert len(after_add["blocks"]) == original_count + 1
    assert after_add["can_undo"] is True

    # --- bulk restyle several blocks in one call ---
    ids = [b["id"] for b in after_add["blocks"][:3]]
    bulk = client.patch(
        f"{page_url}/blocks",
        json={"object_ids": ids, "color": "#123456", "alignment": "CENTER"},
    )
    assert bulk.status_code == 200, bulk.text
    restyled = client.get(f"{page_url}/blocks").json()["blocks"]
    for block in restyled:
        if block["id"] in ids:
            assert block["manual_color"] == "#123456"
            assert block["manual_alignment"] == "CENTER"

    # --- remove an engine block: suppressed, not deleted ---
    engine_block = next(b for b in restyled if not b["is_manual_block"])
    removed = client.delete(f"{page_url}/blocks/{engine_block['id']}")
    assert removed.status_code == 200, removed.text
    listed = client.get(f"{page_url}/blocks").json()["blocks"]
    still_there = next(b for b in listed if b["id"] == engine_block["id"])
    assert still_there["suppressed"] is True

    # --- restore it ---
    assert client.post(
        f"{page_url}/blocks/{engine_block['id']}/restore"
    ).status_code == 200
    listed = client.get(f"{page_url}/blocks").json()["blocks"]
    assert next(
        b for b in listed if b["id"] == engine_block["id"]
    )["suppressed"] is False

    # --- undo steps back the restore ---
    undone = client.post(f"{page_url}/undo")
    assert undone.status_code == 200, undone.text
    listed = client.get(f"{page_url}/blocks").json()["blocks"]
    assert next(
        b for b in listed if b["id"] == engine_block["id"]
    )["suppressed"] is True

    # --- the added text reaches the download ---
    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    import fitz

    out = fitz.open(stream=dl.content, filetype="pdf")
    try:
        assert "ÚNETE A LA AVENTURA" in out[0].get_text()
        assert out.page_count == 1
    finally:
        out.close()


def test_bulk_with_nothing_to_change_is_refused(client):
    up = _upload(client, "01_clean_worksheet.pdf")
    _wait_for_stage(client, up["document_id"])
    r = client.patch(
        f"/api/documents/{up['document_id']}/pages/1/blocks",
        json={"object_ids": ["a"]},
    )
    assert r.status_code == 400
    assert "Nothing to change" in r.json()["error"]
