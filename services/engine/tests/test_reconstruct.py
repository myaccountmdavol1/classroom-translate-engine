"""Reconstruction, verification, and the completion gate."""

from __future__ import annotations

import fitz
import pytest

from conftest import corpus_file
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.fonts.registry import REGISTRY
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import PageStatus, TextSource, TranslationStatus
from ct_engine.pipeline.reconstruct_stage import (
    check_completion,
    reconstruct_and_export,
    verify_page,
)
from ct_engine.pipeline.translate import translate_page
from ct_engine.reconstruct.builder import rebuild_document
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider, PseudoTranslateProvider


def prepared_pages(name: str, provider=None, target="es"):
    data = corpus_file(name)
    pages = []
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            translate_page(
                page, provider or EchoProvider(), source_lang="en", target_lang=target
            )
            pages.append(page)
    return data, pages


@pytest.mark.parametrize(
    "name",
    ["01_clean_worksheet.pdf", "04_science_diagram.pdf", "05_tables.pdf",
     "08_exam_multiple_choice.pdf", "10_mixed_orientation.pdf",
     "13_rotated_pages.pdf"],
)
def test_output_page_count_always_matches_source(name):
    """The headline promise: same pages in, same pages out (spec sec. 46)."""
    data, pages = prepared_pages(name)
    out, _ = rebuild_document(data, pages, target_lang="es")
    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        assert dst.page_count == src.page_count
    finally:
        src.close()
        dst.close()


def test_output_pdf_reopens_and_has_selectable_text():
    """Rasterizing everything would be easier and is explicitly disallowed."""
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    out, _ = rebuild_document(data, pages, target_lang="es")
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert doc.page_count == 1
        assert len(doc[0].get_text().strip()) > 50
    finally:
        doc.close()


def test_page_dimensions_are_unchanged():
    data, pages = prepared_pages("10_mixed_orientation.pdf")
    out, _ = rebuild_document(data, pages, target_lang="es")
    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        for i in range(src.page_count):
            assert src[i].rect.width == pytest.approx(dst[i].rect.width, abs=0.5)
            assert src[i].rect.height == pytest.approx(dst[i].rect.height, abs=0.5)
    finally:
        src.close()
        dst.close()


def test_images_are_preserved_in_the_output():
    """Original artwork is reused, never regenerated (spec sec. 14)."""
    data, pages = prepared_pages("04_science_diagram.pdf")
    out, _ = rebuild_document(data, pages, target_lang="es")
    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        assert len(dst[0].get_images()) == len(src[0].get_images()) == 1
        src_bytes = src.extract_image(src[0].get_images()[0][0])["image"]
        dst_bytes = dst.extract_image(dst[0].get_images()[0][0])["image"]
        # Byte-identical: the asset was carried over, not re-encoded.
        assert sha256_bytes(src_bytes) == sha256_bytes(dst_bytes)
    finally:
        src.close()
        dst.close()


def test_rotated_pages_keep_their_look():
    """A /Rotate page is drawn on in its visual space (drawing takes
    unrotated coordinates; the model's boxes are visual), so its rotation is
    folded into the page: the output reads /Rotate 0 but looks the same --
    same visual size, same artwork where nothing was translated."""
    data, pages = prepared_pages("13_rotated_pages.pdf")
    out, _ = rebuild_document(data, pages, target_lang="es")
    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        for a, b in zip(src, dst):
            assert (a.rect.width, a.rect.height) == pytest.approx((b.rect.width, b.rect.height))
        # The heating-curve chart on the 90-degree page is untouched.
        clip = fitz.Rect(345, 118, 507, 397)
        before = src[1].get_pixmap(clip=clip, dpi=36).samples
        after = dst[1].get_pixmap(clip=clip, dpi=36).samples
        assert before == after
    finally:
        src.close()
        dst.close()


@pytest.mark.parametrize("index,english,angle", [
    (1, "Wide Data Table", 90.0), (3, "Rotated Answer Sheet", -90.0),
])
def test_text_turned_with_its_page_is_replaced_and_drawn_turned(index, english, angle):
    """Corpus 13: on /Rotate pages the English stayed and the translation
    was drawn level in two-letter columns. It is covered and drawn along
    the original's direction, as one readable line."""
    data, pages = prepared_pages(
        "13_rotated_pages.pdf", EchoProvider(lambda t: "XX " + t.upper())
    )
    out, _ = rebuild_document(data, pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as dst:
        page = dst[index]
        text = page.get_text()
        assert english not in text
        lines = [
            line for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if english.upper() in "".join(s["text"] for s in line["spans"])
        ]
        assert lines, text
        dx, dy = lines[0]["dir"]
        import math

        assert math.degrees(math.atan2(dy, dx)) == pytest.approx(angle, abs=1.0)


def test_pages_with_nothing_translated_are_carried_over_untouched():
    """A page we cannot improve must not be degraded."""
    data = corpus_file("02_scanned_worksheet.pdf")
    with PdfParser().open_bytes(data) as doc:
        pages = [doc.parse_page(0)]
    out, stats = rebuild_document(data, pages, target_lang="es")
    assert stats[0].written == 0 and stats[0].redacted == 0
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        assert dst.page_count == 1
        assert len(dst[0].get_images()) == 1
    finally:
        dst.close()


def test_source_text_is_removed_not_left_underneath():
    """Leaving the original text behind would make the PDF unsearchable junk.

    Uses a provider that emits no source words, so the assertion tests
    redaction rather than the test double's echo.
    """
    data, pages = prepared_pages(
        "01_clean_worksheet.pdf", provider=PseudoTranslateProvider()
    )
    out, _ = rebuild_document(data, pages, target_lang="es")
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        text = doc[0].get_text()
    finally:
        doc.close()
    assert "Explain why chlorophyll is necessary" not in text


def test_a_native_block_gets_its_background_colour_recorded():
    """Regression: the field was only ever set inside the raster-covering
    loop, so on a native PDF -- the common case -- every block came back
    None and the editor's patch fell back to white on every banner.
    """
    import re

    from ct_engine.model.enums import TextSource

    data, pages = prepared_pages("01_clean_worksheet.pdf")
    rebuild_document(data, pages, target_lang="es")
    native = [
        obj
        for page in pages
        for obj in page.text_objects
        if obj.source is TextSource.NATIVE
        and obj.translation_status
        in (TranslationStatus.TRANSLATED, TranslationStatus.MANUAL_EDIT)
        and obj.translated_text
    ]
    assert native, "fixture has no native block to assert on"
    assert all(
        obj.background_color and re.fullmatch(r"#[0-9a-f]{6}", obj.background_color)
        for obj in native
    )


def test_font_failure_keeps_source_text_rather_than_blanking_the_page(monkeypatch):
    """If we cannot draw the translation, we must not delete the original."""
    from ct_engine.fonts import registry as reg
    from ct_engine.reconstruct import builder

    data, pages = prepared_pages("01_clean_worksheet.pdf")

    def boom(self, text, *, script="Latin", bold=False, italic=False, family=None,
             look=None):
        raise reg.FontResolutionError("no font", missing="x")

    monkeypatch.setattr(reg.FontRegistry, "resolve", boom)
    out, stats = rebuild_document(data, pages, target_lang="es")
    assert stats[0].redacted == 0, "text was redacted with no replacement available"
    assert stats[0].font_failures > 0
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert "chlorophyll" in doc[0].get_text()
    finally:
        doc.close()


# --- Verification ---------------------------------------------------------
def test_clean_page_verifies_as_ready():
    _, pages = prepared_pages("01_clean_worksheet.pdf")
    qa = verify_page(pages[0])
    assert qa.status in (PageStatus.READY, PageStatus.REVIEW_SUGGESTED)
    assert qa.layout_confidence > 0.5
    assert "protected_content_intact" in qa.checks_run


def test_lost_protected_content_is_reported_as_an_error():
    _, pages = prepared_pages("03_math_worksheet.pdf")
    page = pages[0]
    # Simulate a formula vanishing from a translated block.
    target = next(
        (t for t in page.text_objects if t.translation_status is TranslationStatus.TRANSLATED),
        None,
    )
    if target is None:
        pytest.skip("no translated block on this page")
    page.protected_regions.append(
        type(page.protected_regions[0])(
            object_id=target.id, start=0, end=5, text="ZZZZ-NOT-PRESENT",
            category="math", placeholder="#9#",
        )
    )
    qa = verify_page(page)
    assert any(f.code == "protected_content_lost" for f in qa.errors)
    assert qa.status is PageStatus.REVIEW_REQUIRED


def test_verification_records_page_status_and_score():
    _, pages = prepared_pages("08_exam_multiple_choice.pdf")
    for page in pages:
        qa = verify_page(page)
        assert 0.0 <= qa.layout_confidence <= 1.0
        assert qa.status is not PageStatus.PENDING
        assert page.verified


# --- Completion gate ------------------------------------------------------
def _manifest(pages, name="x.pdf") -> DocumentManifest:
    m = DocumentManifest(
        document_id="gate", filename=name, content_type="application/pdf",
        byte_size=1, sha256="a" * 64, source_key="documents/gate/source.pdf",
    )
    m.pages = pages
    m.analysis.page_count = len(pages)
    return m


def test_completion_gate_passes_when_counts_match(store):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    gate = check_completion(manifest, export, output_opens=True)
    assert gate.passed, gate.reasons
    assert export.page_counts_match


def test_completion_gate_blocks_on_page_count_mismatch(store):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    export.output_page_count = 0  # simulate a dropped page
    gate = check_completion(manifest, export, output_opens=True)
    assert not gate.passed
    assert any("pages" in r for r in gate.reasons)


def test_completion_gate_blocks_when_output_will_not_reopen(store):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    gate = check_completion(manifest, export, output_opens=False)
    assert not gate.passed
    assert any("reopened" in r for r in gate.reasons)


def test_export_writes_translated_previews(store):
    data, pages = prepared_pages("07_reading_passage.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    assert store.exists(export.export_key)
    for page in manifest.pages:
        assert page.renders.translated_preview_key
        assert store.exists(page.renders.translated_preview_key)
        assert store.exists(page.renders.translated_thumb_key)


def test_manual_bold_overrides_the_source_in_both_directions():
    """None means 'as the source had it'. False must turn bold OFF."""
    from ct_engine.model.enums import ObjectKind
    from ct_engine.model.geometry import NormRect, Rect
    from ct_engine.model.objects import TextObject, TextStyle

    obj = TextObject(
        id="o",
        kind=ObjectKind.TEXT,
        bbox=Rect(x0=0, y0=0, x1=10, y1=10),
        norm_bbox=NormRect(x0=0, y0=0, x1=1, y1=1),
        style=TextStyle(bold=True),
    )
    assert obj.render_bold is True
    obj.manual_bold = False
    assert obj.render_bold is False
    obj.manual_bold = True
    assert obj.render_bold is True


# --- End-to-end weight fidelity (builder wiring, not just the model) ------
#
# The test above proves render_bold/render_italic compute the right value.
# It never touches builder.py, so it is not evidence that a rebuilt PDF
# actually comes back bold. These exercise rebuild_document/rebuild_page
# directly and assert on what reaches REGISTRY.resolve() -- the call that
# determines the font handed to the drawing layer -- rather than on a model
# property or on raw PDF bytes.


def _first_native_translated_block(pages):
    """The first ordinary (non-raster, non-manual) translated block on page
    1 of a prepared document -- a safe, generic target to mutate style on."""
    return next(
        t
        for t in pages[0].text_objects
        if t.translation_status == TranslationStatus.TRANSLATED
        and t.translated_text
        and t.source is TextSource.NATIVE
        and not t.is_manual_block
    )


def _spy_on_resolve(monkeypatch):
    """Wrap FontRegistry.resolve to record every call's weight request and
    the ResolvedFont it returned, while still running the real resolution.

    Real registry, not a fake: MuPDF's builtin Bold/Italic Helvetica
    variants ("hebo"/"heit"/"hebi") are base-14 fonts available on every
    platform this runs on, so for plain Latin ASCII text a genuine weight
    match is deterministic even on a machine with no matching font files
    installed -- there is no need to fake the registry to get a reliable
    answer.

    The registry is a process-wide singleton reused by the whole suite, so
    its resolution cache is reset to empty for the duration of the test:
    otherwise an earlier test resolving the same characters under the same
    weight could serve this test a stale hit (or vice versa).
    """
    from ct_engine.fonts import registry as reg

    calls: list[tuple[str, bool, bool, "reg.ResolvedFont"]] = []
    original = reg.FontRegistry.resolve

    def spy(self, text, *, script="Latin", bold=False, italic=False, family=None,
            look=None):
        result = original(
            self, text, script=script, bold=bold, italic=italic, family=family,
            look=look,
        )
        calls.append((text, bold, italic, result))
        return result

    monkeypatch.setattr(reg.FontRegistry, "resolve", spy)
    monkeypatch.setattr(REGISTRY, "_resolved", {})
    return calls


def _calls_for(calls, text):
    target = text.strip()
    return [c for c in calls if c[0].strip() == target]


def test_source_bold_block_draws_with_a_genuinely_bold_face(monkeypatch):
    """The fidelity repair this task exists for: a source-bold block must
    reach the drawing layer with a genuinely bold face, not merely report
    render_bold=True as a model property."""
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    obj = _first_native_translated_block(pages)
    obj.style.bold = True

    calls = _spy_on_resolve(monkeypatch)
    rebuild_document(data, pages, target_lang="es")

    matching = _calls_for(calls, obj.translated_text)
    assert matching, "the block's text never reached REGISTRY.resolve"
    assert all(bold for _, bold, _, _ in matching), "bold was not requested"
    resolved = matching[-1][3]
    assert resolved.weight_matched, "no bold face was found for plain Latin text"

    # A distinct face actually reached the drawing layer -- not just a flag.
    regular = REGISTRY.resolve(obj.translated_text.strip(), script="Latin")
    assert resolved.key != regular.key


def test_manual_bold_false_draws_regular_even_though_source_was_bold(monkeypatch):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    obj = _first_native_translated_block(pages)
    obj.style.bold = True
    obj.manual_bold = False

    calls = _spy_on_resolve(monkeypatch)
    rebuild_document(data, pages, target_lang="es")

    matching = _calls_for(calls, obj.translated_text)
    assert matching
    assert all(not bold for _, bold, _, _ in matching), (
        "manual_bold=False did not turn the drawn weight off"
    )


def test_manual_bold_true_draws_bold_even_though_source_was_regular(monkeypatch):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    obj = _first_native_translated_block(pages)
    obj.style.bold = False
    obj.manual_bold = True

    calls = _spy_on_resolve(monkeypatch)
    rebuild_document(data, pages, target_lang="es")

    matching = _calls_for(calls, obj.translated_text)
    assert matching
    assert all(bold for _, bold, _, _ in matching), (
        "manual_bold=True did not turn the drawn weight on"
    )


def test_source_italic_block_draws_with_a_genuinely_italic_face(monkeypatch):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    obj = _first_native_translated_block(pages)
    obj.style.italic = True

    calls = _spy_on_resolve(monkeypatch)
    rebuild_document(data, pages, target_lang="es")

    matching = _calls_for(calls, obj.translated_text)
    assert matching
    assert all(italic for _, _, italic, _ in matching)
    resolved = matching[-1][3]
    assert resolved.weight_matched, "no italic face was found for plain Latin text"
    regular = REGISTRY.resolve(obj.translated_text.strip(), script="Latin")
    assert resolved.key != regular.key


def test_unavailable_weight_falls_back_to_regular_and_is_reported(monkeypatch):
    """The never-drop-text rule: when no bold/italic face can be found, the
    text must still be drawn -- in the regular face -- and the page must
    say so, naming the block, under its own code (not the hard-failure
    "font_unavailable" a teacher would read as "this text vanished").

    `_resolve_weighted` is monkeypatched to always come up empty, rather
    than removing font files, because whether a real bold face happens to
    be installed is a platform fact this test cannot control. This forces
    the one step that searches for a weighted match to fail deterministically
    on any machine, while leaving the real fallback-to-regular path
    (`_resolve_uncached`) untouched and real -- the text is still drawn by
    genuine font resolution, just not a bold one.
    """
    from ct_engine.fonts import registry as reg

    data, pages = prepared_pages("01_clean_worksheet.pdf")
    obj = _first_native_translated_block(pages)
    obj.style.bold = True

    monkeypatch.setattr(reg.FontRegistry, "_resolve_weighted", lambda *a, **k: None)
    monkeypatch.setattr(REGISTRY, "_resolved", {})

    out, stats = rebuild_document(data, pages, target_lang="es")

    weight_findings = [
        f for f in stats[0].findings if f.code == "font_weight_unavailable"
    ]
    assert weight_findings, "no finding was raised for the missing bold face"
    assert obj.id in weight_findings[0].object_ids
    assert not any(f.code == "font_unavailable" for f in stats[0].findings), (
        "a weight substitution must not be reported as a hard font failure"
    )

    first_word = obj.translated_text.strip("<> ").split()[0]
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert first_word in doc[0].get_text(), (
            "text was dropped instead of falling back to the regular face"
        )
    finally:
        doc.close()


def test_translated_paragraph_stays_inside_the_documents_margin():
    """Regression: every wrapped block was widened toward the paper's edge.

    The widening meant for short labels also fired for any paragraph with
    blank margin beside it, so a longer translation ran to within a few
    points of the edge -- past the original margin, where printers clip.
    """
    body = fitz.Rect(72, 80, 540, 200)
    src = fitz.open()
    page = src.new_page(width=612, height=792)
    page.insert_htmlbox(
        body,
        "<p>We will be holding our traditional parent-teacher conferences on "
        "Monday for grades 7-12. Our afternoon conferences will be held from "
        "12:45 p.m. to 4:00 p.m. for grades 7 and 8. Students will follow an "
        "early release time schedule on this day.</p>",
        css="p{font-family:sans-serif;font-size:11.5pt;margin:0}",
    )
    data = src.tobytes()
    margin = max(w[2] for w in page.get_text("words"))

    pages = []
    with PdfParser().open_bytes(data) as doc:
        parsed = doc.parse_page(0)
        split_page_blocks(parsed)
        apply_reading_order(parsed)
        classify_page(parsed, is_first_page=True)
        translate_page(
            parsed,
            EchoProvider(lambda t: "TRADUCIDO " + t.upper()),
            source_lang="en",
            target_lang="es",
        )
        pages.append(parsed)

    out, _ = rebuild_document(data, pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as result:
        drawn = max(w[2] for w in result[0].get_text("words"))
    assert drawn <= margin + 1.0


def test_the_box_the_editor_shows_contains_every_word_drawn():
    """The editor's frame is what a teacher grabs. When the rebuild widens a
    box into empty space (or reflows it down), the frame must follow; it
    used to stay at the source box, with the longer translation spilling out
    of it and the handles sitting on top of the first letter."""
    longer = EchoProvider(lambda t: f"{t} {t}")
    data, pages = prepared_pages("04_science_diagram.pdf", provider=longer)
    out, _ = rebuild_document(data, pages, target_lang="es")
    page = pages[0]
    drawn = [t for t in page.text_objects if t.drawn_bbox is not None]
    assert drawn, "nothing reported where it was drawn"
    assert any(t.drawn_bbox.width > t.bbox.width + 4 for t in drawn), (
        "fixture should widen at least one block"
    )
    with fitz.open(stream=out, filetype="pdf") as doc:
        words = doc[0].get_text("words")
    for t in drawn:
        box = fitz.Rect(*t.drawn_bbox.as_tuple())
        inside = [w for w in words if fitz.Rect(w[:4]).intersects(box)]
        for w in inside:
            word = fitz.Rect(w[:4])
            # A word that starts in the frame must finish in it.
            if box.contains(fitz.Point(word.x0 + 0.5, (word.y0 + word.y1) / 2)):
                assert word.x1 <= box.x1 + 1.5, (t.translated_text, w[4], word, box)


def test_translations_printed_over_each_other_are_an_error():
    """What a teacher saw on page 33: two translations drawn on one spot.
    The check must use where they were really drawn, both ways round."""
    from ct_engine.model.enums import ObjectKind, TextSource, TranslationStatus
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan
    from ct_engine.model.document import Page
    from ct_engine.pipeline.reconstruct_stage import verify_page

    def block(oid, box, drawn):
        r = Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
        return TextObject(
            id=oid, kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(612, 792),
            source=TextSource.OCR, translated_text="译文",
            translation_status=TranslationStatus.TRANSLATED,
            lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r)])],
            drawn_bbox=Rect(x0=drawn[0], y0=drawn[1], x1=drawn[2], y1=drawn[3]),
        )

    # A short line inside a long paragraph: under 30% of the paragraph, so
    # the old one-directional check on source boxes said nothing.
    para = block("para", (72, 601, 559, 632), (72, 601, 559, 640))
    line = block("line", (72, 619, 144, 632), (72, 619, 160, 632))
    page = Page(index=0, number=33, width=612, height=792, text_objects=[para, line])
    found = [f for f in verify_page(page).findings if f.code == "collision_text"]
    assert found and found[0].severity == "error"
    assert set(found[0].object_ids) == {"para", "line"}


# --- Backstop: a page nobody translated -------------------------------------
def test_completion_gate_blocks_a_page_never_translated(store):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    # As if translation had silently skipped page 1: its words still wait.
    page = manifest.pages[0]
    page.translated = False
    waiting = next(t for t in page.text_objects if t.flat_text.strip())
    waiting.translation_status = TranslationStatus.PENDING
    waiting.keep_original = False
    waiting.suppressed = False
    gate = check_completion(manifest, export, output_opens=True)
    assert not gate.passed
    assert any("Page 1 was not translated" in r for r in gate.reasons), gate.reasons

    # A page that failed openly is already reported; the backstop is quiet.
    page.error = "Page 1 could not be translated."
    gate = check_completion(manifest, export, output_opens=True)
    assert not any("not translated" in r for r in gate.reasons)


# --- Memory: rebuilding from a file into a file -----------------------------
def test_rebuild_from_a_file_into_a_file_matches_the_bytes_route(tmp_path):
    data, pages = prepared_pages("01_clean_worksheet.pdf")
    as_bytes, _ = rebuild_document(data, pages, target_lang="es")

    source = tmp_path / "source.pdf"
    source.write_bytes(data)
    out = tmp_path / "out.pdf"
    written, stats = rebuild_document(source, pages, target_lang="es", out_path=out)

    assert written == out
    assert stats
    # The drawing file is gone; only the source and the output remain.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.pdf", "source.pdf"]
    a = fitz.open(stream=as_bytes, filetype="pdf")
    b = fitz.open(str(out))
    try:
        assert a.page_count == b.page_count
        assert [p.get_text() for p in a] == [p.get_text() for p in b]
    finally:
        a.close()
        b.close()


def test_a_stopped_rebuild_writes_nothing(tmp_path):
    from ct_engine.jobs.runner import JobCancelled

    data, pages = prepared_pages("01_clean_worksheet.pdf")
    source = tmp_path / "source.pdf"
    source.write_bytes(data)

    def stop():
        raise JobCancelled()

    with pytest.raises(JobCancelled):
        rebuild_document(
            source, pages, target_lang="es", before_page=stop,
            out_path=tmp_path / "out.pdf",
        )
    assert [p.name for p in tmp_path.iterdir()] == ["source.pdf"]


def test_subset_fonts_file_keeps_the_source_when_the_text_would_change(
    tmp_path, monkeypatch
):
    from ct_engine.reconstruct.builder import subset_fonts_file

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hola", fontname="china-s")
    src = tmp_path / "in.pdf"
    doc.save(str(src))
    doc.close()
    before = src.read_bytes()

    def corrupting(self, *args, **kwargs):
        self[0].insert_text((72, 120), "extra", fontname="helv")

    monkeypatch.setattr(fitz.Document, "subset_fonts", corrupting)
    dst = tmp_path / "out.pdf"
    assert subset_fonts_file(src, dst) is False
    assert not dst.exists()
    assert src.read_bytes() == before


def test_export_thumbnails_are_drawn_at_thumbnail_width(store):
    from ct_engine.config import get_settings

    data, pages = prepared_pages("13_rotated_pages.pdf")
    store.put_bytes("documents/gate/source.pdf", data)
    manifest = _manifest(pages)
    reconstruct_and_export(manifest, store=store, target_lang="es")
    limit = get_settings().thumbnail_width_px
    for page in manifest.pages:
        with fitz.open(stream=store.get_bytes(page.renders.translated_thumb_key)) as im:
            width = im[0].rect.width
        assert 0 < width <= limit, (page.number, width)
