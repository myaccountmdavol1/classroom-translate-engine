"""Manual correction of one translation (spec sec. 36)."""

from __future__ import annotations

import time

import fitz
import pytest

from conftest import corpus_file
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import TranslationStatus
from ct_engine.pipeline.edit import (
    EditError,
    apply_edit,
    editable_blocks,
    reset_edit,
)
from ct_engine.pipeline.reconstruct_stage import reconstruct_and_export
from ct_engine.pipeline.translate import translate_page
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider


def prepared(store, name="01_clean_worksheet.pdf"):
    data = corpus_file(name)
    store.put_bytes("documents/edit/source.pdf", data)
    manifest = DocumentManifest(
        document_id="edit",
        filename=name,
        content_type="application/pdf",
        byte_size=len(data),
        sha256=sha256_bytes(data),
        source_key="documents/edit/source.pdf",
    )
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
            manifest.pages.append(page)
    manifest.analysis.page_count = len(manifest.pages)
    reconstruct_and_export(manifest, store=store, target_lang="es")
    return manifest


def first_editable(manifest, page_number=1):
    return editable_blocks(manifest.page(page_number - 1))[0]


# --- Basics ---------------------------------------------------------------
def test_editable_blocks_excludes_protected(store):
    manifest = prepared(store, "03_math_worksheet.pdf")
    page = manifest.page(0)
    blocks = editable_blocks(page)
    assert blocks
    assert all(not b.protected for b in blocks)
    # The formula blocks on this worksheet must not be offered for editing.
    assert any(t.protected for t in page.text_objects)


def test_edit_replaces_the_translation(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    machine = target.translated_text

    result = apply_edit(
        manifest, 1, target.id, "Texto corregido", store=store, target_lang="es"
    )
    assert target.translated_text == "Texto corregido"
    assert target.manually_edited
    assert target.translation_status is TranslationStatus.MANUAL_EDIT
    # The machine output is retained so Reset needs no translator call.
    assert target.machine_translated_text == machine
    assert result.page.revision == 1


def test_reset_restores_the_machine_translation(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    machine = target.translated_text

    apply_edit(manifest, 1, target.id, "Algo distinto", store=store, target_lang="es")
    reset_edit(manifest, 1, target.id, store=store, target_lang="es")

    assert target.translated_text == machine
    assert not target.manually_edited
    assert target.translation_status is TranslationStatus.TRANSLATED


def test_reset_without_a_prior_edit_is_refused(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    with pytest.raises(EditError):
        reset_edit(manifest, 1, target.id, store=store, target_lang="es")


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_edits_are_refused(store, text):
    manifest = prepared(store)
    target = first_editable(manifest)
    with pytest.raises(EditError):
        apply_edit(manifest, 1, target.id, text, store=store, target_lang="es")


def test_overlong_edits_are_refused(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    with pytest.raises(EditError):
        apply_edit(manifest, 1, target.id, "x" * 9000, store=store, target_lang="es")


def test_protected_blocks_cannot_be_edited(store):
    """A formula must stay exactly as it is, so it is not editable at all."""
    manifest = prepared(store, "03_math_worksheet.pdf")
    protected = next(t for t in manifest.page(0).text_objects if t.protected)
    with pytest.raises(EditError) as exc:
        apply_edit(manifest, 1, protected.id, "3x + 7 = 99", store=store,
                   target_lang="es")
    assert "cannot be edited" in exc.value.message


def test_unknown_object_and_page_are_refused(store):
    manifest = prepared(store)
    with pytest.raises(EditError):
        apply_edit(manifest, 1, "text_doesnotexist", "hola", store=store,
                   target_lang="es")
    with pytest.raises(EditError):
        apply_edit(manifest, 99, "whatever", "hola", store=store, target_lang="es")


# --- Rendering and scope --------------------------------------------------
def test_edit_rerenders_only_the_edited_page(store):
    """A one-word fix must not re-render a whole document (spec sec. 36)."""
    manifest = prepared(store, "08_exam_multiple_choice.pdf")
    target = first_editable(manifest, page_number=2)

    apply_edit(manifest, 2, target.id, "Corregido", store=store, target_lang="es")

    assert manifest.page(1).revision == 1
    for index in (0, 2, 3):
        assert manifest.page(index).revision == 0


def test_edit_is_fast_on_a_large_document(store):
    """The editor has to feel immediate, so the export is rebuilt lazily."""
    manifest = prepared(store, "11_long_50p.pdf")
    target = first_editable(manifest, page_number=25)

    start = time.perf_counter()
    apply_edit(manifest, 25, target.id, "Correccion rapida", store=store,
               target_lang="es")
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"a single edit took {elapsed:.1f}s"


def test_edited_text_appears_in_the_rebuilt_page(store):
    from ct_engine.pipeline.reconstruct_stage import rebuild_export_only

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_edit(manifest, 1, target.id, "FRASECORREGIDA", store=store,
               target_lang="es")

    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        assert "FRASECORREGIDA" in doc[0].get_text()
    finally:
        doc.close()


def test_edit_reruns_page_qa(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    result = apply_edit(manifest, 1, target.id, "Corto", store=store,
                        target_lang="es")
    page = manifest.page(0)
    assert page.verified
    assert page.qa.checks_run
    assert result.status == page.qa.status
    assert 0.0 <= result.layout_confidence <= 1.0


def test_an_overlong_edit_is_reported_not_silently_clipped(store):
    """A correction that cannot fit must warn rather than look fine."""
    manifest = prepared(store)
    page = manifest.page(0)
    # A short block with a very long replacement.
    target = min(editable_blocks(page), key=lambda t: t.bbox.area)
    result = apply_edit(
        manifest,
        1,
        target.id,
        "palabra " * 120,
        store=store,
        target_lang="es",
    )
    assert result.warnings or result.status.value != "READY"


def test_protected_content_is_rechecked_after_an_edit(store):
    """A human must not be able to quietly delete a preserved formula."""
    manifest = prepared(store, "03_math_worksheet.pdf")
    page = manifest.page(0)
    editable = editable_blocks(page)
    target = editable[0]
    # Attribute a protected value to this block, then edit it away.
    page.protected_regions.append(
        type(page.protected_regions[0])(
            object_id=target.id, start=0, end=5, text="ZZZ-MUST-SURVIVE",
            category="math", placeholder="#9#",
        )
    )
    result = apply_edit(manifest, 1, target.id, "sin la formula", store=store,
                        target_lang="es")
    assert result.lost_protected, "losing a protected value was not reported"
    assert result.status.value == "REVIEW_REQUIRED"


def test_manual_edits_survive_a_later_translation_pass(store):
    """Re-running translation must not discard a teacher's correction."""
    manifest = prepared(store)
    page = manifest.page(0)
    target = editable_blocks(page)[0]
    apply_edit(manifest, 1, target.id, "MIA Y DEFINITIVA", store=store,
               target_lang="es")

    translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
    assert target.translated_text == "MIA Y DEFINITIVA"
    assert target.translation_status is TranslationStatus.MANUAL_EDIT


def test_edit_survives_a_manifest_round_trip(store):
    from ct_engine.pipeline.manifest_store import ManifestStore

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_edit(manifest, 1, target.id, "Persistido", store=store, target_lang="es")

    ms = ManifestStore(store)
    ms.save_manifest(manifest)
    loaded = ms.load_manifest("edit")
    block = loaded.page(0).find_text_object(target.id)
    assert block is not None
    assert block.translated_text == "Persistido"
    assert block.manually_edited
    assert block.machine_translated_text


# ==========================================================================
# Manual layout: move, resize, colour, text size
# ==========================================================================
def test_move_sets_a_manual_box_without_touching_the_original(store):
    """The original box still governs redaction: moving must not leave the
    source text showing where it was."""
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    original = target.bbox.as_tuple()

    apply_layout(
        manifest, 1, target.id,
        bbox={"x0": 300, "y0": 600, "x1": 560, "y1": 640},
        store=store, target_lang="es",
    )
    assert target.manual_bbox is not None
    assert target.manual_bbox.as_tuple() == (300.0, 600.0, 560.0, 640.0)
    assert target.bbox.as_tuple() == original, "the original box was mutated"
    assert target.render_bbox.as_tuple() == (300.0, 600.0, 560.0, 640.0)
    assert target.has_manual_layout


def test_colour_and_size_are_applied_and_size_is_locked(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_layout(
        manifest, 1, target.id, color="#CC2200", font_size=18.0,
        store=store, target_lang="es",
    )
    assert target.manual_color == "#cc2200"
    assert target.manual_font_size == 18.0
    # A size a person chose must not be silently shrunk by the fitter.
    assert target.manual_size_locked


def test_manual_layout_reaches_the_rendered_pdf(store):
    import fitz

    from ct_engine.pipeline.edit import apply_layout
    from ct_engine.pipeline.reconstruct_stage import rebuild_export_only

    manifest = prepared(store)
    target = first_editable(manifest)
    # One word: where the line wraps depends on the face it is set in.
    needle = (target.translated_text or "").split()[0]

    apply_layout(
        manifest, 1, target.id,
        bbox={"x0": 300, "y0": 600, "x1": 560, "y1": 645},
        color="#cc2200", font_size=18.0,
        store=store, target_lang="es",
    )
    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        matches = [
            span
            for block in doc[0].get_text("dict")["blocks"]
            if block["type"] == 0
            for line in block["lines"]
            for span in line["spans"]
            if needle and needle in span["text"] and span["bbox"][0] > 250
        ]
        assert matches, "the moved text was not found at its new position"
        span = matches[0]
        assert span["bbox"][0] == pytest.approx(300, abs=3)
        assert span["size"] == pytest.approx(18.0, abs=0.6)
        assert (span["color"] & 0xFFFFFF) == 0xCC2200
    finally:
        doc.close()


@pytest.mark.parametrize(
    "bbox",
    [
        {"x0": 100, "y0": 100, "x1": 103, "y1": 140},   # too narrow
        {"x0": 100, "y0": 100, "x1": 300, "y1": 103},   # too short
    ],
)
def test_a_box_too_small_for_text_is_refused(store, bbox):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    with pytest.raises(EditError):
        apply_layout(manifest, 1, target.id, bbox=bbox, store=store,
                     target_lang="es")


def test_a_box_dragged_off_the_page_is_clamped_not_rejected(store):
    """A drag that overshoots should snap back, not lose the edit."""
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    page = manifest.page(0)
    target = first_editable(manifest)
    apply_layout(
        manifest, 1, target.id,
        bbox={"x0": 400, "y0": 700, "x1": 900, "y1": 950},
        store=store, target_lang="es",
    )
    assert target.manual_bbox is not None
    assert target.manual_bbox.x1 <= page.width + 0.01
    assert target.manual_bbox.y1 <= page.height + 0.01


@pytest.mark.parametrize("color", ["red", "#ggg", "#12345", "rgb(1,2,3)", ""])
def test_a_bad_colour_is_refused(store, color):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    with pytest.raises(EditError):
        apply_layout(manifest, 1, target.id, color=color, store=store,
                     target_lang="es")


@pytest.mark.parametrize("size", [0.0, 2.0, 200.0, -12.0])
def test_an_out_of_range_text_size_is_refused(store, size):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    with pytest.raises(EditError):
        apply_layout(manifest, 1, target.id, font_size=size, store=store,
                     target_lang="es")


def test_reset_clears_layout_as_well_as_text(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_layout(
        manifest, 1, target.id,
        bbox={"x0": 300, "y0": 600, "x1": 560, "y1": 640},
        color="#cc2200", font_size=18.0, store=store, target_lang="es",
    )
    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_bbox is None
    assert target.manual_color is None
    assert target.manual_font_size is None
    assert not target.manual_size_locked
    assert not target.has_manual_layout


def test_reset_works_when_only_the_layout_changed(store):
    """There is no earlier TEXT to restore, but placement must still revert."""
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    apply_layout(manifest, 1, target.id, color="#cc2200", store=store,
                 target_lang="es")
    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_color is None


def test_reset_also_clears_bold_italic_and_alignment(store):
    """Reset means 'back to how the engine produced it'. A reviewer who
    bolds a block, then presses Reset, must not keep the bold -- the same
    promise the box/colour/size fields already make. Covers the ordinary
    (machine-translated) path, which is what a normal edit takes."""
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_layout(
        manifest, 1, target.id,
        alignment="CENTER", bold=True, italic=True,
        store=store, target_lang="es",
    )
    assert target.manual_bold is True
    assert target.manual_italic is True
    assert target.manual_alignment is not None

    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_bold is None
    assert target.manual_italic is None
    assert target.manual_alignment is None


def test_reset_of_layout_only_change_also_clears_bold_italic_and_alignment(store):
    """The layout-only branch (no prior text edit, so nothing to restore in
    translated_text) has the identical promise to keep."""
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    apply_layout(
        manifest, 1, target.id,
        color="#cc2200", alignment="CENTER", bold=True, italic=True,
        store=store, target_lang="es",
    )
    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_color is None
    assert target.manual_bold is None
    assert target.manual_italic is None
    assert target.manual_alignment is None


# --- Reset of a bold/italic/alignment-only change, nothing else set ------
#
# Both tests above always paired the new fields with a colour or box change,
# which alone made `has_manual_layout` True and let Reset's guard pass. That
# left the actual reported bug uncovered: a block whose translated text was
# never edited (`machine_translated_text is None`) and whose ONLY override
# is bold, italic, or alignment used to fail `has_manual_layout` (which
# checked only bbox/colour/font_size), so `reset_edit` raised "There is no
# earlier version to go back to" instead of clearing the override -- lying
# to a teacher who can see the bold sitting right there on the page.
def test_reset_of_bold_only_change_with_no_prior_edit_succeeds(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    apply_layout(manifest, 1, target.id, bold=True, store=store, target_lang="es")
    assert target.manual_bold is True

    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_bold is None


def test_reset_of_italic_only_change_with_no_prior_edit_succeeds(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    apply_layout(manifest, 1, target.id, italic=True, store=store, target_lang="es")
    assert target.manual_italic is True

    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_italic is None


def test_reset_of_alignment_only_change_with_no_prior_edit_succeeds(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    target.machine_translated_text = None
    apply_layout(manifest, 1, target.id, alignment="CENTER", store=store,
                 target_lang="es")
    assert target.manual_alignment is not None

    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_alignment is None


def test_a_hand_placed_block_is_not_moved_by_automatic_reflow(store):
    """Reflow must never override an explicit human placement."""
    import fitz

    from ct_engine.pipeline.edit import apply_layout
    from ct_engine.reconstruct.builder import rebuild_page

    manifest = prepared(store, "08_exam_multiple_choice.pdf")
    target = first_editable(manifest, page_number=1)
    apply_layout(
        manifest, 1, target.id,
        bbox={"x0": 80, "y0": 620, "x1": 520, "y1": 660},
        store=store, target_lang="es",
    )
    placed = target.manual_bbox.as_tuple()

    doc = fitz.open(stream=corpus_file("08_exam_multiple_choice.pdf"), filetype="pdf")
    try:
        rebuild_page(doc.load_page(0), manifest.page(0), target_lang="es")
    finally:
        doc.close()
    assert target.manual_bbox.as_tuple() == placed


def test_protected_blocks_cannot_be_moved_either(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store, "03_math_worksheet.pdf")
    protected = next(t for t in manifest.page(0).text_objects if t.protected)
    with pytest.raises(EditError):
        apply_layout(
            manifest, 1, protected.id,
            bbox={"x0": 100, "y0": 100, "x1": 300, "y1": 140},
            store=store, target_lang="es",
        )
