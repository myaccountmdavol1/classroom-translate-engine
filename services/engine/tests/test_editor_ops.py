"""Add, delete, bulk edit and undo (completing the review editor).

The driving case: on a real Cub Scouts flyer, OCR never detected
"JOIN THE ADVENTURE" or "OUTDOOR ADVENTURE" -- stylised type on dark
backgrounds. Without the ability to ADD a block, that document cannot be
finished at all.
"""

from __future__ import annotations

import fitz
import pytest

from conftest import corpus_file
from ct_engine.model.enums import TextAlignment, TextSource, TranslationStatus
from ct_engine.pipeline.edit import (
    EditError,
    add_block,
    apply_bulk,
    apply_layout,
    can_undo,
    editable_blocks,
    lock_reason,
    remove_block,
    reset_edit,
    restore_block,
    undo,
)
from ct_engine.pipeline.reconstruct_stage import rebuild_export_only
from test_edit import first_editable, prepared

BOX = {"x0": 120.0, "y0": 300.0, "x1": 420.0, "y1": 330.0}


# --- Adding ---------------------------------------------------------------
def test_add_a_block_where_the_engine_found_nothing(store):
    manifest = prepared(store)
    before = len(manifest.page(0).text_objects)

    _result, created = add_block(
        manifest, 1, bbox=BOX, text="ÚNETE A LA AVENTURA",
        color="#ffffff", font_size=20.0, store=store, target_lang="es",
    )

    assert len(manifest.page(0).text_objects) == before + 1
    assert created.source is TextSource.MANUAL
    assert created.is_manual_block
    assert created.translated_text == "ÚNETE A LA AVENTURA"
    assert created.manual_color == "#ffffff"
    assert created.manual_font_size == 20.0
    assert created.translation_status is TranslationStatus.MANUAL_EDIT


def test_an_added_block_reaches_the_pdf(store):
    manifest = prepared(store)
    add_block(
        manifest, 1, bbox=BOX, text="TEXTOANADIDO", font_size=18.0,
        store=store, target_lang="es",
    )
    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        assert "TEXTOANADIDO" in doc[0].get_text()
    finally:
        doc.close()


def test_an_added_block_removes_nothing_from_the_page(store):
    """It sits on artwork; redacting for it would erase what it was placed on."""
    from ct_engine.reconstruct.builder import rebuild_page

    manifest = prepared(store, "04_science_diagram.pdf")
    page = manifest.page(0)
    for obj in list(page.text_objects):
        obj.suppressed = True          # isolate: only the added block draws
    add_block(
        manifest, 1, bbox=BOX, text="ETIQUETA", font_size=14.0,
        store=store, target_lang="es",
    )
    doc = fitz.open(stream=corpus_file("04_science_diagram.pdf"), filetype="pdf")
    try:
        stats = rebuild_page(doc.load_page(0), page, target_lang="es")
    finally:
        doc.close()
    # The manual block contributed no redaction and no covering.
    assert stats.written >= 1
    assert stats.covered == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": "", "bbox": BOX},
        {"text": "ok", "bbox": {"x0": 10, "y0": 10, "x1": 13, "y1": 40}},
        {"text": "ok", "bbox": BOX, "color": "notacolour"},
        {"text": "ok", "bbox": BOX, "font_size": 300.0},
    ],
)
def test_bad_new_blocks_are_refused(store, kwargs):
    manifest = prepared(store)
    with pytest.raises(EditError):
        add_block(manifest, 1, store=store, target_lang="es", **kwargs)


def test_an_added_block_survives_a_later_translation_pass(store):
    """It has no source text, so translation must leave it completely alone."""
    from ct_engine.pipeline.translate import translate_page
    from test_translate_pipeline import EchoProvider

    manifest = prepared(store)
    _r, created = add_block(
        manifest, 1, bbox=BOX, text="MIO", font_size=14.0,
        store=store, target_lang="es",
    )
    translate_page(
        manifest.page(0), EchoProvider(), source_lang="en", target_lang="es"
    )
    assert created.translated_text == "MIO"


# --- Removing -------------------------------------------------------------
def test_removing_an_engine_block_suppresses_rather_than_deletes(store):
    """Its original glyphs exist and must still come off the page."""
    manifest = prepared(store)
    target = first_editable(manifest)
    before = len(manifest.page(0).text_objects)

    remove_block(manifest, 1, target.id, store=store, target_lang="es")

    assert len(manifest.page(0).text_objects) == before, "the block was deleted"
    assert target.suppressed


def test_a_suppressed_block_is_removed_from_the_page_and_not_redrawn(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    needle = (target.translated_text or "")[:12]
    source_words = target.flat_text.strip()[:18]

    remove_block(manifest, 1, target.id, store=store, target_lang="es")
    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        text = doc[0].get_text()
    finally:
        doc.close()
    assert needle not in text, "suppressed text was still drawn"
    assert source_words not in text, "the original was left behind"


def test_removing_a_manual_block_deletes_it(store):
    manifest = prepared(store)
    _r, created = add_block(
        manifest, 1, bbox=BOX, text="BORRAME", store=store, target_lang="es"
    )
    remove_block(manifest, 1, created.id, store=store, target_lang="es")
    assert all(o.id != created.id for o in manifest.page(0).text_objects)


def test_a_suppressed_block_can_be_restored(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    remove_block(manifest, 1, target.id, store=store, target_lang="es")
    restore_block(manifest, 1, target.id, store=store, target_lang="es")
    assert not target.suppressed


# --- Bulk -----------------------------------------------------------------
def test_bulk_applies_to_every_selected_block(store):
    """A page with dozens of shrunken blocks must not need dozens of calls."""
    manifest = prepared(store)
    blocks = editable_blocks(manifest.page(0))[:4]
    ids = [b.id for b in blocks]

    apply_bulk(
        manifest, 1, ids, color="#123456", font_size=13.0,
        alignment="CENTER", store=store, target_lang="es",
    )
    for b in blocks:
        assert b.manual_color == "#123456"
        assert b.manual_font_size == 13.0
        assert b.manual_alignment is TextAlignment.CENTER


def test_bulk_renders_the_page_once(store):
    manifest = prepared(store)
    ids = [b.id for b in editable_blocks(manifest.page(0))[:5]]
    before = manifest.page(0).revision
    apply_bulk(manifest, 1, ids, color="#222222", store=store, target_lang="es")
    assert manifest.page(0).revision == before + 1


def test_bulk_skips_stale_ids_rather_than_failing(store):
    manifest = prepared(store)
    good = editable_blocks(manifest.page(0))[0]
    apply_bulk(
        manifest, 1, [good.id, "text_gone", "also_gone"],
        color="#654321", store=store, target_lang="es",
    )
    assert good.manual_color == "#654321"


def test_bulk_never_touches_protected_blocks(store):
    manifest = prepared(store, "03_math_worksheet.pdf")
    page = manifest.page(0)
    protected = next(t for t in page.text_objects if t.protected)
    editable = editable_blocks(page)[0]
    apply_bulk(
        manifest, 1, [editable.id, protected.id], color="#abcdef",
        store=store, target_lang="es",
    )
    assert editable.manual_color == "#abcdef"
    assert protected.manual_color is None


def test_bulk_with_no_usable_selection_is_refused(store):
    manifest = prepared(store)
    with pytest.raises(EditError):
        apply_bulk(manifest, 1, ["nope"], color="#000000", store=store,
                   target_lang="es")


def test_every_path_agrees_on_what_a_person_may_edit(store):
    """One predicate, or a block the editor greys out is still changeable.

    `editable_blocks` decides what the editor offers. If the single-block
    and bulk mutators use a looser test, a bulk selection becomes a way in
    through the back: a block drawn as locked, changed anyway.
    """
    manifest = prepared(store)
    page = manifest.page(0)
    offered = {b.id for b in editable_blocks(page)}
    withheld = [t for t in page.text_objects if t.id not in offered]
    assert withheld, "this page has nothing locked, so it proves nothing"

    for obj in withheld:
        before = obj.manual_color
        with pytest.raises(EditError):
            apply_bulk(manifest, 1, [obj.id], color="#abcdef", store=store,
                       target_lang="es")
        assert obj.manual_color == before
        with pytest.raises(EditError):
            apply_layout(manifest, 1, obj.id, color="#abcdef", store=store,
                         target_lang="es")
        assert obj.manual_color == before


def test_a_refusal_says_why_rather_than_only_that_it_failed(store):
    """The reason a block is locked is the same text the editor shows."""
    manifest = prepared(store)
    page = manifest.page(0)
    offered = {b.id for b in editable_blocks(page)}
    locked = next(t for t in page.text_objects if t.id not in offered)

    with pytest.raises(EditError) as exc:
        apply_layout(manifest, 1, locked.id, color="#abcdef", store=store,
                     target_lang="es")
    assert lock_reason(locked) in exc.value.message
    assert "role=" not in exc.value.message


# --- Undo -----------------------------------------------------------------
def test_undo_reverts_a_text_change(store):
    from ct_engine.pipeline.edit import apply_edit

    manifest = prepared(store)
    target = first_editable(manifest)
    original = target.translated_text

    apply_edit(manifest, 1, target.id, "cambiado", store=store, target_lang="es")
    assert target.translated_text == "cambiado"

    undo(manifest, 1, store=store, target_lang="es")
    assert target.translated_text == original


def test_undo_reverts_a_move(store):
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_layout(manifest, 1, target.id, bbox=BOX, store=store, target_lang="es")
    assert target.manual_bbox is not None

    undo(manifest, 1, store=store, target_lang="es")
    assert target.manual_bbox is None


def test_undo_removes_an_added_block(store):
    manifest = prepared(store)
    _r, created = add_block(
        manifest, 1, bbox=BOX, text="TEMPORAL", store=store, target_lang="es"
    )
    undo(manifest, 1, store=store, target_lang="es")
    assert all(o.id != created.id for o in manifest.page(0).text_objects)


def test_undo_restores_a_removed_block(store):
    manifest = prepared(store)
    target = first_editable(manifest)
    remove_block(manifest, 1, target.id, store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")
    assert not target.suppressed


def test_undo_steps_back_one_change_at_a_time(store):
    from ct_engine.pipeline.edit import apply_edit

    manifest = prepared(store)
    target = first_editable(manifest)
    original = target.translated_text
    apply_edit(manifest, 1, target.id, "uno", store=store, target_lang="es")
    apply_edit(manifest, 1, target.id, "dos", store=store, target_lang="es")

    undo(manifest, 1, store=store, target_lang="es")
    assert target.translated_text == "uno"
    undo(manifest, 1, store=store, target_lang="es")
    assert target.translated_text == original


def test_undo_after_a_reset_goes_back_to_before_the_reset(store):
    """Reset is a change like any other, so Undo has to step over it.

    Drag a block to P1, drag it to P2, press Reset, press Undo. Reset used
    to record nothing, so Undo popped the snapshot from the FIRST drag and
    put the block back at P1 -- an intermediate position she had already
    moved on from, offered as if it were the state she had just cleared.
    """
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    target = first_editable(manifest)

    p1 = {"x0": 100.0, "y0": 100.0, "x1": 300.0, "y1": 130.0}
    p2 = {"x0": 150.0, "y0": 400.0, "x1": 350.0, "y1": 430.0}
    apply_layout(manifest, 1, target.id, bbox=p1, store=store, target_lang="es")
    apply_layout(manifest, 1, target.id, bbox=p2, store=store, target_lang="es")
    assert target.manual_bbox is not None
    assert target.manual_bbox.y0 == pytest.approx(p2["y0"])

    reset_edit(manifest, 1, target.id, store=store, target_lang="es")
    assert target.manual_bbox is None

    undo(manifest, 1, store=store, target_lang="es")
    assert target.manual_bbox is not None
    assert target.manual_bbox.y0 == pytest.approx(p2["y0"]), (
        "Undo after Reset restored a position from before the last drag"
    )


def test_undo_with_nothing_to_undo_is_refused(store):
    manifest = prepared(store)
    assert not can_undo(manifest.page(0))
    with pytest.raises(EditError):
        undo(manifest, 1, store=store, target_lang="es")


def test_history_is_bounded(store):
    from ct_engine.pipeline.edit import MAX_HISTORY, apply_edit

    manifest = prepared(store)
    target = first_editable(manifest)
    for i in range(MAX_HISTORY + 8):
        apply_edit(manifest, 1, target.id, f"v{i}", store=store, target_lang="es")
    assert len(manifest.page(0).edit_history) <= MAX_HISTORY


def test_history_survives_a_manifest_round_trip(store):
    from ct_engine.pipeline.edit import apply_edit
    from ct_engine.pipeline.manifest_store import ManifestStore

    manifest = prepared(store)
    target = first_editable(manifest)
    original = target.translated_text
    apply_edit(manifest, 1, target.id, "persistido", store=store, target_lang="es")

    ms = ManifestStore(store)
    ms.save_manifest(manifest)
    loaded = ms.load_manifest("edit")
    assert can_undo(loaded.page(0))
    undo(loaded, 1, store=store, target_lang="es")
    assert loaded.page(0).find_text_object(target.id).translated_text == original


def test_undo_restores_typed_values_after_a_manifest_round_trip(store):
    """Regression: a snapshotted Rect came back as a dict and broke the render.

    Snapshots persist in the manifest, so a Rect becomes a plain dict and an
    enum a plain string; assigning those with setattr bypasses validation.
    """
    from ct_engine.model.geometry import Rect
    from ct_engine.pipeline.edit import apply_layout
    from ct_engine.pipeline.manifest_store import ManifestStore

    manifest = prepared(store)
    target = first_editable(manifest)
    apply_layout(
        manifest, 1, target.id, bbox=BOX, alignment="CENTER",
        store=store, target_lang="es",
    )
    apply_layout(
        manifest, 1, target.id, bbox={"x0": 50, "y0": 50, "x1": 250, "y1": 90},
        store=store, target_lang="es",
    )

    ms = ManifestStore(store)
    ms.save_manifest(manifest)
    loaded = ms.load_manifest("edit")

    # Must not raise, and must restore a real Rect rather than a dict.
    undo(loaded, 1, store=store, target_lang="es")
    restored = loaded.page(0).find_text_object(target.id)
    assert isinstance(restored.manual_bbox, Rect)
    assert restored.manual_bbox.as_tuple() == (
        BOX["x0"], BOX["y0"], BOX["x1"], BOX["y1"],
    )
    assert restored.manual_alignment is TextAlignment.CENTER


def test_redo_added_block_survives_project_round_trip(store):
    from ct_engine.pipeline.edit import redo
    from ct_engine.model.document import DocumentManifest
    manifest = prepared(store)
    _, added = add_block(manifest, 1, bbox=BOX, text="Texto recuperado", store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")
    assert manifest.page(0).find_text_object(added.id) is None
    manifest = DocumentManifest.model_validate_json(manifest.model_dump_json())
    redo(manifest, 1, store=store, target_lang="es")
    assert manifest.page(0).find_text_object(added.id).translated_text == "Texto recuperado"
    undo(manifest, 1, store=store, target_lang="es")
    add_block(manifest, 1, bbox=BOX, text="Nueva idea", store=store, target_lang="es")
    with pytest.raises(EditError, match="nothing to redo"):
        redo(manifest, 1, store=store, target_lang="es")



def test_source_correction_preserves_teacher_translation_and_original_geometry(store):
    from ct_engine.pipeline.edit import correct_source, apply_edit, redo
    manifest = prepared(store)
    obj = first_editable(manifest)
    original_text = obj.flat_text
    original_lines = [line.model_dump() for line in obj.lines]
    apply_edit(manifest, 1, obj.id, "Mi traducción revisada", store=store, target_lang="es")
    result = correct_source(manifest, 1, obj.id, "Corrected original words", store=store, target_lang="es")
    assert obj.flat_text == "Corrected original words"
    assert obj.translated_text == "Mi traducción revisada"
    assert [line.model_dump() for line in obj.lines] == original_lines
    assert result.warnings
    undo(manifest, 1, store=store, target_lang="es")
    assert obj.flat_text == original_text
    assert obj.translated_text == "Mi traducción revisada"
    redo(manifest, 1, store=store, target_lang="es")
    assert obj.flat_text == "Corrected original words"


def test_source_correction_retranslates_only_untouched_machine_text(store, monkeypatch):
    from ct_engine.pipeline.edit import correct_source
    from ct_engine.translate.one_off import OneOffTranslation
    import ct_engine.translate.one_off as one_off
    manifest = prepared(store)
    obj = first_editable(manifest)
    before = obj.translated_text
    calls = []
    def translate(text, **kwargs):
        calls.append(text)
        return OneOffTranslation(text="Nueva traducción", provider="fixture", protected_intact=True)
    monkeypatch.setattr(one_off, "translate_one", translate)
    correct_source(manifest, 1, obj.id, "New approved source", store=store, target_lang="es")
    assert calls == ["New approved source"]
    assert obj.translated_text == "Nueva traducción"
    undo(manifest, 1, store=store, target_lang="es")
    assert obj.translated_text == before


def test_several_blocks_move_in_one_step_and_one_undo(store):
    from ct_engine.pipeline.edit import apply_positions, undo
    from test_edit import prepared, editable_blocks

    manifest = prepared(store)
    a, b = editable_blocks(manifest.page(0))[:2]
    before = (a.render_bbox, b.render_bbox)
    moves = [
        {"id": o.id, "bbox": {"x0": o.bbox.x0 + 20, "y0": o.bbox.y0 + 30,
                              "x1": o.bbox.x1 + 20, "y1": o.bbox.y1 + 30}}
        for o in (a, b)
    ]
    apply_positions(manifest, 1, moves, store=store, target_lang="es")
    assert round(a.render_bbox.x0 - before[0].x0) == 20
    assert round(b.render_bbox.y0 - before[1].y0) == 30
    undo(manifest, 1, store=store, target_lang="es")
    a2 = manifest.page(0).find_text_object(a.id)
    b2 = manifest.page(0).find_text_object(b.id)
    assert (a2.render_bbox, b2.render_bbox) == before


def test_group_move_route_is_not_mistaken_for_a_block_id(client):
    """It once lived at /blocks/positions, which the single-block route
    claimed as a block called "positions" and answered 400."""
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id)
    _wait_terminal(client, doc_id)
    blocks = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]
    movable = [b for b in blocks if b["editable"]][:2]
    moves = [
        {"id": b["id"], "bbox": {k: v + (10 if k in ("x0", "x1") else 0) for k, v in b["render_bbox"].items()}}
        for b in movable
    ]
    r = client.post(f"/api/documents/{doc_id}/pages/1/moves", json={"moves": moves})
    assert r.status_code == 200, r.text
