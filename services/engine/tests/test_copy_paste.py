"""Duplicate, copy and paste, cut, and lining boxes up (Google Slides parity).

Every copy is an ordinary box she made (source MANUAL): it is drawn where it
is put and takes nothing off the page, it keeps the original's look and its
English so it can be corrected and translated again, and making, cutting or
lining up any number of boxes is one undo step.
"""

from __future__ import annotations

from types import SimpleNamespace

import fitz
import pytest

from conftest import corpus_file
from ct_engine.api.routes import _block_out
from ct_engine.model.enums import TextAlignment, TextSource, TranslationStatus
from ct_engine.model.geometry import Rect
from ct_engine.pipeline.edit import (
    EditError,
    _restore_snapshot,
    _snapshot,
    add_blocks,
    apply_edit,
    apply_positions,
    redo,
    remove_block,
    remove_blocks,
    reset_edit,
    translate_from_english,
    undo,
)
from ct_engine.pipeline.reconstruct_stage import rebuild_export_only
from ct_engine.pipeline.word_pictures import _labels_in
from ct_engine.reconstruct.builder import _source_look, turned_frame_of
from ct_engine.translate.one_off import OneOffError
from test_box_styles import _label_manifest
from test_edit import first_editable, prepared
from test_english_editing import fake_translate  # noqa: F401  (fixture)

OFFSET = 10.0


def spec_of(obj, page, *, dx=OFFSET, dy=OFFSET, client_id="c1", **changes) -> dict:
    """The spec the editor sends to copy `obj` (lib/boxes.ts specOf), read
    from the block as the editor has it, moved by (dx, dy)."""
    b = _block_out(obj, page.width, page.height)
    f = b["render_bbox"]
    alignment = (
        "CENTER" if b["centred"]
        else (None if b["alignment"] == "UNKNOWN" else b["alignment"])
    )
    spec = {
        "client_id": client_id,
        "bbox": {"x0": f["x0"] + dx, "y0": f["y0"] + dy,
                 "x1": f["x1"] + dx, "y1": f["y1"] + dy},
        "angle": b["angle"] or 0.0,
        "text": b["translation"] or "",
        "source_text": b["copy_source"],
        "machine_text": b["machine_translation"],
        "fit": bool(b["fit"]),
        "font_size": None if b["fit"] else round(b["font_size"] or 11, 2),
        "font_look": b["font_look"],
        "font_family": b["font_family"] or "auto",
        "bold": bool(b["bold"]),
        "italic": bool(b["italic"]),
        "color": b["manual_color"] or b["color"],
        "alignment": alignment,
        "vertical_align": b["vertical_align"],
        "line_spacing": b["line_spacing"] or 0,
        "underline": b["underline"],
        "highlight": b["highlight"] or "none",
        "fill": b["fill"] or "none",
        "border_color": b["border_color"] or "none",
        "border_width": b["border_width"],
        "border_dash": b["border_dash"],
        "padding": b["padding"],
    }
    spec.update(changes)
    return spec


def _add(manifest, specs, page_number=1, store=None):
    return add_blocks(manifest, page_number, specs, store=store, target_lang="es")


def _editables(manifest, n=1):
    page = manifest.page(n - 1)
    return [t for t in page.text_objects if t.translated_text and not t.is_manual_block
            and t.translation_status is TranslationStatus.TRANSLATED]


def _close(a: Rect, b: Rect, tol=0.05):
    return all(abs(x - y) <= tol for x, y in zip(a.as_tuple(), b.as_tuple()))


# --- 1. One step, stacked on top --------------------------------------------
def test_a_batch_is_one_step_with_the_copies_on_top_in_order(store):
    manifest = prepared(store)
    page = manifest.page(0)
    a, b = _editables(manifest)[:2]
    top = max(t.z_order for t in page.text_objects)
    steps = len(page.edit_history)

    _r, (ca, cb) = _add(manifest, [spec_of(a, page), spec_of(b, page, client_id="c2")], store=store)

    assert len(page.edit_history) == steps + 1
    for c in (ca, cb):
        assert c.source is TextSource.MANUAL and c.is_manual_block
        assert c.translation_status is TranslationStatus.MANUAL_EDIT
        assert c.manually_edited
    assert top < ca.z_order < cb.z_order
    assert page.text_objects[-2:] == [ca, cb]


# --- 2. A copy of an engine block keeps its English and its Reset ----------
def test_a_copy_carries_the_english_and_the_machine_translation(store, fake_translate):
    manifest = prepared(store)
    page = manifest.page(0)
    orig = first_editable(manifest)
    apply_edit(manifest, 1, orig.id, "Texto corregido", store=store, target_lang="es")

    _r, (copy,) = _add(manifest, [spec_of(orig, page)], store=store)
    frame = copy.render_bbox

    assert copy.approved_source_text == orig.flat_text
    assert copy.flat_text == orig.flat_text
    assert copy.translated_text == "Texto corregido"
    assert copy.machine_translated_text == orig.machine_translated_text

    reset_edit(manifest, 1, copy.id, store=store, target_lang="es")
    assert copy.translated_text == orig.machine_translated_text
    assert _close(copy.render_bbox, frame)  # a reset copy does not jump

    translate_from_english(manifest, 1, copy.id, "Write here.", store=store, target_lang="es")
    assert copy.translated_text == "[es] Write here."
    assert orig.translated_text == "Texto corregido"


def test_a_copy_of_a_box_typed_in_the_translation_has_no_english(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (typed,) = _add(manifest, [{
        "bbox": {"x0": 100, "y0": 600, "x1": 300, "y1": 630}, "text": "Hola", "font_size": 14,
    }], store=store)
    b = _block_out(typed, page.width, page.height)
    assert b["copy_source"] is None
    assert b["original"] == "Hola"  # why `original` cannot be used instead
    _r, (copy,) = _add(manifest, [spec_of(typed, page)], store=store)
    assert copy.approved_source_text is None
    assert copy.translated_text == "Hola"


# --- 3. A copy takes nothing off the page -----------------------------------
def test_a_copy_removes_nothing_from_the_page(store):
    from ct_engine.reconstruct.builder import rebuild_page

    manifest = prepared(store, "04_science_diagram.pdf")
    page = manifest.page(0)
    orig = next(t for t in page.text_objects if t.translated_text and t.drawn_layout)
    spec = spec_of(orig, page, dx=40, dy=60)
    for obj in list(page.text_objects):
        obj.suppressed = True  # isolate: only the copy draws
    _r, (copy,) = _add(manifest, [spec], store=store)

    assert copy.oriented is None and copy.cover_boxes == [] and copy.anchor is None
    doc = fitz.open(stream=corpus_file("04_science_diagram.pdf"), filetype="pdf")
    try:
        stats = rebuild_page(doc.load_page(0), page, target_lang="es")
    finally:
        doc.close()
    assert stats.written >= 1
    assert stats.covered == 0


# --- 4. A tilted picture label copies at its angle --------------------------
def test_a_copy_of_a_tilted_label_is_turned_the_same_about_its_new_frame(store):
    manifest, label = _label_manifest(store)
    page = manifest.page(0)
    assert label.drawn_layout.quad

    _r, (copy,) = _add(manifest, [spec_of(label, page)], store=store)

    lay = copy.drawn_layout
    assert lay.quad
    assert abs(lay.angle - label.drawn_layout.angle) < 0.01
    frame, width, height = turned_frame_of(lay)
    orig_frame = turned_frame_of(label.drawn_layout)[0]
    assert abs(frame.width - orig_frame.width) < 0.05
    assert abs(frame.height - orig_frame.height) < 0.05
    cx, cy = orig_frame.center
    assert abs(frame.center[0] - (cx + OFFSET)) < 0.05
    assert abs(frame.center[1] - (cy + OFFSET)) < 0.05
    assert abs(lay.size - label.drawn_layout.size) < 0.01


# --- 5. Size and spacing ----------------------------------------------------
@pytest.mark.parametrize("spacing, ratio", [(0, 1.18), (1.5, 1.5)])
def test_a_locked_size_is_drawn_at_that_size_and_spacing(store, spacing, ratio):
    manifest = prepared(store)
    page = manifest.page(0)
    orig = first_editable(manifest)
    _r, (copy,) = _add(
        manifest, [spec_of(orig, page, font_size=14.0, line_spacing=spacing)], store=store
    )
    assert copy.manual_size_locked and copy.manual_font_size == 14.0
    assert copy.drawn_layout.size == 14.0
    assert abs(copy.drawn_layout.leading - 14.0 * ratio) < 0.02


def test_fit_fills_the_frame(store):
    manifest = prepared(store)
    page = manifest.page(0)
    orig = first_editable(manifest)
    big = {"x0": 72, "y0": 560, "x1": 400, "y1": 640}
    _r, (copy,) = _add(
        manifest, [spec_of(orig, page, bbox=big, fit=True, font_size=None, text="Hola")],
        store=store,
    )
    assert copy.manual_fit and copy.manual_font_size is None
    assert copy.drawn_layout.size > 30  # far past any original size: it fills


def test_a_size_and_fit_together_are_refused(store):
    manifest = prepared(store)
    page = manifest.page(0)
    with pytest.raises(EditError):
        _add(manifest, [spec_of(first_editable(manifest), page, fit=True, font_size=12)], store=store)


# --- 6. The automatic face --------------------------------------------------
def test_a_copy_of_a_serif_block_is_drawn_in_the_same_face(store):
    manifest = prepared(store)
    page = manifest.page(0)
    orig = first_editable(manifest)
    orig.style.serif = True
    apply_edit(manifest, 1, orig.id, orig.translated_text, store=store, target_lang="es")
    assert _block_out(orig)["font_look"] == "serif"

    _r, (copy,) = _add(manifest, [spec_of(orig, page)], store=store)

    assert _source_look(copy) == "serif"
    assert copy.manual_font_family is None  # still automatic
    assert copy.drawn_layout.font_builtin == orig.drawn_layout.font_builtin
    assert copy.drawn_layout.font_path == orig.drawn_layout.font_path


# --- 7. Every style round-trips; a bad one adds nothing ---------------------
def test_every_style_is_kept_on_the_copy(store):
    manifest = prepared(store)
    page = manifest.page(0)
    orig = first_editable(manifest)
    spec = spec_of(
        orig, page, font_family="serif", bold=True, italic=True, color="#AA0000",
        alignment="RIGHT", vertical_align="bottom", line_spacing=1.15,
        underline=True, highlight="#ffff00", fill="#eeeeee",
        border_color="#0000ff", border_width=2.0, border_dash="dashed", padding=4.0,
        angle=30.0,
    )
    _r, (copy,) = _add(manifest, [spec], store=store)
    assert copy.manual_font_family == "serif"
    assert copy.manual_bold is True and copy.manual_italic is True
    assert copy.manual_color == "#aa0000"
    assert copy.manual_alignment is TextAlignment.RIGHT
    assert copy.manual_vertical_align == "bottom"
    assert copy.manual_line_spacing == 1.15
    assert copy.manual_underline is True
    assert copy.manual_highlight == "#ffff00"
    assert copy.manual_fill == "#eeeeee"
    assert copy.manual_border_color == "#0000ff"
    assert copy.manual_border_width == 2.0
    assert copy.manual_border_dash == "dashed"
    assert copy.manual_padding == 4.0
    assert copy.manual_angle == 30.0
    lay = copy.drawn_layout
    assert lay.underline and lay.highlight and lay.border and lay.fill
    assert lay.angle == 30.0
    # The frame she sent is the unturned frame, as for any turned box.
    assert _close(copy.manual_bbox, Rect(**spec["bbox"]))


def test_a_border_weight_with_no_border_stays_without_one(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (copy,) = _add(
        manifest, [spec_of(first_editable(manifest), page, border_color="none", border_width=3.0)],
        store=store,
    )
    assert copy.manual_border_color is None
    assert copy.manual_border_width == 3.0


@pytest.mark.parametrize(
    "change",
    [
        {"color": "red"}, {"highlight": "yellow"}, {"fill": "#12"},
        {"border_color": "blue"}, {"border_width": 5.0}, {"border_dash": "wavy"},
        {"padding": 3.0}, {"line_spacing": 1.3}, {"vertical_align": "centre"},
        {"font_family": "comic"}, {"font_look": "script"}, {"alignment": "MIDDLE"},
        {"font_size": 200.0}, {"angle": float("nan")},
        {"bbox": {"x0": 10, "y0": 10, "x1": 12, "y1": 40}},
        {"text": "", "fill": "none", "border_color": "none"},
    ],
)
def test_a_value_that_cannot_be_read_adds_nothing(store, change):
    manifest = prepared(store)
    page = manifest.page(0)
    good = spec_of(first_editable(manifest), page)
    bad = spec_of(first_editable(manifest), page, client_id="c2", **change)
    before = [t.id for t in page.text_objects]
    steps, revision = len(page.edit_history), page.revision
    with pytest.raises(EditError):
        _add(manifest, [good, bad], store=store)
    assert [t.id for t in page.text_objects] == before
    assert len(page.edit_history) == steps
    assert page.revision == revision


def test_a_cover_with_only_a_border_can_be_copied(store):
    manifest = prepared(store)
    _r, (cover,) = _add(manifest, [{
        "bbox": {"x0": 100, "y0": 600, "x1": 300, "y1": 650},
        "text": "", "border_color": "#ff0000", "border_width": 2.0,
    }], store=store)
    assert cover.drawn_layout is not None and cover.drawn_layout.border


# --- 8. Text pasted from outside --------------------------------------------
def test_outside_text_is_translated_line_by_line_and_sized_to_its_rows(store, fake_translate):
    from ct_engine.reconstruct.fitting import DEFAULT_LEADING

    manifest = prepared(store)
    english = "Name the planets.\nWhich one is the largest?"
    frame = {"x0": 100, "y0": 500, "x1": 400, "y1": 520}
    _r, (box,) = _add(manifest, [{
        "bbox": frame, "translate": True, "source_text": english, "text": "ignored",
        "font_size": 12, "fit_height": True, "alignment": "LEFT",
    }], store=store)

    assert fake_translate == ["Name the planets.", "Which one is the largest?"]
    assert box.translated_text == "[es] Name the planets.\n[es] Which one is the largest?"
    assert box.approved_source_text == english
    assert box.machine_translated_text == box.translated_text
    rows = len(box.drawn_layout.rows)
    assert rows >= 2
    assert abs(box.render_bbox.height - (rows * 12 * DEFAULT_LEADING + 2)) < 0.01
    # The box holds exactly what is drawn.
    assert box.drawn_layout.size == 12


def test_a_translation_failure_adds_nothing(store, monkeypatch):
    def broken(text, *, target_lang, source_lang=None, provider=None):
        raise OneOffError("The translation service could not be reached.")

    monkeypatch.setattr("ct_engine.translate.one_off.translate_one", broken)
    manifest = prepared(store)
    page = manifest.page(0)
    before, steps = len(page.text_objects), len(page.edit_history)
    good = spec_of(first_editable(manifest), page)
    with pytest.raises(EditError, match="could not be reached"):
        _add(manifest, [good, {
            "bbox": {"x0": 100, "y0": 500, "x1": 400, "y1": 520},
            "translate": True, "source_text": "Hello", "font_size": 12,
        }], store=store)
    assert len(page.text_objects) == before
    assert len(page.edit_history) == steps


def test_outside_text_translated_beforehand_is_not_translated_again(store, fake_translate):
    # The batch route translates before it takes the document lock
    # (translate_specs), so a slow translator holds up no other save; the
    # box is then made from that translation alone.
    from ct_engine.pipeline.edit import translate_specs

    manifest = prepared(store)
    spec = {"bbox": {"x0": 100, "y0": 500, "x1": 400, "y1": 520}, "translate": True,
            "source_text": "Good morning", "font_size": 12}
    translate_specs(manifest, [spec], target_lang="es")
    assert spec["translated"] == "[es] Good morning"
    _r, (box,) = _add(manifest, [spec], store=store)
    assert fake_translate == ["Good morning"]
    assert box.translated_text == "[es] Good morning"


# --- 9. Undo and redo -------------------------------------------------------
def test_undo_removes_every_copy_and_redo_brings_them_back(store):
    manifest = prepared(store)
    page = manifest.page(0)
    a, b = _editables(manifest)[:2]
    before = {t.id for t in page.text_objects}
    _r, created = _add(manifest, [spec_of(a, page), spec_of(b, page)], store=store)
    ids = [c.id for c in created]

    undo(manifest, 1, store=store, target_lang="es")
    assert {t.id for t in page.text_objects} == before

    redo(manifest, 1, store=store, target_lang="es")
    again = [t for t in page.text_objects if t.id in ids]
    assert [t.id for t in again] == ids
    assert all(t.drawn_layout is not None for t in again)
    assert again[0].translated_text == a.translated_text


# --- 10. Cut / delete several -----------------------------------------------
def test_removing_several_is_one_step_and_undo_brings_each_back(store):
    manifest = prepared(store)
    page = manifest.page(0)
    engine = _editables(manifest)[0]
    _r, (made,) = _add(manifest, [spec_of(engine, page, underline=True)], store=store)
    steps = len(page.edit_history)

    remove_blocks(manifest, 1, [engine.id, made.id, "missing"], store=store, target_lang="es")

    assert len(page.edit_history) == steps + 1
    assert engine.suppressed
    assert page.find_text_object(made.id) is None

    undo(manifest, 1, store=store, target_lang="es")
    back = page.find_text_object(made.id)
    assert back is not None and back.manual_underline and back.drawn_layout is not None
    assert not page.find_text_object(engine.id).suppressed


def test_removing_only_locked_text_says_why(store):
    manifest = prepared(store, "03_math_worksheet.pdf")
    page = manifest.page(0)
    locked = next(t for t in page.text_objects if t.protected)
    with pytest.raises(EditError, match="cannot be removed. "):
        remove_blocks(manifest, 1, [locked.id], store=store, target_lang="es")
    with pytest.raises(EditError, match="None of the selected"):
        remove_blocks(manifest, 1, [locked.id, "missing"], store=store, target_lang="es")


def test_the_last_box_on_a_page_can_be_removed(store):
    manifest = prepared(store)
    page = manifest.page(0)
    page.text_objects = []  # a page with no text of its own
    _r, (made,) = _add(manifest, [{
        "bbox": {"x0": 100, "y0": 500, "x1": 300, "y1": 530}, "text": "Hola", "font_size": 14,
    }], store=store)
    result = remove_block(manifest, 1, made.id, store=store, target_lang="es")
    assert page.text_objects == [] and result.object_id == ""

    _r, (made,) = _add(manifest, [{
        "bbox": {"x0": 100, "y0": 500, "x1": 300, "y1": 530}, "text": "Hola", "font_size": 14,
    }], store=store)
    remove_blocks(manifest, 1, [made.id], store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")
    assert [t.id for t in page.text_objects] == [made.id]
    undo(manifest, 1, store=store, target_lang="es")  # back before the add
    assert page.text_objects == []


# --- 11. Lining up: moves and alignment in one step ------------------------
def test_moves_with_an_alignment_are_one_step(store):
    manifest = prepared(store)
    page = manifest.page(0)
    a, b = _editables(manifest)[:2]
    was = (a.render_bbox, b.render_bbox, a.manual_alignment, b.manual_alignment)
    steps = len(page.edit_history)
    left = min(a.render_bbox.x0, b.render_bbox.x0) - 5

    moves = [
        {"id": o.id, "bbox": {"x0": left, "y0": o.render_bbox.y0,
                              "x1": left + o.render_bbox.width, "y1": o.render_bbox.y1}}
        for o in (a, b)
    ]
    apply_positions(manifest, 1, moves, alignment="RIGHT", align_ids=[a.id, b.id],
                    store=store, target_lang="es")

    assert len(page.edit_history) == steps + 1
    assert a.manual_alignment is TextAlignment.RIGHT is b.manual_alignment
    assert a.render_bbox.x0 == b.render_bbox.x0 == left

    undo(manifest, 1, store=store, target_lang="es")
    assert (a.render_bbox, b.render_bbox, a.manual_alignment, b.manual_alignment) == was


def test_a_bad_alignment_records_nothing(store):
    manifest = prepared(store)
    page = manifest.page(0)
    a = first_editable(manifest)
    steps = len(page.edit_history)
    with pytest.raises(EditError):
        apply_positions(manifest, 1, [{"id": a.id, "bbox": a.render_bbox.model_dump()}],
                        alignment="SIDEWAYS", align_ids=[a.id], store=store, target_lang="es")
    assert len(page.edit_history) == steps


# --- 12. Snapshots keep boxes without their drawing ------------------------
def test_snapshots_leave_the_drawing_out_and_still_redraw(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (made,) = _add(manifest, [spec_of(first_editable(manifest), page)], store=store)
    snap = _snapshot(page)
    raw = next(r for r in snap["manual_objects"] if r["id"] == made.id)
    assert "drawn_layout" not in raw and "drawn_bbox" not in raw

    page.text_objects.remove(made)
    _restore_snapshot(manifest, page, snap, store=store, target_lang="es")
    back = page.find_text_object(made.id)
    assert back.drawn_layout is not None and back.drawn_bbox is not None


def test_an_old_snapshot_with_the_drawing_still_restores(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (made,) = _add(manifest, [spec_of(first_editable(manifest), page)], store=store)
    snap = _snapshot(page)
    snap["manual_objects"] = [made.model_dump(mode="json")]  # as saved before
    assert snap["manual_objects"][0]["drawn_layout"] is not None
    page.text_objects.remove(made)
    _restore_snapshot(manifest, page, snap, store=store, target_lang="es")
    assert page.find_text_object(made.id).drawn_layout is not None


# --- 13. The PDF ------------------------------------------------------------
def test_a_copy_reaches_the_pdf(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _add(manifest, [spec_of(first_editable(manifest), page, dy=400, text="COPIADO",
                            font_size=14.0)], store=store)
    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        assert "COPIADO" in doc[0].get_text()
    finally:
        doc.close()


# --- 14. Office pictures ----------------------------------------------------
def test_a_box_drawn_inside_a_picture_goes_into_it(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (inside, straddle) = _add(manifest, [
        {"bbox": {"x0": 120, "y0": 520, "x1": 220, "y1": 540}, "text": "Dentro", "font_size": 10},
        {"bbox": {"x0": 280, "y0": 520, "x1": 380, "y1": 540}, "text": "Borde", "font_size": 10},
    ], store=store)
    image = SimpleNamespace(bbox=Rect(x0=100, y0=500, x1=300, y1=600))
    page.text_objects = [inside, straddle]  # only the made boxes

    areas = _labels_in(page, image)

    assert areas == [inside.drawn_bbox.expanded(2.5)]  # its drawn box only


def test_a_turned_box_inside_a_picture_goes_in_by_its_turned_frame(store):
    manifest = prepared(store)
    page = manifest.page(0)
    _r, (turned,) = _add(manifest, [
        {"bbox": {"x0": 150, "y0": 540, "x1": 250, "y1": 560}, "text": "Girado",
         "font_size": 10, "angle": 20},
    ], store=store)
    page.text_objects = [turned]
    (area,) = _labels_in(page, SimpleNamespace(bbox=Rect(x0=100, y0=500, x1=300, y1=600)))
    assert isinstance(area, list) and len(area) == 4
    assert _labels_in(page, SimpleNamespace(bbox=Rect(x0=160, y0=500, x1=300, y1=600))) == []


def test_a_box_over_the_pages_own_text_stays_out_of_a_full_page_picture(store):
    # A slide whose background is one full-slide picture: every box she
    # made is "inside" it. One over a paragraph the file keeps as text must
    # not be copied in, or that paragraph would be baked into the picture
    # and written as text as well.
    manifest = prepared(store)
    page = manifest.page(0)
    native = first_editable(manifest)
    assert native.source is TextSource.NATIVE
    n = native.drawn_bbox or native.bbox
    _r, (over, clear) = _add(manifest, [
        {"bbox": {"x0": n.x0, "y0": n.y0, "x1": n.x0 + 80, "y1": n.y0 + 16},
         "text": "Encima", "font_size": 10},
        {"bbox": {"x0": 40, "y0": page.height - 60, "x1": 140, "y1": page.height - 40},
         "text": "Aparte", "font_size": 10},
    ], store=store)
    page.text_objects = [native, over, clear]
    image = SimpleNamespace(bbox=Rect(x0=0, y0=0, x1=page.width, y1=page.height))

    areas = _labels_in(page, image)

    assert areas == [clear.drawn_bbox.expanded(2.5)]


# --- 15. The API ------------------------------------------------------------
def _api_doc(client):
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    assert _translate(client, doc_id).status_code in (200, 202)
    _wait_terminal(client, doc_id)
    page = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    return doc_id, page


def _api_spec(b, client_id, dx=OFFSET):
    f = b["render_bbox"]
    return {
        "client_id": client_id,
        "bbox": {"x0": f["x0"] + dx, "y0": f["y0"] + dx, "x1": f["x1"] + dx, "y1": f["y1"] + dx},
        "angle": b["angle"] or 0, "text": b["translation"] or "",
        "source_text": b["copy_source"], "machine_text": b["machine_translation"],
        "font_size": round(b["font_size"], 2), "font_look": b["font_look"],
        "bold": b["bold"], "italic": b["italic"], "color": b["color"],
    }


def test_the_batch_endpoint_makes_boxes_and_names_them_back(client):
    doc_id, page = _api_doc(client)
    blocks = [b for b in page["blocks"] if b["editable"] and b["translation"]][:2]
    assert all(b["font_look"] in ("sans", "serif", "mono") for b in page["blocks"])
    assert all("copy_source" in b for b in page["blocks"])
    url = f"/api/documents/{doc_id}/pages/1/blocks/batch"

    r = client.post(url, json={"blocks": [_api_spec(blocks[0], "tmp-a"), _api_spec(blocks[1], "tmp-b")]})

    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["client_id"] for c in body["created"]] == ["tmp-a", "tmp-b"]
    made = [c["block"] for c in body["created"]]
    assert all(m["editable"] is True and m["locked_reason"] is None for m in made)
    assert all(m["is_manual_block"] and m["sprite_rev"] for m in made)
    assert made[0]["copy_source"] == blocks[0]["copy_source"]
    assert made[0]["translation"] == blocks[0]["translation"]
    assert body["can_undo"] is True
    after = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    assert {m["id"] for m in made} <= {b["id"] for b in after["blocks"]}

    # One undo step takes both away.
    assert client.post(f"/api/documents/{doc_id}/pages/1/undo").status_code == 200
    after = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    assert not {m["id"] for m in made} & {b["id"] for b in after["blocks"]}


def test_the_batch_endpoint_refuses_too_many_none_or_a_bad_one(client):
    doc_id, page = _api_doc(client)
    b = next(b for b in page["blocks"] if b["editable"] and b["translation"])
    url = f"/api/documents/{doc_id}/pages/1/blocks/batch"
    assert client.post(url, json={"blocks": []}).status_code == 422
    assert client.post(url, json={"blocks": [_api_spec(b, f"c{i}") for i in range(201)]}).status_code == 422
    bad = _api_spec(b, "x") | {"color": "red"}
    r = client.post(url, json={"blocks": [_api_spec(b, "ok"), bad]})
    assert r.status_code == 400
    assert "colour" in r.text
    after = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    assert len(after["blocks"]) == len(page["blocks"])


def test_remove_and_realigned_moves_through_the_api(client):
    doc_id, page = _api_doc(client)
    a, b = [b for b in page["blocks"] if b["editable"] and b["translation"]][:2]

    r = client.post(f"/api/documents/{doc_id}/pages/1/moves", json={
        "moves": [{"id": a["id"], "bbox": a["render_bbox"]}, {"id": b["id"], "bbox": b["render_bbox"]}],
        "alignment": "CENTER", "align_ids": [a["id"], b["id"]],
    })
    assert r.status_code == 200, r.text
    after = {x["id"]: x for x in client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]}
    assert after[a["id"]]["manual_alignment"] == after[b["id"]]["manual_alignment"] == "CENTER"

    r = client.post(f"/api/documents/{doc_id}/pages/1/blocks/remove",
                    json={"object_ids": [a["id"], b["id"]]})
    assert r.status_code == 200, r.text
    after = {x["id"]: x for x in client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]}
    assert after[a["id"]]["suppressed"] and after[b["id"]]["suppressed"]
    assert client.post(f"/api/documents/{doc_id}/pages/1/blocks/remove",
                       json={"object_ids": []}).status_code == 422
    r = client.post(f"/api/documents/{doc_id}/pages/1/blocks/remove", json={"object_ids": ["nope"]})
    assert r.status_code == 400


def test_a_bulk_look_paints_every_field_of_the_format_painter(store):
    """The format painter sends the whole look through the bulk edit; every
    field lands, and a border colour "none" with a weight stays without a
    border."""
    from ct_engine.pipeline.edit import apply_bulk

    manifest = prepared(store)
    page = manifest.page(0)
    a, b = _editables(manifest)[:2]
    steps = len(page.edit_history)
    apply_bulk(
        manifest, 1, [a.id, b.id], font_family="auto", font_size=13.0, bold=True,
        italic=False, underline=True, color="#123456", highlight="none",
        alignment="CENTER", vertical_align="middle", line_spacing=0, fill="none",
        border_color="none", border_width=1.0, border_dash="solid", padding=0,
        store=store, target_lang="es",
    )
    assert len(page.edit_history) == steps + 1
    for o in (a, b):
        assert o.manual_font_size == 13.0 and o.manual_bold is True
        assert o.manual_underline and o.manual_color == "#123456"
        assert o.manual_alignment is TextAlignment.CENTER
        assert o.manual_vertical_align == "middle"
        assert o.manual_border_color is None


# --- Format painter onto a block the engine placed ---------------------------
def _long_block(manifest):
    """The longest block the engine placed: the one most likely to need
    more rows once its type is made bigger."""
    return max(_editables(manifest), key=lambda t: len(t.translated_text or ""))


def test_a_bigger_size_never_drops_words(store):
    # Painting the title's look (bold 16pt, space 4) onto a question left it
    # two rows in a box that needed three: the last word went missing from
    # the page and its picture, and nothing was flagged. Bigger still here,
    # so the box needs several more rows than it has.
    from ct_engine.pipeline.edit import apply_bulk

    manifest = prepared(store)
    obj = _long_block(manifest)
    apply_bulk(manifest, 1, [obj.id], font_size=28.0, bold=True, padding=4.0,
               store=store, target_lang="es")
    # A number the text starts with ("3.") is drawn as its row's marker,
    # hung apart from the words, so it is read from there.
    drawn = " ".join(
        f"{r.marker} {r.text}" if r.marker else r.text
        for r in obj.drawn_layout.rows if r.text
    )
    assert drawn.split() == obj.translated_text.split()
    assert obj.drawn_layout.rows[-1].end
    # The frame holds every row it draws.
    last = obj.drawn_layout.rows[-1].baseline
    assert obj.drawn_bbox.y1 >= last


def test_a_size_change_keeps_the_frame_the_editor_shows(store):
    # The editor keeps the frame it shows while a new size is on its way;
    # laid out again in its narrower source box, the block shrank sideways
    # and re-wrapped once the picture arrived.
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    obj = _long_block(manifest)
    before = obj.drawn_bbox
    assert obj.manual_bbox is None
    apply_layout(manifest, 1, obj.id, font_size=9.0, store=store, target_lang="es")
    assert obj.manual_bbox == before.clamped_to(manifest.page(0).bounds)
    assert abs(obj.drawn_bbox.x0 - before.x0) < 0.01
    assert abs(obj.drawn_bbox.x1 - before.x1) < 0.01
    # With space inside set too, the frame grows outward by it, as it does
    # for a block left where the engine put it.
    other = [t for t in _editables(manifest) if t.id != obj.id and t.manual_bbox is None][0]
    was = other.drawn_bbox
    apply_layout(manifest, 1, other.id, fit=True, padding=4.0, store=store, target_lang="es")
    assert abs(other.manual_bbox.x0 - (was.x0 - 4)) < 0.01 or other.manual_bbox.x0 == 0
    assert abs(other.manual_bbox.x1 - (was.x1 + 4)) < 0.01
    # Reset puts it back where the engine had it.
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert obj.manual_bbox is None


def test_a_box_grown_for_its_rows_keeps_that_height(store):
    # The editor outlines her frame; once the engine has grown a box to
    # hold the rows of a size she set, that height is the box's own, so the
    # outline takes in every word, and a drag hands back the same box.
    from ct_engine.pipeline.edit import apply_layout

    manifest = prepared(store)
    obj = _long_block(manifest)
    apply_layout(manifest, 1, obj.id, font_size=28.0, store=store, target_lang="es")
    assert obj.manual_bbox.y1 >= obj.drawn_layout.rows[-1].baseline
    frame, rows = obj.manual_bbox, len(obj.drawn_layout.rows)
    # Laid out again in that frame: the same rows, and no further growth.
    apply_layout(manifest, 1, obj.id, color="#112233", store=store, target_lang="es")
    assert len(obj.drawn_layout.rows) == rows
    assert obj.manual_bbox == frame
    # Undo puts the frame back as it was before the size.
    undo(manifest, 1, store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")
    assert manifest.page(0).find_text_object(obj.id).manual_bbox is None
