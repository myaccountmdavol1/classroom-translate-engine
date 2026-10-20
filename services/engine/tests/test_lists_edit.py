"""Lists, numbering and indents as edits: one box, many boxes, undo, Reset,
copies, English retranslation, and what the editor is told."""

from __future__ import annotations

import io
import math

import fitz
import numpy as np
import pytest
from PIL import Image

from ct_engine.api.routes import _block_out
from ct_engine.model.document import DocumentManifest
from ct_engine.model.objects import TextObject
from ct_engine.pipeline.edit import (
    EditError,
    _restore_snapshot,
    _snapshot,
    add_blocks,
    apply_bulk,
    apply_edit,
    apply_layout,
    redo,
    reset_edit,
    translate_from_english,
    undo,
)
from ct_engine.reconstruct.builder import rebuild_document
from ct_engine.reconstruct.lists import NUMBER_CHANGED, REFUSE_BULLETS, REFUSE_NUMBERS
from ct_engine.reconstruct.sprite import render_sprite, sprite_area
from test_copy_paste import spec_of
from test_edit import prepared
from test_english_editing import fake_translate  # noqa: F401  (fixture)
from test_text_styles import BOX, _directions, _styled

BOX_X0 = BOX["x0"]

LIST_FIELDS = ("manual_list", "manual_list_style", "manual_list_start", "manual_indents")
THREE = "Primera línea del texto\nSegunda línea del texto\nTercera línea del texto"


def _edit(manifest, obj, store, **kw):
    return apply_layout(manifest, 1, obj.id, store=store, target_lang="es", **kw)


def _question(manifest, n):
    return next(t for t in manifest.page(0).text_objects if t.flat_text.startswith(f"{n}. "))


def test_turning_lists_on_and_off(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="bullet")
    assert obj.manual_list == "bullet" and obj.manual_list_style is None
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["•", "•", "•"]
    _edit(manifest, obj, store, list_style="lower-alpha")
    assert obj.manual_list == "number" and obj.manual_list_style == "lower-alpha"
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["a.", "b.", "c."]
    _edit(manifest, obj, store, list_start=3)
    assert obj.manual_list_start == 3
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["c.", "d.", "e."]
    _edit(manifest, obj, store, list="none")
    assert all(getattr(obj, f) is None for f in LIST_FIELDS)
    assert obj.drawn_layout.marks == []


def test_indent_moves_every_line_and_stops_at_the_ends(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="number")
    _edit(manifest, obj, store, indent=-1)
    assert obj.manual_indents is None
    for expected in ([1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]):
        _edit(manifest, obj, store, indent=1)
        assert obj.manual_indents == expected
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["1.", "2.", "3."]
    _edit(manifest, obj, store, list_levels=[0, 1, 2])
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["1.", "a.", "i."]
    _edit(manifest, obj, store, list_levels=[0, 0, 0])
    assert obj.manual_indents is None


@pytest.mark.parametrize("edit,message", [
    ({"list_levels": [0, 4]}, "Each line's indent must be from 0 to 3."),
    ({"list_levels": [-1]}, "Each line's indent must be from 0 to 3."),
    ({"indent": 2}, "Indent can only go in or out one step."),
    ({"list_start": 0}, "Numbering can start from 1 to 999."),
    ({"list_start": 1000}, "Numbering can start from 1 to 999."),
    ({"list": "bullet", "list_style": "decimal"}, "That list style is not available."),
    ({"list_style": "stars"}, "That list style is not available."),
    ({"list": "tick"}, "That list could not be read."),
])
def test_values_that_cannot_be_read_are_refused(store, edit, message):
    manifest, obj = _styled(store, text=THREE)
    steps = len(manifest.page(0).edit_history)
    with pytest.raises(EditError) as exc:
        _edit(manifest, obj, store, **edit)
    assert exc.value.message == message
    assert len(manifest.page(0).edit_history) == steps


def test_turning_a_list_off_keeps_the_indents(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="bullet", list_levels=[0, 1, 1])
    _edit(manifest, obj, store, list="none")
    assert obj.manual_indents == [0, 1, 1]
    rows = [r for r in obj.drawn_layout.rows if r.text]
    assert rows[-1].indent > rows[0].indent


def test_a_refused_toggle_leaves_no_undo_step(store):
    manifest, obj = _styled(store, text="1. uno\na) dos")
    steps = len(manifest.page(0).edit_history)
    with pytest.raises(EditError) as exc:
        _edit(manifest, obj, store, list="number")
    assert exc.value.message == REFUSE_NUMBERS
    with pytest.raises(EditError) as exc:
        _edit(manifest, obj, store, list="bullet")
    assert exc.value.message == REFUSE_BULLETS
    assert len(manifest.page(0).edit_history) == steps
    assert obj.manual_list is None


def test_numbering_takes_over_typed_numbers_without_doubling(store):
    manifest, obj = _styled(store, text="3. uno\n4. dos\n5. tres")
    _edit(manifest, obj, store, list="number")
    assert obj.manual_list_start == 3 and obj.manual_list_style == "decimal"
    rows = [r for r in obj.drawn_layout.rows if r.text]
    assert [r.marker for r in rows] == ["3.", "4.", "5."]
    assert [r.text for r in rows] == ["uno", "dos", "tres"]
    # The translation itself keeps them, so turning the list off shows them.
    assert obj.translated_text.startswith("3. uno")


def test_a_changed_typed_number_is_flagged(store):
    manifest, obj = _styled(store, text="5. uno\n6. dos")
    result = _edit(manifest, obj, store, list="number", list_start=3)
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["3.", "4."]
    assert NUMBER_CHANGED in result.warnings


def test_bulk_skips_a_box_with_its_own_numbers_and_says_so(store):
    manifest = prepared(store)
    q1, q2 = _question(manifest, 1), _question(manifest, 2)
    directions = _directions(manifest)
    apply_edit(manifest, 1, q2.id, "1. uno\na) dos", store=store, target_lang="es")
    result = apply_bulk(manifest, 1, [directions.id, q1.id, q2.id], list="bullet",
                        store=store, target_lang="es")
    assert directions.manual_list == "bullet"
    assert q1.manual_list is None and q2.manual_list is None
    assert result.warnings[:2] == [REFUSE_BULLETS, "Some boxes kept their own numbers."]


def test_bulk_where_every_box_is_refused_changes_nothing(store):
    manifest = prepared(store)
    q1, q2 = _question(manifest, 1), _question(manifest, 2)
    steps = len(manifest.page(0).edit_history)
    with pytest.raises(EditError) as exc:
        apply_bulk(manifest, 1, [q1.id, q2.id], list="bullet", store=store, target_lang="es")
    assert exc.value.message == REFUSE_BULLETS
    assert len(manifest.page(0).edit_history) == steps


def test_bulk_indent_and_style(store):
    manifest = prepared(store)
    q1, q2 = _question(manifest, 1), _question(manifest, 2)
    apply_bulk(manifest, 1, [q1.id, q2.id], list_style="decimal-paren",
               store=store, target_lang="es")
    assert q1.manual_list == "number" and q1.manual_list_start == 1 or q1.manual_list_start is None
    assert q2.manual_list_start == 2
    assert [r.marker for r in q2.drawn_layout.rows if r.marker] == ["2)"]
    apply_bulk(manifest, 1, [q1.id, q2.id], indent=1, store=store, target_lang="es")
    assert q1.manual_indents == [1] and q2.manual_indents == [1]


def test_undo_and_redo_bring_back_every_list_field(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list_style="upper-alpha", list_start=4, list_levels=[0, 1, 0])
    values = {f: getattr(obj, f) for f in LIST_FIELDS}
    assert values == {
        "manual_list": "number", "manual_list_style": "upper-alpha",
        "manual_list_start": 4, "manual_indents": [0, 1, 0],
    }
    undo(manifest, 1, store=store, target_lang="es")
    assert all(getattr(obj, f) is None for f in LIST_FIELDS)
    redo(manifest, 1, store=store, target_lang="es")
    assert {f: getattr(obj, f) for f in LIST_FIELDS} == values


def test_a_snapshot_from_before_lists_restores_the_defaults(store):
    manifest, obj = _styled(store, text=THREE)
    snapshot = _snapshot(manifest.page(0))
    for f in LIST_FIELDS:
        del snapshot["blocks"][obj.id][f]
    _edit(manifest, obj, store, list="bullet", list_levels=[1, 1, 1])
    _restore_snapshot(manifest, manifest.page(0), snapshot, store=store, target_lang="es")
    assert all(getattr(obj, f) is None for f in LIST_FIELDS)


@pytest.mark.parametrize("edited", [False, True])
def test_reset_clears_the_list(store, edited):
    manifest = prepared(store)
    obj = _directions(manifest)
    if edited:
        apply_edit(manifest, 1, obj.id, THREE, store=store, target_lang="es")
    else:
        obj.machine_translated_text = None
    _edit(manifest, obj, store, list="number", list_levels=[1])
    assert obj.has_manual_layout
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert all(getattr(obj, f) is None for f in LIST_FIELDS)


def test_a_list_alone_counts_as_something_to_reset(store):
    manifest = prepared(store)
    obj = _directions(manifest)
    obj.machine_translated_text = None
    obj.manual_indents = [1]
    assert obj.has_manual_layout
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert obj.manual_indents is None


def test_an_old_manifest_without_list_fields_loads(store):
    manifest = prepared(store)
    raw = manifest.model_dump(mode="json")
    for t in raw["pages"][0]["text_objects"]:
        for f in LIST_FIELDS:
            t.pop(f, None)
        layout = t.get("drawn_layout")
        if layout:
            for key in ("marks", "face", "face_fallback"):
                layout.pop(key, None)
            for row in layout.get("rows", []):
                for key in ("indent", "hang", "marker", "marker_x0", "marker_x1", "marker_face"):
                    row.pop(key, None)
    loaded = DocumentManifest.model_validate(raw)
    obj = loaded.page(0).text_objects[0]
    assert all(getattr(obj, f) is None for f in LIST_FIELDS)
    assert obj.drawn_layout.marks == [] and obj.drawn_layout.face is None


def test_text_and_line_levels_are_saved_together(store):
    manifest, obj = _styled(store, text=THREE)
    steps = len(manifest.page(0).edit_history)
    apply_edit(manifest, 1, obj.id, "uno\ndos", list_levels=[0, 2],
               store=store, target_lang="es")
    assert obj.manual_indents == [0, 2]
    assert len(manifest.page(0).edit_history) == steps + 1


def test_new_english_with_levels_keeps_them(store, fake_translate):
    manifest = prepared(store)
    obj = _directions(manifest)
    translate_from_english(manifest, 1, obj.id, "One\nTwo", store=store,
                           target_lang="es", levels=[0, 1])
    assert obj.manual_indents == [0, 1]
    with pytest.raises(EditError):
        translate_from_english(manifest, 1, obj.id, "One", store=store,
                               target_lang="es", levels=[5])


def test_levels_survive_a_translation_with_more_or_fewer_lines(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="bullet", list_levels=[0, 1, 2])
    apply_edit(manifest, 1, obj.id, "uno\ndos", store=store, target_lang="es")
    assert obj.manual_indents == [0, 1, 2]
    assert _block_out(obj)["list_levels"] == [0, 1]
    apply_edit(manifest, 1, obj.id, "uno\ndos\ntres\ncuatro", store=store, target_lang="es")
    assert _block_out(obj)["list_levels"] == [0, 1, 2, 2]


def test_a_copy_keeps_the_list(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list_style="dash", list_levels=[0, 1, 0])
    page = manifest.page(0)
    b = _block_out(obj, page.width, page.height)
    spec = spec_of(obj, page, list=b["list"], list_style=b["list_style"],
                   list_start=b["list_start"], list_levels=b["list_levels"])
    _, (copy,) = add_blocks(manifest, 1, [spec], store=store, target_lang="es")
    assert copy.manual_list == "bullet" and copy.manual_list_style == "dash"
    assert copy.manual_indents == [0, 1, 0]
    assert [r.marker for r in copy.drawn_layout.rows if r.marker] == ["–", "◦", "–"]


def test_a_copy_with_a_refused_list_is_not_made(store):
    manifest, obj = _styled(store, text="1. uno\na) dos")
    page = manifest.page(0)
    count = len(page.text_objects)
    with pytest.raises(EditError):
        add_blocks(manifest, 1, [spec_of(obj, page, list="number")], store=store, target_lang="es")
    assert len(page.text_objects) == count


def test_block_out_tells_the_editor_the_plan(store):
    manifest, obj = _styled(store, text=THREE)
    before = _block_out(obj)
    assert before["list"] is None and before["list_style"] is None
    assert before["list_start"] == 1 and before["list_levels"] == [0, 0, 0]
    assert before["own_markers"] == {
        "kind": "none", "number_ok": True, "bullet_ok": True, "message": None,
        "number_message": None, "bullet_message": None,
        "number_style": None, "number_start": None, "bullet_style": None,
    }
    assert before["font_face"] is None and before["font_fallback"] is False
    _edit(manifest, obj, store, list="bullet", list_levels=[0, 1, 0])
    after = _block_out(obj)
    assert after["list"] == "bullet" and after["list_style"] == "disc"
    assert after["list_levels"] == [0, 1, 0]
    assert [p["marker"] for p in after["paragraphs"]] == ["•", "◦", "•"]
    assert [p["face"] for p in after["paragraphs"]] == ["text", "marks", "text"]
    first = after["rows"][0]
    assert first["marker"] == "•" and first["marker_face"] == "text"
    assert first["hang"] > 0 and first["indent"] == pytest.approx(first["hang"])
    assert after["sprite_rev"] != before["sprite_rev"]


def test_block_out_reports_typed_markers_and_the_source_hang(store):
    manifest = prepared(store)
    q3 = _question(manifest, 3)
    b = _block_out(q3)
    assert b["own_markers"]["kind"] == "numbers"
    assert b["own_markers"]["number_ok"] and not b["own_markers"]["bullet_ok"]
    assert b["own_markers"]["message"] == REFUSE_BULLETS
    assert b["paragraphs"][0] == {
        "marker": "3.", "face": "text", "level": 0, "typed": True, "source_hang": 21.6,
    }


def test_markers_are_real_text_in_the_pdf(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="bullet", list_levels=[0, 1, 0])
    out, _ = rebuild_document(store.get_bytes(manifest.source_key), manifest.pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as doc:
        text = doc[0].get_text()
    assert "•" in text and "◦" in text
    _edit(manifest, obj, store, list="number")
    out, _ = rebuild_document(store.get_bytes(manifest.source_key), manifest.pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as doc:
        text = doc[0].get_text()
    assert "1." in text and "a." in text


def _alpha(png):
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    return np.asarray(img)[:, :, 3].astype(int)


def test_the_sprite_adds_only_the_markers(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="bullet", list_style="check")
    marked = _alpha(render_sprite(obj))
    bare = obj.model_copy(update={
        "drawn_layout": obj.drawn_layout.model_copy(update={"marks": []}),
    })
    plain = _alpha(render_sprite(bare))
    assert plain.shape == marked.shape
    diff = np.argwhere(plain != marked)
    assert diff.size
    # Every changed pixel is left of the words' column (2x scale).
    area = sprite_area(obj)
    column = (BOX_X0 + obj.drawn_layout.rows[0].indent - area.x0) * 2.0
    assert diff[:, 1].max() < column
    assert all(m.font_path.endswith("NotoSansSymbols2-Regular.ttf") for m in obj.drawn_layout.marks)


@pytest.mark.parametrize("style,glyph", [("circle", "◦"), ("square", "▪"), ("check", "✓")])
def test_symbol_bullets_use_the_marks_face(store, style, glyph):
    manifest, obj = _styled(store, text=THREE)
    result = _edit(manifest, obj, store, list_style=style)
    assert [m.text for m in obj.drawn_layout.marks] == [glyph] * 3
    assert all(m.font_path.endswith("NotoSansSymbols2-Regular.ttf") for m in obj.drawn_layout.marks)
    assert not any("could not be" in w for w in result.warnings)
    render_sprite(obj)  # draws with each mark's own font


def test_a_turned_box_turns_its_markers(store):
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="number", angle=30.0)
    out, _ = rebuild_document(store.get_bytes(manifest.source_key), manifest.pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as doc:
        lines = [line for b in doc[0].get_text("dict")["blocks"] for line in b.get("lines", [])]
    words = next(ln for ln in lines if ln["spans"][0]["text"].startswith("Primera"))
    marker = next(
        ln for ln in lines
        if ln["spans"][0]["text"].strip() == "1." and abs(ln["dir"][1]) > 0.01
    )
    # The marker's writing direction is turned 30 degrees, as the words are.
    assert marker["dir"] == pytest.approx(words["dir"], abs=1e-3)
    angle = math.degrees(math.atan2(marker["dir"][1], marker["dir"][0]))
    assert angle == pytest.approx(30.0, abs=0.5)


def test_the_api_round_trip(client):
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    assert _translate(client, doc_id).status_code in (200, 202)
    assert _wait_terminal(client, doc_id)["stage"] == "COMPLETE"
    blocks = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]
    block = next(b for b in blocks if b["original"].startswith("Directions"))
    question = next(b for b in blocks if b["original"].startswith("1. "))
    url = f"/api/documents/{doc_id}/pages/1/blocks/{block['id']}"

    r = client.patch(url, json={"list": "number", "list_style": "upper-roman"})
    assert r.status_code == 200, r.text
    out = r.json()["block"]
    assert out["list"] == "number" and out["list_style"] == "upper-roman"
    assert out["paragraphs"][0]["marker"] == "I."

    r = client.patch(url, json={"indent": 1})
    assert r.status_code == 200 and r.json()["block"]["list_levels"][0] == 1
    r = client.patch(url, json={"indent": 3})
    assert r.status_code == 400 and "one step" in r.text
    r = client.patch(url, json={"text": "uno\ndos", "list_levels": [0, 2]})
    assert r.status_code == 200, r.text
    assert r.json()["block"]["list_levels"] == [0, 2]

    r = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks",
        json={"object_ids": [question["id"]], "list": "bullet"},
    )
    assert r.status_code == 400 and "Numbered list" in r.text

    r = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks",
        json={"object_ids": [block["id"], question["id"]], "list": "none"},
    )
    assert r.status_code == 200, r.text

    r = client.post(f"{url}/reset")
    assert r.status_code == 200
    reset = r.json()["block"]
    assert reset["list"] is None and reset["list_levels"] == [0]


def test_turning_a_list_on_keeps_the_rest_of_the_box(store):
    manifest, obj = _styled(store, text=THREE, alignment="CENTER")
    _edit(manifest, obj, store, list="bullet")
    assert obj.manual_alignment.value == "CENTER"
    assert TextObject.model_fields["manual_list"].get_default() is None


def test_a_list_already_on_can_change_style_over_mixed_typed_numbers(store):
    """Text edited after the list went on may type mixed numbers; that
    refuses turning a list on, but not restyling one already on."""
    manifest, obj = _styled(store, text=THREE)
    _edit(manifest, obj, store, list="number")
    apply_edit(manifest, 1, obj.id, "1. uno\na) dos", store=store, target_lang="es")
    _edit(manifest, obj, store, list_style="upper-alpha")
    assert obj.manual_list == "number" and obj.manual_list_style == "upper-alpha"
    assert [r.marker for r in obj.drawn_layout.rows if r.marker] == ["A.", "B."]
    _edit(manifest, obj, store, list_start=4)
    assert obj.manual_list_start == 4


def test_numbering_a_question_keeps_its_words_on_the_source_tab(store):
    """Q3's words start 21.6pt after its "3.", in line with questions 1
    and 2 and the answer lines; a list taking over the "3." keeps that."""
    manifest = prepared(store)
    q3 = _question(manifest, 3)
    _edit(manifest, q3, store, list="number")
    assert q3.manual_list_start == 3
    rows = [r for r in q3.drawn_layout.rows if r.text]
    assert rows[0].marker == "3." and q3.drawn_layout.marks[0].x == pytest.approx(72.0)
    assert all(r.x0 == pytest.approx(93.6, abs=0.05) for r in rows)
