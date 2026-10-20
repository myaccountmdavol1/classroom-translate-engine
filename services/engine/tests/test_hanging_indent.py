"""Markers hang apart from their words, so wrapped lines sit under the words.

The long-standing fault: a numbered question whose translation wrapped
drew its second line under the "3.", not under the words, as the source
did not. Every block now splits a marker it starts with and hangs it --
and a box with a list, or an indent, wraps each line in its own column.
A block with neither draws exactly as before, so no checked page moves.
"""

from __future__ import annotations

import pytest

from ct_engine.fonts.registry import REGISTRY
from ct_engine.model.enums import TextAlignment
from ct_engine.pipeline.edit import _fitted_height, apply_edit, apply_layout
from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.lists import MIN_COLUMN_EM
from test_edit import prepared
from test_text_styles import BOX, SPANISH, _directions, _styled

LONG_Q3 = (
    "3. Se coloca una planta en un armario oscuro durante una semana. Prediga qué "
    "les pasará a sus hojas y explique su razonamiento con detalle."
)
# Drawn before lists existed (commit f4b24f8), for blocks without markers.
FROZEN_DIRECTIONS = {
    "lines": [
        (72.0, 117.405, "<Directions: Read each question carefully. Write your answers in complete sentences on the lines"),
        (72.0, 131.405, "provided.>"),
    ],
    "rows": [
        (72.0, 525.726, 117.405, "<Directions: Read each question carefully. Write your answers in complete sentences on the lines", False),
        (72.0, 121.318, 131.405, "provided.>", True),
    ],
}
FROZEN_JUSTIFIED_ROWS = [
    (72.0, 330.0, 428.8, "Instrucciones: lea cada pregunta con cuidado y", False),
    (72.0, 330.0, 441.78, "escriba sus respuestas en oraciones completas", False),
    (72.0, 245.041, 454.76, "usando el vocabulario de la unidad.", True),
    (72.0, 330.0, 467.74, "Use evidencia del texto para apoyar cada respuesta", False),
    (72.0, 131.917, 480.72, "que escriba.", True),
]


def _rows(obj):
    return [(r.x0, r.x1, r.baseline, r.text, r.end) for r in obj.drawn_layout.rows]


def _question(manifest, n):
    return next(
        t for t in manifest.page(0).text_objects if t.flat_text.startswith(f"{n}. ")
    )


def test_a_wrapped_question_sits_under_its_words(store):
    manifest = prepared(store)
    q3 = _question(manifest, 3)
    apply_edit(manifest, 1, q3.id, LONG_Q3, store=store, target_lang="es")
    layout = q3.drawn_layout
    rows = [r for r in layout.rows if r.text]
    assert len(rows) >= 2
    # The source hung its words 21.6pt after the "3." (72.0 -> 93.6).
    assert rows[0].marker == "3." and rows[0].hang == pytest.approx(21.6)
    assert layout.marks[0].x == pytest.approx(72.0) and layout.marks[0].text == "3."
    assert all(r.x0 == pytest.approx(93.6, abs=0.05) for r in rows)
    assert all(r.marker is None for r in rows[1:])
    # The marker is ink on the page but not part of the words' rows.
    assert not any(text.startswith("3.") for _, _, text in layout.lines)


def test_blocks_without_markers_draw_as_they_always_have(store):
    manifest = prepared(store)
    directions = _directions(manifest)
    assert directions.drawn_layout.lines == FROZEN_DIRECTIONS["lines"]
    assert _rows(directions) == FROZEN_DIRECTIONS["rows"]
    assert directions.drawn_layout.marks == []
    manifest, obj = _styled(store, alignment="JUSTIFY")
    assert _rows(obj) == FROZEN_JUSTIFIED_ROWS


def test_a_typed_marker_without_a_source_gap_keeps_its_place(store):
    # A box she typed "3. ..." into: the words start where they did
    # ("3. " then the text), and wrapped lines line up under them.
    manifest, obj = _styled(store, text=LONG_Q3)
    font = REGISTRY.resolve(LONG_Q3, script="Latin")
    hang = REGISTRY.measure(font, "3.", 11.0) + REGISTRY.measure(font, " ", 11.0)
    rows = [r for r in obj.drawn_layout.rows if r.text]
    assert obj.drawn_layout.marks[0].x == pytest.approx(BOX["x0"], abs=0.01)
    assert rows[0].x0 == pytest.approx(BOX["x0"] + hang, abs=0.01)
    assert all(r.x0 == pytest.approx(rows[0].x0, abs=0.01) for r in rows)


def test_a_right_to_left_list_hangs_its_markers_on_the_right(store):
    arabic = "النص الأول في القائمة\nالنص الثاني في القائمة"
    manifest, obj = _styled(store, text=arabic, lang="ar")
    apply_layout(manifest, 1, obj.id, list="number", store=store, target_lang="ar")
    layout = obj.drawn_layout
    inked = [r for r in layout.rows if r.text]
    for row, mark in zip(inked, layout.marks):
        assert mark.text == ".%d" % (inked.index(row) + 1)
        # Words end at the column's right edge, the marker past them.
        assert row.x1 == pytest.approx(BOX["x1"] - row.indent, abs=0.05)
        assert row.marker_x1 == pytest.approx(row.x1 + row.hang, abs=0.05)
        assert mark.x > row.x1


def test_justified_list_rows_spread_across_their_column(store):
    manifest, obj = _styled(store, alignment="JUSTIFY")
    apply_layout(manifest, 1, obj.id, list="bullet", store=store, target_lang="es")
    spread = [r for r in obj.drawn_layout.rows if r.spread]
    assert spread
    for r in spread:
        assert r.x0 == pytest.approx(BOX["x0"] + r.indent, abs=0.01)
        assert r.x1 == pytest.approx(BOX["x1"], abs=0.01)


def test_a_centred_marker_hangs_before_its_words(store):
    manifest, obj = _styled(store, alignment="CENTER")
    apply_layout(manifest, 1, obj.id, list="number", store=store, target_lang="es")
    for r in obj.drawn_layout.rows:
        if r.marker:
            assert r.marker_x0 == pytest.approx(r.x0 - r.hang, abs=0.01)


def test_levels_move_the_column_and_change_the_marker(store):
    manifest, obj = _styled(store)
    apply_layout(manifest, 1, obj.id, list="bullet", list_levels=[0, 1],
                 store=store, target_lang="es")
    firsts = [r for r in obj.drawn_layout.rows if r.marker]
    assert [r.marker for r in firsts] == ["•", "◦"]
    assert firsts[1].indent - firsts[0].indent == pytest.approx(2.0 * 11.0, abs=0.01)
    assert obj.drawn_layout.marks[1].font_path.endswith("NotoSansSymbols2-Regular.ttf")


def test_a_plain_indent_moves_the_words_without_a_marker(store):
    manifest, obj = _styled(store)
    apply_layout(manifest, 1, obj.id, indent=1, store=store, target_lang="es")
    rows = [r for r in obj.drawn_layout.rows if r.text]
    assert all(r.x0 == pytest.approx(BOX["x0"] + 22.0, abs=0.01) for r in rows)
    assert obj.drawn_layout.marks == []


def test_fit_wraps_in_the_narrower_column(store):
    manifest, plain = _styled(store, size=None)
    rows_plain = len(plain.drawn_layout.rows)
    size_plain = plain.drawn_layout.size
    apply_layout(manifest, 1, plain.id, indent=1, store=store, target_lang="es")
    apply_layout(manifest, 1, plain.id, indent=1, store=store, target_lang="es")
    layout = plain.drawn_layout
    assert layout.size < size_plain or len(layout.rows) > rows_plain
    # No word is wider than its own column.
    font = REGISTRY.resolve(SPANISH, script="Latin")
    for r in layout.rows:
        for word in r.text.split():
            assert REGISTRY.measure(font, word, layout.size) <= (BOX["x1"] - BOX["x0"]) - r.indent + 0.01


def test_a_locked_size_grows_the_box_for_indented_rows(store):
    manifest, obj = _styled(store, box={"x0": 72.0, "y0": 420.0, "x1": 200.0, "y1": 440.0})
    before = len(obj.drawn_layout.rows)
    apply_layout(manifest, 1, obj.id, list="bullet", list_levels=[2, 2],
                 store=store, target_lang="es")
    assert len(obj.drawn_layout.rows) > before
    last = obj.drawn_layout.rows[-1].baseline
    assert obj.drawn_bbox.y1 >= last


def test_the_narrow_column_keeps_room_for_words(store):
    manifest, obj = _styled(store, box={"x0": 72.0, "y0": 420.0, "x1": 120.0, "y1": 560.0})
    apply_layout(manifest, 1, obj.id, list="bullet", list_levels=[3, 3],
                 store=store, target_lang="es")
    for r in obj.drawn_layout.rows:
        assert 48.0 - r.indent >= MIN_COLUMN_EM * 11.0 - 0.01


def test_fitted_height_counts_indented_rows():
    box = Rect(x0=0, y0=0, x1=200, y1=20)
    spec = {"bold": False, "italic": False}
    plain = _fitted_height(box, SPANISH, spec, 11.0, "sans", None, "es")
    listed = _fitted_height(
        box, SPANISH, spec | {"_listed": {"manual_list": "bullet", "manual_indents": [2]}},
        11.0, "sans", None, "es",
    )
    assert listed.height > plain.height


def test_the_drawn_alignment_is_unchanged_by_a_list(store):
    manifest, obj = _styled(store, alignment="RIGHT")
    apply_layout(manifest, 1, obj.id, list="bullet", store=store, target_lang="es")
    assert obj.drawn_alignment is TextAlignment.RIGHT
    for r in obj.drawn_layout.rows:
        if r.text:
            assert r.x1 == pytest.approx(BOX["x1"], abs=0.01)


def test_the_editor_is_told_how_far_a_block_may_widen(store):
    """The live words widen into the same room before the picture comes,
    so a bullet does not squeeze them first (builder.lay_out)."""
    from ct_engine.api.routes import _block_out

    manifest = prepared(store)
    unit = next(
        t for t in manifest.page(0).text_objects if t.flat_text.startswith("Living Environment")
    )
    assert unit.widen_x1 is not None and unit.widen_x1 > unit.bbox.x1 + 4.0
    assert _block_out(unit)["widen_x1"] == unit.widen_x1
    apply_layout(
        manifest, 1, unit.id, store=store, target_lang="es",
        bbox={"x0": unit.bbox.x0, "y0": unit.bbox.y0, "x1": unit.bbox.x1 + 10, "y1": unit.bbox.y1},
    )
    # Placed by hand: never widened, so nothing to mirror.
    assert unit.widen_x1 is None and _block_out(unit)["widen_x1"] is None
