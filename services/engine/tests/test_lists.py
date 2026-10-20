"""The list rules on their own (reconstruct/lists.py): markers, counting,
levels, the numbers a worksheet typed itself, and where markers hang."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ct_engine.fonts.registry import REGISTRY
from ct_engine.reconstruct import lists
from ct_engine.reconstruct.lists import (
    REFUSE_BULLETS,
    REFUSE_NUMBERS,
    MarkerFonts,
    check_list,
    effective_levels,
    marker_visual,
    offsets,
    plan_paragraphs,
    typed_marker,
)


def box(kind=None, style=None, start=None, indents=None):
    return SimpleNamespace(
        manual_list=kind, manual_list_style=style, manual_list_start=start,
        manual_indents=indents, lines=[], source=None,
    )


def markers(text, **kw):
    return [p.marker for p in plan_paragraphs(text, box(**kw))]


def test_numbers_count_by_level_and_continue_across_the_box():
    text = "uno\ndos\ntres\ncuatro\ncinco\nseis\nsiete"
    got = markers(text, kind="number", indents=[0, 0, 1, 1, 0, 1, 2])
    assert got == ["1.", "2.", "a.", "b.", "3.", "a.", "i."]


def test_numbering_can_start_anywhere():
    assert markers("a\nb", kind="number", start=3) == ["3.", "4."]


def test_a_blank_line_has_no_marker_and_is_not_counted():
    assert markers("a\n\nb", kind="number") == ["1.", None, "2."]


def test_roman_and_letters_past_z():
    assert lists.to_roman(14) == "xiv" and lists.to_roman(1999) == "mcmxcix"
    assert lists.to_alpha(26) == "z" and lists.to_alpha(27) == "aa" and lists.to_alpha(28) == "ab"
    assert lists.format_number("upper-roman", 4) == "IV."
    assert lists.format_number("lower-alpha-paren", 2) == "b)"
    text = "\n".join("x" * 1 for _ in range(28))
    assert markers(text, kind="number", style="lower-alpha")[-2:] == ["aa.", "ab."]


@pytest.mark.parametrize("style,expected", [
    ("disc", ["•", "◦", "▪", "•"]),
    ("dash", ["–", "◦", "▪", "•"]),
    ("check", ["✓", "◦", "▪", "•"]),
    ("square", ["▪", "•", "◦", "▪"]),
])
def test_bullets_change_with_the_level(style, expected):
    assert markers("a\nb\nc\nd", kind="bullet", style=style, indents=[0, 1, 2, 3]) == expected


def test_paren_numbers_keep_the_bracket_at_every_level():
    got = markers("a\nb\nc", kind="number", style="decimal-paren", indents=[0, 1, 2])
    assert got == ["1)", "a)", "i)"]


def test_a_right_to_left_marker_shows_reversed():
    assert marker_visual("1.", True) == ".1"
    assert marker_visual("1.", False) == "1."


@pytest.mark.parametrize("text,marker", [
    ("1. x", "1."), ("12) x", "12)"), ("(a) x", "(a)"), ("b. x", "b."),
    ("IV. x", "IV."), ("• x", "•"), ("– x", "–"), ("ii. x", "ii."),
])
def test_typed_markers_are_found(text, marker):
    found = typed_marker(text)
    assert found is not None and found.marker == marker and found.rest == "x"


@pytest.mark.parametrize("text", [
    "3 apples", "1.5 cm", "e.g. x", "Mr. Smith", "U.S. x", "1.", "civil. x", "-5 grados",
])
def test_ordinary_text_is_not_a_marker(text):
    assert typed_marker(text) is None


def test_check_list_without_typed_markers():
    assert check_list("uno\ndos", "number").ok
    assert check_list("uno\ndos", "bullet").ok


def test_consecutive_typed_numbers_are_taken_over():
    got = check_list("3. uno\n4. dos\n5. tres", "number")
    assert got.ok and got.style == "decimal" and got.start == 3
    got = check_list("a) uno\nb) dos", "number")
    assert got.ok and got.style == "lower-alpha-paren" and got.start is None
    got = check_list("i. uno\nii. dos\niii. tres", "number")
    assert got.ok and got.style == "lower-roman"


def test_mixed_or_broken_numbering_is_refused():
    got = check_list("1. uno\na) dos", "number")
    assert not got.ok and got.message == REFUSE_NUMBERS
    assert not check_list("1. uno\n3. dos", "number").ok
    assert not check_list("1. uno\ndos", "number").ok


def test_bullets_over_numbers_are_refused_numbers_over_bullets_are_not():
    got = check_list("1. uno\n2. dos", "bullet")
    assert not got.ok and got.message == REFUSE_BULLETS
    assert check_list("• uno\n• dos", "number").ok
    got = check_list("– uno\n– dos", "bullet")
    assert got.ok and got.style == "dash"


def test_own_markers_says_what_the_editor_can_offer():
    out = lists.own_markers("1. uno\na) dos")
    assert out == {
        "kind": "numbers", "number_ok": False, "bullet_ok": False, "message": REFUSE_NUMBERS,
        "number_message": REFUSE_NUMBERS, "bullet_message": REFUSE_BULLETS,
        "number_style": None, "number_start": None, "bullet_style": None,
    }
    out = lists.own_markers("3. uno\n4. dos")
    assert out["number_style"] == "decimal" and out["number_start"] == 3
    assert out["number_message"] is None and out["bullet_message"] == REFUSE_BULLETS
    assert lists.own_markers("uno")["kind"] == "none"
    assert lists.own_markers("• uno\n1. dos")["kind"] == "mixed"


def test_levels_follow_the_line_index():
    assert effective_levels([1, 2], 1) == [1]
    assert effective_levels([1, 2], 4) == [1, 2, 2, 2]
    assert effective_levels(None, 3) == [0, 0, 0]
    assert lists.normalise_levels([0, 0]) is None


def test_typed_markers_are_split_off_without_a_list():
    paras = plan_paragraphs("3. Una planta\nsin número", box())
    assert paras[0].typed and paras[0].marker == "3." and paras[0].body == "Una planta"
    assert paras[1].marker is None and paras[1].body == "sin número"


def test_a_list_strips_what_it_replaces_and_flags_a_changed_number():
    paras = plan_paragraphs("5. uno\n6. dos", box(kind="number"))
    assert [p.marker for p in paras] == ["1.", "2."]
    assert [p.body for p in paras] == ["uno", "dos"]
    assert all(p.replaced for p in paras)
    same = plan_paragraphs("5. uno\n6. dos", box(kind="number", start=5))
    assert not any(p.replaced for p in same)
    bullets = plan_paragraphs("• uno\n2. dos", box(kind="bullet"))
    assert [p.body for p in bullets] == ["uno", "2. dos"]


def test_marker_faces_are_predictable():
    assert lists.marker_face("•", "bullet", None) == "text"
    assert lists.marker_face("◦", "bullet", lambda c: True) == "marks"
    assert lists.marker_face("•", "bullet", lambda c: False) == "marks"
    assert lists.marker_face("–", "bullet", lambda c: False) == "sans"
    assert lists.marker_face("1.", "number", lambda c: False) == "sans"


def _fonts():
    return MarkerFonts(REGISTRY.resolve("Hola", script="Latin"))


def test_managed_markers_share_a_hang_per_level():
    paras = plan_paragraphs("a\nb\nc", box(kind="number", indents=[0, 0, 1]))
    fonts = _fonts()
    out = offsets(paras, fonts, 10.0, 300.0)
    widest = max(fonts.measure("text", m, 10.0) for m in ("1.", "2."))
    assert out[0].hang == out[1].hang == pytest.approx(max(12.5, widest + 5.0))
    # Level 1 holds only "a.": MIN_HANG_EM wins there.
    assert out[2].marker_offset == pytest.approx(20.0)
    assert out[2].hang == pytest.approx(max(12.5, fonts.measure("text", "a.", 10.0) + 5.0))


def test_a_narrow_box_keeps_room_for_the_words():
    paras = plan_paragraphs("a", box(kind="bullet", indents=[3]))
    out = offsets(paras, _fonts(), 10.0, 50.0)
    assert out[0].text_offset == pytest.approx(30.0)  # 50 - MIN_COLUMN_EM * 10
    assert out[0].marker_offset + out[0].hang == pytest.approx(30.0)


def test_a_plain_paragraph_is_indented_by_its_level():
    paras = plan_paragraphs("a\nb", box(indents=[0, 2]))
    out = offsets(paras, _fonts(), 10.0, 300.0)
    assert out[1].text_offset == pytest.approx(40.0) and out[1].hang == 0.0


def test_a_missing_marks_face_never_draws_a_missing_glyph(monkeypatch):
    from ct_engine.fonts import catalog

    monkeypatch.setattr(catalog, "marks_face", lambda: None)
    paras = plan_paragraphs("a", box(kind="bullet", style="check"))
    marker, face = lists.drawable_marker(paras[0], _fonts())
    assert face == "text" and marker == "•"  # Helvetica has a bullet


@pytest.mark.parametrize("text", [
    "A. Lincoln fue presidente de los Estados Unidos.",
    "J. K. Rowling escribió muchos libros.",
    "- Hola, dijo ella.",
    "* Nota: lee con cuidado.",
])
def test_prose_that_starts_like_a_marker_stays_in_the_words(text):
    """An initial or a dialogue dash alone is the start of a sentence: it
    is not hung, and numbering does not read it as a letter."""
    para = plan_paragraphs(text, box())[0]
    assert para.marker is None and para.body == text
    assert check_list(text, "number").ok and check_list(text, "number").start is None
    assert lists.own_markers(text)["kind"] == "none"


def test_weak_markers_count_beside_others_of_their_family():
    paras = plan_paragraphs("a. uno\nb. dos", box())
    assert [p.marker for p in paras] == ["a.", "b."]
    paras = plan_paragraphs("- uno\n- dos", box())
    assert [p.marker for p in paras] == ["-", "-"]
    # Brackets are never prose, so "a)" alone still hangs.
    assert plan_paragraphs("a) uno", box())[0].marker == "a)"


def test_a_weak_marker_counts_when_the_source_set_it_apart():
    assert lists.typed_markers(["A. Lincoln"], first_hang=20.0)[0] is not None
    assert lists.typed_markers(["A. Lincoln"])[0] is None


def test_a_lone_roman_numeral_reads_as_roman():
    got = check_list("I) Introducción", "number")
    assert got.ok and got.style == "upper-roman-paren" and got.start is None
    got = check_list("V) quinto", "number")
    assert got.style == "upper-roman-paren" and got.start == 5
    # A letter that is a large roman numeral is a letter: "C)" is the third.
    got = check_list("C) tercero", "number")
    assert got.style == "upper-alpha-paren" and got.start == 3
    got = check_list("I. uno\nII. dos", "number")
    assert got.style == "upper-roman" and got.start is None


def test_a_list_keeps_the_source_hang_of_the_number_it_takes_over():
    paras = plan_paragraphs("3. uno\n4. dos", box(kind="number", start=3))
    assert paras[0].source_hang is None  # no source lines here
    fonts = MarkerFonts(REGISTRY.resolve("Hola", script="Latin"))
    paras[0].source_hang = 21.6
    got = offsets(paras, fonts, 10.0, 300.0)
    assert got[0].text_offset == pytest.approx(21.6)
    # One hang per level, so the next item's words line up with it.
    assert got[1].text_offset == pytest.approx(21.6)
