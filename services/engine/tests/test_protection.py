"""Protection layer: the exam-integrity guarantee (spec sec. 20/23/24)."""

from __future__ import annotations

import pytest

from ct_engine.translate.protection import (
    find_protected_spans,
    has_translatable_content,
    is_page_number,
    mask_text,
    peel_edges,
    placeholder,
    restore_text,
    split_leading_label,
)


def categories(text: str) -> set[str]:
    return {s.category for s in find_protected_spans(text)}


def protected_strings(text: str) -> list[str]:
    return [s.text for s in find_protected_spans(text)]


# --- Detection ------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Solve 3x + 7 = 22 for x.", "3x + 7 = 22"),
        ("Use 5(y - 2) = 35 to find y.", "5(y - 2) = 35"),
        ("Solve -4m + 9 = -7.", "-4m + 9 = -7"),
        ("Compare 0.5t + 1.25 = 4.75 carefully.", "0.5t + 1.25 = 4.75"),
        ("The area is A = πr² exactly.", "A = πr²"),
        ("Recall E = mc² from physics.", "E = mc²"),
        ("Use 2a/3 + 4 = 10 here.", "2a/3 + 4 = 10"),
    ],
)
def test_equations_are_protected_whole(text, expected):
    """A partial match would corrupt the expression, which is worse than none."""
    assert expected in protected_strings(text)


def test_hyphen_minus_does_not_fragment_an_equation():
    """ASCII '-' is the minus sign in real documents, not a word separator."""
    spans = protected_strings("Use 5(y - 2) = 35 to solve.")
    assert spans == ["5(y - 2) = 35"]


@pytest.mark.parametrize(
    "text,expected",
    [("Water is H₂O.", "H₂O"), ("Carbon dioxide is CO2.", "CO2"),
     ("Glucose is C6H12O6.", "C6H12O6"), ("Vitamin B12 helps.", "B12")],
)
def test_chemical_formulas_are_protected(text, expected):
    assert expected in protected_strings(text)


def test_fill_in_blanks_are_protected():
    """Blanks are the student's writing space; a translator must not eat them."""
    assert "blank" in categories("Name: ______________________")
    assert "______" in protected_strings("Date: ______")


def test_urls_emails_measurements_citations():
    assert "url" in categories("Visit www.example.org/lab for details.")
    assert "email" in categories("Email teacher@school.edu today.")
    assert "measurement" in categories("The beaker holds 250 mL of water.")
    assert "citation" in categories("As shown in (Smith, 2019), rates rose.")


@pytest.mark.parametrize(
    "text",
    [
        "Read each question carefully and write your answer.",
        "The dog ran home - it was tired.",
        "This is a well-known fact about photosynthesis.",
        "The USA and the PDF file are fine.",
        "Turn to page 32 and look at question 6.",
        "Part A - Answer all questions in this part.",
        "Section B - Read the passage below.",
    ],
)
def test_ordinary_prose_is_not_protected(text):
    """False positives leave real sentences untranslated, so they matter too."""
    assert find_protected_spans(text) == []


def test_part_a_dash_is_not_mistaken_for_an_equation():
    """Regression: 'A -' was protected as math, silently killing directions."""
    text = "Part A - Answer all questions in this part."
    assert not find_protected_spans(text)
    assert has_translatable_content(text)


# --- Masking and restoration ---------------------------------------------
def test_mask_then_restore_is_lossless():
    text = "Solve 3x + 7 = 22 and visit www.example.org for help."
    masked = mask_text(text)
    assert "#0#" in masked.masked_text
    assert "3x + 7 = 22" not in masked.masked_text
    restored = restore_text(masked.masked_text, masked.spans)
    assert restored.ok
    assert restored.text == text


def test_restore_repairs_a_missing_leading_delimiter():
    """Observed NMT damage: '#0#' comes back as '0#'."""
    masked = mask_text("Solve 3x + 7 = 22 now.")
    damaged = masked.masked_text.replace("#0#", "0#")
    restored = restore_text(damaged, masked.spans)
    assert restored.ok and restored.repaired == [0]
    assert "3x + 7 = 22" in restored.text


def test_restore_repairs_adjacent_to_cjk_without_word_boundary():
    """Regression: '\\b' never matches in '\\u4e3a0#', so CJK repairs were missed."""
    masked = mask_text("Water is H₂O.")
    damaged = "水为0#。"
    restored = restore_text(damaged, masked.spans)
    assert restored.ok
    assert "H₂O" in restored.text


def test_restore_refuses_to_guess_when_the_index_is_destroyed():
    """'###' lost the digit: repairing it would be a guess, so it must fail."""
    masked = mask_text("The beaker holds 250 mL at 37 °C.")
    damaged = masked.masked_text.replace(masked.spans[0].token, "###")
    restored = restore_text(damaged, masked.spans)
    assert not restored.ok
    assert 0 in restored.missing


def test_restore_never_rewrites_a_bare_number():
    """A digit with no surviving '#' must not be treated as a placeholder."""
    masked = mask_text("Solve 3x + 7 = 22 now.")
    damaged = "Resuelve 0 ahora."   # bare digit, no delimiter
    restored = restore_text(damaged, masked.spans)
    assert not restored.ok
    assert "3x + 7 = 22" not in restored.text


def test_literal_sentinel_in_source_is_protected_first():
    """Source text containing '#1#' must not collide with our placeholders."""
    text = "Reference #1# applies here with 3x + 7 = 22."
    masked = mask_text(text)
    restored = restore_text(masked.masked_text, masked.spans)
    assert restored.ok and restored.text == text


def test_placeholder_format_is_the_measured_one():
    """Changing this without re-measuring NMT survival is a regression."""
    assert placeholder(3) == "#3#"


# --- Block-level helpers ---------------------------------------------------
@pytest.mark.parametrize("text", ["1", "14", "Page 7", "  23  "])
def test_page_numbers_are_recognized(text):
    assert is_page_number(text)


@pytest.mark.parametrize("text", ["1. Explain why", "Chapter 4 begins", "2019"])
def test_non_page_numbers_are_not(text):
    assert not is_page_number(text) or text == "2019"


def test_leading_question_number_is_split_off():
    """Renumbering a question breaks the answer key, so the label never moves."""
    label, body = split_leading_label("3. Describe one way organisms depend on each other.")
    assert label == "3. "
    assert body.startswith("Describe")


def test_leading_answer_letter_is_split_off():
    label, body = split_leading_label("(2) mitochondrion")
    assert label.strip() == "(2)"
    assert body == "mitochondrion"


def test_peel_edges_moves_trailing_blank_out_of_translation():
    """The most common worksheet pattern must never depend on a placeholder."""
    prefix, core, suffix = peel_edges("Name: ______________________")
    assert prefix == ""
    assert core == "Name: "
    assert suffix == "______________________"


def test_peel_edges_leaves_interior_protection_alone():
    prefix, core, suffix = peel_edges("Solve 3x + 7 = 22 for x.")
    assert prefix == "" and suffix == ""
    assert core == "Solve 3x + 7 = 22 for x."


def test_has_translatable_content():
    assert has_translatable_content("Explain your reasoning.")
    assert not has_translatable_content("3x + 7 = 22")
    assert not has_translatable_content("12345")
    assert not has_translatable_content("")


# --- Symbols the translation engine mangles -------------------------------


def test_standalone_ampersand_becomes_a_word_the_engine_can_translate():
    """Measured on a real packet cover: `&` came back as `&amp;`/`&quot;`.

    Masking it was worse -- `"Mrs. Rivera & Mrs. O'Connor"` came back as
    `"Sra. Rivera #0#"`, dropping a name -- so the conjunction is substituted
    and the engine renders it naturally in the target language.
    """
    from ct_engine.translate.protection import normalize_for_translation

    assert (
        normalize_for_translation("Simplifying Expressions & Solving Equations")
        == "Simplifying Expressions and Solving Equations"
    )
    assert (
        normalize_for_translation("Mrs. Rivera & Mrs. O'Connor")
        == "Mrs. Rivera and Mrs. O'Connor"
    )


def test_ampersand_inside_a_name_is_left_alone():
    """"AT&T" is a name, not a conjunction."""
    from ct_engine.translate.protection import normalize_for_translation

    for name in ("AT&T", "R&D budget", "Barnes&Noble", "Q&A"):
        assert normalize_for_translation(name) == name


def test_html_entities_from_the_engine_are_decoded():
    """A literal `&amp;` must never reach a student."""
    from ct_engine.translate.protection import repair_entities

    assert repair_entities("Sra. Rivera &amp; Sra. O", "Mrs. Rivera & Mrs. O") == (
        "Sra. Rivera & Sra. O"
    )
    assert repair_entities("5 &lt; 7", "5 < 7") == "5 < 7"
    assert repair_entities("caf&#233;", "cafe") == "café"
    assert repair_entities("caf&#xe9;", "cafe") == "café"


def test_an_entity_the_source_really_contained_is_preserved():
    """A worksheet teaching HTML must keep saying `&amp;`."""
    from ct_engine.translate.protection import repair_entities

    source = "Write &amp; to show an ampersand."
    assert repair_entities("Escribe &amp; para mostrar.", source) == (
        "Escribe &amp; para mostrar."
    )


def test_a_malformed_numeric_entity_is_left_as_is():
    from ct_engine.translate.protection import repair_entities

    assert repair_entities("&#999999;", "x") == "&#999999;"
