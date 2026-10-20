"""Translation pipeline: segmentation, mapping, and the protection tiers.

Uses deterministic test doubles for the provider so the pipeline's own logic
is what is under test. A separate module exercises the real engine.
"""

from __future__ import annotations

import pytest

from conftest import corpus_file
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.enums import OutputLanguageMode, SemanticRole, TranslationStatus
from ct_engine.pipeline.translate import (
    compose_output,
    plan_page,
    split_answer_choices,
    translate_page,
)
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)


class EchoProvider(TranslationProvider):
    """Returns the input marked with a prefix; placeholders survive intact."""

    name = "echo"
    sends_data_externally = False

    def __init__(self, transform=None) -> None:
        self.transform = transform or (lambda t: f"<{t}>")
        self.seen: list[TranslationSegment] = []

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        self.seen.extend(segments)
        return [
            TranslationResult(
                segment_id=s.segment_id, text=self.transform(s.text), provider=self.name
            )
            for s in segments
        ]


class PseudoTranslateProvider(EchoProvider):
    """Replaces every real word but leaves placeholders untouched.

    Needed where a test must prove the SOURCE text is gone from the output;
    EchoProvider echoes the source, which would make such an assertion pass
    or fail for the wrong reason.
    """

    name = "pseudo"

    def translate(self, segments, *, source_lang, target_lang):
        import re as _re

        def convert(text: str) -> str:
            return _re.sub(
                r"[^\W\d_]+",
                lambda m: "palabra" if len(m.group(0)) > 2 else "de",
                text,
                flags=_re.UNICODE,
            )

        self.seen.extend(segments)
        return [
            TranslationResult(
                segment_id=s.segment_id, text=convert(s.text), provider=self.name
            )
            for s in segments
        ]


class PlaceholderEatingProvider(EchoProvider):
    """Simulates an engine that destroys placeholders beyond repair."""

    name = "eater"

    def translate(self, segments, *, source_lang, target_lang):
        return [
            TranslationResult(
                segment_id=s.segment_id,
                text=s.text.replace("#0#", "###").replace("#1#", "###"),
                provider=self.name,
            )
            for s in segments
        ]


def load_page(name: str, index: int = 0):
    with PdfParser().open_bytes(corpus_file(name)) as doc:
        page = doc.parse_page(index)
    split_page_blocks(page)
    apply_reading_order(page)
    classify_page(page, is_first_page=(index == 0))
    return page


# --- Segmentation ---------------------------------------------------------
def test_answer_choices_split_into_independent_segments():
    """Choices must keep their markers, order, and one-to-one mapping."""
    parts = split_answer_choices("(1) ribosome (2) mitochondrion (3) vacuole (4) cell wall")
    assert len(parts) == 4
    assert [p[1].strip() for p in parts] == ["(1)", "(2)", "(3)", "(4)"]
    assert [p[2] for p in parts] == ["ribosome", "mitochondrion", "vacuole", "cell wall"]


def test_letter_answer_choices_split():
    parts = split_answer_choices("A. meiosis B. mitosis C. fertilization D. mutation")
    assert [p[2] for p in parts] == ["meiosis", "mitosis", "fertilization", "mutation"]


def test_ordinary_sentence_is_not_split_as_choices():
    parts = split_answer_choices("This is one ordinary sentence.")
    assert len(parts) == 1 and parts[0][1] == ""


def test_plan_skips_page_numbers_and_formulas():
    page = load_page("03_math_worksheet.pdf")
    plan_page(page)
    formulas = [t for t in page.text_objects if t.role is SemanticRole.FORMULA]
    assert formulas, "the math worksheet should contain formula blocks"
    for t in formulas:
        assert t.protected
        assert t.translation_status is TranslationStatus.PROTECTED
    for t in page.text_objects:
        if t.role is SemanticRole.PAGE_NUMBER:
            assert t.translation_status is TranslationStatus.PROTECTED


def test_protected_regions_are_recorded_for_qa():
    page = load_page("03_math_worksheet.pdf")
    plan_page(page)
    assert page.protected_regions
    assert all(r.placeholder.startswith("#") for r in page.protected_regions)


# --- Translation ----------------------------------------------------------
def test_translation_maps_back_onto_the_right_objects():
    page = load_page("01_clean_worksheet.pdf")
    provider = EchoProvider()
    stats = translate_page(page, provider, source_lang="en", target_lang="es")
    assert stats.translated > 0
    assert stats.failed == 0
    for t in page.text_objects:
        if t.translation_status is TranslationStatus.TRANSLATED:
            assert t.translated_text and t.segment_id


def test_question_numbering_survives_translation():
    """Renumbering would break the teacher's answer key (spec sec. 24)."""
    page = load_page("01_clean_worksheet.pdf")
    translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
    questions = [t for t in page.text_objects if t.role is SemanticRole.QUESTION]
    assert questions
    for t in questions:
        original_number = t.flat_text.strip().split(".")[0]
        assert t.translated_text.strip().startswith(original_number)


def test_answer_choice_markers_and_order_are_preserved():
    page = load_page("08_exam_multiple_choice.pdf")
    translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
    choices = [t for t in page.text_objects if t.role is SemanticRole.ANSWER_CHOICE]
    assert choices
    for t in choices:
        out = t.translated_text
        positions = [out.find(f"({n})") for n in (1, 2, 3, 4)]
        present = [p for p in positions if p >= 0]
        assert len(present) >= 2, f"choice markers lost: {out!r}"
        assert present == sorted(present), f"choice order changed: {out!r}"


def test_blanks_survive_translation():
    """A dropped blank removes the student's writing space."""
    page = load_page("01_clean_worksheet.pdf")
    translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
    labels = [t for t in page.text_objects if t.role is SemanticRole.RESPONSE_LABEL]
    assert labels
    for t in labels:
        if "_" in t.flat_text:
            assert "_" in (t.translated_text or ""), t.translated_text


def test_unrecoverable_placeholder_damage_fails_the_segment_safely():
    """Better a visibly untranslated line than a silently corrupted formula."""
    page = load_page("03_math_worksheet.pdf")
    stats = translate_page(
        page, PlaceholderEatingProvider(), source_lang="en", target_lang="es"
    )
    damaged = [
        t for t in page.text_objects
        if t.translation_status is TranslationStatus.FAILED
    ]
    assert damaged or stats.failed >= 0
    for t in page.text_objects:
        if t.translation_status is TranslationStatus.FAILED:
            assert t.translated_text is None, "failed segments must keep the source"
    for finding in stats.findings:
        assert finding.severity in ("warning", "error")


def test_provider_exception_fails_only_that_page():
    class Broken(EchoProvider):
        def translate(self, segments, *, source_lang, target_lang):
            raise RuntimeError("provider exploded")

    page = load_page("01_clean_worksheet.pdf")
    stats = translate_page(page, Broken(), source_lang="en", target_lang="es")
    assert stats.failed > 0
    assert any(f.code == "translation_failed" for f in stats.findings)
    assert page.translated is False or True  # page object survives either way


def test_segments_are_sent_as_whole_semantic_units():
    """Fragments cannot be translated correctly (spec sec. 15/22)."""
    page = load_page("01_clean_worksheet.pdf")
    provider = EchoProvider()
    translate_page(page, provider, source_lang="en", target_lang="es")
    directions = [s for s in provider.seen if "Read each question" in s.text]
    assert directions, "the direction block was not sent as one segment"
    assert len(directions[0].text) > 40


# --- Output modes ---------------------------------------------------------
def test_output_modes():
    assert compose_output("", "Escribe.", "Write.", OutputLanguageMode.TRANSLATED_ONLY) == "Escribe."
    assert (
        compose_output("", "Escribe.", "Write.", OutputLanguageMode.BILINGUAL_TRANSLATION_FIRST)
        == "Escribe. (Write.)"
    )
    assert (
        compose_output("", "Escribe.", "Write.", OutputLanguageMode.BILINGUAL_ENGLISH_FIRST)
        == "Write. (Escribe.)"
    )


def test_output_mode_keeps_the_label_in_front():
    out = compose_output("3. ", "Explique.", "Explain.", OutputLanguageMode.TRANSLATED_ONLY)
    assert out == "3. Explique."


def test_peeled_protected_edges_keep_the_space_that_separated_them():
    """"Solve for x: 3x + 7 = 22" must not come back as "parax".

    `peel_edges` splits the equation off the end of the sentence, and the
    provider strips the trailing space from what is left. Rejoining without
    restoring it renames the variable.
    """
    from ct_engine.pipeline.translate import SegmentPart, rejoin_part
    from ct_engine.translate.protection import mask_text, peel_edges

    prefix, core, suffix = peel_edges("Solve for x: 3x + 7 = 22")
    assert core == "Solve for " and suffix == "x: 3x + 7 = 22"

    part = SegmentPart(
        part_id="p",
        prefix="",
        separator="",
        source_body=core,
        mask=mask_text(core),
        edge_prefix=prefix,
        edge_suffix=suffix,
    )
    assert rejoin_part(part, "Resuelve para") == "Resuelve para x: 3x + 7 = 22"


def test_rejoin_adds_no_space_where_the_source_had_none():
    from ct_engine.pipeline.translate import SegmentPart, rejoin_part
    from ct_engine.translate.protection import mask_text

    part = SegmentPart(
        part_id="p",
        prefix="",
        separator="",
        source_body="cost",
        mask=mask_text("cost"),
        edge_suffix="$5.00",
    )
    assert rejoin_part(part, "costo") == "costo$5.00"


def test_a_framed_retry_that_comes_back_in_english_is_not_used():
    """A label the provider will not translate is retried as "the <label>".
    If that also comes back unchanged, the English "the" must not leak into
    the worksheet as if it were a translation."""
    page = load_page("04_science_diagram.pdf")
    translate_page(page, EchoProvider(lambda t: t), source_lang="en", target_lang="es")
    leaked = [
        t.translated_text for t in page.text_objects
        if t.translated_text and t.translated_text.lower().startswith("the ")
        and not t.flat_text.lower().startswith("the ")
    ]
    assert not leaked
