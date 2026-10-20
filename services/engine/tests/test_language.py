"""Language detection and the target-language registry."""

from __future__ import annotations

import pytest

from ct_engine.analysis.language import detect_language, sample_document_text
from ct_engine.languages import LANGUAGES, get_language, source_language_name


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Read each question carefully and write your answer on the lines below.", "en"),
        ("Lee cada pregunta con atención y escribe tu respuesta en las líneas.", "es"),
        ("请仔细阅读每个问题，并在下面的横线上写出你的答案。", "zh"),
        ("اقرأ كل سؤال بعناية واكتب إجابتك على السطور أدناه", "ar"),
        ("Прочитайте каждый вопрос внимательно и напишите свой ответ ниже.", "ru"),
        ("각 질문을 주의 깊게 읽고 아래 줄에 답을 쓰십시오.", "ko"),
    ],
)
def test_detects_common_languages(text, expected):
    code, confidence, _ = detect_language(text)
    assert code == expected
    assert confidence > 0.5


def test_japanese_is_not_mistaken_for_chinese():
    """Kana is the distinguishing evidence; Han alone is ambiguous."""
    code, _, _ = detect_language("この文章を読んで、下の行に答えを書きなさい。")
    assert code == "ja"


def test_refuses_to_guess_on_short_input():
    """A wrong source language degrades OCR and translation; unknown is safer."""
    for text in ["", "hi", "ok", "12 34", "   "]:
        code, confidence, _ = detect_language(text)
        assert code is None and confidence == 0.0


def test_detection_is_deterministic():
    text = "Write a short paragraph explaining your reasoning using evidence."
    results = {detect_language(text)[0] for _ in range(5)}
    assert len(results) == 1


def test_sampling_spans_the_whole_document():
    pages = ["English text on the first page."] * 3 + ["Texto en español aquí."] * 20
    sample = sample_document_text(pages)
    assert "español" in sample, "sampling ignored the body of the document"


def test_registry_entries_are_well_formed():
    for lang in LANGUAGES:
        assert lang.code and lang.base and lang.english_name and lang.native_name
        assert 0.3 <= lang.expansion_factor <= 2.0
        if lang.rtl:
            assert lang.complex_shaping or lang.script in {"Arabic", "Hebrew"}


def test_chinese_variants_are_distinct():
    hans, hant = get_language("zh-Hans"), get_language("zh-Hant")
    assert hans.english_name != hant.english_name
    assert hans.native_name != hant.native_name
    # A bare "zh" must still resolve, defaulting to the most common variant.
    assert get_language("zh").code == "zh-Hans"


def test_portuguese_variants_are_distinct():
    assert get_language("pt-BR").english_name != get_language("pt-PT").english_name


def test_cjk_expansion_factors_are_below_one():
    """CJK is more compact than English; the layout engine relies on this."""
    for code in ("zh-Hans", "zh-Hant", "ja", "ko"):
        assert get_language(code).expansion_factor < 1.0


def test_unknown_code_returns_none():
    assert get_language("klingon") is None
    assert get_language(None) is None


def test_source_language_names_cover_detection_output():
    assert source_language_name("en") == "English"
    assert source_language_name("ht") == "Haitian Creole"
    assert source_language_name(None) is None
