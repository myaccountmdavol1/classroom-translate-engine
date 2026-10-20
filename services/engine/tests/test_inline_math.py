"""Inline math inside prose: exponents, subscripts and symbol-font glyphs.

The fixture was typed in Microsoft Word and saved as PDF by Word itself
(tests/fixtures/make_word_math_worksheet.py), because that is how a teacher's
worksheet reaches us. Word marks a superscript with PyMuPDF's superscript flag,
but a subscript only by a smaller size and a lower baseline, and it writes π as
the letter "p" in the SymbolMT font. Before this was handled, "x²" reached the
translator as "x2" and was redrawn on the baseline, and "2πr" came back "2pr".
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.math_runs import normalize_line_spans, symbol_to_unicode
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextSpan, TextStyle
from ct_engine.pipeline.translate import translate_page
from ct_engine.reconstruct.builder import rebuild_document
from ct_engine.translate.protection import find_protected_spans
from test_translate_pipeline import EchoProvider

FIXTURE = Path(__file__).parent / "fixtures" / "word_math_worksheet.pdf"


def _texts() -> list[str]:
    with PdfParser().open_bytes(FIXTURE.read_bytes()) as doc:
        page = doc.parse_page(0)
    return [t.flat_text for t in page.text_objects]


def _joined() -> str:
    return "\n".join(_texts())


# --- Reading --------------------------------------------------------------
@pytest.mark.parametrize(
    "expected",
    ["area is x² square", "H₂O", "CO₂.", "is 2πr.", "3 × 10⁸ meters", "aᵐ · aⁿ"],
)
def test_word_math_is_read_as_the_characters_the_reader_sees(expected):
    assert expected in _joined()


def test_flattened_forms_are_gone():
    text = _joined()
    for wrong in ("x2 square", "H2O", "2pr", "108 meters"):
        assert wrong not in text


def test_raised_ordinal_suffixes_stay_ordinary_words():
    """"1st" must reach the translator as a word it can localise (1.º, 1er)."""
    text = _joined()
    assert "in 1st place" in text and "was 2nd." in text


# --- Span normalisation ---------------------------------------------------
def _span(text, size, y_base, *, font="Cambria", x0=0.0):
    return TextSpan(
        text=text,
        bbox=Rect(x0=x0, y0=y_base - size, x1=x0 + 5 * len(text), y1=y_base + 2),
        style=TextStyle(font_family=font, font_name_raw=font, font_size=size),
    )


def test_superscript_is_converted_and_marked():
    spans = [_span("area x", 11, 100), _span("2", 7, 97.6)]
    normalize_line_spans(spans, origins=[100, 97.6], flags=[4, 5])
    assert spans[1].text == "²" and spans[1].script == "super"


def test_subscript_needs_no_flag_only_size_and_lower_baseline():
    spans = [_span("H", 11, 100), _span("2", 7, 101)]
    normalize_line_spans(spans, origins=[100, 101], flags=[4, 4])
    assert spans[1].text == "₂" and spans[1].script == "sub"


def test_superscript_with_no_unicode_form_is_kept_and_marked():
    """There is no superscript q. Keep the glyph text and say it is raised,
    so the page can be flagged rather than silently flattened."""
    spans = [_span("x", 11, 100), _span("q", 7, 97.6)]
    normalize_line_spans(spans, origins=[100, 97.6], flags=[4, 5])
    assert spans[1].text == "q" and spans[1].script == "super"
    assert not spans[1].script_converted


def test_same_size_text_is_never_treated_as_a_script():
    spans = [_span("Total", 11, 100), _span("2", 11, 100)]
    normalize_line_spans(spans, origins=[100, 100], flags=[4, 4])
    assert spans[1].text == "2" and spans[1].script is None


@pytest.mark.parametrize(
    "font,raw,expected",
    [("SymbolMT", "p", "π"), ("Symbol", "a", "α"), ("Symbol", "", "π"),
     ("Symbol", "\xb4", "×"), ("Symbol", "\xa3", "≤"), ("Cambria", "p", "p")],
)
def test_symbol_font_letters_become_their_symbols(font, raw, expected):
    assert symbol_to_unicode(raw, font) == expected


# --- Protection -----------------------------------------------------------
@pytest.mark.parametrize("term", ["x²", "aᵐ · aⁿ", "2πr", "3 × 10⁸"])
def test_converted_math_is_protected(term):
    protected = "".join(s.text for s in find_protected_spans(f"Find {term} now."))
    assert term.replace(" ", "") in protected.replace(" ", "")


# --- End to end -----------------------------------------------------------
def test_math_survives_translation_and_rebuild():
    """Uppercasing stands in for translation: anything sent to the provider
    unprotected comes back changed, which is how "x2" became "X2"."""
    data = FIXTURE.read_bytes()
    pages = []
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(str.upper), source_lang="en", target_lang="es")
        pages.append(page)
    out, _ = rebuild_document(data, pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as rebuilt:
        text = rebuilt[0].get_text()
    for kept in ("x²", "H₂O", "CO₂", "2πr", "10⁸", "aᵐ", "aⁿ"):
        assert kept in text, kept
    assert "1ST PLACE" in text  # the ordinal really was translated


# --- Honest reporting -----------------------------------------------------
def _page_with(span_texts):
    """One translated block whose second span was drawn raised."""
    from ct_engine.model.enums import ObjectKind, TextSource, TranslationStatus
    from ct_engine.model.objects import TextLine, TextObject
    from ct_engine.model.document import Page

    spans = [_span(span_texts[0], 11, 100), _span(span_texts[1], 7, 97.6, x0=30)]
    normalize_line_spans(spans, origins=[100, 97.6], flags=[4, 5])
    box = Rect(x0=0, y0=80, x1=200, y1=110)
    obj = TextObject(id="text_q", kind=ObjectKind.TEXT, bbox=box,
                     norm_bbox=box.normalized(612, 792), source=TextSource.NATIVE,
                     lines=[TextLine(bbox=box, spans=spans)],
                     translated_text="traducido",
                     translation_status=TranslationStatus.TRANSLATED)
    return Page(index=0, number=1, width=612, height=792, text_objects=[obj])


def test_a_raised_glyph_that_could_not_be_kept_is_reported():
    from ct_engine.pipeline.reconstruct_stage import verify_page

    codes = {f.code for f in verify_page(_page_with(["x", "q"])).findings}
    assert "math_script_flattened" in codes


def test_converted_scripts_raise_no_warning():
    from ct_engine.pipeline.reconstruct_stage import verify_page

    codes = {f.code for f in verify_page(_page_with(["x", "2"])).findings}
    assert "math_script_flattened" not in codes


# --- Math-font glyphs (Word equations, LaTeX) ------------------------------
# Word exports an equation in a subset of Cambria Math whose text layer is
# nonsense: the fraction 3/4 + 1/8 extracts as '! " + # $'. Only the drawn
# glyphs are right, so they must be left exactly as drawn and never be
# translated or redrawn from their "text".
EQUATION_CLIP = fitz.Rect(123, 263, 144, 288)  # "3/4 + 1/8" on line 7, glyphs only


def _rebuilt_fixture():
    data = FIXTURE.read_bytes()
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
    split_page_blocks(page)
    apply_reading_order(page)
    classify_page(page, is_first_page=True)
    translate_page(page, EchoProvider(str.upper), source_lang="en", target_lang="es")
    out, _ = rebuild_document(data, [page], target_lang="es")
    return data, page, out


def test_equation_glyphs_are_never_sent_for_translation():
    _, page, _ = _rebuilt_fixture()
    drawn = " ".join(t.translated_text or "" for t in page.text_objects)
    for garbage in ("$", "!", '"', "#"):
        assert garbage not in drawn


def test_the_words_around_an_equation_are_still_translated():
    _, page, _ = _rebuilt_fixture()
    drawn = [t.translated_text or "" for t in page.text_objects]
    assert any(t.startswith("7. ADD") for t in drawn)
    assert any("WRITE THE ANSWER IN SIMPLEST FORM" in t for t in drawn)


def test_equation_is_left_exactly_as_drawn():
    data, _, out = _rebuilt_fixture()

    def pixels(pdf):
        with fitz.open(stream=pdf, filetype="pdf") as d:
            return d[0].get_pixmap(dpi=144, clip=EQUATION_CLIP, alpha=False).samples

    assert pixels(out) == pixels(data)
