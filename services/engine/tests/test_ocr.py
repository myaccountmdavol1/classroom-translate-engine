"""OCR: recognition, coordinate mapping, confidence, and raster reconstruction."""

from __future__ import annotations

import shutil

import fitz
import pytest

import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
from conftest import corpus_file
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.enums import TextSource
from ct_engine.pipeline.ocr_stage import (
    DISCARD_CONFIDENCE,
    REVIEW_CONFIDENCE,
    ocr_page,
    pages_needing_ocr,
)
from ct_engine.pipeline.translate import translate_page
from ct_engine.providers import OCR_PROVIDERS, get_ocr_provider
from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult, OCRWord
from ct_engine.reconstruct.builder import estimate_background, rebuild_document
from test_translate_pipeline import EchoProvider

has_tesseract = shutil.which("tesseract") is not None
needs_tesseract = pytest.mark.skipif(
    not has_tesseract, reason="tesseract binary not installed"
)


def scanned_page(name: str = "02_scanned_worksheet.pdf", index: int = 0):
    with PdfParser().open_bytes(corpus_file(name)) as doc:
        return doc.parse_page(index)


# --- Registry and availability -------------------------------------------
def test_local_ocr_provider_is_registered():
    assert "local" in OCR_PROVIDERS
    assert OCR_PROVIDERS["local"].sends_data_externally is False


def test_scanned_pages_are_the_ones_selected_for_ocr():
    pages = []
    for name in ("01_clean_worksheet.pdf", "02_scanned_worksheet.pdf"):
        with PdfParser().open_bytes(corpus_file(name)) as doc:
            pages.append(doc.parse_page(0))
    needing = pages_needing_ocr(pages)
    assert len(needing) == 1
    assert needing[0].is_scanned


def test_a_page_already_read_is_not_read_again():
    page = scanned_page()
    page.ocr_done = True
    assert pages_needing_ocr([page]) == []


# --- Recognition ----------------------------------------------------------
@needs_tesseract
def test_ocr_reads_a_scanned_worksheet():
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        png = doc.render_page_png(0, dpi=300)
    result = provider.recognize(png, language="en")
    assert result.error is None
    assert len(result.regions) >= 5
    assert result.mean_confidence > 0.7
    joined = " ".join(r.text for r in result.regions).lower()
    assert "photosynthesis" in joined
    assert "chlorophyll" in joined


@needs_tesseract
def test_regions_carry_word_level_boxes():
    """Word boxes are what let a line be split and covered precisely."""
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        png = doc.render_page_png(0, dpi=300)
    result = provider.recognize(png, language="en")
    with_words = [r for r in result.regions if r.words]
    assert with_words
    for region in with_words:
        for word in region.words:
            assert word.text.strip()
            assert 0.0 <= word.confidence <= 1.0
            x0, y0, x1, y1 = word.bbox
            assert x1 > x0 and y1 > y0
            # Each word must sit inside its line.
            assert x0 >= region.bbox[0] - 2 and x1 <= region.bbox[2] + 2


@needs_tesseract
def test_confidence_is_reported_and_bounded():
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("14_scanned_exam.pdf")) as doc:
        png = doc.render_page_png(0, dpi=300)
    result = provider.recognize(png, language="en")
    for region in result.regions:
        assert 0.0 <= region.confidence <= 1.0
    assert 0.0 < result.mean_confidence <= 1.0


@needs_tesseract
def test_low_confidence_words_are_kept_not_silently_deleted():
    """Regression: a 30.0 floor deleted a correctly-read 'A student' (conf 9)."""
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("14_scanned_exam.pdf")) as doc:
        png = doc.render_page_png(0, dpi=300)
    result = provider.recognize(png, language="en")
    joined = " ".join(r.text for r in result.regions).lower()
    assert "student" in joined, "a correctly-read word was dropped by the threshold"


def test_provider_reports_missing_tesseract_rather_than_pretending():
    from ct_engine.providers.ocr import local_tesseract as lt

    lt.tesseract_binary.cache_clear()
    original = lt.shutil.which
    lt.shutil.which = lambda _name: None
    try:
        result = lt.LocalTesseractOCRProvider().recognize(b"", language="en")
        assert result.error and "not installed" in result.error
        assert result.regions == []
    finally:
        lt.shutil.which = original
        lt.tesseract_binary.cache_clear()


# --- Stage: coordinate mapping and flags ---------------------------------
@needs_tesseract
def test_ocr_objects_land_in_page_coordinates():
    page = scanned_page()
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        stats = ocr_page(doc, page, provider, language="en")

    assert page.ocr_done
    assert stats.regions > 0
    assert page.text_objects
    for obj in page.text_objects:
        assert obj.source is TextSource.OCR
        assert obj.confidence is not None
        # Pixel coordinates would be ~4x the page size; points must fit.
        assert 0 <= obj.bbox.x0 < page.width
        assert 0 <= obj.bbox.y0 < page.height
        assert obj.bbox.x1 <= page.width + 1
        assert obj.bbox.y1 <= page.height + 1
        assert 4.0 <= obj.style.font_size <= 72.0


@needs_tesseract
def test_recognized_text_matches_the_original_documents_words():
    """The scan came from a known worksheet, so we can check real content."""
    page = scanned_page()
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        ocr_page(doc, page, provider, language="en")
    text = " ".join(t.flat_text for t in page.text_objects).lower()
    for word in ("photosynthesis", "directions", "chlorophyll", "glucose"):
        assert word in text, f"OCR missed {word!r}"


class _StubOCR(OCRProvider):
    name = "stub"
    sends_data_externally = False

    def __init__(self, regions):
        self._regions = regions

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, *, language=None, detect_handwriting=False):
        total = sum(len(r.text) for r in self._regions) or 1
        return OCRResult(
            regions=self._regions,
            mean_confidence=sum(r.confidence * len(r.text) for r in self._regions)
            / total,
            provider=self.name,
        )


def test_very_low_confidence_regions_are_discarded_and_reported():
    page = scanned_page()
    regions = [
        OCRRegion(text="clearly readable", bbox=(100, 100, 600, 140), confidence=0.95),
        OCRRegion(text="~|noise|~", bbox=(100, 200, 300, 240),
                  confidence=DISCARD_CONFIDENCE - 0.1),
    ]
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        stats = ocr_page(doc, page, _StubOCR(regions), language="en")
    assert stats.discarded_regions == 1
    assert stats.regions == 1
    assert any(f.code == "ocr_discarded" for f in stats.findings)


def test_low_confidence_regions_are_flagged_for_review():
    page = scanned_page()
    regions = [
        OCRRegion(text="hard to read line", bbox=(100, 100, 600, 140),
                  confidence=REVIEW_CONFIDENCE - 0.1),
    ]
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        stats = ocr_page(doc, page, _StubOCR(regions), language="en")
    assert stats.low_confidence_regions == 1
    finding = next(f for f in stats.findings if f.code == "ocr_low_confidence")
    assert finding.severity == "warning"
    assert finding.object_ids


def test_empty_ocr_result_is_reported_not_silently_ignored():
    page = scanned_page()
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        stats = ocr_page(doc, page, _StubOCR([]), language="en")
    assert stats.regions == 0
    assert any(f.code == "ocr_empty" for f in stats.findings)


def test_ocr_provider_error_becomes_a_page_finding_not_an_exception():
    class Broken(OCRProvider):
        name = "broken"

        def supports_language(self, lang):
            return True

        def recognize(self, image_png, *, language=None, detect_handwriting=False):
            return OCRResult(provider="broken", error="engine unavailable")

    page = scanned_page()
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        stats = ocr_page(doc, page, Broken(), language="en")
    assert stats.regions == 0
    assert any(f.code == "ocr_failed" for f in stats.findings)
    assert page.ocr_done


def test_unreadable_glyph_boxes_are_kept_for_covering():
    """Otherwise a scanned line renders half translated, half original."""
    page = scanned_page()
    regions = [
        OCRRegion(
            text="readable part",
            bbox=(100, 100, 600, 140),
            confidence=0.9,
            words=[OCRWord(text="readable", bbox=(100, 100, 300, 140), confidence=0.9),
                   OCRWord(text="part", bbox=(310, 100, 420, 140), confidence=0.9)],
            discarded_boxes=[(430, 100, 600, 140)],
        )
    ]
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        ocr_page(doc, page, _StubOCR(regions), language="en")
    obj = page.text_objects[0]
    assert obj.cover_boxes, "discarded glyph boxes were not carried through"


# --- Background estimation and reconstruction ----------------------------
def test_background_estimation_finds_paper_and_refuses_artwork():
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        pass
    doc = fitz.open(stream=corpus_file("02_scanned_worksheet.pdf"), filetype="pdf")
    try:
        page = doc.load_page(0)
        from ct_engine.model.geometry import Rect

        # An empty area of the scan: uniform paper, should yield a colour.
        color, spread = estimate_background(page, Rect(x0=80, y0=600, x1=300, y1=615))
        assert color is not None
        assert all(c > 0.7 for c in color), "paper should be light"
    finally:
        doc.close()

    # A busy photograph must be refused rather than painted over.
    doc = fitz.open(stream=corpus_file("06_photographs.pdf"), filetype="pdf")
    try:
        page = doc.load_page(0)
        from ct_engine.model.geometry import Rect

        color, spread = estimate_background(page, Rect(x0=100, y0=200, x1=300, y1=240))
        assert color is None, "painting over a photograph must be refused"
        assert spread > 0
    finally:
        doc.close()


@needs_tesseract
def test_scanned_worksheet_translates_end_to_end():
    """Phase 3 success criterion: no selectable source text required."""
    data = corpus_file("02_scanned_worksheet.pdf")
    provider = get_ocr_provider("local")
    pages = []
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        assert not page.text_objects, "the scan must start with no text layer"
        ocr_page(doc, page, provider, language="en")
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
        pages.append(page)

    out, stats = rebuild_document(data, pages, target_lang="es")
    assert stats[0].written > 0
    # Raster text is covered, never redacted.
    assert stats[0].covered > 0
    assert stats[0].redacted == 0

    result = fitz.open(stream=out, filetype="pdf")
    try:
        assert result.page_count == 1
        # The rebuilt page now carries selectable text where it had none.
        assert len(result[0].get_text().strip()) > 50
        # And the scan itself is still there.
        assert len(result[0].get_images()) == 1
    finally:
        result.close()


@needs_tesseract
def test_scanned_page_keeps_its_image_and_page_size():
    data = corpus_file("14_scanned_exam.pdf")
    provider = get_ocr_provider("local")
    pages = []
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            ocr_page(doc, page, provider, language="en")
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
            pages.append(page)
    out, _ = rebuild_document(data, pages, target_lang="es")

    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        assert dst.page_count == src.page_count == 4
        for i in range(src.page_count):
            assert src[i].rect.width == pytest.approx(dst[i].rect.width, abs=0.5)
            assert len(dst[i].get_images()) == len(src[i].get_images())
    finally:
        src.close()
        dst.close()


def test_tesseract_misreads_of_print_are_repaired():
    from ct_engine.providers.ocr.base import OCRWord
    from ct_engine.providers.ocr.local_tesseract import _repair_words

    def words(*pairs):
        return [OCRWord(text=t, bbox=(i * 10.0, 0.0, i * 10.0 + 8, 10.0), confidence=c)
                for i, (t, c) in enumerate(pairs)]

    kept, junk = _repair_words(words(
        ("[«", 0.42), ("}}", 0.37), ("lf", 0.78), ("|", 0.72), ("will", 0.9),
        ("go", 0.9), ("|'m", 0.9), ("here", 0.9), ("|", 0.9), ("Name", 0.9),
        ("—", 0.8), ("|have", 0.9),
    ))
    assert " ".join(w.text for w in kept) == "If I will go I'm here | Name — I have"
    assert [w.text for w in junk] == ["[«", "}}"]


def test_doubted_letters_are_never_dropped_as_junk():
    from ct_engine.providers.ocr.base import OCRWord
    from ct_engine.providers.ocr.local_tesseract import _repair_words

    kept, junk = _repair_words([
        OCRWord(text="A", bbox=(0, 0, 8, 10), confidence=0.09),
        OCRWord(text="student", bbox=(10, 0, 50, 10), confidence=0.96),
    ])
    assert [w.text for w in kept] == ["A", "student"] and junk == []


def test_icon_scrap_before_a_sentence_is_dropped():
    from ct_engine.providers.ocr.base import OCRWord
    from ct_engine.providers.ocr.local_tesseract import _repair_words

    def words(*pairs):
        return [OCRWord(text=t, bbox=(i * 10.0, 0.0, i * 10.0 + 8, 10.0), confidence=c)
                for i, (t, c) in enumerate(pairs)]

    # The real slide: "[« }} a | will never share my iPad..."
    kept, junk = _repair_words(words(
        ("[«", 0.42), ("}}", 0.37), ("a", 0.3), ("|", 0.94), ("will", 0.95),
        ("never", 0.95), ("share", 0.95),
    ))
    assert " ".join(w.text for w in kept) == "I will never share"
    assert [w.text for w in junk] == ["[«", "}}", "a"]
    # A confidently read "a" at the start of a line is a word, not a scrap.
    kept, _ = _repair_words(words(("a", 0.9), ("Tree", 0.9)))
    assert [w.text for w in kept] == ["a", "Tree"]
