"""Text inside images (spec sec. 18 / 19).

The governing rule is that a diagram must never be damaged to replace a
label: anything we cannot read confidently, or cannot paint over safely, is
left exactly as it was and reported.
"""

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
from ct_engine.model.enums import SemanticRole, TextSource
from ct_engine.pipeline.image_text import (
    MIN_LABEL_CONFIDENCE,
    extract_image_text,
    looks_instructional,
)
from ct_engine.pipeline.translate import translate_page
from ct_engine.providers import get_ocr_provider
from ct_engine.reconstruct.builder import rebuild_document, uniform_growth_right
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Cell Wall", True), ("Nucleus", True), ("Atlantic Ocean", True),
        ("Step 1", True), ("Mitochondrion", True), ("Time (hours)", True),
        ("NN", False), ("oN", False), ("@", False), ("3:", False),
        ("12", False), ("Aue", False), ("", False),
    ],
)
def test_instructional_filter(text, expected):
    """OCR debris must never be painted onto a diagram as a 'translation'."""
    assert looks_instructional(text) is expected


@needs_tesseract
def test_diagram_labels_are_found():
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(doc, page, provider, language="en")

    assert stats.images_examined == 1
    assert stats.images_with_text == 1
    labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert labels, "no labels were read from the diagram"
    found = " ".join(t.flat_text for t in labels).lower()
    assert "cell wall" in found
    for label in labels:
        assert label.role is SemanticRole.IMAGE_LABEL
        assert label.confidence is not None
        assert label.confidence >= MIN_LABEL_CONFIDENCE
        # Labels must land inside the image they came from.
        image = page.image_objects[0]
        assert image.bbox.contains(label.bbox, tolerance=2.0)


@needs_tesseract
def test_photograph_yields_no_false_labels():
    """A continuous-tone photo has no text; inventing some would be worse."""
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("06_photographs.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(doc, page, provider, language="en")
    assert stats.labels_accepted == 0
    assert not [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]


@needs_tesseract
def test_unreadable_diagram_labels_are_rejected_and_reported():
    """A photocopied figure must be left alone, not guessed at."""
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("15_lowres_diagram.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(doc, page, provider, language="en")
    assert stats.labels_rejected > 0, "low-quality labels should not all pass"
    if stats.labels_rejected:
        assert any(f.code == "image_label_unreadable" for f in stats.findings)
    for label in [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]:
        assert label.confidence >= MIN_LABEL_CONFIDENCE


@needs_tesseract
def test_full_page_scan_is_left_to_the_page_ocr_stage():
    """Otherwise a scan would be read twice and labels duplicated."""
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(doc, page, provider, language="en")
    assert stats.images_examined == 0


def test_background_probe_stops_at_artwork():
    """Sideways growth must halt where the diagram begins."""
    from ct_engine.model.geometry import Rect

    doc = fitz.open(stream=corpus_file("04_science_diagram.pdf"), filetype="pdf")
    try:
        page = doc.load_page(0)
        clear = uniform_growth_right(page, Rect(x0=79, y0=163, x1=110, y1=172))
        across = uniform_growth_right(page, Rect(x0=120, y0=250, x1=140, y1=262))
        assert clear > across, "growth should be larger over blank paper"
    finally:
        doc.close()


@needs_tesseract
def test_labels_are_covered_not_redacted_and_artwork_survives():
    """The end-to-end promise: translated labels, same diagram."""
    data = corpus_file("04_science_diagram.pdf")
    provider = get_ocr_provider("local")
    pages = []
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        extract_image_text(doc, page, provider, language="en")
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
        pages.append(page)

    out, stats = rebuild_document(data, pages, target_lang="es")
    # Raster text is painted over; only native text is redacted.
    assert stats[0].covered > 0, "image labels were not covered"

    src = fitz.open(stream=data, filetype="pdf")
    dst = fitz.open(stream=out, filetype="pdf")
    try:
        assert dst.page_count == src.page_count == 1
        # The original artwork is reused byte-for-byte, never re-encoded.
        src_img = src.extract_image(src[0].get_images()[0][0])["image"]
        dst_img = dst.extract_image(dst[0].get_images()[0][0])["image"]
        assert sha256_bytes(src_img) == sha256_bytes(dst_img)
    finally:
        src.close()
        dst.close()


@needs_tesseract
def test_image_labels_translate_and_are_marked():
    data = corpus_file("04_science_diagram.pdf")
    provider = get_ocr_provider("local")
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        extract_image_text(doc, page, provider, language="en")
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")

    labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert labels
    assert any(t.translated_text for t in labels)
    assert page.image_objects[0].contains_text is True


def test_contains_text_is_tri_state_until_examined():
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        page = doc.parse_page(0)
    # "Not examined" must never read as "no text".
    assert page.image_objects[0].contains_text is None
