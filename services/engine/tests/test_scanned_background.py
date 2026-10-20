"""Scanned text on plain paper is replaced; scanned text on artwork is not.

The quick paper check samples a thin ring just outside a block. On a
single-spaced scan that ring runs through the ascenders and descenders of the
neighbouring lines, so ordinary paragraphs were refused as "on a picture" and
left in English. Those strokes are recognised text, not artwork, so the
fallback check ignores them and judges only what remains.
"""
from __future__ import annotations

import shutil

import fitz
import pytest

from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.builder import (
    estimate_background,
    estimate_scanned_background,
)

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)

PARAGRAPH = (
    "We will be holding our traditional parent-teacher conferences on Monday, "
    "November 23, 2026, for grades 7-12. Our afternoon conferences will be held "
    "from 12:45 p.m.-4:00 p.m. for grades 7 and 8, and 1:45 p.m.-4:00 p.m. for "
    "grades 9-12. Students will follow an early release time schedule on this "
    "day. Grades 7 and 8 will be dismissed at 12:20 p.m. and grades 9-12 will be "
    "dismissed at 1:20 p.m. Additional information about this important event "
    "will be sent to you separately."
)
BODY = fitz.Rect(72, 80, 540, 200)


def _scan(draw_under=None, draw_over=None) -> tuple[bytes, list[Rect], list[Rect]]:
    """A single-spaced justified paragraph, flattened to a greyscale scan.

    Returns the image-only PDF, one rect per printed line, and every word box
    -- the geometry OCR would report, taken from the text before flattening.
    """
    src = fitz.open()
    page = src.new_page(width=612, height=792)
    if draw_under:
        draw_under(page)
    page.insert_htmlbox(
        BODY,
        f"<p>{PARAGRAPH}</p>",
        css="p{font-family:sans-serif;font-size:11.5pt;line-height:1.0;"
        "text-align:justify;margin:0}",
    )
    if draw_over:
        draw_over(page)
    # Glyph-tight boxes, like Tesseract's, not line-height boxes.
    fitz.TOOLS.set_small_glyph_heights(True)
    try:
        words = page.get_text("words")
    finally:
        fitz.TOOLS.set_small_glyph_heights(False)
    lines: dict[tuple[int, int], list] = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    line_rects = [
        Rect(
            x0=min(w[0] for w in ws),
            y0=min(w[1] for w in ws),
            x1=max(w[2] for w in ws),
            y1=max(w[3] for w in ws),
        )
        for ws in lines.values()
    ]
    word_rects = [Rect(x0=w[0], y0=w[1], x1=w[2], y1=w[3]) for w in words]
    pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    out = fitz.open()
    out.new_page(width=612, height=792).insert_image(
        fitz.Rect(0, 0, 612, 792), stream=pix.tobytes("png")
    )
    return out.tobytes(), line_rects, word_rects


def _middle_line(line_rects):
    return sorted(line_rects, key=lambda r: r.y0)[len(line_rects) // 2]


def test_quick_check_refuses_a_single_spaced_line():
    """Pins the bug: the thin ring alone calls plain paper a picture."""
    data, lines, _ = _scan()
    with fitz.open(stream=data, filetype="pdf") as doc:
        color, _ = estimate_background(doc[0], _middle_line(lines))
    assert color is None


def test_single_spaced_line_on_paper_is_accepted_once_neighbours_are_ignored():
    data, lines, words = _scan()
    with fitz.open(stream=data, filetype="pdf") as doc:
        color = estimate_scanned_background(doc[0], _middle_line(lines), words)
    assert color is not None
    assert min(color) > 0.9, "paper should read as white"


def test_line_on_a_flat_coloured_panel_is_accepted_in_that_colour():
    data, lines, words = _scan(
        draw_under=lambda p: p.draw_rect(
            fitz.Rect(60, 70, 552, 210), color=None, fill=(0.8, 0.8, 0.8)
        )
    )
    with fitz.open(stream=data, filetype="pdf") as doc:
        color = estimate_scanned_background(doc[0], _middle_line(lines), words)
    assert color is not None
    assert all(abs(c - 0.8) < 0.05 for c in color)


def _photograph(page):
    """Hard-edged tiles of varied tone: nothing a flat patch could match."""
    for i in range(0, 480, 12):
        for j in range(0, 130, 6):
            tone = ((i * 7 + j * 13) % 100) / 100
            page.draw_rect(
                fitz.Rect(66 + i, 76 + j, 78 + i, 82 + j),
                color=None,
                fill=(tone, tone, tone),
            )


def test_line_on_a_photograph_is_refused():
    data, lines, words = _scan(draw_under=_photograph)
    with fitz.open(stream=data, filetype="pdf") as doc:
        assert estimate_scanned_background(doc[0], _middle_line(lines), words) is None


def test_line_crossed_by_a_diagram_rule_is_refused():
    """A patch would cut the rule where it passes through the words."""
    data, lines, words = _scan(
        draw_over=lambda p: p.draw_line((300, 60), (300, 220), width=1.2)
    )
    with fitz.open(stream=data, filetype="pdf") as doc:
        assert estimate_scanned_background(doc[0], _middle_line(lines), words) is None


def test_long_rule_beside_the_text_is_refused():
    """A leader or arrow running along the line is artwork, not paper."""
    data, lines, words = _scan()
    target = _middle_line(lines)

    def rule(page):
        y = target.y1 + 1.5
        page.draw_line((target.x0, y), (target.x1, y), width=1.0)

    data, _, _ = _scan(draw_over=rule)
    with fitz.open(stream=data, filetype="pdf") as doc:
        assert estimate_scanned_background(doc[0], target, words) is None


def test_no_known_text_means_no_second_opinion():
    """Without word boxes there is nothing to set aside; defer to the quick check."""
    data, lines, _ = _scan()
    with fitz.open(stream=data, filetype="pdf") as doc:
        assert estimate_scanned_background(doc[0], _middle_line(lines), []) is None


@needs_tesseract
def test_single_spaced_scanned_letter_is_translated_not_left_in_english():
    import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
    from ct_engine.analysis.block_split import split_page_blocks
    from ct_engine.analysis.classify import classify_page
    from ct_engine.analysis.reading_order import apply_reading_order
    from ct_engine.ingest.pdf_parser import PdfParser
    from ct_engine.pipeline.ocr_stage import ocr_page
    from ct_engine.pipeline.translate import translate_page
    from ct_engine.providers import get_ocr_provider
    from ct_engine.reconstruct.builder import rebuild_document
    from test_translate_pipeline import EchoProvider

    data, _, _ = _scan()
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        ocr_page(doc, page, get_ocr_provider("local"), language="en")
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")

    _, stats = rebuild_document(data, [page], target_lang="es")
    refused = [
        f for f in stats[0].findings if f.code == "scanned_text_not_replaced"
    ]
    assert refused == []
    assert stats[0].covered > 0


@needs_tesseract
def test_a_white_box_replaces_scanned_text_the_paper_check_refused():
    """Real packet: "Solve" and "Check" are underlined, so the paper behind
    them is not plain and the translation was never drawn -- the page kept
    the English, while the editor showed the translation in the text box.
    A teacher who gives the box a background has answered the question the
    paper check could not: cover it with this colour."""
    import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
    from ct_engine.analysis.block_split import split_page_blocks
    from ct_engine.analysis.classify import classify_page
    from ct_engine.analysis.reading_order import apply_reading_order
    from ct_engine.ingest.pdf_parser import PdfParser
    from ct_engine.pipeline.ocr_stage import ocr_page
    from ct_engine.pipeline.translate import translate_page
    from ct_engine.providers import get_ocr_provider
    from ct_engine.reconstruct.builder import rebuild_document
    from test_translate_pipeline import EchoProvider

    data, _, _ = _scan(draw_under=_photograph)

    def build(fill):
        with PdfParser().open_bytes(data) as doc:
            page = doc.parse_page(0)
            ocr_page(doc, page, get_ocr_provider("local"), language="en")
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=True)
            translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
        for t in page.text_objects:
            t.manual_fill = fill
        _, stats = rebuild_document(data, [page], target_lang="es")
        return [f for f in stats[0].findings if f.code == "scanned_text_not_replaced"], stats[0]

    refused, _ = build(None)
    assert refused, "fixture must be refused without a fill"
    refused, stats = build("#ffffff")
    assert refused == [] and stats.covered > 0
