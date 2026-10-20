"""Ruled tables are read, not dropped.

Tesseract treats a text cell boxed in by 1-2pt rules as part of a picture:
on a scanned table it recovered as few as 0 of 40 words, at every OCR
resolution. Removing the grid from the copy it reads -- never from the page
-- lets it see the cells as text again.
"""
from __future__ import annotations

import io
import re
import shutil
import tracemalloc

import fitz
import numpy as np
import pytest
from PIL import Image

from ct_engine.pipeline.ocr_prep import remove_grid_lines

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)

ROWS = [
    ["Planet", "Distance", "Moons", "Day length"],
    ["Mercury", "58 million km", "0", "59 days"],
    ["Venus", "108 million km", "0", "243 days"],
    ["Earth", "150 million km", "1", "24 hours"],
    ["Mars", "228 million km", "2", "25 hours"],
    ["Jupiter", "778 million km", "95", "10 hours"],
]
COLUMNS = [72, 172, 312, 392, 520]
TOP, ROW_HEIGHT = 90, 24


def _page(*, table=True, blank=False, rule_width=1.0) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 70), "Use the table to answer the questions.", fontsize=12)
    if table:
        for r, row in enumerate(ROWS):
            for c, cell in enumerate(row):
                page.insert_text(
                    (COLUMNS[c] + 5, TOP + r * ROW_HEIGHT + 16), cell, fontsize=11
                )
        bottom = TOP + len(ROWS) * ROW_HEIGHT
        for r in range(len(ROWS) + 1):
            y = TOP + r * ROW_HEIGHT
            page.draw_line((COLUMNS[0], y), (COLUMNS[-1], y), width=rule_width)
        for x in COLUMNS:
            page.draw_line((x, TOP), (x, bottom), width=rule_width)
    if blank:
        page.insert_text((72, 300), "Name:", fontsize=12)
        page.draw_line((110, 302), (320, 302), width=1.0)
    return doc


def _png(doc: fitz.Document, dpi: int = 300) -> bytes:
    return doc[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY).tobytes("png")


def _grey(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"))


def _px(points: float, dpi: int = 300) -> int:
    return int(round(points * dpi / 72))


def test_grid_lines_are_removed_from_the_ocr_copy():
    png = _png(_page())
    cleaned = remove_grid_lines(png, dpi=300)
    assert cleaned is not None
    img = _grey(cleaned)
    # Along a row rule, between cells: nothing dark should remain.
    y = _px(TOP + ROW_HEIGHT)
    x0, x1 = _px(COLUMNS[1] + 10), _px(COLUMNS[2] - 10)
    assert (img[y - 3 : y + 4, x0:x1] < 128).sum() == 0
    # Along a column rule, inside a row.
    x = _px(COLUMNS[2])
    y0, y1 = _px(TOP + 3), _px(TOP + ROW_HEIGHT - 3)
    assert (img[y0:y1, x - 3 : x + 4] < 128).sum() == 0


def test_cell_text_survives_rule_removal():
    png = _png(_page())
    before = _grey(png)
    after = _grey(remove_grid_lines(png, dpi=300))
    # Inside each cell of the first data row, clear of every rule.
    y0, y1 = _px(TOP + ROW_HEIGHT + 3), _px(TOP + 2 * ROW_HEIGHT - 3)
    cells = [(_px(a + 3), _px(b - 3)) for a, b in zip(COLUMNS, COLUMNS[1:])]
    ink_before = sum((before[y0:y1, a:b] < 128).sum() for a, b in cells)
    ink_after = sum((after[y0:y1, a:b] < 128).sum() for a, b in cells)
    assert ink_before > 0
    assert ink_after >= 0.9 * ink_before


def test_a_lone_answer_blank_is_not_a_grid():
    """An answer line has nothing crossing it; it must reach OCR unchanged."""
    png = _png(_page(table=False, blank=True))
    assert remove_grid_lines(png, dpi=300) is None


def test_a_page_with_no_lines_is_left_alone():
    assert remove_grid_lines(_png(_page(table=False)), dpi=300) is None


def test_rule_removal_stays_within_a_small_memory_budget():
    """The engine runs in 512 MB with ~390 MB already used."""
    png = _png(_page())
    tracemalloc.start()
    try:
        remove_grid_lines(png, dpi=300)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 80 * 1024 * 1024, f"peak {peak / 2**20:.0f} MB"


@needs_tesseract
@pytest.mark.parametrize("rule_width", [1.0, 2.0])
def test_scanned_ruled_table_is_read(rule_width):
    import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
    from ct_engine.ingest.pdf_parser import PdfParser
    from ct_engine.pipeline.ocr_stage import ocr_page
    from ct_engine.providers import get_ocr_provider

    scan = fitz.open()
    scan.new_page(width=612, height=792).insert_image(
        fitz.Rect(0, 0, 612, 792), stream=_png(_page(rule_width=rule_width))
    )
    with PdfParser().open_bytes(scan.tobytes()) as doc:
        page = doc.parse_page(0)
        ocr_page(doc, page, get_ocr_provider("local"), language="en")
    read = re.findall(r"\w+", " ".join(t.flat_text for t in page.text_objects))
    truth = [w for row in ROWS for cell in row for w in cell.split()]
    found = sum(1 for w in truth if w in read)
    assert found >= 0.9 * len(truth), f"read {found} of {len(truth)} table words"


def test_one_tall_junk_word_does_not_set_the_type_size():
    """Real packet, page 10: the HW starburst beside "Name:" was read as a
    65pt-tall "2". The line's height became its type size, the label was
    later cut down to its 12pt-tall words, and the translation was drawn at
    40pt over the instructions below it."""
    from ct_engine.model.document import Page
    from ct_engine.pipeline.ocr_stage import _region_to_object
    from ct_engine.providers.ocr.base import OCRRegion, OCRWord

    region = OCRRegion(
        text="Name: 2",
        bbox=(165, 20, 361, 85),
        confidence=0.62,
        words=[
            OCRWord(text="Name:", bbox=(165, 72, 199, 84), confidence=0.96),
            OCRWord(text="2", bbox=(300, 20, 312, 85), confidence=0.28),
        ],
    )
    page = Page(index=0, number=1, width=612, height=792)
    obj = _region_to_object(region, scale=1.0, page=page, index=0)
    assert obj.style.font_size <= 14
