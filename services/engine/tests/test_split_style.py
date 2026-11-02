"""A block split from its neighbour takes the type size it is printed in."""
import fitz

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.ingest.pdf_parser import PdfParser


def _page():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 100), "01 Smoke Alarms and Carbon Monoxide Detectors", fontsize=13)
    page.insert_text((500, 100), "Section N/A", fontsize=7.4)
    page.draw_line((50, 108), (560, 108), width=0.5)
    page.insert_text((50, 200), "An ordinary paragraph that is never split.", fontsize=11)
    data = doc.tobytes()
    with PdfParser().open_bytes(data) as d:
        return d.parse_page(0)


def test_split_label_keeps_its_own_size():
    page = _page()
    merged = [b for b in page.text_objects if "Section N/A" in b.text and "Smoke" in b.text]
    assert merged, [b.text for b in page.text_objects]
    plain = next(b for b in page.text_objects if "ordinary" in b.text)
    plain_style = plain.style.model_copy()
    split_page_blocks(page)
    label = next(b for b in page.text_objects if b.text.strip() == "Section N/A")
    heading = next(b for b in page.text_objects if "Smoke" in b.text)
    assert label.style.font_size == 7.4
    assert heading.style.font_size == 13
    assert next(b for b in page.text_objects if "ordinary" in b.text).style == plain_style
