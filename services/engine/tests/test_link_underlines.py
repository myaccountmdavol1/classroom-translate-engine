"""A link's underline goes with its words.

Canva draws a link's underline as a thin bar, apart from the words. The
words were redrawn translated and the bar stayed where the English was: on
the Colorín Colorado slide it ran through the Spanish paragraph like a
strike-through.
"""
from __future__ import annotations

import fitz

from test_text_effects import _translated_pdf

LINK = "The Plant Library"
SPOT = (60, 100)


def _page(writing_line: bool = False) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(SPOT, LINK + " has many lessons for every class.", fontsize=14)
    width = fitz.get_text_length(LINK, fontsize=14)
    # The link's underline, drawn as a bar just under the baseline.
    page.draw_rect(fitz.Rect(SPOT[0], SPOT[1] + 2, SPOT[0] + width, SPOT[1] + 3.2),
                   color=None, fill=(0, 0, 0))
    if writing_line:
        page.insert_text((60, 300), "Name:", fontsize=14)
        page.draw_line((110, 302), (400, 302), color=(0, 0, 0), width=1)
    data = doc.tobytes()
    doc.close()
    return data


def _bars(page: fitz.Page, area: fitz.Rect) -> list[fitz.Rect]:
    # A stroked line's box has no height, which `intersects` counts as empty.
    return [r for r in (d["rect"] for d in page.get_drawings())
            if r.height <= 2.5 and area.y0 <= r.y0 <= area.y1 and r.x0 < area.x1 and r.x1 > area.x0]


def test_the_underline_of_a_redrawn_link_goes(store):
    with _translated_pdf(store, _page()) as out:
        line = fitz.Rect(SPOT[0] - 2, SPOT[1] - 14, 560, SPOT[1] + 6)
        assert not _bars(out[0], line), "the English link's underline is left under the translation"


def test_a_writing_line_beside_words_stays(store):
    with _translated_pdf(store, _page(writing_line=True)) as out:
        assert _bars(out[0], fitz.Rect(100, 295, 420, 310)), "the line to write a name on was taken away"


def test_an_underlined_heading_is_underlined_translated(store):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 100), "Course Guide", fontsize=18)
    width = fitz.get_text_length("Course Guide", fontsize=18)
    page.draw_rect(fitz.Rect(60, 102, 60 + width, 103.5), color=None, fill=(0, 0, 0))
    page.insert_text((60, 160), "Pick two classes for next year.", fontsize=12)
    with _translated_pdf(store, doc.tobytes()) as out:
        bars = _bars(out[0], fitz.Rect(40, 95, 560, 112))
        assert bars, "the heading lost its underline"
        assert bars[0].width > width, "drawn under the longer translation, not the English"
        assert not _bars(out[0], fitz.Rect(40, 150, 560, 170)), "a plain line was underlined"


def test_a_table_border_under_a_cell_stays(store):
    """A one-line cell's bottom border sits right under its words, but it
    meets the lines between the columns."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for x0, x1 in [(60, 200), (200, 340)]:
        page.draw_rect(fitz.Rect(x0, 117, x1, 118), color=None, fill=(0, 0, 0))
    for x in (60, 200, 340):
        page.draw_rect(fitz.Rect(x - 0.5, 98, x + 0.5, 118), color=None, fill=(0, 0, 0))
    page.insert_text((62, 115), "Monday morning class time", fontsize=11)
    page.insert_text((202, 115), "Tuesday evening club time", fontsize=11)
    with _translated_pdf(store, doc.tobytes()) as out:
        assert len(_bars(out[0], fitz.Rect(55, 112, 345, 122))) == 2, "the table's border was taken away"
