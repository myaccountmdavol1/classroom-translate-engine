"""Table cells that a PDF producer packed into one text block.

From the real Unit 1 packet, page 4: Word exported a 2x2 table of questions
with the top-right cell and the bottom-left cell in ONE block. The gutter
between the columns is 9pt -- under the 22pt a column split needs -- so the
two questions were translated as one sentence and drawn as one paragraph
across both cells. The table's own ruling lines say where each cell ends.
"""

from __future__ import annotations

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, VectorObject


def _line(box, text):
    r = Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    return TextLine(bbox=r, spans=[TextSpan(text=text, bbox=r)])


def _rule(box):
    r = Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    return VectorObject(id=f"v{box}", bbox=r, norm_bbox=r.normalized(612, 792),
                        line_count=1, looks_like_rule=True)


def _page():
    lines = [
        _line((311, 372, 557, 391), "What number can you triple and then "),
        _line((311, 391, 473, 411), "subtract 1 to get to 29? "),
        _line((311, 411, 315, 430), " "),
        _line((41, 548, 302, 568), "What can you double and then subtract "),
        _line((41, 568, 154, 587), "from 9 to get 5? "),
    ]
    box = Rect(x0=41, y0=372, x1=557, y1=587)
    block = TextObject(id="t1", kind=ObjectKind.TEXT, bbox=box,
                       norm_bbox=box.normalized(612, 792), source=TextSource.NATIVE,
                       lines=lines)
    rules = [_rule(b) for b in (
        (36, 371, 36, 724), (306, 371, 306, 724), (576, 371, 576, 724),
        (36, 371, 576, 371), (36, 547, 576, 547), (36, 723, 576, 723),
    )]
    return Page(index=3, number=4, width=612, height=792,
                text_objects=[block], vector_objects=rules)


def test_cells_divided_by_table_rules_become_separate_blocks():
    page = _page()
    split_page_blocks(page)
    texts = sorted(t.flat_text for t in page.text_objects if t.flat_text.strip())
    assert texts == [
        "What can you double and then subtract from 9 to get 5?",
        "What number can you triple and then subtract 1 to get to 29?",
    ]


def test_a_paragraph_with_no_rule_through_it_is_untouched():
    page = _page()
    page.vector_objects = []
    split_page_blocks(page)
    assert len([t for t in page.text_objects if t.flat_text.strip()]) == 1


def test_widening_stops_at_a_table_rule():
    """Page 4: the bottom-left question's translation was widened up to and
    over the cell's right-hand rule, because only text, pictures and writing
    areas counted as obstacles."""
    from ct_engine.reconstruct.builder import _obstacles
    from ct_engine.reconstruct.fitting import available_growth_right

    page = _page()
    split_page_blocks(page)
    cell = next(t for t in page.text_objects if t.flat_text.startswith("What can you double"))
    room = available_growth_right(cell.bbox, page.bounds, _obstacles(page, cell.id))
    assert cell.bbox.x1 + room <= 306
