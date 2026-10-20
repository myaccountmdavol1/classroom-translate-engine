"""Collapsing text the engine read twice.

Page OCR and image-text extraction can both read the same pixels. Before
this pass, both readings were translated and both were drawn, so the reader
got two translations printed on top of each other.
"""

from __future__ import annotations

from ct_engine.analysis.dedupe import drop_duplicate_text
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import NormRect, Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan


_next_id = iter(range(1, 10_000))


def _obj(text, box, source=TextSource.OCR, confidence=0.9, oid=None):
    rect = Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    return TextObject(
        id=oid or f"o_{next(_next_id):04d}",
        kind=ObjectKind.TEXT,
        bbox=rect,
        norm_bbox=NormRect(x0=0, y0=0, x1=1, y1=1),
        source=source,
        confidence=confidence,
        lines=[TextLine(bbox=rect, spans=[TextSpan(text=text, bbox=rect)])],
    )


def _page(objects):
    return Page(index=0, number=1, width=612, height=792, text_objects=list(objects))


def test_the_same_words_read_twice_collapse_to_one():
    """The real case: one icon's caption read by both OCR stages."""
    page = _page([
        _obj("ie Solve the equations.", (45, 98, 210, 125),
             TextSource.IMAGE_OCR, 0.902),
        _obj("Ry Solve the equations.", (46, 98, 210, 125),
             TextSource.OCR, 0.898),
    ])
    dropped = drop_duplicate_text(page)
    assert len(dropped) == 1
    assert len(page.text_objects) == 1
    # The more confident reading survives.
    assert page.text_objects[0].text.startswith("ie ")


def test_the_dropped_reading_is_still_covered():
    """Its pixels must be painted over, or the original shows through."""
    page = _page([
        _obj("Date:", (371, 72, 399, 84), TextSource.IMAGE_OCR, 0.96),
        _obj("Date:", (371, 72, 399, 84), TextSource.OCR, 0.80),
    ])
    drop_duplicate_text(page)
    kept = page.text_objects[0]
    assert kept.cover_boxes, "the dropped block's area is no longer covered"


def test_native_text_always_wins_over_a_recognition():
    page = _page([
        _obj("Solve the equations.", (45, 98, 210, 125), TextSource.OCR, 0.99),
        _obj("Solve the equations.", (45, 98, 210, 125), TextSource.NATIVE, None),
    ])
    drop_duplicate_text(page)
    assert len(page.text_objects) == 1
    assert page.text_objects[0].source is TextSource.NATIVE


def test_two_native_blocks_are_never_merged():
    """The parser does not report the same glyphs twice."""
    page = _page([
        _obj("Total", (45, 98, 210, 125), TextSource.NATIVE, None, oid="a"),
        _obj("Total", (45, 98, 210, 125), TextSource.NATIVE, None, oid="b"),
    ])
    assert drop_duplicate_text(page) == []
    assert len(page.text_objects) == 2


def test_different_text_in_the_same_place_is_kept():
    """A label over a diagram is not a duplicate of the diagram's caption."""
    page = _page([
        _obj("Chloroplast", (45, 98, 210, 125), TextSource.OCR, 0.9),
        _obj("Mitochondrion", (45, 98, 210, 125), TextSource.IMAGE_OCR, 0.9),
    ])
    assert drop_duplicate_text(page) == []
    assert len(page.text_objects) == 2


def test_the_same_text_far_apart_is_kept():
    """"Name:" appears twice on a page; both are real."""
    page = _page([
        _obj("Name:", (40, 70, 90, 84), TextSource.OCR, 0.9),
        _obj("Name:", (400, 500, 450, 514), TextSource.OCR, 0.9),
    ])
    assert drop_duplicate_text(page) == []
    assert len(page.text_objects) == 2


def test_a_short_label_inside_a_long_line_is_kept():
    """Containment alone is not duplication; overlap must be mutual."""
    page = _page([
        _obj("Name", (40, 70, 80, 84), TextSource.OCR, 0.9),
        _obj("Name: ______ Date: ______", (40, 70, 400, 84), TextSource.OCR, 0.9),
    ])
    assert drop_duplicate_text(page) == []


def test_a_person_created_block_is_never_dropped():
    page = _page([
        _obj("JOIN THE ADVENTURE", (45, 98, 210, 125), TextSource.OCR, 0.9),
        _obj("JOIN THE ADVENTURE", (45, 98, 210, 125), TextSource.MANUAL, None),
    ])
    assert drop_duplicate_text(page) == []
    assert len(page.text_objects) == 2


# --- One reading split into lines, the other not ---------------------------
# From the real Unit 1 packet, page 33: page OCR read question 6 as one
# two-line block; image OCR read the same pixels as two one-line blocks.
# Neither line covers half of the paragraph, so the pairwise rule kept all
# three, and the page printed two Chinese translations over each other.
def test_lines_inside_a_paragraph_that_already_says_them_are_dropped():
    para = _obj(
        "sold 100 more angelfish than guppies. If the total sales was $3,608, "
        "how many angelfish did they sell?",
        (72, 601, 559, 632), source=TextSource.OCR, confidence=0.95, oid="para",
    )
    line1 = _obj(
        "sold 100 more angelfish than guppies. If the total sales was $3,608, how many angelfish",
        (72, 601, 559, 615), source=TextSource.IMAGE_OCR, confidence=0.95, oid="l1",
    )
    line2 = _obj("did they sell?", (72, 619, 144, 632),
                 source=TextSource.IMAGE_OCR, confidence=0.96, oid="l2")
    page = _page([para, line1, line2])
    dropped = drop_duplicate_text(page)
    assert sorted(dropped) == ["l1", "l2"]
    assert [t.id for t in page.text_objects] == ["para"]
    # The paragraph still covers every pixel the dropped lines would have.
    assert len(page.text_objects[0].cover_boxes) == 2


def test_a_label_inside_a_longer_reading_of_it_is_dropped():
    """Page 10: "Name:" read again inside "Name: 2"."""
    longer = _obj("Name: 2", (165, 72, 361, 84), confidence=0.62, oid="long")
    label = _obj("Name:", (165, 72, 199, 84), source=TextSource.IMAGE_OCR,
                 confidence=0.96, oid="label")
    page = _page([longer, label])
    assert drop_duplicate_text(page) == ["label"]


def test_a_different_word_inside_a_box_is_not_a_duplicate():
    para = _obj("Label the parts of the cell below.", (50, 100, 400, 140), oid="p")
    other = _obj("Nucleus", (60, 105, 120, 118), source=TextSource.IMAGE_OCR, oid="n")
    page = _page([para, other])
    assert drop_duplicate_text(page) == []
