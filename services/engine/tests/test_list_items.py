"""A heading and a numbered list that Word stored as one text block.

From the real Unit 1 packet, page 16: "To Solve Equations with Variables on
Both Sides:" in bold, then five numbered steps, all one block. Translated as
one, the steps came back as a single run-on paragraph and the list was gone.
"""

from __future__ import annotations

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle


def _line(box, text, bold=False):
    r = Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    return TextLine(bbox=r, spans=[TextSpan(text=text, bbox=r, style=TextStyle(bold=bold))])


def _page(lines):
    box = Rect.bounding([ln.bbox for ln in lines])
    block = TextObject(id="t1", kind=ObjectKind.TEXT, bbox=box,
                       norm_bbox=box.normalized(612, 792), source=TextSource.NATIVE,
                       lines=lines)
    return Page(index=15, number=16, width=612, height=792, text_objects=[block])


def _texts(page):
    return [t.flat_text for t in page.text_objects]


def test_heading_and_each_numbered_step_become_their_own_blocks():
    page = _page([
        _line((36, 205, 472, 230), "To Solve Equations with Variables on Both Sides:", bold=True),
        _line((36, 230, 47, 255), "​  "),
        _line((36, 256, 390, 281), "1. Use the ________ Property "),
        _line((36, 282, 489, 307), "2. Combine Like Terms ON THE _____ SIDE "),
        _line((36, 334, 573, 359), "4. Bring all ______ (numbers without a variable) to the "),
        _line((36, 359, 334, 384), "other side using inverse operations "),
        _line((36, 385, 311, 410), "5. Solve like a one-step equation "),
    ])
    split_page_blocks(page)
    assert _texts(page) == [
        "To Solve Equations with Variables on Both Sides:",
        "1. Use the ________ Property",
        "2. Combine Like Terms ON THE _____ SIDE",
        "4. Bring all ______ (numbers without a variable) to the other side using inverse operations",
        "5. Solve like a one-step equation",
    ]


def test_a_wrapped_paragraph_is_left_whole():
    page = _page([
        _line((36, 100, 500, 120), "The ticket costs $"),
        _line((36, 120, 500, 140), "1.50 for students and more for adults "),
        _line((36, 140, 300, 160), "who come to the game."),
    ])
    split_page_blocks(page)
    assert len(page.text_objects) == 1


def test_a_tab_aligned_worked_example_keeps_its_rows_and_columns():
    """Page 26: "Equation:<tab>a + a + 3 = 15" and the rows under it."""
    page = _page([
        _line((36, 498, 103, 518), "Example: "),
        _line((36, 519, 567, 538), "Andrea and Patrick ran a total of 15 miles. "),
        _line((36, 538, 267, 558), "How many miles did they each run? "),
        _line((72, 620, 148, 640), "Equation: "),
        _line((180, 620, 268, 640), "a + a + 3 = 15 "),
        _line((72, 641, 148, 660), "Solution: "),
        _line((180, 641, 215, 660), "a = 6 "),
    ])
    split_page_blocks(page)
    assert _texts(page) == [
        "Example: Andrea and Patrick ran a total of 15 miles. How many miles did they each run?",
        "Equation:", "a + a + 3 = 15", "Solution:", "a = 6",
    ]


def test_a_numbered_question_keeps_its_number():
    page = _page([
        _line((36, 100, 50, 120), "1."),
        _line((62, 100, 400, 120), "Explain why chlorophyll is needed."),
        _line((36, 130, 50, 150), "2."),
        _line((62, 130, 400, 150), "Describe what happens to glucose."),
        _line((62, 150, 300, 170), "Use full sentences."),
    ])
    split_page_blocks(page)
    assert [t.flat_text for t in page.text_objects][0].startswith("1. Explain")


def test_a_step_cut_off_one_line_late_gets_its_last_word_back():
    """Page 6: "operations." ended step 2 but opened the block with step 3."""
    def block(oid, lines):
        box = Rect.bounding([ln.bbox for ln in lines])
        return TextObject(id=oid, kind=ObjectKind.TEXT, bbox=box,
                          norm_bbox=box.normalized(612, 792), source=TextSource.NATIVE,
                          lines=lines)
    page = Page(index=5, number=6, width=612, height=792, text_objects=[
        block("a", [
            _line((54, 117, 331, 142), "1.​Combine ______ terms "),
            _line((54, 142, 573, 167), "2.​Undo the ____ or ____ using inverse "),
        ]),
        block("b", [
            _line((72, 167, 170, 192), "operations. "),
            _line((54, 192, 573, 218), "3.​Undo the ____ or ____ using inverse "),
        ]),
        block("c", [
            _line((72, 218, 170, 243), "operations. "),
            _line((54, 243, 410, 268), "4.​Get your answer and check if necessary "),
        ]),
    ])
    split_page_blocks(page)
    assert [t.flat_text.replace("​", " ") for t in page.text_objects] == [
        "1. Combine ______ terms",
        "2. Undo the ____ or ____ using inverse operations.",
        "3. Undo the ____ or ____ using inverse operations.",
        "4. Get your answer and check if necessary",
    ]


def test_printed_rows_offer_one_line_per_step_and_label():
    """What "Break into lines like the original" fills the English with."""
    from ct_engine.api.routes import _printed_rows
    lines = [
        _line((36, 205, 472, 230), "To Solve Equations:​"),
        _line((36, 334, 573, 359), "4. Bring all ___ (numbers without a variable) to the "),
        _line((36, 359, 334, 384), "other side using inverse operations "),
        _line((72, 620, 148, 640), "Equation: "),
        _line((180, 620, 268, 640), "a + a + 3 = 15 "),
        _line((252, 641, 469, 660), "Let a + 3 = Patrick's running "),
    ]
    box = Rect.bounding([ln.bbox for ln in lines])
    obj = TextObject(id="t", kind=ObjectKind.TEXT, bbox=box,
                     norm_bbox=box.normalized(612, 792), source=TextSource.NATIVE, lines=lines)
    assert _printed_rows(obj) == [
        "To Solve Equations:",
        "4. Bring all ___ (numbers without a variable) to the other side using inverse operations",
        "Equation:   a + a + 3 = 15",
        "Let a + 3 = Patrick's running",
    ]
