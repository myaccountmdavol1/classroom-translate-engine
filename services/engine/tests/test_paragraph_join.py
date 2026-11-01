"""Joining a PDF paragraph the reader handed back in pieces.

A slide made in Canva keeps a centred paragraph's lines touching, but
PyMuPDF returned it as one block per line or two. Each piece was
translated alone, as a fragment, and the longer translations overlapped.
The geometry below is a real slide's (Magma Math Academy, 810x1012).
"""

from __future__ import annotations

from ct_engine.analysis.paragraph_join import join_paragraph_pieces
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextAlignment, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle, VectorObject

FONT = "Lexend"


def piece(lines: list[tuple[str, float, float, float, float]], *, size: float = 21.8,
          bold: bool = False, color: str = "#000000", source: TextSource = TextSource.NATIVE,
          protected: bool = False) -> TextObject:
    """A block of (text, x0, y0, x1, y1) lines, as the reader gives one."""
    style = TextStyle(font_family=FONT, font_size=size, bold=bold, color=color)
    made = [
        TextLine(bbox=Rect(x0=x0, y0=y0, x1=x1, y1=y1),
                 spans=[TextSpan(text=t, bbox=Rect(x0=x0, y0=y0, x1=x1, y1=y1), style=style)])
        for t, x0, y0, x1, y1 in lines
    ]
    bbox = Rect.bounding([ln.bbox for ln in made])
    return TextObject(
        id=f"text_{lines[0][0][:8]}_{int(lines[0][2])}",
        kind=ObjectKind.TEXT,
        bbox=bbox,
        norm_bbox=bbox.normalized(810, 1012),
        lines=made,
        source=source,
        style=style,
        protected=protected,
    )


def page_with(blocks, rules=()) -> Page:
    page = Page(index=0, number=1, width=810.0, height=1012.0)
    page.text_objects = list(blocks)
    page.vector_objects = [
        VectorObject(id=f"rule{i}", bbox=r, norm_bbox=r.normalized(810, 1012), line_count=1)
        for i, r in enumerate(rules)
    ]
    return page


# The intro paragraph, in the four pieces the reader made of it.
INTRO = [
    piece([("Magma Math Academy provides various 30-45", 27.9, 85.4, 466.1, 117.7),
           ("minute courses for our teachers, no matter their", 30.0, 115.4, 464.0, 147.7)]),
    piece([("level. Select the most appropriate level for your", 31.1, 145.4, 462.9, 177.7)]),
    piece([("needs and Learn. Explore. Apply. Process. Earn", 34.8, 175.4, 459.2, 207.7)]),
    piece([("your certificates.", 170.9, 205.4, 323.1, 237.7)]),
]


def test_a_centred_paragraph_in_pieces_is_one_block_again():
    page = page_with(INTRO)

    removed = join_paragraph_pieces(page)

    assert removed == 3
    (block,) = page.text_objects
    assert block.flat_text.startswith("Magma Math Academy provides")
    assert block.flat_text.endswith("Earn your certificates.")
    assert len(block.lines) == 5
    assert block.alignment is TextAlignment.CENTER, "five centred lines now say how they are set"
    assert block.bbox.y0 == 85.4 and block.bbox.y1 == 237.7


def test_the_leap_paragraph_is_joined():
    page = page_with([
        piece([("Reflect on what you", 452.9, 685.9, 664.7, 721.2)], size=23.8),
        piece([("gleaned from this LEAP and share your", 446.1, 712.2, 671.5, 773.7)], size=23.8),
        piece([("reflection on the form linked here.", 468.7, 764.7, 648.9, 826.2)], size=23.8),
    ])
    assert join_paragraph_pieces(page) == 2
    assert page.text_objects[0].flat_text.startswith("Reflect on what you gleaned")


def test_paragraphs_with_space_between_stay_apart():
    """'Take this time to plan.' is its own paragraph, 16pt above the next."""
    page = page_with([
        piece([("Take this time to plan.", 70.9, 724.1, 344.8, 767.6)], size=29.3),
        piece([("Incorporate what you learned into your upcoming", 40.7, 784.1, 374.9, 822.0)], size=29.3),
    ])
    assert join_paragraph_pieces(page) == 0
    assert len(page.text_objects) == 2


def test_a_heading_over_body_text_stays_apart():
    page = page_with([
        piece([("LEVEL - YOU DECIDE!", 92.9, 51.7, 403.2, 85.2)], size=33.4, bold=True),
        INTRO[0],
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_different_colour_stays_apart():
    page = page_with([
        piece([("Answer key", 72.0, 100.0, 300.0, 120.0)], color="#c0392b"),
        piece([("Check your work against it", 72.0, 119.0, 300.0, 139.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_numbered_question_starts_its_own_block():
    page = page_with([
        piece([("Read the passage below carefully.", 72.0, 100.0, 400.0, 120.0)]),
        piece([("1. What is the main idea?", 72.0, 119.0, 400.0, 139.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_side_by_side_columns_stay_apart():
    page = page_with([
        piece([("LEARN", 145.5, 270.2, 285.8, 326.7)], size=47.1),
        piece([("EXPLORE", 507.4, 270.2, 700.8, 326.7)], size=47.1),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_table_rule_between_keeps_cells_apart():
    upper = piece([("Monday", 72.0, 100.0, 200.0, 118.0)])
    lower = piece([("Tuesday", 72.0, 119.0, 200.0, 137.0)])
    rule = Rect(x0=60.0, y0=118.2, x1=300.0, y1=118.8)
    page = page_with([upper, lower], rules=[rule])
    assert join_paragraph_pieces(page) == 0


def test_scanned_lines_and_protected_math_are_left_alone():
    page = page_with([
        piece([("A line read from a scan", 72.0, 100.0, 400.0, 120.0)], source=TextSource.OCR),
        piece([("and the next one", 72.0, 119.0, 400.0, 139.0)], source=TextSource.OCR),
        piece([("x = 2y + 3", 72.0, 200.0, 200.0, 220.0)], protected=True),
        piece([("y = 4", 72.0, 219.0, 200.0, 239.0)], protected=True),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_heading_set_like_its_paragraph_stays_apart():
    """Same font, size and colour, touching: still a heading, as its
    capitals and the capital after it show."""
    page = page_with([
        piece([("SUBSTITUTION", 72.0, 100.0, 300.0, 120.0)]),
        piece([("Technology acts as a direct tool substitute", 72.0, 119.0, 400.0, 139.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_contact_card_keeps_its_lines():
    page = page_with([
        piece([("Austen Cortese", 100.0, 100.0, 260.0, 120.0)]),
        piece([("IT Manager", 120.0, 119.0, 240.0, 139.0)]),
        piece([("516-622-6406", 115.0, 138.0, 245.0, 158.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_sentence_carries_on_into_a_name():
    page = page_with([
        piece([("Select the one that is most appropriate for your current", 52.2, 444.6, 365.5, 513.7)],
              size=26.8),
        piece([("Magma Math needs.", 96.8, 503.1, 320.9, 542.9)], size=26.8),
    ])
    assert join_paragraph_pieces(page) == 1


def test_a_list_does_not_take_the_paragraph_under_it():
    """Bullets drawn as pictures: the list's last item has no full stop."""
    page = page_with([
        piece([("Create custom activities", 72.0, 100.0, 400.0, 120.0),
               ("Lead with an instructional framework", 72.0, 119.0, 400.0, 139.0),
               ("Ask the class to increase rigor and authentic learning", 72.0, 138.0, 400.0, 158.0)]),
        piece([("Diffit encourages educators to use frameworks.", 72.0, 157.0, 400.0, 177.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_title_over_a_paragraph_stays_apart():
    page = page_with([
        piece([("Tech Tools Handbook and Important Tools", 72.0, 100.0, 400.0, 120.0)]),
        piece([("This is the hub for all things tech", 72.0, 119.0, 400.0, 139.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_a_question_over_its_answer_stays_apart():
    page = page_with([
        piece([("What do we want students to do?", 72.0, 100.0, 400.0, 120.0)]),
        piece([("technology gives students greater agency", 72.0, 119.0, 400.0, 139.0)]),
    ])
    assert join_paragraph_pieces(page) == 0


def test_two_columns_each_join_down_their_own_side():
    left = [
        piece([("Review the Magma Academy", 37.2, 356.9, 380.5, 396.7)], size=26.8),
        piece([("course options, #1 - #4.", 54.1, 386.1, 363.6, 425.9)], size=26.8),
    ]
    right = [
        piece([("Complete the Magma Academy course", 436.9, 354.9, 746.6, 386.0)], size=25.4),
        piece([("that you selected to earn a certificate", 436.9, 385.0, 746.6, 416.0)], size=25.4),
    ]
    page = page_with([left[0], right[0], left[1], right[1]])
    assert join_paragraph_pieces(page) == 2
    texts = sorted(b.flat_text for b in page.text_objects)
    assert texts == [
        "Complete the Magma Academy course that you selected to earn a certificate",
        "Review the Magma Academy course options, #1 - #4.",
    ]
