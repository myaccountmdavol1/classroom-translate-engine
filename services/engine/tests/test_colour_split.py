"""A title whose lines are two colours keeps both.

"COLORÍN COLORADO" (dark green) over "ELL STRATEGY LIBRARY" (light green)
came from the PDF as one block, and a block is drawn in one colour, so the
translated title was all light green. Lines that change colour where the
sentence does not run on are two blocks, each in its own colour.
"""
from __future__ import annotations

from ct_engine.analysis.block_split import split_at_colour_change, split_page_blocks
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle

DARK, LIGHT = "#005026", "#62b21f"


def block(*lines: tuple[str, str]) -> TextObject:
    made = []
    y = 20.0
    for text, colour in lines:
        style = TextStyle(font_family="Futura", font_size=30, bold=True, color=colour)
        box = Rect(x0=90, y0=y, x1=410, y1=y + 40)
        made.append(TextLine(bbox=box, spans=[TextSpan(text=text, bbox=box, style=style)]))
        y += 42
    bbox = Rect.bounding([ln.bbox for ln in made])
    return TextObject(id="text_title", kind=ObjectKind.TEXT, bbox=bbox,
                      norm_bbox=bbox.normalized(810, 1012), lines=made,
                      source=TextSource.NATIVE, style=made[-1].spans[0].style)


def test_a_two_colour_title_is_two_blocks_in_their_colours():
    parts = split_at_colour_change(block(("COLORÍN COLORADO", DARK), ("ELL STRATEGY LIBRARY", LIGHT)))
    assert [p.flat_text for p in parts] == ["COLORÍN COLORADO", "ELL STRATEGY LIBRARY"]
    assert [p.style.color for p in parts] == [DARK, LIGHT]


def test_a_sentence_running_on_into_another_colour_stays_whole():
    whole = block(("Reflect on what you learned and share your", DARK),
                  ("reflection with the class.", LIGHT))
    assert split_at_colour_change(whole) == [whole]


def test_a_coloured_phrase_in_brackets_stays_in_its_sentence():
    whole = block(("Critical Thinking (e.g., Digging", LIGHT), ("Deeper, Perspective Taking)", DARK))
    assert split_at_colour_change(whole) == [whole]


def test_a_shadow_drawn_as_text_is_not_split():
    """The same words in two colours, offset: a shadow, not two parts."""
    whole = block(("INSTRUCTIONAL TECHNOLOGY", DARK), ("INSTRUCTIONAL TECHNOLOGY", LIGHT))
    assert split_at_colour_change(whole) == [whole]


def test_a_shadow_line_a_little_off_its_words_is_dropped():
    """Entry Plan: "within our school community?" in black, 6 points off
    the same words in yellow -- a shadow, in another block from them."""
    from ct_engine.analysis.dedupe import drop_duplicate_text

    shadow = block(("within our school community?", "#000000"))
    words = block(("within our school community?", "#ffde59"))
    words.id = "text_words"
    for ln in words.lines:
        ln.bbox = Rect(x0=ln.bbox.x0, y0=ln.bbox.y0 + 6, x1=ln.bbox.x1, y1=ln.bbox.y1 + 6)
    words.bbox = words.lines[0].bbox
    page = Page(index=0, number=1, width=1280.0, height=720.0)
    page.text_objects = [shadow, words]
    drop_duplicate_text(page)
    assert [t.style.color for t in page.text_objects] == ["#ffde59"]


def test_lines_of_one_colour_stay_whole():
    whole = block(("COLORÍN COLORADO", DARK), ("ELL STRATEGY LIBRARY", DARK))
    assert split_at_colour_change(whole) == [whole]


def test_the_page_split_gives_each_its_colour():
    page = Page(index=0, number=1, width=810.0, height=1012.0)
    page.text_objects = [block(("COLORÍN COLORADO", DARK), ("ELL STRATEGY LIBRARY", LIGHT))]
    split_page_blocks(page)
    assert sorted(t.style.color for t in page.text_objects) == sorted([DARK, LIGHT])
