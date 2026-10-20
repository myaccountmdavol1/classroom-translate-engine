"""Page-local reflow (spec sec. 25).

Bilingual output is ~2.4x the source length. When the bounded concessions run
out there is nowhere to put it, and the result was text drawn over its
neighbour. Reflow pushes following blocks down into whitespace instead.

What it must never do: move an image, touch a response area, push anything off
the page, or reorder questions.
"""

from __future__ import annotations

import pytest

from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TranslationStatus
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import (
    ImageObject,
    ResponseArea,
    TextLine,
    TextObject,
    TextSpan,
    TextStyle,
)
from ct_engine.reconstruct.reflow import (
    BOTTOM_MARGIN,
    MAX_SHIFT,
    plan_reflow,
)


def block(name: str, y: float, height: float = 14.0, x0: float = 72.0,
          x1: float = 520.0,
          status: TranslationStatus = TranslationStatus.TRANSLATED) -> TextObject:
    bbox = Rect(x0=x0, y0=y, x1=x1, y1=y + height)
    style = TextStyle(font_size=11.0)
    return TextObject(
        id=name,
        kind=ObjectKind.TEXT,
        bbox=bbox,
        norm_bbox=bbox.normalized(612, 792),
        lines=[TextLine(bbox=bbox, spans=[TextSpan(text="x", bbox=bbox, style=style)])],
        style=style,
        translated_text="translated",
        translation_only_text="translated",
        translation_status=status,
    )


def make_page(blocks: list[TextObject]) -> Page:
    page = Page(index=0, number=1, width=612.0, height=792.0)
    page.text_objects = list(blocks)
    return page


def test_following_blocks_are_pushed_down_when_there_is_room():
    a, b, c = block("a", 100), block("b", 130), block("c", 160)
    page = make_page([a, b, c])
    plan = plan_reflow(page, {"a": 30.0}, [a, b, c])

    assert plan.extra_height == {"a": 30.0}
    assert plan.shifts["a"] == 0.0
    assert plan.shifts["b"] == 30.0
    assert plan.shifts["c"] == 30.0
    assert not plan.translation_only
    assert plan.moved == 2


def test_shifts_accumulate_across_several_overflows():
    a, b, c = block("a", 100), block("b", 140), block("c", 180)
    page = make_page([a, b, c])
    plan = plan_reflow(page, {"a": 20.0, "b": 20.0}, [a, b, c])
    assert plan.shifts["a"] == 0.0
    assert plan.shifts["b"] == 20.0
    assert plan.shifts["c"] == 40.0


def test_a_response_area_below_blocks_reflow():
    """Answer lines are inviolable, so the pairing is dropped instead."""
    a, b = block("a", 100), block("b", 130)
    page = make_page([a, b])
    area = Rect(x0=72, y0=150, x1=520, y1=300)
    page.response_areas = [
        ResponseArea(bbox=area, norm_bbox=area.normalized(612, 792),
                     kind="writing_lines", confidence=0.9)
    ]
    plan = plan_reflow(page, {"a": 40.0}, [a, b])
    assert plan.translation_only == {"a"}
    assert not plan.extra_height
    assert any("only the translation" in n for n in plan.notes)


def test_an_image_below_blocks_reflow():
    a, b = block("a", 100), block("b", 130)
    page = make_page([a, b])
    box = Rect(x0=72, y0=150, x1=400, y1=400)
    page.image_objects = [
        ImageObject(id="img", kind=ObjectKind.IMAGE, bbox=box,
                    norm_bbox=box.normalized(612, 792))
    ]
    plan = plan_reflow(page, {"a": 60.0}, [a, b])
    assert plan.translation_only == {"a"}


def test_nothing_is_pushed_off_the_bottom_of_the_page():
    a = block("a", 700)
    b = block("b", 730)
    page = make_page([a, b])
    plan = plan_reflow(page, {"a": 80.0}, [a, b])
    assert plan.translation_only == {"a"}
    # And whatever is granted must keep b on the page.
    for obj_id, shift in plan.shifts.items():
        target = a if obj_id == "a" else b
        assert target.bbox.y1 + shift <= page.height - BOTTOM_MARGIN + 0.01


def test_a_side_by_side_block_is_not_dragged_down():
    """Only the same column moves; a neighbour keeps its position."""
    left = block("left", 100, x0=72, x1=280)
    right = block("right", 100, x0=320, x1=530)
    below = block("below", 140, x0=72, x1=280)
    page = make_page([left, right, below])
    plan = plan_reflow(page, {"left": 25.0}, [left, right, below])
    assert plan.extra_height.get("left") == 25.0
    assert plan.shifts["below"] == 25.0
    # The right-hand column is untouched.
    assert plan.shifts["right"] == 0.0


def test_an_excessive_shift_is_refused():
    """A large displacement stops resembling the original document."""
    a, b = block("a", 100), block("b", 140)
    page = make_page([a, b])
    plan = plan_reflow(page, {"a": MAX_SHIFT + 50}, [a, b])
    assert plan.translation_only == {"a"}


def test_protected_text_is_treated_as_fixed():
    """Its original glyphs stay on the page, so it cannot be overlapped."""
    a = block("a", 100)
    formula = block("f", 130, status=TranslationStatus.PROTECTED)
    formula.protected = True
    page = make_page([a, formula])
    plan = plan_reflow(page, {"a": 40.0}, [a])
    assert plan.translation_only == {"a"}


def test_no_overflow_means_no_reflow():
    """A page that fits must never be rearranged."""
    a, b = block("a", 100), block("b", 130)
    page = make_page([a, b])
    plan = plan_reflow(page, {}, [a, b])
    assert plan.is_empty
    assert plan.moved == 0
    assert not plan.shifts or all(v == 0.0 for v in plan.shifts.values())


def test_question_order_is_never_changed():
    a, b, c = block("a", 100), block("b", 140), block("c", 180)
    page = make_page([a, b, c])
    plan = plan_reflow(page, {"a": 20.0}, [a, b, c])
    positions = sorted(
        (blk.bbox.y0 + plan.shifts.get(blk.id, 0.0), blk.id) for blk in (a, b, c)
    )
    assert [name for _y, name in positions] == ["a", "b", "c"]


@pytest.mark.slow
def test_bilingual_packet_no_longer_overflows():
    """End to end on a dense real-world layout."""
    import fitz

    from conftest import corpus_file
    from ct_engine.analysis.block_split import split_page_blocks
    from ct_engine.analysis.classify import classify_page
    from ct_engine.analysis.reading_order import apply_reading_order
    from ct_engine.ingest.pdf_parser import PdfParser
    from ct_engine.model.enums import OutputLanguageMode
    from ct_engine.pipeline.translate import translate_page
    from ct_engine.reconstruct.builder import rebuild_document
    from test_translate_pipeline import EchoProvider

    data = corpus_file("08_exam_multiple_choice.pdf")
    pages = []
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            translate_page(
                page, EchoProvider(), source_lang="en", target_lang="es",
                output_mode=OutputLanguageMode.BILINGUAL_TRANSLATION_FIRST,
            )
            pages.append(page)

    out, stats = rebuild_document(data, pages, target_lang="es")
    total_overflow = sum(s.overflowed for s in stats.values())
    assert total_overflow == 0, f"{total_overflow} block(s) still overflow"
    result = fitz.open(stream=out, filetype="pdf")
    try:
        assert result.page_count == 4
    finally:
        result.close()
