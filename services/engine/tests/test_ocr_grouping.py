"""Grouping OCR line regions into paragraph blocks.

Regression cover for a real failure on a 37-page classroom packet: in
bilingual mode, scanned pages produced overlapping text. OCR had reported one
region per LINE (~14pt tall), so bilingual output -- roughly 1.4x the
characters -- had nowhere to go and spilled onto the next line.
"""

from __future__ import annotations

from ct_engine.analysis.ocr_grouping import group_ocr_lines
from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle


def line_block(text: str, y: float, *, x0: float = 72.0, x1: float = 520.0,
               height: float = 14.0, size: float = 11.0,
               source: TextSource = TextSource.OCR) -> TextObject:
    bbox = Rect(x0=x0, y0=y, x1=x1, y1=y + height)
    style = TextStyle(font_size=size)
    return TextObject(
        id=f"ocr_{text[:6]}_{int(y)}",
        kind=ObjectKind.TEXT,
        bbox=bbox,
        norm_bbox=bbox.normalized(612, 792),
        lines=[TextLine(bbox=bbox, spans=[TextSpan(text=text, bbox=bbox, style=style)])],
        source=source,
        style=style,
        confidence=0.9,
    )


def page_with(blocks: list[TextObject]) -> Page:
    page = Page(index=0, number=1, width=612.0, height=792.0)
    page.text_objects = list(blocks)
    return page


def test_consecutive_lines_of_one_sentence_are_merged():
    """The exact failure: one sentence split across two OCR lines."""
    page = page_with([
        line_block("The cost of hiring the band Bling for a dance is $400 plus", 100),
        line_block("$1.50 for each person attending. How much does it cost?", 117),
    ])
    removed = group_ocr_lines(page)
    assert removed == 1
    assert len(page.text_objects) == 1
    merged = page.text_objects[0]
    assert len(merged.lines) == 2
    # The block now spans both lines, so the fitter has real space to work in.
    assert merged.bbox.height > 25
    assert "Bling" in merged.flat_text and "attending" in merged.flat_text
    # And the translator receives a whole sentence, not a fragment.
    assert merged.flat_text.count(" ") > 15


def test_a_new_numbered_question_starts_a_new_block():
    """Merging two questions would be worse than not merging."""
    page = page_with([
        line_block("7) The cost of hiring the band Bling for a dance is", 100),
        line_block("$400 plus $1.50 for each person attending.", 117),
        line_block("8) The cost of hiring the band Rockin Nation is", 134),
    ])
    group_ocr_lines(page)
    texts = [t.flat_text for t in page.text_objects]
    assert len(page.text_objects) == 2
    assert any(t.startswith("7)") for t in texts)
    assert any(t.startswith("8)") for t in texts)


def test_answer_choices_start_a_new_block():
    page = page_with([
        line_block("How many people attended the dance?", 100),
        line_block("A) 228", 117, x1=140),
    ])
    group_ocr_lines(page)
    assert len(page.text_objects) == 2


def test_a_wide_vertical_gap_breaks_the_group():
    page = page_with([
        line_block("End of the first question.", 100),
        line_block("Start of something unrelated.", 190),
    ])
    group_ocr_lines(page)
    assert len(page.text_objects) == 2


def test_different_columns_are_not_merged():
    page = page_with([
        line_block("Left column line", 100, x0=72, x1=280),
        line_block("Right column line", 104, x0=320, x1=530),
    ])
    group_ocr_lines(page)
    assert len(page.text_objects) == 2


def test_each_table_column_is_grouped_although_its_lines_interleave():
    """A rubric scanned as a table: sorted top to bottom, the lines of four
    cells alternate, so comparing each line only with the one before it in
    that order never found a match, and every line was translated alone
    ("El profesor / comprensión del contenido")."""
    page = page_with([
        line_block("The teacher lacks", 119, x0=142, x1=230, height=8, size=8),
        line_block("The teacher's", 120, x0=246, x1=330, height=8, size=8),
        line_block("sufficient knowledge of", 129, x0=142, x1=232, height=8, size=8),
        line_block("understanding of content", 130, x0=246, x1=334, height=8, size=8),
        line_block("content and pedagogy.", 139, x0=142, x1=228, height=8, size=8),
        line_block("partially supports it.", 140, x0=246, x1=326, height=8, size=8),
    ])
    group_ocr_lines(page)
    texts = sorted(" ".join(ln.text for ln in o.lines) for o in page.text_objects)
    assert texts == [
        "The teacher lacks sufficient knowledge of content and pedagogy.",
        "The teacher's understanding of content partially supports it.",
    ]


def test_a_heading_is_not_absorbed_into_body_text():
    """A size change marks a different kind of content."""
    page = page_with([
        line_block("UNIT 1 REVIEW", 100, height=24.0, size=20.0),
        line_block("Answer each question below.", 126, size=11.0),
    ])
    group_ocr_lines(page)
    assert len(page.text_objects) == 2


def test_image_labels_are_never_grouped_together():
    """Diagram labels are scattered by nature; joining them is meaningless."""
    page = page_with([
        line_block("Cell Wall", 100, x0=80, x1=130, source=TextSource.IMAGE_OCR),
        line_block("Nucleus", 112, x0=80, x1=128, source=TextSource.IMAGE_OCR),
    ])
    removed = group_ocr_lines(page)
    assert removed == 0
    assert len(page.text_objects) == 2


def test_native_text_is_untouched():
    page = page_with([
        line_block("Native line one", 100, source=TextSource.NATIVE),
        line_block("Native line two", 114, source=TextSource.NATIVE),
    ])
    assert group_ocr_lines(page) == 0
    assert len(page.text_objects) == 2


def test_merged_block_keeps_confidence_and_cover_boxes():
    a = line_block("First half of the sentence", 100)
    b = line_block("second half of the sentence", 117)
    a.confidence, b.confidence = 0.8, 0.6
    b.cover_boxes = [Rect(x0=300, y0=117, x1=340, y1=131)]
    page = page_with([a, b])
    group_ocr_lines(page)
    merged = page.text_objects[0]
    # Confidence is averaged, not silently taken from the first line.
    assert 0.6 <= merged.confidence <= 0.8
    # Unreadable-glyph boxes must survive, or covering leaves ghosts behind.
    assert merged.cover_boxes


def test_merged_block_records_line_spacing():
    page = page_with([
        line_block("First line of the paragraph here", 100),
        line_block("Second line of the paragraph here", 117),
        line_block("Third line of the paragraph here", 134),
    ])
    group_ocr_lines(page)
    merged = page.text_objects[0]
    assert len(merged.lines) == 3
    assert merged.line_spacing is not None
    assert 15 <= merged.line_spacing <= 19


def test_grouping_is_recorded_in_diagnostics():
    page = page_with([
        line_block("One line of a sentence that continues", 100),
        line_block("onto the following line right here", 117),
    ])
    group_ocr_lines(page)
    assert page.diagnostics.extra.get("ocr_lines_grouped") == 1


def test_single_block_page_is_a_no_op():
    page = page_with([line_block("Only one line", 100)])
    assert group_ocr_lines(page) == 0


def test_a_line_crossing_two_columns_is_split_before_grouping():
    """Tesseract page-mode reads across a gutter on multi-column designs.

    On a real flyer it emitted the single line "SERVICE TO HANDS-ON" from two
    separate captions; grouping then stacked those into a block no later
    split could undo. Splitting wide-gap lines FIRST prevents that.
    """
    from ct_engine.analysis.block_split import split_page_blocks
    from ct_engine.model.objects import TextLine, TextSpan, TextStyle

    style = TextStyle(font_size=7.0)

    def crossed(first: str, second: str, y: float) -> TextObject:
        # Two captions on one reported line, separated by a wide gutter.
        left = Rect(x0=300.0, y0=y, x1=346.0, y1=y + 7.0)
        right = Rect(x0=430.0, y0=y, x1=490.0, y1=y + 7.0)
        bbox = left.union(right)
        return TextObject(
            id=f"x{int(y)}",
            kind=ObjectKind.TEXT,
            bbox=bbox,
            norm_bbox=bbox.normalized(612, 792),
            lines=[
                TextLine(
                    bbox=bbox,
                    spans=[
                        TextSpan(text=first + " ", bbox=left, style=style),
                        TextSpan(text=second, bbox=right, style=style),
                    ],
                )
            ],
            source=TextSource.OCR,
            style=style,
            confidence=0.9,
        )

    page = page_with([crossed("OUTDOOR", "SERVICE TO", 240.0),
                      crossed("ADVENTURE", "OTHERS", 251.5)])

    split_page_blocks(page)
    group_ocr_lines(page)

    texts = [t.flat_text.replace("\n", " ") for t in page.text_objects]
    # No block may contain captions from both columns.
    for text in texts:
        assert not ("OUTDOOR" in text and "SERVICE" in text), text
        assert not ("ADVENTURE" in text and "OTHERS" in text), text
    # And each column's captions are joined. This used to be a known gap:
    # each line was compared only with the one before it in y-then-x order,
    # so on a multi-column grid every run broke at the column boundary. Each
    # line is now compared with the last line of every open run, by the same
    # pairwise rules (not transitive x-overlap clustering, which let wide
    # body text bridge unrelated columns).
    assert sorted(" ".join(t.split()) for t in texts if t.strip()) == [
        "OUTDOOR ADVENTURE", "SERVICE TO OTHERS"]
