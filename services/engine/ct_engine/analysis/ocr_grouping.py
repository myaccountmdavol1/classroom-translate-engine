"""Group adjacent OCR lines into paragraph blocks.

OCR reports one region per visual LINE. Left that way, two problems follow,
and both were visible on a real 37-page packet:

  1. **Fragments get translated.** "The cost of hiring the band Bling for a"
     and "attending. If 264 people attend the dance" are one sentence split
     across two lines. Translated separately, each is a fragment, and no
     engine can do that well.

  2. **Bilingual output has nowhere to go.** A scanned line box is ~14pt
     tall with the next line immediately below it. Bilingual text needs two
     or three lines in that space, so it overflows onto its neighbour --
     which is exactly the overlapping text a teacher sees.

Grouping gives the block the paragraph's full height, so the fitter has real
space, the covering covers the whole paragraph, and the translator gets a
whole sentence.

Grouping is conservative: a new numbered question, a new answer choice, a
change of column or a change of type size all start a new block. Merging two
unrelated questions would be worse than not merging at all.
"""

from __future__ import annotations

import re

from ..model.document import Page
from ..model.enums import TextSource
from ..model.geometry import Rect
from ..model.objects import TextLine, TextObject

# A following line joins the current block only if the vertical gap is under
# this multiple of the line height. Normal leading is ~1.2x; a paragraph
# break or a new question is further away.
MAX_GAP_RATIO = 0.75
# After a line that stops mid-sentence. Still under a blank line's worth.
MAX_UNFINISHED_GAP_RATIO = 1.0
# Horizontal overlap required, as a fraction of the narrower line.
MIN_X_OVERLAP = 0.55
# Type sizes must be within this ratio to belong to the same paragraph.
MAX_SIZE_RATIO = 1.35
# The same, for a line that carries on an unfinished sentence. Measured
# word heights put one rubric cell's lines 1.48 apart; a 20pt heading over
# 11pt body text is 1.8.
MAX_CONTINUATION_SIZE_RATIO = 1.6

# Lines that always begin a new block: a numbered question, a lettered or
# numbered answer choice, or a bulleted item.
_NEW_ITEM = re.compile(
    r"^\s*(?:\(?\d{1,3}[).]|\(\s*[A-Da-d1-9]\s*\)|[A-Da-d][).]\s|[•‣●\-\*]\s)"
)


def _starts_new_item(text: str) -> bool:
    return bool(_NEW_ITEM.match(text))


def _x_overlap_ratio(a: Rect, b: Rect) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.width, b.width) or 1.0
    return overlap / narrower


def _unfinished(previous: TextObject) -> bool:
    """The line stops without ending a sentence."""
    end = previous.flat_text.rstrip().rstrip("\"'”’)]")
    return bool(end) and end[-1] not in ".!?:"


def _mid_sentence(previous: TextObject, block: TextObject) -> bool:
    """The line above stops mid-sentence and this one carries on in lower case."""
    start = next((c for c in block.flat_text if c.isalpha()), "")
    return _unfinished(previous) and start.islower()


def _continues(previous: TextObject, block: TextObject) -> bool:
    """Whether `block` is the next line of the paragraph ending in `previous`."""
    gap = block.bbox.y0 - previous.bbox.y1
    # The taller of the two: a line's box is as tall as its letters, and
    # "Resources" (nothing below the line) allowed a gap 0.4pt too small
    # for "Effectively" under it on a scanned rubric.
    height = max(previous.bbox.height, block.bbox.height, 1.0)
    size_a = previous.style.font_size or height
    size_b = block.style.font_size or block.bbox.height
    ratio = max(size_a, size_b) / max(1.0, min(size_a, size_b))
    # Sizes come from the words' heights, so a line of short letters
    # ("assessments.") reads a third smaller than the line above it. When
    # the sentence plainly runs on, that is not a heading meeting body text.
    size_limit = (
        MAX_CONTINUATION_SIZE_RATIO if _mid_sentence(previous, block) else MAX_SIZE_RATIO
    )
    # Two lines with no letter below the line ("materials and other" over
    # "resources to meet") have short boxes, so the gap between them looks
    # a descender wider than it is. A line that has not finished its
    # sentence may reach that much further for the rest of it.
    gap_limit = MAX_UNFINISHED_GAP_RATIO if _unfinished(previous) else MAX_GAP_RATIO
    return (
        gap <= height * gap_limit
        and _x_overlap_ratio(previous.bbox, block.bbox) >= MIN_X_OVERLAP
        and ratio <= size_limit
    )


def _merge(blocks: list[TextObject], page: Page) -> TextObject:
    """Fuse a run of single-line blocks into one paragraph block."""
    first = blocks[0]
    lines: list[TextLine] = []
    for b in blocks:
        lines.extend(b.lines)
    bbox = Rect.bounding([ln.bbox for ln in lines]) or first.bbox

    confidences = [b.confidence for b in blocks if b.confidence is not None]
    cover: list[Rect] = []
    for b in blocks:
        cover.extend(b.cover_boxes)

    return first.model_copy(
        update={
            "lines": lines,
            "bbox": bbox,
            "norm_bbox": bbox.normalized(page.width, page.height),
            "confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
            "cover_boxes": cover,
            "line_spacing": _mean_spacing(lines),
        }
    )


def _mean_spacing(lines: list[TextLine]) -> float | None:
    if len(lines) < 2:
        return None
    tops = sorted(ln.bbox.y0 for ln in lines)
    gaps = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0.1]
    return round(sum(gaps) / len(gaps), 2) if gaps else None


def group_ocr_lines(page: Page) -> int:
    """Merge adjacent OCR line blocks into paragraphs. Returns blocks removed."""
    ocr_blocks = [
        t
        for t in page.text_objects
        if t.source in (TextSource.OCR, TextSource.IMAGE_OCR)
    ]
    if len(ocr_blocks) < 2:
        return 0

    others = [t for t in page.text_objects if t not in ocr_blocks]
    # Image labels are scattered by nature and must not be joined to each
    # other; only page-scan lines are grouped.
    labels = [t for t in ocr_blocks if t.source is TextSource.IMAGE_OCR]
    scan_lines = [t for t in ocr_blocks if t.source is TextSource.OCR]
    if len(scan_lines) < 2:
        return 0

    scan_lines.sort(key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))

    # Every open run is a candidate, not only the latest. Sorted top to
    # bottom, the lines of side-by-side table cells alternate (cell A, cell
    # B, cell A...), so on a scanned rubric no line ever followed its own
    # cell's line and each was translated alone. The closest run above wins.
    runs: list[list[TextObject]] = [[scan_lines[0]]]
    for block in scan_lines[1:]:
        best: list[TextObject] | None = None
        if not _starts_new_item(block.flat_text):
            for run in runs:
                if _continues(run[-1], block) and (
                    best is None or run[-1].bbox.y1 > best[-1].bbox.y1
                ):
                    best = run
        if best is not None:
            best.append(block)
        else:
            runs.append([block])

    merged = [_merge(run, page) if len(run) > 1 else run[0] for run in runs]
    removed = len(scan_lines) - len(merged)

    page.text_objects = others + labels + merged
    if removed:
        page.diagnostics.extra["ocr_lines_grouped"] = removed
    return removed
