"""Reading order for text blocks.

PDF content-stream order is producer order, not reading order: a two-column
worksheet often emits the right column first, and footers can precede body
text. Translation quality depends on neighbouring context, and the reviewer
UI lists blocks in this order, so it needs to match how a person reads.

Approach: a light XY-cut. Detect column bands from a vertical-gap projection,
then sort within each band top-to-bottom with a row tolerance so that items
on the same visual line stay left-to-right.
"""

from __future__ import annotations

from ..model.document import Page
from ..model.objects import TextObject

# A column gutter must be at least this wide (points) and span at least this
# fraction of the occupied height to count as a real column split.
MIN_GUTTER_WIDTH = 24.0
MIN_GUTTER_COVERAGE = 0.55
MAX_COLUMNS = 3


def _row_tolerance(blocks: list[TextObject]) -> float:
    """Vertical slack within which two blocks count as the same visual row."""
    sizes = [t.style.font_size for t in blocks if t.style.font_size > 0]
    if not sizes:
        return 6.0
    sizes.sort()
    median = sizes[len(sizes) // 2]
    return max(4.0, median * 0.6)


def detect_column_bands(
    blocks: list[TextObject], page_width: float
) -> list[tuple[float, float]]:
    """Find vertical column bands as (x_start, x_end) ranges."""
    if len(blocks) < 4:
        return [(0.0, page_width)]

    # Occupancy histogram across the page width, 2pt buckets.
    bucket = 2.0
    n = max(1, int(page_width / bucket) + 1)
    occupied = [0] * n
    for t in blocks:
        lo = max(0, int(t.bbox.x0 / bucket))
        hi = min(n - 1, int(t.bbox.x1 / bucket))
        for i in range(lo, hi + 1):
            occupied[i] += 1

    # Ignore margins: bands are found between the first and last occupied bucket.
    filled = [i for i, c in enumerate(occupied) if c > 0]
    if not filled:
        return [(0.0, page_width)]
    left, right = filled[0], filled[-1]

    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    for i in range(left, right + 1):
        if occupied[i] == 0:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) * bucket >= MIN_GUTTER_WIDTH:
                gaps.append((run_start, i))
            run_start = None

    if not gaps:
        return [(0.0, page_width)]

    # A gutter must separate blocks over most of the content height, or it is
    # just a wide space inside one column (e.g. a "Name: ____ Date: ____" row).
    y_top = min(t.bbox.y0 for t in blocks)
    y_bottom = max(t.bbox.y1 for t in blocks)
    total_height = max(1.0, y_bottom - y_top)

    real_gaps: list[tuple[int, int]] = []
    for g0, g1 in gaps:
        x0, x1 = g0 * bucket, g1 * bucket
        # Height spanned by blocks that cross this gutter -- must be small.
        crossing = [t for t in blocks if t.bbox.x0 < x0 and t.bbox.x1 > x1]
        crossed_height = sum(t.bbox.height for t in crossing)
        if crossed_height / total_height < (1.0 - MIN_GUTTER_COVERAGE):
            real_gaps.append((g0, g1))

    if not real_gaps or len(real_gaps) >= MAX_COLUMNS:
        return [(0.0, page_width)]

    bands: list[tuple[float, float]] = []
    cursor = 0.0
    for g0, g1 in real_gaps:
        bands.append((cursor, g0 * bucket))
        cursor = g1 * bucket
    bands.append((cursor, page_width))
    return bands


def sort_reading_order(blocks: list[TextObject], page_width: float) -> list[TextObject]:
    """Return blocks in the order a person would read them."""
    if len(blocks) <= 1:
        return list(blocks)

    bands = detect_column_bands(blocks, page_width)
    tol = _row_tolerance(blocks)

    def band_index(t: TextObject) -> int:
        cx = t.bbox.center[0]
        for i, (x0, x1) in enumerate(bands):
            if x0 <= cx < x1:
                return i
        return len(bands) - 1

    def key(t: TextObject) -> tuple[int, int, float]:
        # Quantising y by the row tolerance keeps same-line items together,
        # then x orders them left to right.
        return (band_index(t), int(t.bbox.y0 / tol), t.bbox.x0)

    return sorted(blocks, key=key)


def apply_reading_order(page: Page) -> None:
    """Reorder a page's text blocks in place and record the column count."""
    if not page.text_objects:
        return
    ordered = sort_reading_order(page.text_objects, page.width)
    page.text_objects = ordered
    for i, t in enumerate(ordered):
        t.reading_index = i
    bands = detect_column_bands(ordered, page.width)
    page.diagnostics.extra["columns"] = len(bands)
