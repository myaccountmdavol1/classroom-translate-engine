"""Split text blocks that PyMuPDF merged across a column gap.

PyMuPDF groups text into blocks by proximity, which merges items that are
side by side on the same visual row -- two math problems in a two-column
worksheet, or a "Name: ___  Date: ___" header row.

Left merged, reconstruction would lay the translation out as one wide box and
smear it across both columns, and translation would receive two unrelated
sentences as one segment. So blocks are split at wide internal gaps before
classification and translation.
"""

from __future__ import annotations

import re

from ..ingest.math_runs import MATH_FONT_REASON, is_math_font
from ..model.document import Page
from ..model.geometry import Rect
from ..model.enums import TextSource
from ..model.objects import TextLine, TextObject, TextSpan, _new_id

# A gap must exceed both thresholds to split: a multiple of the local space
# width (scales with font size) and an absolute floor (stops ordinary
# inter-word spacing in large type from splitting a sentence).
GAP_SPACE_MULTIPLE = 3.5
GAP_MIN_POINTS = 22.0


def _space_width(line: TextLine) -> float:
    sizes = [s.style.font_size for s in line.spans if s.style.font_size > 0]
    base = (sum(sizes) / len(sizes)) if sizes else 10.0
    # Roughly a quarter em for most text faces.
    return max(2.0, base * 0.25)


def _split_line(line: TextLine) -> list[TextLine]:
    """Split one line at wide internal gaps between spans."""
    if len(line.spans) < 2:
        return [line]

    threshold = max(GAP_MIN_POINTS, _space_width(line) * GAP_SPACE_MULTIPLE)
    groups: list[list[TextSpan]] = [[line.spans[0]]]
    for prev, cur in zip(line.spans, line.spans[1:]):
        gap = cur.bbox.x0 - prev.bbox.x1
        if gap >= threshold:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    if len(groups) == 1:
        return [line]

    out: list[TextLine] = []
    for spans in groups:
        bbox = Rect.bounding([s.bbox for s in spans])
        if bbox is None:
            continue
        out.append(
            TextLine(id=_new_id("line"), bbox=bbox, spans=spans, rotation=line.rotation)
        )
    return out


def _x_overlap(a: Rect, b: Rect) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.width, b.width) or 1.0
    return overlap / narrower


def split_text_block(block: TextObject) -> list[TextObject]:
    """Split a block into column-aligned sub-blocks. Returns [block] if intact.

    Two mechanisms, both needed:
      * a single LINE containing a wide internal gap (spans side by side), and
      * a block whose LINES sit in separate horizontal bands, which is how
        PyMuPDF reports two math problems drawn at the same y.
    """
    pieces: list[TextLine] = []
    for line in block.lines:
        pieces.extend(_split_line(line))

    if len(pieces) < 2:
        return [block]

    # Group lines into columns by horizontal overlap, preserving vertical
    # order within each column.
    columns: list[list[TextLine]] = []
    for line in sorted(pieces, key=lambda ln: (ln.bbox.y0, ln.bbox.x0)):
        placed = False
        for col in columns:
            if any(_x_overlap(existing.bbox, line.bbox) > 0.35 for existing in col):
                col.append(line)
                placed = True
                break
        if not placed:
            columns.append([line])

    if len(columns) <= 1:
        return [block]

    columns.sort(key=lambda c: min(ln.bbox.x0 for ln in c))
    extents = [
        (min(ln.bbox.x0 for ln in c), max(ln.bbox.x1 for ln in c)) for c in columns
    ]

    # Require a real gutter between every adjacent pair. Without this, an
    # indented first line or a leading "1." would be torn off its own
    # sentence, which costs translation quality rather than improving it.
    for (_, prev_end), (next_start, _) in zip(extents, extents[1:]):
        if next_start - prev_end < GAP_MIN_POINTS:
            return [block]

    out: list[TextObject] = []
    for col in columns:
        bbox = Rect.bounding([ln.bbox for ln in col])
        if bbox is None:
            continue
        clone = block.model_copy(
            deep=True,
            update={
                "id": _new_id("text"),
                "lines": sorted(col, key=lambda ln: ln.bbox.y0),
                "bbox": bbox,
            },
        )
        out.append(clone)
    return out or [block]


def _is_math_span(span: TextSpan) -> bool:
    return is_math_font(span.style.font_name_raw)


def isolate_math_glyphs(block: TextObject) -> list[TextObject]:
    """Cut runs drawn in a math font out of a block, leaving them in place.

    A Word equation's text layer is nonsense ('! " + # $' for 3/4 + 1/8), so
    those glyphs can be neither translated nor redrawn. They become their own
    protected pieces, which the rebuild never touches; the prose on either
    side becomes separate pieces that translate as usual.

    Lines with no math stay together as one paragraph piece. A line that
    holds an equation is cut at it, so prose that wraps around an inline
    equation translates as fragments; that is the price of never moving the
    equation.
    """
    if not any(_is_math_span(s) for ln in block.lines for s in ln.spans):
        return [block]

    pieces: list[tuple[bool, list[TextLine]]] = []  # (is_math, lines)

    def add(is_math: bool, line: TextLine, *, merge: bool) -> None:
        if merge and pieces and pieces[-1][0] == is_math:
            pieces[-1][1].append(line)
        else:
            pieces.append((is_math, [line]))

    for line in block.lines:
        runs: list[tuple[bool, list[TextSpan]]] = []
        for span in line.spans:
            # A space between two math glyphs belongs to the equation.
            is_math = _is_math_span(span) or (
                not span.text.strip() and bool(runs) and runs[-1][0]
            )
            if runs and runs[-1][0] == is_math:
                runs[-1][1].append(span)
            else:
                runs.append((is_math, [span]))
        whole_line = len(runs) == 1
        for is_math, spans in runs:
            if not any(s.text.strip() for s in spans):
                continue
            bbox = Rect.bounding([s.bbox for s in spans])
            if bbox is None:
                continue
            if not is_math and not whole_line:
                # The space that separated prose from the equation is part of
                # the prose span's box. Drawing from the box edge would butt
                # the translation against the equation ("Add¾"), so keep it.
                space = spans[0].style.font_size * 0.25
                if spans[0].text[:1].isspace():
                    bbox = Rect(x0=bbox.x0 + space, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1)
                if spans[-1].text[-1:].isspace():
                    bbox = Rect(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1 - space, y1=bbox.y1)
            piece = TextLine(
                id=_new_id("line"), bbox=bbox, spans=spans, rotation=line.rotation
            )
            # Only an uncut prose line continues the paragraph before it.
            add(is_math, piece, merge=whole_line and not is_math
                and bool(pieces) and pieces[-1][1][-1].bbox.x1 >= bbox.x0)

    out: list[TextObject] = []
    for is_math, lines in pieces:
        bbox = Rect.bounding([ln.bbox for ln in lines])
        if bbox is None:
            continue
        update = {"id": _new_id("text"), "lines": lines, "bbox": bbox}
        if is_math:
            update.update(protected=True, protected_reason=MATH_FONT_REASON)
        out.append(block.model_copy(deep=True, update=update))
    return out or [block]


# A rule is a stroke this thin in one direction.
RULE_THICKNESS = 3.0


def _ruled_apart(a: Rect, b: Rect, vertical: list[Rect], horizontal: list[Rect]) -> bool:
    """True when a table rule runs BETWEEN two lines.

    Not merely near them: a writing line under a question sits below both
    the number "1." and its sentence, and must not divide them.
    """
    upper, lower = (a, b) if a.y0 <= b.y0 else (b, a)
    if upper.y1 <= lower.y0 + 1:
        for r in horizontal:
            if (
                upper.y1 - 1 <= r.y0 <= lower.y0 + 1
                and r.x0 < upper.x1 and r.x1 > upper.x0
                and r.x0 < lower.x1 and r.x1 > lower.x0
            ):
                return True
    left, right = (a, b) if a.x0 <= b.x0 else (b, a)
    if left.x1 <= right.x0 + 1:
        for r in vertical:
            if (
                left.x1 - 1 <= r.x0 <= right.x0 + 1
                and r.y0 < left.y1 and r.y1 > left.y0
                and r.y0 < right.y1 and r.y1 > right.y0
            ):
                return True
    return False


def split_at_table_rules(block: TextObject, rules: list[Rect]) -> list[TextObject]:
    """Split a block whose lines sit in different cells of a ruled table.

    PDF producers pack neighbouring cells into one block. On a real packet
    the top-right and bottom-left cells of a 2x2 table were one block with a
    9pt gutter -- too narrow for the column split -- so two questions were
    translated as one sentence and drawn across both cells. The table's own
    rules say where each cell ends, so lines are grouped by the rules that
    enclose them.
    """
    if len(block.lines) < 2 or not rules:
        return [block]
    vertical = [r for r in rules if r.width <= RULE_THICKNESS and r.height > RULE_THICKNESS]
    horizontal = [r for r in rules if r.height <= RULE_THICKNESS and r.width > RULE_THICKNESS]
    if not vertical and not horizontal:
        return [block]
    groups: list[list[TextLine]] = []
    for line in block.lines:
        if not line.text.strip():
            continue  # spacer lines in an empty cell carry nothing
        if groups and not _ruled_apart(
            groups[-1][-1].bbox, line.bbox, vertical, horizontal
        ):
            groups[-1].append(line)
        else:
            groups.append([line])
    if len(groups) < 2:
        return [block]
    out: list[TextObject] = []
    for lines in groups:
        bbox = Rect.bounding([ln.bbox for ln in lines])
        if bbox is None:
            continue
        out.append(
            block.model_copy(
                deep=True, update={"id": _new_id("text"), "lines": lines, "bbox": bbox}
            )
        )
    return out or [block]


# Tab stops, not word spacing: the gap Word leaves between "Equation:" and
# "a + a + 3 = 15" on a real packet was 28-57pt.
TAB_GAP = 12.0
_MARKER_ONLY = re.compile(r"^\(?(?:\d{1,2}|[a-zA-Z])[.)]$")


def _visible(line: TextLine) -> str:
    return line.text.replace("\u200b", "").strip()


def _clone(block: TextObject, lines: list[TextLine]) -> TextObject | None:
    bbox = Rect.bounding([ln.bbox for ln in lines])
    if bbox is None:
        return None
    return block.model_copy(
        deep=True, update={"id": _new_id("text"), "lines": lines, "bbox": bbox}
    )


def split_tabbed_rows(block: TextObject) -> list[TextObject]:
    """Keep a tab-aligned layout in its rows and columns.

    Page 26 of the real packet sets its worked example out with tabs --
    "Equation:  a + a + 3 = 15", "Solution:  a = 6", each on its own row --
    and Word exported all of it, with the question above, as one block. It
    was translated as one paragraph and every label ran into every answer.
    When two or more rows of a block are split by tab-sized gaps, each piece
    of such a row becomes its own block, and the lines between them stay
    together as the paragraph they are. A bare "1." stays with its words.
    """
    lines = sorted(
        (ln for ln in block.lines if _visible(ln)), key=lambda ln: (ln.bbox.y0, ln.bbox.x0)
    )
    if len(lines) < 3:
        return [block]
    rows: list[list[TextLine]] = []
    for line in lines:
        mid = (line.bbox.y0 + line.bbox.y1) / 2
        if rows and rows[-1][0].bbox.y0 <= mid <= rows[-1][0].bbox.y1:
            rows[-1].append(line)
        else:
            rows.append([line])

    def pieces(row: list[TextLine]) -> list[list[TextLine]]:
        row = sorted(row, key=lambda ln: ln.bbox.x0)
        out = [[row[0]]]
        for prev, line in zip(row, row[1:]):
            marker = len(out[-1]) == 1 and _MARKER_ONLY.match(_visible(out[-1][0]))
            if line.bbox.x0 - prev.bbox.x1 >= TAB_GAP and not marker:
                out.append([line])
            else:
                out[-1].append(line)
        return out

    split_rows = [pieces(row) for row in rows]
    if sum(len(p) > 1 for p in split_rows) < 2:
        return [block]
    out: list[TextObject] = []
    run: list[TextLine] = []
    for parts in split_rows:
        if len(parts) == 1:
            run.extend(parts[0])
            continue
        if run:
            out.append(_clone(block, run))
            run = []
        out.extend(_clone(block, part) for part in parts)
    if run:
        out.append(_clone(block, run))
    return [o for o in out if o is not None] or [block]


# A line that opens a new list item: "1. ", "12) ", "(3) ", "b. ", "• ".
# The marker must be followed by a space, so a wrapped "1.50" or "3.14" is
# not taken for one. Word often writes a zero-width space there instead
# ("1.\u200bCombine"), which counts.
_ITEM_START = re.compile(
    r"^[\s\u200b]*(?:\(?\d{1,2}[.)]|\([a-zA-Z]\)|[a-hA-H][.)]|[•●▪◦‣\-–])[\s\u200b]+\S"
)


def _bold_share(line: TextLine) -> float | None:
    """Fraction of a line's visible characters set in bold, or None if blank."""
    total = bold = 0
    for span in line.spans:
        n = len(span.text.replace("\u200b", "").strip())
        total += n
        if span.style.bold:
            bold += n
    return bold / total if total else None


def rejoin_item_tails(page: Page) -> None:
    """Give the last line of a list item back to the item it ends.

    Page 6 of the real packet: Word cut its numbered steps into blocks one
    line late, so "operations." -- the end of step 2 -- opened the block
    holding step 3, and the same again for step 4. Split by item, each step
    still began with the previous step's last word. Lines that open a block
    before its first item move up to the block directly above, when that
    block is a list too.
    """
    blocks = sorted(
        (t for t in page.text_objects if not t.protected and t.lines),
        key=lambda t: t.bbox.y0,
    )
    for upper, lower in zip(blocks, blocks[1:]):
        lines = lower.lines
        first_item = next(
            (i for i, ln in enumerate(lines) if _ITEM_START.match(ln.text)), None
        )
        if not first_item:
            continue  # starts with an item, or is not a list at all
        if not any(_ITEM_START.match(ln.text) for ln in upper.lines):
            continue
        tail, last = lines[:first_item], upper.lines[-1]
        gap = tail[0].bbox.y0 - last.bbox.y1
        if gap > max(4.0, last.bbox.height * 0.6) or _x_overlap(tail[0].bbox, last.bbox) <= 0:
            continue
        upper.lines = [*upper.lines, *tail]
        lower.lines = lines[first_item:]
        upper.bbox = Rect.bounding([ln.bbox for ln in upper.lines]) or upper.bbox
        lower.bbox = Rect.bounding([ln.bbox for ln in lower.lines]) or lower.bbox
        for t in (upper, lower):
            t.norm_bbox = t.bbox.normalized(page.width, page.height)


def split_at_list_items(block: TextObject) -> list[TextObject]:
    """Give a heading and each numbered or bulleted item its own block.

    Word stores "To Solve Equations...:" and the five steps under it as one
    block. Translated as one, the line breaks were gone: the steps came back
    as one run-on paragraph and were drawn that way. A new block starts at a
    line that opens a list item, and where an all-bold line meets regular
    lines (a heading above its text). A wrapped item keeps its second line.
    """
    lines = [ln for ln in block.lines if ln.text.replace("\u200b", "").strip()]
    if len(lines) < 2:
        return [block]
    groups: list[list[TextLine]] = [[lines[0]]]
    for prev, line in zip(lines, lines[1:]):
        a, b = _bold_share(prev), _bold_share(line)
        # A bold "1." on its own is a marker, not a heading: it stays with
        # the question it numbers.
        worded = all(sum(c.isalpha() for c in ln.text) >= 3 for ln in (prev, line))
        heading_edge = worded and (
            (a >= 0.9 and b <= 0.1) or (a <= 0.1 and b >= 0.9)
        )
        if _ITEM_START.match(line.text) or heading_edge:
            groups.append([line])
        else:
            groups[-1].append(line)
    if len(groups) < 2:
        return [block]
    out: list[TextObject] = []
    for group in groups:
        bbox = Rect.bounding([ln.bbox for ln in group])
        if bbox is None:
            continue
        out.append(
            block.model_copy(
                deep=True, update={"id": _new_id("text"), "lines": group, "bbox": bbox}
            )
        )
    return out or [block]


def split_page_blocks(page: Page) -> int:
    """Split merged blocks on a page. Returns how many new blocks were created."""
    if not page.text_objects:
        return 0
    rejoin_item_tails(page)
    result: list[TextObject] = []
    rules = [
        v.bbox for v in page.vector_objects
        if min(v.bbox.width, v.bbox.height) <= RULE_THICKNESS
    ]
    for block in page.text_objects:
        if block.source is TextSource.IMAGE_OCR:
            # A line read from a picture is already one line. Split at the
            # wide gaps a blurry screenshot leaves between words, it became
            # half-sentences, and lost the "hard to read" note made for it.
            result.append(block)
            continue
        for piece in isolate_math_glyphs(block):
            if piece.protected:
                result.append(piece)
                continue
            for cell in split_at_table_rules(piece, rules):
                for row in split_tabbed_rows(cell):
                    for item in split_at_list_items(row):
                        result.extend(split_text_block(item))
    created = len(result) - len(page.text_objects)
    if created:
        for t in result:
            t.norm_bbox = t.bbox.normalized(page.width, page.height)
        page.text_objects = result
        page.diagnostics.extra["blocks_split"] = created
    return created
