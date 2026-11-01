"""Join a PDF paragraph the reader handed back in pieces.

PyMuPDF groups a page's text into blocks, and on slides made in Canva it
returned a centred paragraph as one block per line or two. Each piece was
translated alone, as a fragment ("needs and Learn. Explore. Apply.
Process. Earn"), and the longer translations ran into each other.

The pieces of one paragraph touch: Canva sets its lines so close that one
line's box overlaps the next. A new paragraph leaves space (a slide's
"Take this time to plan." sits 16pt above the next paragraph, over a third
of a line). So pieces join only when they touch, look the same (font,
size, weight, colour), share the column, and nothing marks a new item or
a table cell between them.

And only mid-sentence: the piece above stops without ending its sentence
and the next carries on in lower case ("no matter their" / "level."), or
with a name ("for your current" / "Magma Math needs.") after a piece that
reads as a sentence, its words mostly in lower case. On
seven real decks, touching pieces that looked the same were also a
heading over its paragraph ("SUBSTITUTION" / "Technology acts as..."), a
question over its answer, and a name over a title over a phone number;
joined, each lost the line break that made it what it was. Those are left
for the editor's Combine. A scan's lines are left to ocr_grouping, which
has its own, looser rules for words read from pixels.
"""

from __future__ import annotations

from ..ingest.pdf_parser import _detect_alignment, _mean_line_spacing
from ..model.document import Page
from ..model.enums import TextSource
from ..model.geometry import Rect
from ..model.objects import TextObject
from .block_split import RULE_THICKNESS, _ruled_apart
from .ocr_grouping import _mid_sentence, _starts_new_item, _unfinished, _x_overlap_ratio

# The most space between two pieces, as a share of the taller line, for
# them to be one paragraph. Touching or overlapping lines are well under.
MAX_GAP_SHARE = 0.25
# Sizes as the PDF gives them, so one paragraph's are the same; this allows
# for rounding only.
MAX_SIZE_RATIO = 1.05
MIN_X_OVERLAP = 0.55
# A piece reads as a sentence, not a heading or a name, with at least this
# many words, most of them starting in lower case.
MIN_SENTENCE_WORDS = 4
# The pieces the reader made of a slide's paragraph had one or two lines.
# A longer block whose last line has no full stop was a list with its
# bullets drawn as pictures, and the capitalised paragraph under it is not
# its next line.
MAX_PIECE_LINES = 2


def _reads_as_sentence(text: str) -> bool:
    words = [w for w in text.split() if w[0].isalpha()]
    lower = sum(1 for w in words if w[0].islower())
    return len(words) >= MIN_SENTENCE_WORDS and lower * 2 > len(words)


def _runs_on(previous: TextObject, block: TextObject) -> bool:
    """The sentence in `previous` carries on into `block`."""
    if _mid_sentence(previous, block):
        return True
    start = block.flat_text.lstrip()[:1]
    return (
        start.isalpha() and start.isupper()
        and len(previous.lines) <= MAX_PIECE_LINES
        and _unfinished(previous)
        and _reads_as_sentence(previous.flat_text)
    )


def _joinable(block: TextObject) -> bool:
    return (
        block.source is TextSource.NATIVE
        and not block.protected
        and not block.suppressed
        and bool(block.lines)
        and all(abs(ln.rotation) < 0.5 for ln in block.lines)
    )


def _same_look(a: TextObject, b: TextObject) -> bool:
    sa, sb = a.style, b.style
    if not sa.font_size or not sb.font_size:
        return False
    ratio = max(sa.font_size, sb.font_size) / min(sa.font_size, sb.font_size)
    return (
        ratio <= MAX_SIZE_RATIO
        and sa.font_family == sb.font_family
        and sa.bold == sb.bold
        and sa.italic == sb.italic
        and (sa.color or "").lower() == (sb.color or "").lower()
    )


def _continues(previous: TextObject, block: TextObject, horizontal: list[Rect]) -> bool:
    """Whether `block` is the next piece of the paragraph ending in `previous`."""
    last, first = previous.lines[-1].bbox, block.lines[0].bbox
    gap = first.y0 - last.y1
    height = max(last.height, first.height, 1.0)
    return (
        gap <= height * MAX_GAP_SHARE
        # Below it, not beside it: a block starting higher is another column.
        and first.y0 > last.y0
        and _x_overlap_ratio(last, first) >= MIN_X_OVERLAP
        and _same_look(previous, block)
        and _runs_on(previous, block)
        and not _starts_new_item(block.flat_text)
        and not _ruled_apart(previous.bbox, block.bbox, [], horizontal)
    )


def _merge(run: list[TextObject], page: Page) -> TextObject:
    first = run[0]
    lines = [ln for b in run for ln in b.lines]
    bbox = Rect.bounding([ln.bbox for ln in lines]) or first.bbox
    return first.model_copy(
        update={
            "lines": lines,
            "bbox": bbox,
            "norm_bbox": bbox.normalized(page.width, page.height),
            # One line could not say how a piece was set; the paragraph can.
            "alignment": _detect_alignment(lines),
            "line_spacing": _mean_line_spacing(lines, 0.0),
            "cover_boxes": [r for b in run for r in b.cover_boxes],
        }
    )


def join_paragraph_pieces(page: Page) -> int:
    """Join the pieces of each split paragraph. Returns blocks removed."""
    pieces = [t for t in page.text_objects if _joinable(t)]
    if len(pieces) < 2:
        return 0
    horizontal = [
        v.bbox for v in page.vector_objects
        if v.bbox.height <= RULE_THICKNESS < v.bbox.width
    ]

    pieces.sort(key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))
    # Every open run is a candidate, as in ocr_grouping: read top to bottom,
    # two columns' pieces alternate. The closest run above wins.
    runs: list[list[TextObject]] = [[pieces[0]]]
    for block in pieces[1:]:
        best: list[TextObject] | None = None
        for run in runs:
            if _continues(run[-1], block, horizontal) and (
                best is None or run[-1].bbox.y1 > best[-1].bbox.y1
            ):
                best = run
        if best is not None:
            best.append(block)
        else:
            runs.append([block])

    removed = len(pieces) - len(runs)
    if not removed:
        return 0
    joined = {id(run[0]): _merge(run, page) for run in runs if len(run) > 1}
    gone = {id(b) for run in runs for b in run[1:]}
    page.text_objects = [
        joined.get(id(t), t) for t in page.text_objects if id(t) not in gone
    ]
    page.diagnostics.extra["paragraph_pieces_joined"] = removed
    return removed
