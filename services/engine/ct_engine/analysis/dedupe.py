"""Remove text the engine read twice.

Two stages can see the same glyphs. Page OCR rasterizes the whole page and
reads everything on it, including the words inside small pictures; image-text
extraction then renders each picture on its own, at higher density, and reads
it again. Neither is wrong, and neither can be dropped -- page OCR is what
reads a scan, and image OCR is what makes a diagram label legible -- but when
both fire on the same words the page ends up with two blocks describing one
piece of text.

That is not a cosmetic duplicate. Both blocks get translated, and both get
DRAWN, so the reader sees two translations printed over each other. On a real
homework page, "Solve the equations." appeared as "Ry Resuelva las
ecuaciones." with "Es decir Resolver las ecuaciones." overprinted on top of
it, at a slightly different size -- illegible, and impossible for a teacher
to fix in the editor because only one of the two was where they clicked.

So near-identical, near-coincident blocks are collapsed to one before
translation. Which one survives is decided on evidence, not on stage order:
native text beats recognised text because it is exact, and between two
recognitions the more confident one wins.
"""

from __future__ import annotations

import re

from ..model.document import Page
from ..model.enums import TextSource
from ..model.objects import TextObject
from ..util.logging import get_logger

log = get_logger(__name__)

# Both boxes must cover at least this much of each other. Requiring it in
# both directions is what stops a long line from swallowing a short label
# that merely sits inside it.
MIN_MUTUAL_OVERLAP = 0.55

# How alike the words have to be. Recognition of the same pixels twice gives
# similar but rarely identical text ("ie Solve the equations." vs "Ry Solve
# the equations."), so this compares the letters that both readings agree on.
MIN_TEXT_SIMILARITY = 0.6


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _similarity(a: str, b: str) -> float:
    """Character-level agreement between two readings of the same text."""
    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return 1.0 if ka == kb else 0.0
    if ka == kb:
        return 1.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, ka, kb).ratio()


def _confidence(obj: TextObject) -> float:
    if obj.source is TextSource.NATIVE:
        return 2.0  # exact by construction; always wins
    return obj.confidence if obj.confidence is not None else 0.0


def _prefer(a: TextObject, b: TextObject) -> bool:
    """True when `a` is the reading to keep."""
    ca, cb = _confidence(a), _confidence(b)
    if ca != cb:
        return ca > cb
    # Equally confident: keep the longer reading, which has lost less.
    if len(a.text) != len(b.text):
        return len(a.text) > len(b.text)
    return True


# A block counts as inside another when this much of it is covered.
CONTAINED_FRACTION = 0.8
# And as saying the same thing when this much of its text appears, in
# order, in the containing block's text.
CONTAINED_TEXT = 0.85


def _contained_text(inner: str, outer: str) -> float:
    """How much of `inner` the reading `outer` already says, in order."""
    ki, ko = _key(inner), _key(outer)
    if not ki or not ko:
        return 0.0
    if ki in ko:
        return 1.0
    from difflib import SequenceMatcher

    blocks = SequenceMatcher(None, ki, ko, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(ki)


def drop_duplicate_text(page: Page) -> list[str]:
    """Collapse text blocks that describe the same glyphs.

    Returns the ids that were removed. A person-created block is never
    touched: it exists precisely because the engine missed something, and it
    is allowed to sit on top of whatever is already there.

    Candidates are tracked by POSITION, not by id. Ids are expected to be
    unique, but nothing here needs to assume it, and keying a removal set on
    a value that might repeat would quietly delete both copies instead of
    one.
    """
    objects = [t for t in page.text_objects if not t.is_manual_block]
    dropped: set[int] = set()

    for i, a in enumerate(objects):
        if i in dropped or not a.text.strip():
            continue
        for j in range(i + 1, len(objects)):
            if j in dropped:
                continue
            b = objects[j]
            if not b.text.strip():
                continue
            # Two native blocks are two real blocks; the parser does not
            # report the same glyphs twice.
            if a.source is TextSource.NATIVE and b.source is TextSource.NATIVE:
                continue
            if (
                a.bbox.overlap_fraction_of_self(b.bbox) < MIN_MUTUAL_OVERLAP
                or b.bbox.overlap_fraction_of_self(a.bbox) < MIN_MUTUAL_OVERLAP
            ):
                continue
            if _similarity(a.text, b.text) < MIN_TEXT_SIMILARITY:
                continue

            keep_a = _prefer(a, b)
            keep, drop, drop_index = (
                (a, b, j) if keep_a else (b, a, i)
            )
            dropped.add(drop_index)
            # The kept block must still cover the pixels the dropped one
            # would have, or a raster page keeps its original words showing
            # through beside the translation.
            if drop.source in (TextSource.OCR, TextSource.IMAGE_OCR):
                keep.cover_boxes = [*keep.cover_boxes, drop.bbox]
            if drop_index == i:
                break

    # Second pass: a reading that sits inside another and only repeats part
    # of it. Pairwise overlap cannot see this -- one OCR stage reads a
    # paragraph as one block, the other reads it line by line, and no single
    # line covers half the paragraph. Real packet, page 33: both readings of
    # question 6 were translated and drawn, one over the other.
    for i, inner in enumerate(objects):
        if i in dropped or not inner.text.strip():
            continue
        for j, outer in enumerate(objects):
            if j == i or j in dropped or not outer.text.strip():
                continue
            # Only two different readers can see the same glyphs twice; one
            # pass never reports its own words in two overlapping blocks.
            if inner.source is outer.source:
                continue
            if len(_key(outer.text)) <= len(_key(inner.text)):
                continue  # the container is the one that says more
            if inner.bbox.overlap_fraction_of_self(outer.bbox) < CONTAINED_FRACTION:
                continue
            if _contained_text(inner.text, outer.text) < CONTAINED_TEXT:
                continue
            dropped.add(i)
            if inner.source in (TextSource.OCR, TextSource.IMAGE_OCR):
                outer.cover_boxes = [*outer.cover_boxes, inner.bbox]
            break

    if dropped:
        removed_ids = {objects[i].id for i in dropped}
        keep_set = {id(objects[i]) for i in range(len(objects)) if i not in dropped}
        page.text_objects = [
            t
            for t in page.text_objects
            if t.is_manual_block or id(t) in keep_set
        ]
        for image in page.image_objects:
            image.ocr_regions = [
                r for r in image.ocr_regions if id(r) in keep_set
            ]
        page.diagnostics.extra["duplicate_text_dropped"] = len(dropped)
        log.debug(
            "Page %s: dropped %d duplicate text block(s)", page.number, len(dropped)
        )
        return sorted(removed_ids)
    return []
