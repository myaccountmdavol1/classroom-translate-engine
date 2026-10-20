"""Page-local reflow for text that cannot fit in place (spec sec. 25).

Bilingual output runs roughly 2.4x the length of the source. On a dense page
there is nowhere for it to go, and the bounded concessions -- re-wrap, tighten
leading, shrink, widen -- run out. The result is text drawn over its
neighbour, which is worse than any of the alternatives.

This module is the exception path. On a page where something still does not
fit, following text blocks are pushed DOWN into the whitespace below, so the
overflowing block gets the room it needs.

Four things are never moved or covered, and they bound the whole operation:

  * images and diagrams,
  * student response areas (answer lines, boxes) -- inviolable,
  * protected blocks, whose original glyphs we are deliberately keeping,
  * the page itself: nothing may be pushed past the bottom margin.

If the room is not there, the block falls back to TRANSLATED-ONLY for that
block and the page is flagged. Dropping the English from one question is
recoverable in review; overlapping text is not readable at all.

Reflow moves text only. Question order, page count and every fixed element
stay exactly where they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model.document import Page
from ..model.enums import TranslationStatus
from ..model.geometry import Rect
from ..model.objects import TextObject

# Keep this much clear at the foot of the page; pushing text into the very
# last few points looks broken even when it technically fits.
BOTTOM_MARGIN = 36.0
# Blocks closer than this horizontally are treated as the same column, so a
# side-by-side item is not dragged down by its neighbour.
COLUMN_OVERLAP = 0.45
# Never shift a block further than this. A large displacement stops looking
# like the original document.
MAX_SHIFT = 160.0


@dataclass
class ReflowPlan:
    shifts: dict[str, float] = field(default_factory=dict)
    extra_height: dict[str, float] = field(default_factory=dict)
    # Blocks that must drop their bilingual pairing: there was no room.
    translation_only: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    @property
    def moved(self) -> int:
        return sum(1 for v in self.shifts.values() if v > 0.01)

    @property
    def is_empty(self) -> bool:
        return not self.extra_height and not self.translation_only


def fixed_obstacles(page: Page, movable_ids: set[str]) -> list[Rect]:
    """Everything reflow must not touch."""
    boxes = [img.bbox for img in page.image_objects]
    # Response areas are inviolable (spec sec. 28).
    boxes += [area.bbox for area in page.response_areas]
    # Vector boxes that read as answer boxes or writing rules.
    boxes += [
        v.bbox
        for v in page.vector_objects
        if v.looks_like_box or v.looks_like_rule
    ]
    # Text we are not replacing keeps its original glyphs in place.
    boxes += [
        t.bbox
        for t in page.text_objects
        if t.id not in movable_ids
        and t.translation_status
        in (TranslationStatus.PROTECTED, TranslationStatus.FAILED,
            TranslationStatus.NOT_REQUIRED)
    ]
    return boxes


def _same_column(a: Rect, b: Rect) -> bool:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.width, b.width) or 1.0
    return overlap / narrower >= COLUMN_OVERLAP


def _collides(box: Rect, obstacles: list[Rect]) -> bool:
    return any(box.overlap_fraction_of_self(o) > 0.02 for o in obstacles)


def plan_reflow(
    page: Page,
    needed: dict[str, float],
    movable: list[TextObject],
    *,
    bottom_margin: float = BOTTOM_MARGIN,
) -> ReflowPlan:
    """Decide how far to push blocks down so overflowing text fits.

    `needed` maps object id -> extra height required beyond its current box.
    Only blocks in `movable` may be shifted.
    """
    plan = ReflowPlan()
    if not needed:
        return plan

    movable_ids = {t.id for t in movable}
    obstacles = fixed_obstacles(page, movable_ids)
    ordered = sorted(movable, key=lambda t: (t.bbox.y0, t.bbox.x0))

    # Granted growth, in document order. A block inherits displacement only
    # from blocks ABOVE IT IN ITS OWN COLUMN: a single global offset dragged
    # side-by-side neighbours down with it, which wrecks a two-column layout.
    granted: list[tuple[TextObject, float]] = []

    def shift_for(block: TextObject) -> float:
        # A block is NOT displaced by its own growth: it grows downward from
        # its own top edge. Only grants from blocks above it, in its column,
        # move it.
        return sum(
            extra
            for source, extra in granted
            if source is not block
            and source.bbox.y0 <= block.bbox.y0
            and _same_column(source.bbox, block.bbox)
        )

    for index, block in enumerate(ordered):
        shift = shift_for(block)
        plan.shifts[block.id] = shift

        extra = needed.get(block.id, 0.0)
        if extra <= 0.01:
            continue

        if shift + extra > MAX_SHIFT:
            plan.translation_only.add(block.id)
            continue

        # Would the taller block itself run into something fixed?
        grown = Rect(
            x0=block.bbox.x0,
            y0=block.bbox.y0 + shift,
            x1=block.bbox.x1,
            y1=block.bbox.y1 + shift + extra,
        )
        if grown.y1 > page.height - bottom_margin or _collides(grown, obstacles):
            plan.translation_only.add(block.id)
            continue

        # Would the blocks below it, in its column, still fit?
        feasible = True
        for later in ordered[index + 1 :]:
            if not _same_column(block.bbox, later.bbox):
                continue
            later_shift = shift_for(later) + extra
            shifted = Rect(
                x0=later.bbox.x0,
                y0=later.bbox.y0 + later_shift,
                x1=later.bbox.x1,
                y1=later.bbox.y1 + later_shift,
            )
            if shifted.y1 > page.height - bottom_margin or _collides(
                shifted, obstacles
            ):
                feasible = False
                break

        if feasible:
            plan.extra_height[block.id] = round(extra, 2)
            granted.append((block, extra))
        else:
            plan.translation_only.add(block.id)

    # Recompute shifts now that every grant is known, so a block below two
    # grown blocks in its column picks up both.
    for block in ordered:
        plan.shifts[block.id] = round(shift_for(block), 2)

    if plan.extra_height:
        plan.notes.append(
            f"{len(plan.extra_height)} block(s) were given more room and "
            f"{plan.moved} block(s) moved down to make space."
        )
    if plan.translation_only:
        plan.notes.append(
            f"{len(plan.translation_only)} block(s) show only the translation, "
            "because both languages would not fit."
        )
    return plan
