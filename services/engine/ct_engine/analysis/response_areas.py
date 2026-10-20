"""Student response-area detection (spec sec. 28).

Response areas are inviolable: translated text may not consume the space a
student needs to write in. Detection is deliberately conservative -- a missed
area costs a QA flag, while a false positive would block legitimate text
placement.

Evidence used:
  * runs of evenly spaced horizontal rules (writing lines)
  * empty rectangles large enough to write in (answer boxes)
Both come from vector geometry the parser already extracted.
"""

from __future__ import annotations

from ..model.document import Page
from ..model.geometry import Rect
from ..model.objects import ResponseArea, VectorObject

# A writing rule must be at least this long to count; shorter strokes are
# underlines, table borders or leader lines.
MIN_RULE_LENGTH = 90.0
# Consecutive rules further apart than this are unrelated.
MAX_RULE_GAP = 46.0
MIN_RULE_GAP = 8.0
# An answer box must be big enough to actually write in.
MIN_BOX_W, MIN_BOX_H = 60.0, 28.0


def _horizontal_rules(vectors: list[VectorObject]) -> list[VectorObject]:
    out = []
    for v in vectors:
        if not v.looks_like_rule:
            continue
        if v.bbox.width < MIN_RULE_LENGTH:
            continue
        if v.bbox.height > 2.5:  # must be horizontal, not a vertical rule
            continue
        out.append(v)
    return out


def detect_response_areas(page: Page) -> list[ResponseArea]:
    areas: list[ResponseArea] = []
    pw, ph = page.width, page.height

    rules = sorted(_horizontal_rules(page.vector_objects), key=lambda v: v.bbox.y0)
    used: set[str] = set()

    # --- Writing-line groups ---
    i = 0
    while i < len(rules):
        group = [rules[i]]
        j = i + 1
        while j < len(rules):
            gap = rules[j].bbox.y0 - group[-1].bbox.y0
            # Same group only if vertically close AND horizontally aligned:
            # two unrelated columns of lines must not merge into one area.
            x_overlap = min(rules[j].bbox.x1, group[-1].bbox.x1) - max(
                rules[j].bbox.x0, group[-1].bbox.x0
            )
            aligned = x_overlap > 0.55 * min(
                rules[j].bbox.width, group[-1].bbox.width
            )
            if MIN_RULE_GAP <= gap <= MAX_RULE_GAP and aligned:
                group.append(rules[j])
                j += 1
            else:
                break
        if len(group) >= 2:
            bbox = Rect.bounding([g.bbox for g in group])
            assert bbox is not None
            # Extend upward by one line gap: a student writes ON the first
            # rule, so the usable space starts above it.
            gaps = [
                group[k + 1].bbox.y0 - group[k].bbox.y0 for k in range(len(group) - 1)
            ]
            lead = sum(gaps) / len(gaps) if gaps else 18.0
            bbox = Rect(
                x0=bbox.x0, y0=max(0.0, bbox.y0 - lead), x1=bbox.x1, y1=bbox.y1
            )
            areas.append(
                ResponseArea(
                    bbox=bbox,
                    norm_bbox=bbox.normalized(pw, ph),
                    kind="writing_lines",
                    confidence=round(min(0.95, 0.55 + 0.1 * len(group)), 3),
                    line_count=len(group),
                    source_object_ids=[g.id for g in group],
                )
            )
            used.update(g.id for g in group)
        i = j if j > i + 1 else i + 1

    # --- Empty answer boxes ---
    text_boxes = [t.bbox for t in page.text_objects]
    image_boxes = [im.bbox for im in page.image_objects]
    for v in page.vector_objects:
        if v.id in used or not v.looks_like_box:
            continue
        if v.bbox.width < MIN_BOX_W or v.bbox.height < MIN_BOX_H:
            continue
        # A box containing text or artwork is a table cell or a figure frame,
        # not blank writing space.
        occupied = any(
            v.bbox.overlap_fraction_of_self(b) > 0.12 for b in text_boxes + image_boxes
        )
        if occupied:
            continue
        # A box covering most of the page is a border, not an answer box.
        if pw and ph and v.bbox.area > 0.55 * pw * ph:
            continue
        areas.append(
            ResponseArea(
                bbox=v.bbox,
                norm_bbox=v.bbox.normalized(pw, ph),
                kind="answer_box",
                confidence=0.7,
                source_object_ids=[v.id],
            )
        )

    # Merge heavily overlapping detections so the same space is not reported
    # twice by two different rules.
    merged: list[ResponseArea] = []
    for a in sorted(areas, key=lambda r: -r.bbox.area):
        if any(a.bbox.overlap_fraction_of_self(m.bbox) > 0.75 for m in merged):
            continue
        merged.append(a)
    return merged
