"""Shape outlines (Google Slides' Insert > Shape), one definition for all.

A shape is a box she made with a different outline: the frame is still
`manual_bbox`, turned by `manual_angle` about its centre, and this module
says what is drawn in it. The same arithmetic is written in TypeScript in
apps/web/src/lib/shapes.ts, for the editor's live drawing, and the two are
held together by apps/web/src/lib/shapes.golden.json, which both test
suites check (tests/test_shapes_geometry.py, src/lib/shapes.test.mjs). A
change here without the same change there fails one of them.

Paths, defaults and text areas are the standard Office preset definitions
(ECMA-376 presetShapeDefinitions.xml), with each 90 degree `arcTo` written
as one cubic Bezier. Every command is in the frame's own points, origin at
its top-left, y down:

    ("M", x, y)  ("L", x, y)  ("C", x1, y1, x2, y2, x, y)  ("Z",)

"rect" alone does not follow Office's order: it starts at the top-left and
runs clockwise, exactly as a bordered box has always been drawn
(builder.paint_border), so a Rectangle's dashes fall where a border's do.
"""

from __future__ import annotations

import math
from typing import Callable

# A quarter circle as one cubic Bezier: the control points sit this far
# along each end's tangent, as a share of the radius.
KAPPA = 0.5522847498
# Joins are mitred up to this ratio, in the PDF (its default) and in the
# editor's SVG (which defaults to 4, so it sets it).
MITER_LIMIT = 10.0

Command = tuple
Outline = list[Command]


def _pin(lo: float, x: float, hi: float) -> float:
    return max(lo, min(x, hi))


def _arc(cx: float, cy: float, rx: float, ry: float, start: float) -> Command:
    """The Bezier for a quarter of an ellipse, clockwise on the page from
    `start` degrees (0 is +x, 90 is down), centred at (cx, cy)."""
    a0 = math.radians(start)
    a1 = math.radians(start + 90.0)
    c0, s0 = math.cos(a0), math.sin(a0)
    c1, s1 = math.cos(a1), math.sin(a1)
    x0, y0 = cx + rx * c0, cy + ry * s0
    x3, y3 = cx + rx * c1, cy + ry * s1
    return (
        "C",
        x0 - KAPPA * rx * s0, y0 + KAPPA * ry * c0,
        x3 + KAPPA * rx * s1, y3 - KAPPA * ry * c1,
        x3, y3,
    )


# --- rect ---------------------------------------------------------------
def _rect_outline(w: float, h: float, adjust: list[float]) -> Outline:
    return [("M", 0.0, 0.0), ("L", w, 0.0), ("L", w, h), ("L", 0.0, h), ("Z",)]


def _rect_text(w: float, h: float, adjust: list[float]) -> tuple[float, float, float, float]:
    return (0.0, 0.0, w, h)


# --- roundRect ------------------------------------------------------------
def _corner(w: float, h: float, adjust: list[float]) -> float:
    """The corner's radius: `adj` 100000ths of the short side, at most half."""
    a = _pin(0.0, adjust[0], 50000.0)
    return min(w, h) * a / 100000.0


def _round_rect_outline(w: float, h: float, adjust: list[float]) -> Outline:
    x1 = _corner(w, h, adjust)
    x2, y2 = w - x1, h - x1
    return [
        ("M", 0.0, x1),
        _arc(x1, x1, x1, x1, 180.0),
        ("L", x2, 0.0),
        _arc(x2, x1, x1, x1, 270.0),
        ("L", w, y2),
        _arc(x2, y2, x1, x1, 0.0),
        ("L", x1, h),
        _arc(x1, y2, x1, x1, 90.0),
        ("Z",),
    ]


def _round_rect_text(w: float, h: float, adjust: list[float]) -> tuple[float, float, float, float]:
    il = _corner(w, h, adjust) * 29289.0 / 100000.0
    return (il, il, w - il, h - il)


# --- ellipse ---------------------------------------------------------------
def _ellipse_outline(w: float, h: float, adjust: list[float]) -> Outline:
    rx, ry = w / 2.0, h / 2.0
    return [
        ("M", 0.0, ry),
        _arc(rx, ry, rx, ry, 180.0),
        _arc(rx, ry, rx, ry, 270.0),
        _arc(rx, ry, rx, ry, 0.0),
        _arc(rx, ry, rx, ry, 90.0),
        ("Z",),
    ]


def _ellipse_text(w: float, h: float, adjust: list[float]) -> tuple[float, float, float, float]:
    rx, ry = w / 2.0, h / 2.0
    idx = rx * math.cos(math.pi / 4.0)
    idy = ry * math.sin(math.pi / 4.0)
    return (rx - idx, ry - idy, rx + idx, ry + idy)


class ShapeDef:
    def __init__(
        self,
        outline: Callable[[float, float, list[float]], Outline],
        text: Callable[[float, float, list[float]], tuple[float, float, float, float]],
        adjust: list[float],
        name: str,
        click: tuple[float, float],
        corners: Callable[[float, float, list[float]], bool],
    ) -> None:
        self.outline = outline
        self.text = text
        self.adjust = adjust
        self.name = name
        self.click = click
        # Whether the outline has sharp corners (which a thick line mitres
        # past the frame) rather than only smooth turns.
        self.corners = corners


SHAPES: dict[str, ShapeDef] = {
    "rect": ShapeDef(_rect_outline, _rect_text, [], "Rectangle", (72.0, 72.0),
                     lambda w, h, a: True),
    "roundRect": ShapeDef(_round_rect_outline, _round_rect_text, [16667.0],
                          "Rounded rectangle", (72.0, 72.0),
                          lambda w, h, a: _corner(w, h, a) <= 0.0),
    "ellipse": ShapeDef(_ellipse_outline, _ellipse_text, [], "Oval", (72.0, 72.0),
                        lambda w, h, a: False),
}


def known(shape: str | None) -> str:
    """The shape drawn for a stored name: an unknown one (a manifest from a
    later build with more shapes) is drawn as a rectangle."""
    return shape if shape in SHAPES else "rect"


def _adjust(shape: str, adjust: list[float] | None) -> list[float]:
    d = SHAPES[shape].adjust
    if not adjust:
        return list(d)
    return [float(adjust[i]) if i < len(adjust) else d[i] for i in range(len(d))]


def outline(shape: str | None, w: float, h: float, adjust: list[float] | None = None) -> Outline:
    """The shape's outline in a w x h frame, origin top-left."""
    s = known(shape)
    return SHAPES[s].outline(float(w), float(h), _adjust(s, adjust))


def text_area(
    shape: str | None, w: float, h: float, adjust: list[float] | None = None
) -> tuple[float, float, float, float]:
    """(l, t, r, b) in the frame: where the words go, before her inside
    space is taken off."""
    s = known(shape)
    return SHAPES[s].text(float(w), float(h), _adjust(s, adjust))


def ink_reach(
    shape: str | None, w: float, h: float, adjust: list[float] | None, weight: float
) -> float:
    """How far a line of this weight can reach past the frame. Half the
    stroke lies outside; a corner's mitre reaches (weight/2)/sin(angle/2)
    when that is within the miter limit, else it is cut at weight/2. Smooth
    turns reach weight/2. Every corner these shapes have is a right angle."""
    if weight <= 0:
        return 0.0
    s = known(shape)
    if SHAPES[s].corners(float(w), float(h), _adjust(s, adjust)):
        mitre = 1.0 / math.sin(math.pi / 4.0)
        return (weight / 2.0) * (mitre if mitre <= MITER_LIMIT else 1.0)
    return weight / 2.0


def placed(cmds: Outline, x0: float, y0: float) -> Outline:
    """An outline moved from the frame's origin to (x0, y0)."""
    out: Outline = []
    for c in cmds:
        if c[0] == "Z":
            out.append(c)
            continue
        nums = list(c[1:])
        for i in range(0, len(nums), 2):
            nums[i] += x0
            nums[i + 1] += y0
        out.append((c[0], *nums))
    return out


def layout_ink_reach(layout) -> float:
    """How far a drawn block's outline reaches past its frame
    (DrawnLayout): a shape's from ink_reach; a plain box's border its
    whole weight -- half the stroke lies outside the frame, and a turned
    corner's mitre reaches about 0.71 of the weight out. None without an
    outline."""
    if layout is None or layout.border is None:
        return 0.0
    if layout.shape is not None:
        frame = layout.frame
        w, h = (frame.width, frame.height) if frame is not None else (0.0, 0.0)
        return ink_reach(layout.shape, w, h, None, layout.border_width)
    return layout.border_width
