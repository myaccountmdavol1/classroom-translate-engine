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


# --- Lines (Google Slides' Insert > Line) -----------------------------------
# A line is a box she made too: a thin band whose length is the line,
# turned by `manual_angle` about its centre, so every action that moves or
# turns a box moves or turns a line with no endpoints to keep in step.
# Start is the middle of the band's left side, end the middle of its right.
LINE_BAND = 8.0
# A straight line, and one bent into a curve: the curve's middle sits off
# the straight line by `adjust[0]` 100000ths of its length, to the band's
# top (its left, going from start to end) when positive.
LINES: dict[str, str] = {"line": "Line", "curve": "Curved line"}
CURVE_BEND = 25000.0
MAX_BEND = 50000.0
# What a line can have at either end. None is a plain end.
LINE_ENDS: tuple[str, ...] = (
    "arrow", "open_arrow", "circle", "open_circle",
    "square", "open_square", "diamond", "open_diamond",
)
_SQRT2 = math.sqrt(2.0)


def is_line(shape: str | None) -> bool:
    """A line or a curve: drawn as a stroke with ends, never filled, no words."""
    return shape in LINES


def line_frame(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float, float, float], float | None]:
    """((x0, y0, x1, y1), angle) of the band from `start` to `end`: as long
    as the line, LINE_BAND high, centred on its middle, turned by `angle`
    degrees clockwise in (-180, 180] -- None when it runs left to right."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    cx, cy = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
    half = LINE_BAND / 2.0
    frame = (cx - length / 2.0, cy - half, cx + length / 2.0, cy + half)
    angle = math.degrees(math.atan2(dy, dx)) if length > 0 else 0.0
    if angle <= -180.0:
        angle += 360.0
    return frame, (None if abs(angle) < 1e-9 else angle)


def line_ends(
    frame: tuple[float, float, float, float], angle: float | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(start, end) of the line drawn in this band: the middles of its left
    and right sides, turned by `angle` about its centre."""
    x0, y0, x1, y1 = frame
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = (x1 - x0) / 2.0
    t = math.radians(angle or 0.0)
    ux, uy = math.cos(t) * half, math.sin(t) * half
    return (cx - ux, cy - uy), (cx + ux, cy + uy)


def head_size(weight: float) -> float:
    """How big a line's end is: Office's medium, three weights, with a floor
    so a thin line's arrow can still be seen on paper."""
    return max(3.0 * weight, 6.0)


def _bend(shape: str | None, adjust: list[float] | None) -> float:
    """A curve's bend as a share of its length, pinned to +-0.5; 0 for a
    straight line."""
    if shape != "curve":
        return 0.0
    a = float(adjust[0]) if adjust else CURVE_BEND
    return _pin(-MAX_BEND, a, MAX_BEND) / 100000.0


def _quad(p0, q, p2, t):
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * q[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * q[1] + t * t * p2[1],
    )


def _dist(a, b) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _t_at(p0, q, p2, d: float, from_end: bool) -> float:
    """The curve's parameter where it is `d` (straight-line distance) from
    its start (or its end): 60 halvings, the same in both copies. For bends
    up to 0.5 that distance only grows along the curve."""
    if d <= 0:
        return 1.0 if from_end else 0.0
    anchor = p2 if from_end else p0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        t = 1.0 - mid if from_end else mid
        if _dist(_quad(p0, q, p2, t), anchor) < d:
            lo = mid
        else:
            hi = mid
    m = (lo + hi) / 2.0
    return 1.0 - m if from_end else m


def _unit(dx: float, dy: float, fallback: tuple[float, float]) -> tuple[float, float]:
    n = math.sqrt(dx * dx + dy * dy)
    return (dx / n, dy / n) if n > 1e-12 else fallback


def _head(kind: str, tip, u, s: float) -> tuple[Outline, bool]:
    """One end's drawing at `tip`, pointing along unit `u` (outward), size
    `s`: (commands, filled). Open ends are stroked; a closed one ends in Z."""
    ux, uy = u
    nx, ny = -uy, ux
    tx, ty = tip
    h = s / 2.0
    filled = not kind.startswith("open_")
    base = kind[5:] if kind.startswith("open_") else kind
    if base == "arrow":
        a = (tx - s * ux + h * nx, ty - s * uy + h * ny)
        b = (tx - s * ux - h * nx, ty - s * uy - h * ny)
        if filled:
            return [("M", tx, ty), ("L", *a), ("L", *b), ("Z",)], True
        return [("M", *a), ("L", tx, ty), ("L", *b)], False
    if base == "circle":
        return [
            ("M", tx - h, ty),
            _arc(tx, ty, h, h, 180.0),
            _arc(tx, ty, h, h, 270.0),
            _arc(tx, ty, h, h, 0.0),
            _arc(tx, ty, h, h, 90.0),
            ("Z",),
        ], filled
    if base == "square":
        pts = [(h, h), (h, -h), (-h, -h), (-h, h)]
    else:  # diamond
        pts = [(h, 0.0), (0.0, -h), (-h, 0.0), (0.0, h)]
    corners = [(tx + a * ux + b * nx, ty + a * uy + b * ny) for a, b in pts]
    return [("M", *corners[0])] + [("L", *c) for c in corners[1:]] + [("Z",)], filled


def _pull_back(kind: str | None, weight: float, s: float) -> float:
    """How far short of an end the line stops, so it never shows past a
    filled arrow nor through an open end."""
    if kind is None:
        return 0.0
    if kind == "arrow":
        return s
    if kind == "open_arrow":
        return weight / 2.0
    if kind.startswith("open_"):
        return s / 2.0
    return 0.0


def _end_kind(kind: str | None) -> str | None:
    return kind if kind in LINE_ENDS else None


def line_parts(
    length: float,
    weight: float,
    start: str | None,
    end: str | None,
    bend: float = 0.0,
) -> tuple[Outline, list[tuple[Outline, bool]]]:
    """(stroke, heads) of a line `length` long in its band, origin at the
    band's top-left (start at (0, LINE_BAND/2), end at (length, LINE_BAND/2)).

    The stroke is pulled back for the ends; [] when the ends meet. Each
    head is (commands, filled), start first, pointing outward; heads are
    always solid, in the line's colour. `bend` (a share of the length) makes
    the line a quadratic curve whose middle sits `bend * length` toward the
    band's top; its ends follow the curve, each pointing from where the
    curve is one head-size from it, so an arrow meets its line exactly."""
    L = float(length)
    c = LINE_BAND / 2.0
    s = head_size(weight)
    start, end = _end_kind(start), _end_kind(end)
    back_s, back_e = _pull_back(start, weight, s), _pull_back(end, weight, s)
    p0, p2 = (0.0, c), (L, c)
    heads: list[tuple[Outline, bool]] = []
    if bend == 0.0:
        stroke: Outline = (
            [("M", back_s, c), ("L", L - back_e, c)] if back_s + back_e < L else []
        )
        if start:
            heads.append(_head(start, p0, (-1.0, 0.0), s))
        if end:
            heads.append(_head(end, p2, (1.0, 0.0), s))
        return stroke, heads
    q = (L / 2.0, c - 2.0 * bend * L)
    ta = _t_at(p0, q, p2, back_s, False)
    tb = _t_at(p0, q, p2, back_e, True)
    stroke = []
    if ta < tb:
        a, b = _quad(p0, q, p2, ta), _quad(p0, q, p2, tb)
        # The piece of the curve from ta to tb, as a quadratic, written as
        # the one cubic Bezier both PDF and SVG draw.
        m = (
            (1 - ta) * (1 - tb) * p0[0] + ((1 - ta) * tb + ta * (1 - tb)) * q[0] + ta * tb * p2[0],
            (1 - ta) * (1 - tb) * p0[1] + ((1 - ta) * tb + ta * (1 - tb)) * q[1] + ta * tb * p2[1],
        )
        stroke = [
            ("M", a[0], a[1]),
            (
                "C",
                a[0] + 2.0 / 3.0 * (m[0] - a[0]), a[1] + 2.0 / 3.0 * (m[1] - a[1]),
                b[0] + 2.0 / 3.0 * (m[0] - b[0]), b[1] + 2.0 / 3.0 * (m[1] - b[1]),
                b[0], b[1],
            ),
        ]
    if start:
        k = _quad(p0, q, p2, _t_at(p0, q, p2, s, False))
        heads.append(_head(start, p0, _unit(p0[0] - k[0], p0[1] - k[1], (-1.0, 0.0)), s))
    if end:
        k = _quad(p0, q, p2, _t_at(p0, q, p2, s, True))
        heads.append(_head(end, p2, _unit(p2[0] - k[0], p2[1] - k[1], (1.0, 0.0)), s))
    return stroke, heads


def line_drawing(
    shape: str | None,
    length: float,
    weight: float,
    adjust: list[float] | None,
    start: str | None,
    end: str | None,
) -> tuple[Outline, list[tuple[Outline, bool]]]:
    """line_parts for a stored line or curve (its bend from `adjust`)."""
    return line_parts(length, weight, start, end, _bend(shape, adjust))


def bend_point(length: float, bend: float) -> tuple[float, float]:
    """Where a curve's middle is in its band (the bend handle)."""
    return (length / 2.0, LINE_BAND / 2.0 - bend * length)


def _head_reach(kind: str | None, weight: float) -> float:
    """How far an end's ink can reach from its end point."""
    if kind not in LINE_ENDS:
        return 0.0
    s = head_size(weight)
    base = kind[5:] if kind.startswith("open_") else kind
    r = s * _SQRT2 / 2.0 if base == "square" else s / 2.0
    # An open end's mitred corners reach a little further.
    return r + weight / _SQRT2 if kind.startswith("open_") else r


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
    shape: str | None,
    w: float,
    h: float,
    adjust: list[float] | None,
    weight: float,
    ends: tuple[str | None, str | None] = (None, None),
) -> float:
    """How far a line of this weight can reach past the frame. Half the
    stroke lies outside; a corner's mitre reaches (weight/2)/sin(angle/2)
    when that is within the miter limit, else it is cut at weight/2. Smooth
    turns reach weight/2. Every corner these shapes have is a right angle.

    For a line (w its length): half its weight, each end's head (`ends`,
    start and end), and a curve's bulge off its band's middle."""
    if weight <= 0:
        return 0.0
    if is_line(shape):
        bulge = abs(_bend(shape, adjust)) * float(w)
        return max(weight / 2.0 + bulge, *(_head_reach(e, weight) for e in ends))
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
        return ink_reach(
            layout.shape, w, h, layout.shape_adjust, layout.border_width,
            (layout.line_start, layout.line_end),
        )
    return layout.border_width
