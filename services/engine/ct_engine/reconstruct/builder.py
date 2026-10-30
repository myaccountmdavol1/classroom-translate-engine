"""Page reconstruction (spec sec. 29).

Strategy: keep the original page and change only its text.

    1. Redact exactly the source text we are replacing -- text only, with no
       fill painted, so images, vector artwork and background graphics under
       the text survive untouched.
    2. Draw the translated text at the same coordinates, fitted to the same
       box, with a font proven to cover its characters.

The whole page is never rasterized: that would destroy selectable text and
print quality for a short-term gain in implementation ease (spec sec. 29,
engineering rule 6). Protected blocks are never redacted at all -- their
original glyphs are the safest possible rendering of a formula.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, overload

import fitz

from ..fonts.registry import REGISTRY, FontResolutionError, ResolvedFont, stands_alone
from ..fonts.shaping import shape_logical, shaping_available, to_visual
from ..languages import get_language
from ..model.document import Page, QAFinding
from ..model.enums import TextAlignment, TextSource, TranslationStatus
from ..model.geometry import Rect
from ..fonts import catalog
from ..model.objects import DrawnLayout, DrawnMark, DrawnRow, OrientedBox, Point2
from ..util.logging import get_logger
from .fitting import (
    DEFAULT_LEADING,
    inset,
    FitResult,
    RowSpec,
    available_growth,
    available_growth_right,
    fit_text,
    fit_to_box,
    justifiable,
    wrap_rows,
    wrap_with_ends,
)
from .lists import MarkerFonts, NUMBER_CHANGED, drawable_marker, marker_visual, plan_paragraphs
from .reflow import ReflowPlan, plan_reflow
from . import shapes as shape_geo

log = get_logger(__name__)

# Typographic ascent as a fraction of font size, used to place the first
# baseline inside the box.
ASCENT_RATIO = 0.80
# Descent below the baseline, so ascent plus descent is one em: the box a
# row of words is taken to fill, for vertical alignment and the highlight.
DESCENT_RATIO = 0.20
# An underline: the top of the rule this far below the baseline, and this
# thick, both as fractions of the size, never thinner than the minimum.
UNDERLINE_GAP = 0.08
UNDERLINE_WEIGHT = 0.06
MIN_UNDERLINE_PT = 0.5
# How much of the box's spare height goes above the words.
VERTICAL_SHARE = {None: 0.0, "middle": 0.5, "bottom": 1.0}


@dataclass
class PageBuildStats:
    written: int = 0
    redacted: int = 0
    # Scanned-text regions painted over, and regions left alone because
    # artwork was underneath.
    covered: int = 0
    uncovered: int = 0
    # Blocks a person removed: original taken out, nothing drawn back.
    suppressed: int = 0
    # Blocks shifted down to make room, and blocks that had to give up their
    # bilingual pairing because the page could not be made to fit.
    reflowed: int = 0
    pairing_dropped: int = 0
    shrunk: int = 0
    extended: int = 0
    overflowed: int = 0
    font_failures: int = 0
    findings: list[QAFinding] = field(default_factory=list)



# --- Raster text covering -------------------------------------------------
# Text recognized from a scan lives in the page image, so it cannot be
# redacted. It has to be painted over. Two rules keep that safe:
#   * sample the actual local background, so the patch matches the paper
#     rather than being a white rectangle on a grey scan;
#   * refuse to paint when the background is not uniform, because that means
#     artwork is underneath. Never damage a diagram to replace a label
#     (spec sec. 19).
BACKGROUND_UNIFORMITY_LIMIT = 34.0  # max per-channel std-dev, 0-255


def estimate_background(
    pdf_page: "fitz.Page", rect: Rect, *, pad: float = 2.5
) -> tuple[tuple[float, float, float] | None, float]:
    """Return (rgb 0-1, uniformity_stddev) for the area around `rect`.

    Returns (None, stddev) when the surroundings are too varied to paint
    over safely.
    """
    import numpy as np

    probe = fitz.Rect(
        rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad
    ) & pdf_page.rect
    if probe.is_empty or probe.width < 2 or probe.height < 2:
        return None, 999.0
    try:
        pix = pdf_page.get_pixmap(clip=probe, dpi=72, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )[:, :, :3]
    except Exception:
        return None, 999.0
    if arr.size == 0 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return None, 999.0

    # Sample the border ring only: the interior is the glyphs we are removing,
    # so including it would bias the patch toward ink.
    ring = np.concatenate(
        [
            arr[0, :, :].reshape(-1, 3),
            arr[-1, :, :].reshape(-1, 3),
            arr[:, 0, :].reshape(-1, 3),
            arr[:, -1, :].reshape(-1, 3),
        ]
    )
    if ring.size == 0:
        return None, 999.0

    median = np.median(ring, axis=0)
    spread = float(np.max(np.std(ring, axis=0)))
    if spread > BACKGROUND_UNIFORMITY_LIMIT:
        return None, spread
    return (float(median[0]) / 255.0, float(median[1]) / 255.0,
            float(median[2]) / 255.0), spread


# How far a scan's paper may drift from its own median before it counts as
# a mark, and how many marks the band around a block may hold. Scanner noise
# and JPEG ringing stay well under the first; a stray speck stays under the
# second; a photograph or a panel edge blows through both.
SCAN_PAPER_TOLERANCE = 40  # 0-255, largest channel difference
SCAN_MARK_FRACTION = 0.02
# A dark run this long is a rule, leader or arrow, never a glyph fragment.
SCAN_RULE_LENGTH = 24.0  # points
# Fewest paper pixels (at 144 dpi) worth judging: two 4pt strips at the ends
# of one line of 11pt text.
SCAN_MIN_PAPER_PIXELS = 400


def estimate_scanned_background(
    pdf_page: "fitz.Page",
    rect: Rect,
    known_text: list[Rect],
    *,
    pad: float = 4.0,
) -> tuple[float, float, float] | None:
    """Second opinion on the paper around scanned text, ignoring known text.

    `estimate_background` reads a one-pixel ring 2.5pt out. On a
    single-spaced scan that ring runs through the neighbouring lines'
    ascenders and descenders, so a plain paragraph looks like artwork. Those
    strokes are words the OCR engine recognised, so here every known word box
    is set aside and only the band that remains is judged. Artwork is not a
    recognised word, so it still counts against the patch.

    Returns the paper colour (rgb 0-1), or None to leave the original alone.
    """
    import numpy as np

    if not known_text:
        return None
    probe = fitz.Rect(*rect.expanded(pad).as_tuple()) & pdf_page.rect
    if probe.is_empty:
        return None
    zoom = 2.0  # 144 dpi: a 4pt band is 8 pixels deep
    try:
        pix = pdf_page.get_pixmap(
            clip=probe, matrix=fitz.Matrix(zoom, zoom),
            colorspace=fitz.csRGB, alpha=False,
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
    except Exception:
        return None
    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return None

    def pixels(r: Rect) -> tuple[int, int, int, int]:
        return (
            max(0, int((r.x0 - probe.x0) * zoom)),
            max(0, int((r.y0 - probe.y0) * zoom)),
            min(pix.width, int((r.x1 - probe.x0) * zoom) + 1),
            min(pix.height, int((r.y1 - probe.y0) * zoom) + 1),
        )

    # The band: the probe minus the block itself. Its interior is left out on
    # purpose -- the gap between two words may be a student's answer blank.
    band = np.ones(arr.shape[:2], dtype=bool)
    bx0, by0, bx1, by1 = pixels(rect)
    band[by0:by1, bx0:bx1] = False
    judged = band.copy()
    for box in known_text:
        grown = box.expanded(1.0)  # antialiased glyph edges spill past the box
        if not (grown.x1 > probe.x0 and grown.x0 < probe.x1
                and grown.y1 > probe.y0 and grown.y0 < probe.y1):
            continue
        x0, y0, x1, y1 = pixels(grown)
        judged[y0:y1, x0:x1] = False

    # A long rule above the words is the top border of the table cell they
    # head, and the patch over the words never reaches it: set its rows aside
    # like the neighbours' words. On a planning sheet those borders were 12-18%
    # of the band, so "Math" and "Phys. Ed." in every cell stayed in English.
    if judged.any():
        paper = np.median(arr[judged], axis=0)
        above = np.max(np.abs(arr[:by0].astype(np.int16) - paper), axis=2) > (
            2 * SCAN_PAPER_TOLERANCE
        )
        for row in range(above.shape[0]):
            if _longest_run(above[row : row + 1]) >= int(SCAN_RULE_LENGTH * zoom):
                judged[max(0, row - 1) : row + 2] = False

    # Too little left to judge is not evidence of paper. An absolute floor,
    # not a share of the band: on single spacing the neighbours' words fill
    # the band above and below, and only the strips at either end remain --
    # which is where artwork behind a paragraph would show anyway.
    if judged.sum() < SCAN_MIN_PAPER_PIXELS:
        return None
    samples = arr[judged].astype(np.int16)
    median = np.median(samples, axis=0)
    deviation = np.max(np.abs(samples - median), axis=1)
    if np.mean(deviation > SCAN_PAPER_TOLERANCE) > SCAN_MARK_FRACTION:
        return None

    # Rules are few pixels, so they slip under the mark fraction, and where
    # they pass behind a word box the mask above hides them. So they are
    # found on the unmasked page, by the one thing no glyph does: a stroke
    # running the full depth of the probe, or a long run beside the text.
    dark = np.max(np.abs(arr.astype(np.int16) - median), axis=2) > (
        2 * SCAN_PAPER_TOLERANCE
    )
    if _longest_run(dark.T) >= max(4, dark.shape[0] - 2):
        return None  # something crosses the block top to bottom
    # Below the words only: a rule above them is a cell border (see above).
    beside = dark[by1:]
    if beside.size and _longest_run(beside) >= int(SCAN_RULE_LENGTH * zoom):
        return None  # a leader, underline or arrow along the line
    return (float(median[0]) / 255.0, float(median[1]) / 255.0,
            float(median[2]) / 255.0)


def _longest_run(grid) -> int:
    """Longest run of True along the rows of a 2-D boolean array."""
    import numpy as np

    best = 0
    for row in grid:
        if not row.any():
            continue
        edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        best = max(best, int(np.max(
            np.where(edges == -1)[0] - np.where(edges == 1)[0]
        )))
    return best


def _hex(rgb: tuple[float, float, float]) -> str:
    """A 0-1 RGB triple as sRGB hex, for the API and the editor."""
    return "#" + "".join(
        f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb
    )


def uniform_growth_right(
    pdf_page: "fitz.Page", rect: Rect, *, limit: float = 90.0, step: float = 8.0
) -> float:
    """How far a raster label may widen while the background stays uniform.

    A diagram label usually has empty space beside it, and a translated label
    is often longer than the original. Widening into that space keeps the
    label on one line at its original size instead of wrapping to two tiny
    ones. The background test is what keeps this safe: growth stops as soon
    as the area beside the label stops looking like blank paper, which is how
    artwork is protected.
    """
    grown = 0.0
    while grown + step <= limit:
        probe = Rect(
            x0=rect.x1 + grown,
            y0=rect.y0,
            x1=rect.x1 + grown + step,
            y1=rect.y1,
        )
        if probe.x1 > pdf_page.rect.x1:
            break
        color, _ = estimate_background(pdf_page, probe, pad=0.0)
        if color is None:
            break
        grown += step
    return grown



# --- Tilted words in pictures ----------------------------------------------
# A word printed at a slant has an upright box much bigger than itself: on
# the real slide "HARBOR" on a tilted tablet had an upright box holding the
# tablet's black frame and the top of a red letter, so the ring around it
# varied wildly (spread 103) and the cover was refused -- correctly. A band
# around the TURNED box is the tablet's white screen (spread 0-1.5) -- but
# the frame is only ~2.5pt above the word, so a band reaching 2.5pt out
# flipped to "artwork" (spread 22-53) when the measured box moved by a
# quarter of a point. So a 2.5pt band is tried first and a 1.6pt one after
# it. Either fully encloses the 0.6pt patch with plain pixels, and a line
# running into the word would cross either band, so the narrower band
# protects artwork just the same. Inside 0.8pt is the letters' own
# anti-aliasing. Sampled at 288 dpi, so even the narrow band is ~3 pixels.
QUAD_RINGS = ((0.8, 2.5), (0.8, 1.6))
QUAD_ZOOM = 4.0


def _turned_pixels(pdf_page: "fitz.Page", ob: OrientedBox, reach: float, zoom: float):
    """(rgb pixels, distance outside `ob` in points) for the area within
    `reach` of the turned box, or None. Distance is negative inside."""
    import numpy as np

    probe = fitz.Rect(*ob.expanded(reach).aabb().as_tuple()) & pdf_page.rect
    if probe.is_empty or probe.width < 1 or probe.height < 1:
        return None
    try:
        pix = pdf_page.get_pixmap(
            clip=probe, matrix=fitz.Matrix(zoom, zoom),
            colorspace=fitz.csRGB, alpha=False,
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
    except Exception:
        return None
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    # Pixel centres in page points, then in the box's own axes.
    xs = probe.x0 + (np.arange(arr.shape[1]) + 0.5) * (probe.width / arr.shape[1])
    ys = probe.y0 + (np.arange(arr.shape[0]) + 0.5) * (probe.height / arr.shape[0])
    gx, gy = np.meshgrid(xs - ob.cx, ys - ob.cy)
    t = math.radians(ob.angle)
    along = np.abs(gx * math.cos(t) + gy * math.sin(t)) - ob.width / 2.0
    across = np.abs(-gx * math.sin(t) + gy * math.cos(t)) - ob.height / 2.0
    return arr, np.maximum(along, across)


def estimate_background_quad(
    pdf_page: "fitz.Page",
    ob: OrientedBox,
    *,
    rings: tuple[tuple[float | None, float], ...] = QUAD_RINGS,
) -> tuple[tuple[float, float, float] | None, float]:
    """`estimate_background` for a turned box: the colour of the first band
    in `rings` -- (inner, outer) points outside the box -- that is plain,
    or (None, spread) when none is. An inner edge of None judges the whole
    inside too, for probing the empty space a label may widen into."""
    import numpy as np

    reach = max(outer for _, outer in rings)
    found = _turned_pixels(pdf_page, ob, reach, QUAD_ZOOM)
    if found is None:
        return None, 999.0
    arr, dist = found
    spread = 999.0
    for inner, outer in rings:
        ring = dist <= outer
        if inner is not None:
            ring &= dist > inner
        samples = arr[ring]
        if len(samples) < 12:
            continue
        spread = float(np.max(np.std(samples.astype(float), axis=0)))
        if spread <= BACKGROUND_UNIFORMITY_LIMIT:
            median = np.median(samples, axis=0)
            return (float(median[0]) / 255.0, float(median[1]) / 255.0,
                    float(median[2]) / 255.0), spread
    return None, spread


def estimate_ink_color_quad(
    pdf_page: "fitz.Page",
    ob: OrientedBox,
    background: tuple[float, float, float],
    *,
    min_ink_fraction: float = 0.015,
) -> tuple[float, float, float] | None:
    """`estimate_ink_color` inside a turned box only. The upright box of a
    tilted word holds whatever it passes over (the tablet's black frame,
    the red letter below), which is not the word's colour."""
    import numpy as np

    found = _turned_pixels(pdf_page, ob, 0.0, 150 / 72)
    if found is None:
        return None
    arr, dist = found
    inside = arr[dist <= 0].astype(np.int16)
    if len(inside) == 0:
        return None
    bg = np.array([c * 255.0 for c in background], dtype=np.int16)
    ink = inside[np.linalg.norm(inside - bg, axis=1) > 60]
    if len(ink) / len(inside) < min_ink_fraction:
        return None
    median = np.median(ink, axis=0)
    return (float(median[0]) / 255.0, float(median[1]) / 255.0,
            float(median[2]) / 255.0)


def uniform_growth_along(
    pdf_page: "fitz.Page", ob: OrientedBox, *, limit: float = 90.0, step: float = 8.0
) -> float:
    """`uniform_growth_right` along a turned line: how far past its end the
    line may run while the space there stays plain."""
    t = math.radians(ob.angle)
    grown = 0.0
    while grown + step <= limit:
        reach = ob.width / 2.0 + grown + step / 2.0
        probe = ob.model_copy(update={
            "cx": ob.cx + reach * math.cos(t),
            "cy": ob.cy + reach * math.sin(t),
            "width": step,
        })
        box = probe.aabb()
        if (box.x0 < pdf_page.rect.x0 or box.y0 < pdf_page.rect.y0
                or box.x1 > pdf_page.rect.x1 or box.y1 > pdf_page.rect.y1):
            break
        color, _ = estimate_background_quad(pdf_page, probe, rings=((None, 0.0),))
        if color is None:
            break
        grown += step
    return grown


def turned_frame(box: Rect, angle: float, height_hint: float) -> tuple[float, float]:
    """(width, height) of the turned frame whose upright extent is `box`.

    The editor drags and resizes the upright box around a turned block, and
    the rebuild writes that box back as drawn_bbox, so an unchanged box must
    give back the same frame: W = w*c + h*s and H = w*s + h*c, solved for
    w and h (well conditioned while cos 2a is not small, |a| <= ~35).
    A box resized so that solve makes the letters implausibly short --
    widened only, typically -- keeps the measured height and takes the
    width from the box's width.
    """
    c = abs(math.cos(math.radians(angle)))
    s = abs(math.sin(math.radians(angle)))
    width, height = box.width, box.height
    det = c * c - s * s
    # Solvable away from 45 degrees on either side: at 90 (text running down
    # the page) det is -1 and the solve simply swaps the box's sides.
    if abs(det) > 0.3:
        w = (width * c - height * s) / det
        h = (height * c - width * s) / det
        if w > 0 and h >= 0.6 * height_hint:
            return w, h
    if c >= s:
        h = min(height_hint, height / max(c, 1e-6))
        return max(1.0, (width - h * s) / max(c, 1e-6)), h
    # Nearer upright than level: the letters' height lies across the box.
    h = min(height_hint, width / max(s, 1e-6))
    return max(1.0, (height - h * c) / max(s, 1e-6)), h


def _turn_point(x: float, y: float, pivot: Point2, angle: float) -> Point2:
    """(x, y) turned `angle` degrees clockwise (on the page) about `pivot`."""
    t = math.radians(angle)
    dx, dy = x - pivot[0], y - pivot[1]
    return (
        round(pivot[0] + dx * math.cos(t) - dy * math.sin(t), 3),
        round(pivot[1] + dx * math.sin(t) + dy * math.cos(t), 3),
    )


def _fitz_quad(perimeter: list[Point2]) -> "fitz.Quad":
    """A fitz.Quad from corners in perimeter order (ul, ur, lr, ll)."""
    ul, ur, lr, ll = perimeter
    return fitz.Quad(fitz.Point(*ul), fitz.Point(*ur), fitz.Point(*ll), fitz.Point(*lr))


def turn_morph(layout: DrawnLayout, dx: float = 0.0, dy: float = 0.0):
    """The TextWriter morph that draws a turned layout, offset by (dx, dy).

    MuPDF's Matrix(a) turns counter-clockwise on the page, so a clockwise
    angle is Matrix(-a); text drawn so reads back with direction
    (cos a, sin a), the parser's own clockwise convention.
    """
    if not layout.angle or layout.pivot is None:
        return None
    return (
        fitz.Point(layout.pivot[0] + dx, layout.pivot[1] + dy),
        fitz.Matrix(-layout.angle),
    )


def turned_frame_of(layout: DrawnLayout) -> tuple[Rect, float, float]:
    """(frame, dx, dy) for a turned layout: the unturned frame, centred on
    the quad's own centre, and the shift that moves the layout's lines and
    rows into it.

    Turning that frame about its centre Q draws exactly the quad the
    recorded pivot P drew: with C = P + R(-a)(Q - P), the frame's centre
    before the shift, R(x + Q - C - Q) + Q = R(x - P) + P. So the editor
    and the sprite can always turn about the box's centre, as CSS does,
    even for a widened picture label whose pivot is not its centre.
    """
    if not layout.quad:
        raise ValueError("This layout is not turned.")
    (ax, ay), (bx, by), _, (lx, ly) = layout.quad
    qx = sum(x for x, _ in layout.quad) / 4.0
    qy = sum(y for _, y in layout.quad) / 4.0
    w, h = math.hypot(bx - ax, by - ay), math.hypot(lx - ax, ly - ay)
    frame = Rect(x0=qx - w / 2.0, y0=qy - h / 2.0, x1=qx + w / 2.0, y1=qy + h / 2.0)
    px, py = layout.pivot if layout.pivot is not None else (qx, qy)
    t = math.radians(layout.angle)
    ex, ey = qx - px, qy - py
    cx = px + ex * math.cos(t) + ey * math.sin(t)
    cy = py - ex * math.sin(t) + ey * math.cos(t)
    return frame, qx - cx, qy - cy


def frame_corners(rect: Rect, angle: float, pivot: Point2) -> list[Point2]:
    """`rect`'s corners turned `angle` degrees clockwise about `pivot`, in
    perimeter order (ul, ur, lr, ll)."""
    return [
        _turn_point(x, y, pivot, angle)
        for x, y in ((rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1))
    ]


def shape_frame(layout: DrawnLayout, rect: Rect) -> tuple[Rect, float]:
    """(unturned frame, angle) a shape's outline is drawn in: her frame,
    turned about its centre, when turned; else `rect`, upright. The sprite
    hands its own frame in `rect` with the turn taken off."""
    if layout.quad:
        return (layout.frame or turned_frame_of(layout)[0]), layout.angle
    return rect, 0.0


def shape_outline(
    layout: DrawnLayout, rect: Rect, dx: float = 0.0, dy: float = 0.0,
    cmds: list[tuple] | None = None,
) -> list[tuple]:
    """The shape's outline in page points: shapes.outline in its frame,
    every point turned (Bezier control points too) about the frame's
    centre, then moved by (dx, dy). `cmds`, in the frame's own points,
    are placed instead of the outline (a line's stroke and heads)."""
    frame, angle = shape_frame(layout, rect)
    if cmds is None:
        cmds = shape_geo.outline(
            layout.shape, frame.width, frame.height, layout.shape_adjust
        )
    cmds = shape_geo.placed(cmds, frame.x0, frame.y0)
    pivot = frame.center
    out: list[tuple] = []
    for c in cmds:
        if c[0] == "Z":
            out.append(c)
            continue
        nums: list[float] = []
        for i in range(1, len(c), 2):
            x, y = (_turn_point(c[i], c[i + 1], pivot, angle) if angle else (c[i], c[i + 1]))
            nums += [x + dx, y + dy]
        out.append((c[0], *nums))
    return out


def _draw_outline(page: "fitz.Page", cmds: list[tuple], **finish) -> None:
    """One path of joined lines and Beziers, finished as given: closed when
    it ends in Z, open (a line, an open arrowhead) when not."""
    if not cmds:
        return
    shape = page.new_shape()
    cur = None
    for c in cmds:
        if c[0] == "M":
            cur = fitz.Point(c[1], c[2])
        elif c[0] == "L":
            p = fitz.Point(c[1], c[2])
            shape.draw_line(cur, p)
            cur = p
        elif c[0] == "C":
            p = fitz.Point(c[5], c[6])
            shape.draw_bezier(cur, fitz.Point(c[1], c[2]), fitz.Point(c[3], c[4]), p)
            cur = p
    shape.finish(closePath=cmds[-1][0] == "Z", **finish)
    shape.commit(overlay=True)


def paint_line(
    page: "fitz.Page", layout: DrawnLayout, rect: Rect, dx: float = 0.0, dy: float = 0.0
) -> None:
    """A line she drew (shapes.line_drawing): the stroke, pulled back for
    its ends, in her colour, weight and dash; then each end as its own
    solid shape, so the dashes start cleanly at the line's own start, as
    the editor's SVG draws them."""
    frame, _ = shape_frame(layout, rect)
    w = layout.border_width
    stroke, heads = shape_geo.line_drawing(
        layout.shape, frame.width, w, layout.shape_adjust, layout.line_start, layout.line_end
    )
    dashes, cap = border_dashes(w, layout.border_dash)
    _draw_outline(
        page, shape_outline(layout, rect, dx, dy, stroke), color=layout.border, fill=None,
        width=w, dashes=dashes, lineCap=cap, lineJoin=0,
    )
    for cmds, filled in heads:
        placed = shape_outline(layout, rect, dx, dy, cmds)
        if filled:
            _draw_outline(page, placed, color=None, fill=layout.border, width=0)
        else:
            _draw_outline(
                page, placed, color=layout.border, fill=None, width=w, lineCap=0, lineJoin=0
            )


def paint_fill(
    page: "fitz.Page", layout: DrawnLayout, rect: Rect, dx: float = 0.0, dy: float = 0.0
) -> None:
    """Her background: the turned frame (quad) when turned, else `rect`. A
    shape's background fills its outline instead."""
    if layout.fill is None or shape_geo.is_line(layout.shape):
        return
    if layout.shape is not None:
        _draw_outline(
            page, shape_outline(layout, rect, dx, dy), color=None, fill=layout.fill, width=0
        )
        return
    if layout.quad:
        page.draw_quad(
            _fitz_quad([(x + dx, y + dy) for x, y in layout.quad]),
            color=None, fill=layout.fill, width=0, overlay=True,
        )
        return
    page.draw_rect(
        fitz.Rect(rect.x0 + dx, rect.y0 + dy, rect.x1 + dx, rect.y1 + dy),
        color=None, fill=layout.fill, width=0, overlay=True,
    )


def border_dashes(width: float, dash: str | None) -> tuple[str | None, int]:
    """(PDF dash pattern, line cap) for a border: dashes three weights long
    with gaps of two; dots as round caps on zero-length dashes, two weights
    apart. The editor's SVG draws the same pattern."""
    if dash == "dashed":
        return f"[{3 * width:g} {2 * width:g}] 0", 0
    if dash == "dotted":
        return f"[0 {2 * width:g}] 0", 1
    return None, 0


def paint_border(
    page: "fitz.Page", layout: DrawnLayout, rect: Rect, dx: float = 0.0, dy: float = 0.0
) -> None:
    """Her border around the frame (quad when turned, else `rect`).

    A polyline from the top-left corner going clockwise, not draw_quad:
    Shape.draw_quad walks ul, ll, lr, ur, so its dashes would start going
    down, while an SVG <rect> in the editor starts along the top. The
    stroke is centred on the edge, half outside, with mitred corners, as
    the editor's is.
    """
    if layout.border is None or layout.border_width <= 0:
        return
    if shape_geo.is_line(layout.shape):
        paint_line(page, layout, rect, dx, dy)
        return
    if layout.shape is not None:
        # The shape's outline, mitred (PDF's miter limit, 10, which the
        # editor's SVG sets too), dashed from where its path starts.
        dashes, cap = border_dashes(layout.border_width, layout.border_dash)
        _draw_outline(
            page, shape_outline(layout, rect, dx, dy), color=layout.border, fill=None,
            width=layout.border_width, dashes=dashes, lineCap=cap, lineJoin=0,
        )
        return
    corners = layout.quad or [
        (rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)
    ]
    dashes, cap = border_dashes(layout.border_width, layout.border_dash)
    page.draw_polyline(
        [fitz.Point(x + dx, y + dy) for x, y in corners],
        color=layout.border, fill=None, width=layout.border_width,
        dashes=dashes, lineCap=cap, lineJoin=0, closePath=True, overlay=True,
    )


def draw_decorations(
    page: "fitz.Page", layout: DrawnLayout, dx: float = 0.0, dy: float = 0.0
) -> None:
    """Paint a layout's highlight and underline, row by row, offset by
    (dx, dy). The sprite calls this too, so both draw the same shapes.

    The highlight covers each row's inked run from its baseline's ascent
    to its descent; the underline is a thin band just below the baseline.
    Both are filled quads rather than lines: the thickness is exact, a
    turned label turns them with the same pivot and angle as its words,
    and they are the shapes the editor draws in CSS. Both stay inside the
    drawn frame, so the sprite's padding and the Office pictures' copied
    areas never clip them -- except that below about 4pt the underline's
    0.5pt minimum can reach a few hundredths of a point past a middle or
    bottom frame's foot, still well inside that padding and the 2.5pt the
    Office pictures copy around a label. The frame is not grown for it: a
    frame she drags comes back as her box, which would then creep taller
    on every move. A layout drawn before rows existed has none, so
    nothing is painted for it.
    """
    if not layout.rows or not (layout.underline or layout.highlight):
        return
    size = layout.size
    turned = bool(layout.angle) and layout.pivot is not None

    def band(x0: float, y0: float, x1: float, y1: float, fill) -> None:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if turned:
            corners = [_turn_point(x, y, layout.pivot, layout.angle) for x, y in corners]
        page.draw_quad(
            _fitz_quad([(x + dx, y + dy) for x, y in corners]),
            color=None, fill=fill, width=0, overlay=True,
        )

    inked = [r for r in layout.rows if r.x1 > r.x0]
    # Every highlight first, so no row's band covers another row's rule.
    if layout.highlight is not None:
        for r in inked:
            band(
                r.x0, r.baseline - ASCENT_RATIO * size,
                r.x1, r.baseline + DESCENT_RATIO * size, layout.highlight,
            )
    if layout.underline:
        weight = max(MIN_UNDERLINE_PT, UNDERLINE_WEIGHT * size)
        for r in inked:
            top = r.baseline + UNDERLINE_GAP * size
            band(r.x0, top, r.x1, top + weight, layout.color)


# Measured once on a rebuild, for pictures read before slants were.
MAX_LATE_MEASURES = 24


def measure_unmeasured(pdf_page: "fitz.Page", page: Page) -> list[QAFinding]:
    """Measure the slant of picture words read before slants were measured.

    A document translated earlier has its labels but no `oriented` boxes,
    and should not need translating again to get them. But she may already
    have checked those pages, so nothing changes on a guess:

      * Done on the source page, before anything is drawn, without a
        re-read, so only a slant that is `sure_alone` is taken -- many
        letters agreeing, at least 5 degrees, consistent with their shape.
        On a real 182-label screenshot deck that took none; HARBOR passed.
      * A block she placed, filled or sized by hand keeps exactly what she
        did: the slant is recorded with the translation set Straight, and a
        finding tells her how to have it follow the slant.
      * A block drawn automatically is turned, and a finding says so.
      * Curved and steep lines are left as they were drawn: changing words
        she reviewed to "kept in English" is hers to decide on a fresh run.
    """
    from ..analysis.text_angle import measure_on_page

    done = 0
    turned: list[str] = []
    kept: list[str] = []
    for t in page.text_objects:
        if t.source is not TextSource.IMAGE_OCR or t.orientation_checked:
            continue
        if done >= MAX_LATE_MEASURES:
            break
        done += 1
        t.orientation_checked = True
        if t.oriented is not None:
            continue
        try:
            found = measure_on_page(pdf_page, t.bbox)
        except Exception as exc:
            log.warning("Could not measure a label's slant: %r", exc)
            continue
        if found is None or not found.sure_alone:
            continue
        ob = found.page_box()
        t.oriented = ob
        t.rotation = ob.angle
        for line in t.lines:
            line.rotation = ob.angle
        if t.manual_angle is not None:
            # She turned it herself: her angle wins, and there is nothing
            # to tell her. The slant is recorded for Follow the tilt.
            continue
        if (
            t.manual_bbox is not None or t.manual_fill or t.manual_font_size
            or t.manual_rotation is not None
        ):
            # Her box was drawn around the old upright guess; reading it as
            # a turned frame drew a big white tilted shape over the artwork
            # (real slide). It stays exactly as she left it.
            if t.manual_rotation is None:
                t.manual_rotation = 0.0
            kept.append(t.id)
            continue
        size = round(max(4.0, ob.height * 0.92), 1)
        t.style = t.style.model_copy(update={"font_size": size})
        for line in t.lines:
            for span in line.spans:
                span.style = span.style.model_copy(update={"font_size": size})
        t.cover_boxes = []
        turned.append(t.id)
    findings: list[QAFinding] = []
    if turned:
        findings.append(QAFinding(
            severity="warning",
            code="image_text_turned",
            message=(
                f"Words printed at a slant in a picture on page {page.number} "
                "are now translated along the slant. Check them. To draw one "
                "level again, select it and choose Straight."
            ),
            object_ids=turned,
        ))
    if kept:
        findings.append(QAFinding(
            severity="warning",
            code="image_text_tilt_kept_straight",
            message=(
                f"Words in a picture on page {page.number} are printed at a "
                "slant, but you placed their translation by hand, so it was "
                "left straight where you put it. To draw it along the slant "
                "instead, select it and choose Follow the tilt, or use More, "
                "then Reset this box."
            ),
            object_ids=kept,
        ))
    return findings


def _native_turn(t) -> OrientedBox | None:
    """The turned frame of printed (native) text set at a right angle --
    running down or up the page, or upside down -- read off its upright
    box, which for those angles is exact. Other angles cannot be recovered
    from an upright box and are drawn level, with a finding."""
    angle = round(float(t.rotation or 0.0), 2)
    if angle not in (90.0, -90.0, 180.0):
        return None
    if any(round(float(line.rotation or 0.0), 2) != angle for line in t.lines):
        return None
    b = t.bbox
    w, h = (b.height, b.width) if abs(angle) == 90.0 else (b.width, b.height)
    if w < 1 or h < 1:
        return None
    cx, cy = b.center
    return OrientedBox(cx=cx, cy=cy, width=w, height=h, angle=angle)


def estimate_ink_color(
    pdf_page: "fitz.Page",
    rect: Rect,
    background: tuple[float, float, float],
    *,
    min_ink_fraction: float = 0.015,
) -> tuple[float, float, float] | None:
    """Sample the colour of the ORIGINAL glyphs inside `rect`.

    Text recognized from a raster carries no colour: OCR reports characters,
    not styling. Defaulting to black destroys any document whose text is not
    black -- a flyer with white and yellow text on a navy band came back as
    black text on light patches, which is unreadable and ruins the design.

    So the glyph colour is measured: pixels far from the sampled background
    are the ink, and their median is the colour to redraw in.

    MUST be called before the cover patch is painted, or the ink is gone.
    Returns None when no clear foreground exists.
    """
    import numpy as np

    probe = fitz.Rect(*rect.as_tuple()) & pdf_page.rect
    if probe.is_empty or probe.width < 2 or probe.height < 2:
        return None
    try:
        pix = pdf_page.get_pixmap(clip=probe, dpi=150, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )[:, :, :3].astype(np.int16)
    except Exception:
        return None
    if arr.size == 0:
        return None

    bg = np.array([c * 255.0 for c in background], dtype=np.int16)
    distance = np.linalg.norm(arr - bg, axis=2)
    # 60 in RGB space separates glyphs from paper and from antialiasing
    # fringes, without picking up JPEG noise on a scan.
    ink_mask = distance > 60
    fraction = float(ink_mask.mean())
    if fraction < min_ink_fraction:
        return None

    ink = arr[ink_mask]
    median = np.median(ink, axis=0)
    return (
        float(median[0]) / 255.0,
        float(median[1]) / 255.0,
        float(median[2]) / 255.0,
    )


def _relative_luminance(color: tuple[float, float, float]) -> float:
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ensure_contrast(
    ink: tuple[float, float, float],
    background: tuple[float, float, float],
    *,
    minimum: float = 0.28,
) -> tuple[float, float, float]:
    """Guarantee the drawn text is visible against the patch behind it.

    Sampling can land on a colour too close to the background -- an
    antialiasing fringe, or a scan where glyph and paper nearly meet. Rather
    than draw invisible text, fall back to black or white, whichever
    contrasts more.
    """
    if abs(_relative_luminance(ink) - _relative_luminance(background)) >= minimum:
        return ink
    return (0.0, 0.0, 0.0) if _relative_luminance(background) > 0.5 else (1.0, 1.0, 1.0)


def _without_symbols(text: str, missing: str) -> str | None:
    """`text` without the characters no font has, when every one of them is
    an emoji or symbol; None when real letters are missing too.

    A combining mark (a Devanagari or Thai vowel sign) is part of its word,
    not a symbol: losing it changes the word, so it goes down the ordinary
    no-font path instead of being dropped.
    """
    gone = set(missing)
    if not gone or not all(stands_alone(c) for c in gone):
        return None
    kept = "".join(c for c in text if c not in gone and c not in "\ufe0f\u200d")
    return kept.strip() or None


def _source_look(obj) -> str:
    """The kind of face the source text was set in, for its redrawn copy."""
    if obj.style.monospace:
        return "mono"
    return "serif" if obj.style.serif else "sans"


def _alignment_offset(
    alignment: TextAlignment, line_width: float, box_width: float
) -> float:
    if alignment is TextAlignment.CENTER:
        return max(0.0, (box_width - line_width) / 2.0)
    if alignment is TextAlignment.RIGHT:
        return max(0.0, box_width - line_width)
    return 0.0


def _color_tuple(hex_color: str | None) -> tuple[float, float, float]:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return (0.0, 0.0, 0.0)
    try:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
    except ValueError:
        return (0.0, 0.0, 0.0)
    return (r, g, b)


def _border_style(obj) -> dict:
    """DrawnLayout's border fields for a block: none without a colour."""
    if not obj.manual_border_color:
        return {}
    return {
        "border": _color_tuple(obj.manual_border_color),
        "border_width": obj.border_weight,
        "border_dash": obj.manual_border_dash,
    }


def _shape_style(obj, frame: Rect) -> dict:
    """DrawnLayout's shape fields for a block: none for a plain box."""
    if not obj.manual_shape:
        return {}
    return {
        "shape": obj.manual_shape,
        "frame": Rect(
            x0=round(frame.x0, 3), y0=round(frame.y0, 3),
            x1=round(frame.x1, 3), y1=round(frame.y1, 3),
        ),
        "shape_adjust": obj.manual_shape_adjust,
        "line_start": obj.manual_line_start,
        "line_end": obj.manual_line_end,
    }


def _shape_text_box(obj, frame: Rect) -> Rect:
    """Where a shape's words go in its frame (shapes.text_area), before
    her inside space is taken off."""
    l, t, r, b = shape_geo.text_area(obj.manual_shape, frame.width, frame.height)
    return Rect(x0=frame.x0 + l, y0=frame.y0 + t, x1=frame.x0 + r, y1=frame.y0 + b)


def _obstacles(page: Page, exclude_id: str) -> list[Rect]:
    """Everything a growing text box must not cover."""
    boxes = [t.bbox for t in page.text_objects if t.id != exclude_id]
    boxes += [i.bbox for i in page.image_objects]
    # Response areas are inviolable; treating them as obstacles is what stops
    # translated text eating a student's writing space.
    boxes += [r.bbox for r in page.response_areas]
    # A table's vertical rules are cell walls: text widened past one ends up
    # in the next cell. Horizontal rules are left out -- writing lines are
    # already response areas, and the rest must not stop downward growth.
    boxes += [
        v.bbox for v in page.vector_objects
        if v.bbox.width <= 3.0 and v.bbox.height >= 10.0
    ]
    return boxes


def rebuild_page(
    pdf_page: "fitz.Page",
    page: Page,
    *,
    target_lang: str,
    on_plate: "Callable[[fitz.Page], None] | None" = None,
) -> PageBuildStats:
    """Replace translated text on one already-open PDF page, in place."""
    stats = PageBuildStats()
    language = get_language(target_lang)
    rtl = bool(language and language.rtl)
    script = language.script if language else "Latin"

    if rtl and not shaping_available(script):
        stats.findings.append(
            QAFinding(
                severity="warning",
                code="shaping_unavailable",
                message=(
                    "This server cannot shape right-to-left text correctly. "
                    "Letters may appear disconnected."
                ),
                details={"script": script},
            )
        )

    # A suppressed block still has its original removed; it is simply not
    # redrawn. It therefore belongs in the removal lists but not the draw
    # list, which is handled by filtering at the draw step.
    for t in page.text_objects:
        t.drawn_bbox = None
        t.drawn_layout = None
    # On the untouched source page, before anything is drawn on it.
    stats.findings.extend(measure_unmeasured(pdf_page, page))
    targets = [
        t
        for t in page.text_objects
        if t.translation_status
        in (TranslationStatus.TRANSLATED, TranslationStatus.MANUAL_EDIT)
        and t.translated_text and not t.keep_original
    ]
    # Covers: boxes she added with a background or a border and no words.
    # They have no text to fit; they are painted where she put them, over
    # everything, turned about their centre when she turned them.
    # A shape with no words is drawn the same way, whatever its colours.
    covers = [
        t for t in page.text_objects
        if t.is_manual_block and (t.manual_fill or t.manual_border_color or t.manual_shape)
        and not t.suppressed and not (t.translated_text or "").strip()
    ]
    for t in covers:
        t.drawn_bbox = t.render_bbox
        t.drawn_layout = DrawnLayout(
            size=1.0, fill=_color_tuple(t.manual_fill) if t.manual_fill else None,
            **_border_style(t), **_shape_style(t, t.render_bbox),
        )
        if t.manual_angle is not None:
            frame = t.render_bbox
            pivot = (round(frame.center[0], 3), round(frame.center[1], 3))
            corners = frame_corners(frame, t.manual_angle, pivot)
            t.drawn_layout.angle = round(t.manual_angle, 2)
            t.drawn_layout.pivot = pivot
            t.drawn_layout.quad = corners
            xs, ys = zip(*corners)
            t.drawn_bbox = Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    if (targets or covers) and pdf_page.rotation:
        # Drawing, covering and redacting take coordinates in the page's
        # UNROTATED space, but every box in the model is in the visual space
        # (see ingest/pdf_parser.py). On a /Rotate page that put each patch
        # and each translation somewhere else -- the English stayed and the
        # words were drawn sideways in two-letter columns (corpus 13). The
        # page is turned into its visual space once, which keeps how it
        # looks exactly; the output page then has /Rotate 0.
        pdf_page.remove_rotation()

    if not targets:
        # Nothing to change: the page is carried over byte-for-byte, which is
        # the highest-fidelity outcome available.
        if on_plate:
            on_plate(pdf_page)
        for t in sorted(covers, key=lambda c: c.z_order):
            paint_fill(pdf_page, t.drawn_layout, t.drawn_bbox)
            paint_border(pdf_page, t.drawn_layout, t.drawn_bbox)
        return stats

    # --- 1. Resolve fonts before mutating anything -----------------------
    # A page whose font cannot be resolved must not end up with its source
    # text redacted and nothing drawn in its place.
    resolved: dict[str, ResolvedFont] = {}
    shaped_text: dict[str, str] = {}
    for obj in list(targets):
        # Shape first: the glyphs that must exist in the font are the
        # contextual presentation forms, not the raw codepoints.
        shaped = shape_logical(obj.translated_text or "", script)
        shaped_text[obj.id] = shaped
        try:
            resolved[obj.id] = REGISTRY.resolve(
                shaped, script=script, bold=obj.render_bold, italic=obj.render_italic,
                family=obj.manual_font_family, look=_source_look(obj),
            )
            if not resolved[obj.id].weight_matched:
                stats.findings.append(
                    QAFinding(
                        # Recorded, not queued for review: the teacher cannot
                        # supply a font, so this is not something to fix.
                        severity="info",
                        code="font_weight_unavailable",
                        message=(
                            "Bold or italic text on this page was drawn in the "
                            "regular typeface, because no matching font was "
                            "available."
                        ),
                        object_ids=[obj.id],
                    )
                )
        except FontResolutionError as exc:
            # Only an emoji or a symbol has no font ("📘 Installation
            # instructions" in a Word file): the words are drawn without
            # it, rather than the whole heading left in English. Each retry
            # can name another symbol, so a few rounds are allowed.
            dropped = ""
            drawable: str | None = shaped
            while drawable is not None and len(dropped) < 40:
                drawable = _without_symbols(drawable, exc.missing)
                if drawable is None:
                    break
                dropped += exc.missing
                try:
                    resolved[obj.id] = REGISTRY.resolve(
                        drawable, script=script, bold=obj.render_bold,
                        italic=obj.render_italic, family=obj.manual_font_family,
                        look=_source_look(obj),
                    )
                except FontResolutionError as again:
                    exc = again
                    continue
                break
            if drawable is not None and obj.id in resolved:
                shaped_text[obj.id] = drawable
                stats.findings.append(QAFinding(
                    severity="info", code="symbol_not_drawn",
                    message="An emoji or symbol could not be drawn on this page and was left out.",
                    object_ids=[obj.id],
                    details={"missing_characters": dropped[:40]},
                ))
                continue
            stats.font_failures += 1
            stats.findings.append(
                QAFinding(
                    severity="error",
                    code="font_unavailable",
                    message=(
                        "Some translated text could not be shown because this "
                        "server has no font for those characters. The original "
                        "text was kept."
                    ),
                    object_ids=[obj.id],
                    bbox=obj.bbox,
                    details={"missing_characters": exc.missing[:40]},
                )
            )
            targets.remove(obj)

    if not targets:
        # Every translation lost its font: the page is carried over as it
        # is, but her covers and shapes are still drawn, and the editor's
        # clean background is still stored.
        if on_plate:
            on_plate(pdf_page)
        for t in sorted(covers, key=lambda c: c.z_order):
            paint_fill(pdf_page, t.drawn_layout, t.drawn_bbox)
            paint_border(pdf_page, t.drawn_layout, t.drawn_bbox)
        return stats

    # --- 2. Remove the source text ---------------------------------------
    # Native text is redacted. Text that came from OCR is part of the page
    # image and has to be painted over instead.
    # Both OCR sources live in pixels: page scans and text inside images.
    # Routing IMAGE_OCR to the redaction branch is a no-op on a raster, so
    # the original label stayed visible with the translation drawn on top.
    RASTER_SOURCES = (TextSource.OCR, TextSource.IMAGE_OCR)
    # A person-created block has no source text underneath, so removing
    # anything for it would erase the artwork it was placed on.
    native = [
        t
        for t in targets
        if t.source not in RASTER_SOURCES and not t.is_manual_block
    ]
    raster = [t for t in targets if t.source in RASTER_SOURCES]
    manual = [t for t in targets if t.is_manual_block]

    # Printed text set at a right angle is redrawn turned the same way; at
    # any other angle it can't be placed reliably and is drawn level, which
    # she is told about rather than left to find.
    drawn_level: list[str] = []
    for obj in native:
        if obj.oriented is None:
            obj.oriented = _native_turn(obj)
        if obj.oriented is None and abs(float(obj.rotation or 0.0)) >= 1.0:
            drawn_level.append(obj.id)
    if drawn_level:
        stats.findings.append(QAFinding(
            severity="warning",
            code="turned_text_drawn_level",
            message=(
                f"Some words on page {page.number} are printed at a slant. "
                "Their translation is drawn level, because text at that angle "
                "can't be redrawn reliably. Check that it reads well there."
            ),
            object_ids=drawn_level,
        ))

    for obj in native:
        rect = fitz.Rect(*obj.bbox.as_tuple()) & pdf_page.rect
        if rect.is_empty:
            continue
        # fill=None paints nothing, so artwork beneath the text is preserved.
        pdf_page.add_redact_annot(rect, fill=None)
        stats.redacted += 1

    if stats.redacted:
        pdf_page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )

    # Native glyphs were just redacted away above, and a manual block has no
    # source text to remove in the first place, so this is the last moment
    # before the translation goes down that the true page colour -- not ink,
    # not the new text -- is visible for either kind of block.
    for obj in native + manual:
        color, _ = estimate_background(pdf_page, obj.bbox)
        obj.background_color = _hex(color) if color is not None else None

    # Colour sampled from the original glyphs, per object. Populated in the
    # covering loop below, which is the last moment the ink still exists.
    raster_ink: dict[str, tuple[float, float, float]] = {}
    # Every word recognised on the page, translated or not: all of it is ink
    # the paper check may set aside, and none of it is artwork.
    recognised = [
        box
        for t in page.text_objects
        if t.source in RASTER_SOURCES
        for box in [
            *(s.bbox for line in t.lines for s in line.spans if s.text.strip()),
            *t.cover_boxes,
        ]
    ]

    for obj in list(raster):
        if (
            obj.source is TextSource.IMAGE_OCR
            and not obj.manual_fill
            and obj.manual_bbox is None
            and (obj.translated_text or "").strip() == obj.flat_text.strip()
        ):
            # The translator gave back what was read ("COMPIAR", a misread
            # CONFIRM): redrawing it would only replace the picture's own
            # lettering with a worse copy of the same word.
            targets.remove(obj)
            continue
        # Tilted words are judged, covered and colour-sampled by their
        # turned box: their upright box holds the artwork they pass over.
        ob = obj.oriented
        if ob is not None:
            color, spread = estimate_background_quad(pdf_page, ob)
        else:
            color, spread = estimate_background(pdf_page, obj.bbox)
            if color is None:
                color = estimate_scanned_background(pdf_page, obj.bbox, recognised)
        if color is None and obj.retry_background:
            color = estimate_paper_background(pdf_page, obj)
        if color is None and obj.manual_fill:
            # She gave the box a background: that answers what the paper
            # check could not. The original words are covered in her colour
            # and the translation sits on it -- real packet, "Solve" and
            # "Check" underlined, which kept the English on the page.
            color = _color_tuple(obj.manual_fill)
        # Recorded so the editor can patch this block with the right colour
        # while a person is typing into it.
        obj.background_color = _hex(color) if color is not None else None
        if color is None:
            # Artwork underneath: leave the original pixels alone and say so.
            stats.uncovered += 1
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="scanned_text_not_replaced",
                    message=(
                        "We could not replace the printed text cleanly. "
                        "This section still shows the original text."
                    ),
                    object_ids=[obj.id],
                    bbox=obj.bbox,
                    details={"background_spread": round(spread, 1)},
                )
            )
            targets.remove(obj)
            continue
        if ob is not None:
            sampled = estimate_ink_color_quad(pdf_page, ob, color)
            if sampled is not None:
                raster_ink[obj.id] = ensure_contrast(sampled, color)
            # One turned patch over the line's letters. Word boxes and
            # unreadable scraps are upright and would bite into the artwork.
            pdf_page.draw_quad(
                _fitz_quad(ob.expanded(0.6).corners()), color=None, fill=color, width=0
            )
            stats.covered += 1
            continue
        # Paint per word, not per line: the gaps between words on a scan are
        # blank paper, and on a header row they are the student's answer
        # blanks, which a line-wide patch would erase.
        boxes = [
            span.bbox for line in obj.lines for span in line.spans if span.text.strip()
        ]
        # Unreadable glyphs the OCR engine located but could not transcribe.
        boxes.extend(obj.cover_boxes)
        boxes = boxes or [obj.bbox]
        # Measure the glyph colour while it is still on the page.
        sampled = estimate_ink_color(pdf_page, obj.bbox, color)
        if sampled is not None:
            raster_ink[obj.id] = ensure_contrast(sampled, color)

        painted = 0
        for box in boxes:
            rect = fitz.Rect(*box.expanded(0.6).as_tuple()) & pdf_page.rect
            if rect.is_empty:
                continue
            pdf_page.draw_rect(rect, color=None, fill=color, width=0)
            painted += 1
        if not painted:
            targets.remove(obj)
            continue
        stats.covered += 1

    # --- 3. Lay out, reflow if needed, then draw -------------------------
    # One TextWriter PER COLOUR. write_text() applies a single colour to
    # everything a writer holds, so a shared writer silently painted the whole
    # page in whichever colour happened to be last.
    writers: dict[tuple[float, float, float], "fitz.TextWriter"] = {}
    # Which blocks went into each writer, so a writer that fails to draw
    # reports exactly the blocks that are now missing from the page.
    writer_blocks: dict[tuple[float, float, float], list[str]] = {}
    filled: list[tuple[TextObject, "fitz.TextWriter", tuple[float, float, float]]] = []
    # The lowest box with a background. Text stacked above it (brought to
    # the front, or added after it) is drawn in the same layered pass, in
    # order; otherwise every plain block went down first and a white box
    # covered words she had put on top of it.
    layer_floor = min(
        (t.z_order for t in [*covers, *(o for o in targets if o.manual_fill)]),
        default=None,
    )

    def writer_for(rgb: tuple[float, float, float]) -> "fitz.TextWriter":
        key = (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3))
        if key not in writers:
            writers[key] = fitz.TextWriter(pdf_page.rect)
        return writers[key]

    # Where the document's own text stops on the right. Widening a block
    # stops here: past it is margin, not empty space.
    text_margin = max(
        (t.bbox.x1 for t in page.text_objects if not t.is_manual_block),
        default=None,
    )

    # Turned blocks: id -> (angle clockwise, pivot), set by lay_out.
    turns: dict[str, tuple[float, Point2]] = {}
    # The inside space each block was laid out with: id -> (px, py), the
    # points between the frame and the box the words fill (fitting.inset).
    insets: dict[str, tuple[float, float]] = {}
    # Plain (not layered) blocks with an underline or a highlight. Theirs
    # are painted after the plate is taken, so the plate stays clean and
    # they move with the block's picture, and before any words go down.
    decorated: list = []
    # Each block's paragraph plan and marker fonts (reconstruct/lists.py),
    # set by lay_out for the drawing below.
    plans: dict[str, tuple] = {}
    # A shape's frame (unturned), set by lay_out: its outline is drawn in
    # it, and it is the box the editor shows, never grown to the ink.
    frames: dict[str, Rect] = {}

    def lay_out(obj, *, extra_room: float = 0.0, translation_only: bool = False):
        """Fit one block. Returns (text, rfont, font, box, fit) or None.

        `extra_room` is vertical space reflow has granted. `translation_only`
        drops the bilingual pairing when the page could not be made to fit.
        """
        raw = obj.translated_text or ""
        if translation_only and obj.translation_only_text:
            raw = obj.translation_only_text
            shaped = shape_logical(raw, script)
        else:
            shaped = shaped_text.get(obj.id) or raw
        text = shaped.strip()
        if not text:
            return None
        try:
            rfont = REGISTRY.resolve(
                text, script=script, bold=obj.render_bold, italic=obj.render_italic,
                family=obj.manual_font_family, look=_source_look(obj),
            )
        except FontResolutionError:
            return None
        font = REGISTRY.font_for(rfont)
        if font is None:
            return None
        # Lists, indents and markers typed in the text ("3." hung apart
        # from its words, so a wrapped line sits under the words). Every
        # fit below wraps each paragraph in its own column.
        paras = plan_paragraphs(text, obj, font)
        mfonts = MarkerFonts(rfont, bold=obj.render_bold, italic=obj.render_italic)
        plans[obj.id] = (paras, mfonts)
        lists = {
            "paras": paras, "mfonts": mfonts,
            "visual": (lambda m: marker_visual(m, True)) if rtl else None,
        }

        base_size = obj.style.font_size or 10.0
        obstacles = _obstacles(page, obj.id)
        growth = available_growth(obj.bbox, page.bounds, obstacles)

        # A hand-placed box replaces the automatic one, and a hand-set size is
        # honoured rather than fitted: overriding a person's explicit choice
        # would make the editor feel broken.
        placed = obj.manual_bbox is not None or obj.is_manual_block
        obj.widen_x1 = None
        source_box = obj.render_bbox

        # Turned words are fitted in their own unturned frame, centred where
        # they are, and turned about that centre when drawn. The fitter
        # never sees the turn.
        angle = obj.render_angle
        frame: OrientedBox | None = None
        if angle and obj.manual_angle is not None:
            # Her own angle: manual_bbox is the unturned frame itself.
            fr = obj.render_bbox
            cx, cy = fr.center
            frame = OrientedBox(cx=cx, cy=cy, width=fr.width, height=fr.height, angle=angle)
        elif angle:
            # Tilted words (render_angle is only non-zero with `oriented`
            # here): a box she dragged is the turned frame's upright extent.
            ob = obj.oriented
            if obj.manual_bbox is not None:
                w, h = turned_frame(obj.manual_bbox, angle, ob.height)
                cx, cy = obj.manual_bbox.center
                frame = OrientedBox(cx=cx, cy=cy, width=w, height=h, angle=angle)
            else:
                frame = ob.model_copy(update={"angle": angle})
        if frame is not None:
            if obj.source is TextSource.IMAGE_OCR and obj.oriented is not None:
                # Letters as tall as the measured line, never the upright
                # guess's size a block read earlier may still carry -- also
                # once she turns it herself, or drags it (which takes the
                # slant as her angle), so its size stays where it was.
                base_size = min(
                    base_size, round(max(4.0, obj.oriented.height * 0.92), 1)
                )
            source_box = frame.frame()
            turns[obj.id] = (angle, (frame.cx, frame.cy))
            # Room beside or below a turned block runs along the page's
            # axes, not the frame's: it keeps its own size.
            growth = 0.0
        else:
            turns.pop(obj.id, None)

        # A label inside a picture has no neighbouring objects to bound it,
        # and growing DOWNWARD there covers artwork the obstacle test cannot
        # see. Only sideways growth is safe, gated on the pixels beside it.
        raster_widen = 0.0
        if obj.source is TextSource.IMAGE_OCR:
            growth = 0.0
            if frame is not None:
                # Along the line, keeping its start where it is: the frame
                # grows rightward before turning, about the same centre.
                raster_widen = 0.0 if placed else uniform_growth_along(pdf_page, frame)
            else:
                raster_widen = uniform_growth_right(pdf_page, obj.bbox)

        if obj.manual_font_size:
            base_size = obj.manual_font_size
        box = Rect(
            x0=source_box.x0,
            y0=source_box.y0,
            x1=source_box.x1,
            y1=source_box.y1 + (0.0 if placed or frame is not None else extra_room),
        )
        # The words lay out inside her inside space: Fit, wrapping, the
        # vertical share and justified rows all see the inner box. Widening
        # below grows the inner box, and the frame with it. A box she has
        # not placed grows outward by the space instead, as Slides grows a
        # shape round its text: the words keep the room they had, so the
        # type does not shrink to make space.
        if not placed and obj.manual_padding:
            p = obj.manual_padding
            box = Rect(x0=box.x0 - p, y0=box.y0 - p, x1=box.x1 + p, y1=box.y1 + p)
        if obj.manual_shape:
            # A shape's words go in its text area (an oval's inner box), in
            # its unturned frame however it is turned, and never grow it.
            frames[obj.id] = box
            box = _shape_text_box(obj, box)
        box, padx, pady = inset(box, obj.manual_padding)
        insets[obj.id] = (padx, pady)
        # Line spacing she chose. Fit and a locked size use it as given;
        # the automatic fitter keeps it and lets the size give way instead.
        ratio = obj.manual_line_spacing
        if obj.manual_fit:
            return text, rfont, font, box, fit_to_box(
                text, box, rfont, leading=ratio or DEFAULT_LEADING, **lists
            )
        if obj.manual_size_locked and obj.manual_font_size:
            rows = wrap_rows(
                paras, rfont, obj.manual_font_size, max(1.0, box.width), mfonts,
                text=text, visual=lists["visual"],
            )
            lines = [r.text for r in rows]
            leading = obj.manual_font_size * (ratio or DEFAULT_LEADING)
            used = leading * len(lines)
            over = used - box.height
            fit = FitResult(
                lines=lines,
                font_size=obj.manual_font_size,
                leading=round(leading, 2),
                used_height=round(used, 2),
                # A shape is a fixed box: rows past its text area overflow.
                fits=not (obj.manual_shape and over > leading / 2),
                # Her size is kept, so the box grows down to hold every
                # row, as a Slides text box does. Cut to the box instead,
                # the last words simply went missing, from the page and the
                # picture alike, with nothing to say so. Less than half a
                # row over is drawn as it always was (the frame takes in
                # the ink), so a label whose one row is a little taller
                # than its box keeps its box.
                extra_height=(
                    0.0 if obj.manual_shape else round(over, 2) if over > leading / 2 else 0.0
                ),
                shrink_ratio=1.0,
                notes=["size set by hand"],
                rows=rows,
            )
            return text, rfont, font, box, fit

        fit = fit_text(
            text,
            box,
            rfont,
            base_size=base_size,
            source_leading=obj.line_spacing,
            max_extra_height=0.0 if placed else growth,
            leading=ratio,
            **lists,
        )
        if placed:
            return text, rfont, font, box, fit

        upright_text = obj.source is not TextSource.IMAGE_OCR and frame is None
        room_right = (
            available_growth_right(obj.bbox, page.bounds, obstacles, right_margin=text_margin)
            if upright_text else 0.0
        )
        # Told to the editor (see TextObject.widen_x1), needed or not.
        obj.widen_x1 = round(box.x1 + room_right, 2) if room_right > 4.0 else None
        if not fit.fits or fit.shrink_ratio < 1.0 or len(fit.lines) > 1:
            extra_width = room_right if upright_text else raster_widen
            if extra_width > 4.0:
                widened = Rect(
                    x0=box.x0, y0=box.y0, x1=box.x1 + extra_width, y1=box.y1
                )
                wider = fit_text(
                    text,
                    widened,
                    rfont,
                    base_size=base_size,
                    source_leading=obj.line_spacing,
                    max_extra_height=growth,
                    leading=ratio,
                    **lists,
                )
                if wider.fits and (
                    wider.shrink_ratio >= fit.shrink_ratio
                    or len(wider.lines) < len(fit.lines)
                ):
                    box, fit = widened, wider
                    fit.notes.append("text box widened into empty space")
        return text, rfont, font, box, fit

    manual_ids = {
        o.id for o in targets if o.manual_bbox is not None or o.is_manual_block
    }

    # Pass 1: fit everything in place and see what still does not fit.
    # Suppressed blocks are skipped here: their original has already been
    # removed above, and nothing replaces them.
    drawable = [t for t in targets if not t.suppressed]
    stats.suppressed = sum(1 for t in targets if t.suppressed)

    first_pass: dict[str, tuple] = {}
    needed: dict[str, float] = {}
    for obj in drawable:
        result = lay_out(obj)
        if result is None:
            stats.font_failures += 1
            continue
        first_pass[obj.id] = result
        fit = result[4]
        if not fit.fits and fit.overflowed_by > 0.5:
            needed[obj.id] = fit.overflowed_by

    # Pass 2: reflow only if something overflowed. This is the exception
    # path -- a page that fits is never rearranged (spec sec. 25).
    plan = ReflowPlan()
    if needed:
        # A hand-placed block is never moved by the automatic pass.
        plan = plan_reflow(
            page,
            {k: v for k, v in needed.items() if k not in manual_ids},
            [o for o in drawable if o.id in first_pass and o.id not in manual_ids],
        )
        for note in plan.notes:
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="page_reflowed",
                    message=f"Page {page.number}: {note}",
                )
            )
        stats.reflowed = plan.moved

    for obj in drawable:
        if obj.id not in first_pass:
            continue
        shift = plan.shifts.get(obj.id, 0.0)
        extra = plan.extra_height.get(obj.id, 0.0)
        drop_pairing = obj.id in plan.translation_only

        if extra or drop_pairing:
            result = lay_out(obj, extra_room=extra, translation_only=drop_pairing)
            if result is None:
                result = first_pass[obj.id]
        else:
            result = first_pass[obj.id]
        text, rfont, font, layout_box, fit = result

        if drop_pairing:
            stats.pairing_dropped += 1
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="bilingual_pairing_dropped",
                    message=(
                        f"Page {page.number}: some text shows only the "
                        "translation, because both languages would not fit."
                    ),
                    object_ids=[obj.id],
                    bbox=obj.bbox,
                )
            )

        if fit.extra_height:
            stats.extended += 1
        if fit.shrink_ratio < 0.8:
            stats.shrunk += 1
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="font_reduced",
                    message=(
                        f"Translated text on page {page.number} was made smaller "
                        "to fit the original space."
                    ),
                    object_ids=[obj.id],
                    bbox=obj.bbox,
                    details={"shrink_ratio": fit.shrink_ratio},
                )
            )
        if not fit.fits:
            stats.overflowed += 1
            stats.findings.append(
                QAFinding(
                    severity="error",
                    code="text_overflow",
                    message=(
                        f"Translated text on page {page.number} is longer than the "
                        "space available and may be cut off."
                    ),
                    object_ids=[obj.id],
                    bbox=obj.bbox,
                    details={"overflow_points": fit.overflowed_by},
                )
            )

        color = (
            _color_tuple(obj.manual_color)
            if obj.manual_color
            else (raster_ink.get(obj.id) or _color_tuple(obj.style.color))
        )
        if (obj.manual_fill or obj.manual_highlight) and not obj.manual_color:
            # White words read off a blue banner, put on a white box, would
            # vanish; dark words on that banner's own blue would too. A
            # highlight is what sits right behind the words, so it wins.
            color = ensure_contrast(
                color, _color_tuple(obj.manual_highlight or obj.manual_fill)
            )
        turn = turns.get(obj.id)
        # A box with a background is drawn last, fill then text, so it sits
        # over everything beneath it -- including other translations. A
        # turned block always has a writer of its own: a writer's morph
        # turns everything it holds.
        # A border goes on top of the words, so a bordered box is layered
        # too, which also keeps the border off the editor's clean plate.
        # A shape always is: drawn after the plate, it never stays on the
        # editor's clean background.
        layered = bool(obj.manual_fill or obj.manual_border_color or obj.manual_shape) or turn is not None or (
            layer_floor is not None and obj.z_order > layer_floor
        )
        writer = fitz.TextWriter(pdf_page.rect) if layered else writer_for(color)
        if layered:
            filled.append((obj, writer, color))
        else:
            writer_blocks.setdefault(
                (round(color[0], 3), round(color[1], 3), round(color[2], 3)), []
            ).append(obj.id)
        # A picture's turned label is centred along its frame: a short
        # translation then sits where the printed word did ("HARBOR" ran
        # across the tablet's screen; left-aligned, its translation covered
        # the left 45% of that and sat over the red letter).
        # A level label she turned herself keeps its own alignment.
        centred = (
            turn is not None and obj.source is TextSource.IMAGE_OCR
            and obj.oriented is not None and obj.manual_alignment is None
        )
        padx, pady = insets.get(obj.id, (0.0, 0.0))
        align = TextAlignment.CENTER if centred else obj.drawn_alignment
        size, step = fit.font_size, fit.leading
        # Up and down: the spare height under the rows goes above them by
        # this share. At the top (k = 0) the first baseline is exactly where
        # it has always been, so no document already checked moves. Bottom
        # puts the last baseline a descent above the box's bottom; middle
        # centres the rows' em boxes. A block that overflows has no spare
        # height, stays at the top and spills downward as before. The box
        # includes any height the fitter grew it by, and a turned label
        # does the same in its unturned frame. The box is the inner one,
        # inside her inside space.
        k = VERTICAL_SHARE.get(obj.manual_vertical_align, 0.0)
        box_height = layout_box.height + fit.extra_height
        slack = max(0.0, box_height - len(fit.lines) * step)
        y = layout_box.y0 + shift + size * ASCENT_RATIO + k * (slack + step - size)
        # Each row with its column and marker. Rows from a fit always
        # carry them; should one ever not, the words are wrapped as they
        # were before lists, and only the final row ends a paragraph.
        row_specs = fit.rows
        if len(row_specs) != len(fit.lines):
            _, ends = wrap_with_ends(text, rfont, size, max(1.0, layout_box.width))
            if len(ends) != len(fit.lines):
                ends = [i == len(fit.lines) - 1 for i in range(len(fit.lines))]
            row_specs = [RowSpec(text=t, end=e) for t, e in zip(fit.lines, ends)]
        paras, mfonts = plans.get(obj.id, (None, None))
        # Only a JUSTIFY she chose (see drawn_alignment), and only in a
        # script whose words are spaced (see fitting.justifiable).
        justify = align is TextAlignment.JUSTIFY and justifiable(text)
        bottom_limit = layout_box.y1 + shift + fit.extra_height + 0.5
        ink_x0, ink_x1, last_baseline = layout_box.x1, layout_box.x0, y
        drawn_lines: list[tuple[float, float, str]] = []
        rows: list[DrawnRow] = []
        marks: list[DrawnMark] = []

        def put(x: float, y: float, words: str) -> None:
            drawn_lines.append((round(x, 3), round(y, 3), words))
            try:
                writer.append(fitz.Point(x, y), words, font=font, fontsize=size)
            except Exception as exc:
                log.warning("Could not write line on page %s: %r", page.number, exc)

        for spec in row_specs:
            line = spec.text
            if not line:
                # Still a row, so the editor counts the same rows.
                rows.append(DrawnRow(
                    x0=round(layout_box.x0, 3), x1=round(layout_box.x0, 3),
                    baseline=round(y, 3), text="", end=True,
                ))
                y += step
                continue
            if y > bottom_limit + step:
                break  # already flagged as overflow; do not spill down the page
            # The row's text column: the box less the paragraph's offset
            # from its start edge (the right, for right-to-left text).
            col_x0 = layout_box.x0 + (0.0 if rtl else spec.indent)
            col_x1 = layout_box.x1 - (spec.indent if rtl else 0.0)
            col_w = max(0.0, col_x1 - col_x0)
            visual = to_visual(line, script)
            width = REGISTRY.measure(rfont, visual, size)
            # A justified row is spread on its visual string, so right-to-
            # left words are already in their final order.
            words = (
                [w for w in visual.split(" ") if w] if justify and not spec.end else []
            )
            gap = 0.0
            if len(words) >= 2:
                widths = [REGISTRY.measure(rfont, w, size) for w in words]
                gap = (col_w - sum(widths)) / (len(words) - 1)
            spread = len(words) >= 2 and gap > 0
            if spread:
                # Spread: one entry per word, the first at the column's
                # start and the last ending at the box's far edge.
                x = col_x0
                for w, wd in zip(words, widths):
                    put(x, y, w)
                    x += wd + gap
                run = (col_x0, col_x1)
            else:
                # Also a justified paragraph's last row: from the start, so
                # the left, or the right for right-to-left text.
                x = col_x0 + _alignment_offset(align, width, col_w)
                if rtl and not centred:
                    # Right-align RTL text so the line starts at the reading edge.
                    x = max(col_x0, col_x1 - width)
                put(x, y, visual)
                run = (x, x + width)
            ink_x0, ink_x1 = min(ink_x0, run[0]), max(ink_x1, run[1])
            marker_at: tuple[float, float] | None = None
            marker_text, marker_face = None, None
            if spec.marker and paras is not None and mfonts is not None:
                # The marker hangs `hang` before the words' start: left of
                # them, or right of them for right-to-left text. Same
                # writer, so the same colour and the same turn.
                marker_text, marker_face = drawable_marker(paras[spec.para], mfonts)
                shown = marker_visual(marker_text, rtl)
                mres = mfonts.resolved(marker_face)
                mw = REGISTRY.measure(mres, shown, size)
                mx = run[1] + spec.hang - mw if rtl else run[0] - spec.hang
                try:
                    writer.append(
                        fitz.Point(mx, y), shown, font=mfonts.font(marker_face), fontsize=size
                    )
                except Exception as exc:
                    log.warning("Could not write a list marker on page %s: %r", page.number, exc)
                marks.append(DrawnMark(
                    x=round(mx, 3), baseline=round(y, 3), text=shown,
                    font_builtin=mres.builtin,
                    font_path=str(mres.path) if mres.path else None,
                ))
                marker_at = (round(mx, 3), round(mx + mw, 3))
                ink_x0, ink_x1 = min(ink_x0, mx), max(ink_x1, mx + mw)
            last_baseline = y
            rows.append(DrawnRow(
                x0=round(run[0], 3), x1=round(run[1], 3), baseline=round(y, 3),
                text=line, end=spec.end, spread=spread,
                indent=round(spec.indent, 3), hang=round(spec.hang, 3),
                marker=marker_text,
                marker_x0=marker_at[0] if marker_at else None,
                marker_x1=marker_at[1] if marker_at else None,
                marker_face=marker_face,
            ))
            y += step

        if paras and any(p.replaced for p in paras):
            stats.findings.append(QAFinding(
                severity="warning", code="list_number_changed",
                message=NUMBER_CHANGED, object_ids=[obj.id], bbox=obj.bbox,
            ))
        if rfont.fell_back:
            chosen = catalog.face(obj.manual_font_family)
            stats.findings.append(QAFinding(
                # Recorded, not queued: the words are drawn, in the face
                # that has their letters, and the editor's menu says so.
                severity="info", code="font_fallback",
                message=(
                    f"{chosen.name if chosen else 'This font'} has no letters for "
                    "this language, so this text is drawn in the standard font for it."
                ),
                object_ids=[obj.id],
            ))
        obj.translated_font_size = fit.font_size
        obj.drawn_layout = DrawnLayout(
            font_builtin=rfont.builtin,
            font_path=str(rfont.path) if rfont.path else None,
            size=fit.font_size,
            color=color,
            lines=drawn_lines,
            fill=_color_tuple(obj.manual_fill) if obj.manual_fill else None,
            leading=fit.leading,
            rows=rows,
            underline=obj.manual_underline,
            highlight=_color_tuple(obj.manual_highlight) if obj.manual_highlight else None,
            centred=centred,
            marks=marks,
            face=rfont.face,
            face_fallback=rfont.fell_back,
            **_border_style(obj),
            **(_shape_style(obj, frames[obj.id]) if obj.id in frames else {}),
        )
        if not layered and (obj.manual_underline or obj.manual_highlight):
            decorated.append(obj)
        top = layout_box.y0 + shift
        if turn is not None:
            angle, (cx, cy) = turn
            pivot = (round(cx, 3), round(cy + shift, 3))
            # Any height the fitter added is part of the frame, so the sprite
            # (sized from drawn_bbox) never clips what the PDF shows. The
            # frame is the inner box grown back by the inside space.
            bottom = layout_box.y1 + shift + fit.extra_height
            corners = frame_corners(
                frames[obj.id] if obj.id in frames else Rect(
                    x0=layout_box.x0 - padx, y0=top - pady,
                    x1=layout_box.x1 + padx, y1=bottom + pady,
                ),
                angle, pivot,
            )
            obj.drawn_layout.angle = round(angle, 2)
            obj.drawn_layout.pivot = pivot
            obj.drawn_layout.quad = corners
            # Exactly the turned frame's upright extent, not grown to the
            # ink: the editor hands this box back as manual_bbox after a
            # drag, and turned_frame() must recover the same frame from it.
            xs, ys = zip(*corners)
            obj.drawn_bbox = Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
            stats.written += 1
            continue
        if obj.id in frames:
            # A shape's frame is exactly hers, not grown to the ink: the
            # editor hands it back as the box on the next drag, so a grown
            # one would creep larger on every move.
            obj.drawn_bbox = frames[obj.id]
            stats.written += 1
            continue
        # The frame is the laid-out box, grown to whatever ink fell outside
        # it, so it is never smaller than what it claims to hold.
        # The inside space is added back around it, so a frame handed back
        # unchanged as her box lays out the same inner box again.
        obj.drawn_bbox = Rect(
            x0=min(layout_box.x0 - padx, ink_x0),
            y0=top - pady,
            x1=max(layout_box.x1 + padx, ink_x1),
            # Text at the top keeps the old allowance below its last row. In
            # the middle or at the bottom exactly a descent: the editor hands
            # this frame back as the box on the next drag, and anything more
            # would grow the box a little on every move.
            y1=max(
                layout_box.y1 + shift + fit.extra_height,
                last_baseline + fit.font_size * (0.3 if k == 0 else DESCENT_RATIO),
            ) + pady,
        )
        stats.written += 1

    # Everything but the translations is now on the page: the live editor's
    # clean background is exactly this.
    if on_plate:
        on_plate(pdf_page)
    # If a colour's writer then fails (text_not_drawn below), these stay on
    # the page without their words. Rare, and reported with the failure.
    for t in sorted(decorated, key=lambda d: d.z_order):
        draw_decorations(pdf_page, t.drawn_layout)
    for rgb, writer in writers.items():
        try:
            writer.write_text(pdf_page, color=rgb)
        except Exception as exc:
            log.warning(
                "Could not write text on page %s in colour %s: %r",
                page.number, rgb, exc,
            )
            # Never silent: every block in this writer is missing from the
            # page, and the teacher has to know before sharing it.
            lost = writer_blocks.get(rgb, [])
            stats.findings.append(
                QAFinding(
                    severity="error",
                    code="text_not_drawn",
                    message=(
                        "Some translated text could not be drawn on this page. "
                        "Try a different font, or edit the text again."
                    ),
                    object_ids=lost,
                    details={"reason": f"{type(exc).__name__}"[:80]},
                )
            )
            for t in page.text_objects:
                if t.id in lost:
                    t.drawn_bbox = None
                    t.drawn_layout = None
    layered = [*filled, *((c, None, None) for c in covers)]
    for t, writer, rgb in sorted(layered, key=lambda f: f[0].z_order):
        try:
            # Her background (turned with a turned block), then highlight
            # and underline, then the words, then her border on top.
            paint_fill(pdf_page, t.drawn_layout, t.drawn_bbox)
            if writer is not None:
                draw_decorations(pdf_page, t.drawn_layout)
                writer.write_text(pdf_page, color=rgb, morph=turn_morph(t.drawn_layout))
            paint_border(pdf_page, t.drawn_layout, t.drawn_bbox)
        except Exception as exc:
            log.warning("Could not draw a filled box on page %s: %r", page.number, exc)
            stats.findings.append(
                QAFinding(
                    severity="error",
                    code="text_not_drawn",
                    message=(
                        "Some translated text could not be drawn on this page. "
                        "Try a different font, or edit the text again."
                    ),
                    object_ids=[t.id],
                )
            )
    page.reconstructed = True
    return stats


@overload
def rebuild_document(
    source: bytes | Path,
    pages: list[Page],
    *,
    target_lang: str,
    on_plate: "Callable[[int, fitz.Page], None] | None" = None,
    before_page: "Callable[[], None] | None" = None,
    out_path: None = None,
) -> tuple[bytes, dict[int, PageBuildStats]]: ...


@overload
def rebuild_document(
    source: bytes | Path,
    pages: list[Page],
    *,
    target_lang: str,
    on_plate: "Callable[[int, fitz.Page], None] | None" = None,
    before_page: "Callable[[], None] | None" = None,
    out_path: Path,
) -> tuple[Path, dict[int, PageBuildStats]]: ...


def rebuild_document(
    source: bytes | Path,
    pages: list[Page],
    *,
    target_lang: str,
    on_plate: "Callable[[int, fitz.Page], None] | None" = None,
    before_page: "Callable[[], None] | None" = None,
    out_path: Path | None = None,
) -> tuple[bytes | Path, dict[int, PageBuildStats]]:
    """Rebuild the whole document. Page count is preserved unconditionally.

    A page that fails to rebuild is carried over from the source unchanged
    rather than dropped, so the output always has the same number of pages as
    the input (spec sec. 46).

    `source` is the PDF's bytes or its file. With `out_path` the result is
    written to that file and its path returned, so neither the source nor
    the output is ever held in memory whole -- on the free server a large
    scanned packet does not fit twice. Without it the bytes are returned.

    `before_page` runs before each page is drawn and may raise to stop the
    rebuild (a cancelled job); nothing is written then.
    """
    if isinstance(source, Path):
        doc = fitz.open(str(source), filetype="pdf")
    else:
        doc = fitz.open(stream=source, filetype="pdf")
    stats: dict[int, PageBuildStats] = {}
    drawn: Path | None = None
    try:
        by_index = {p.index: p for p in pages}
        for index in range(doc.page_count):
            page_model = by_index.get(index)
            if page_model is None:
                continue
            if before_page is not None:
                before_page()
            try:
                stats[index] = rebuild_page(
                    doc.load_page(index),
                    page_model,
                    target_lang=target_lang,
                    on_plate=(
                        (lambda p, i=index: on_plate(i, p)) if on_plate else None
                    ),
                )
            except Exception as exc:
                log.exception("Page %s could not be rebuilt", index + 1)
                s = PageBuildStats()
                s.findings.append(
                    QAFinding(
                        severity="error",
                        code="reconstruction_failed",
                        message=(
                            f"Page {index + 1} kept its original text because it "
                            "could not be rebuilt."
                        ),
                        details={"reason": f"{type(exc).__name__}: {exc}"[:200]},
                    )
                )
                stats[index] = s
                page_model.error = s.findings[0].message
                page_model.error_stage = "RECONSTRUCTING"
        if out_path is None:
            out = doc.tobytes(garbage=3, deflate=True, clean=True)
        else:
            fd, name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".pdf")
            os.close(fd)
            drawn = Path(name)
            doc.save(str(drawn), garbage=3, deflate=True, clean=True)
    except BaseException:
        if drawn is not None:
            drawn.unlink(missing_ok=True)
        raise
    finally:
        doc.close()

    if out_path is None:
        return _with_subset_fonts(out), stats
    assert drawn is not None
    if subset_fonts_file(drawn, out_path):
        drawn.unlink(missing_ok=True)
    else:
        os.replace(drawn, out_path)
    return out_path, stats


def _page_texts(doc: "fitz.Document") -> list[str]:
    return [page.get_text() for page in doc]


def _same_text(before: list[str], doc: "fitz.Document") -> bool:
    """Whether `doc` reads exactly as `before` did, page for page."""
    return doc.page_count == len(before) and all(
        doc[i].get_text() == text for i, text in enumerate(before)
    )


def _with_subset_fonts(written: bytes) -> bytes:
    """The document saved with each embedded font cut down to the glyphs
    its pages use, or `written` unchanged if that cannot be done safely.

    Drawing text embeds the whole font file: a one-page Spanish worksheet
    came out at 2.4 MB, a Hebrew one at 16 MB. The subsetting runs on the
    saved file reopened, not the document that was drawn on: MuPDF only
    subsets fonts that are in the file, not those just added by drawing
    (measured: no change on the drawn document, Droid Sans Fallback 1.7 MB
    to 9 KB on the reopened file).

    The text of every page is compared before and after, so a subset that
    would make the PDF's words copy or search differently is thrown away
    and the full fonts kept. A slow or heavy download is better than a
    broken one. `subset_fonts_file` is the same step for files on disk.
    """
    try:
        doc = fitz.open(stream=written, filetype="pdf")
        try:
            before = _page_texts(doc)
            doc.subset_fonts()
            out = doc.tobytes(garbage=3, deflate=True, clean=True)
        finally:
            doc.close()
        check = fitz.open(stream=out, filetype="pdf")
        try:
            same = _same_text(before, check)
        finally:
            check.close()
    except Exception as exc:
        log.warning("Could not shrink the fonts in the output: %r", exc)
        return written
    if not same:
        log.warning("Shrinking the fonts changed the text; full fonts kept")
        return written
    return out if len(out) < len(written) else written


def subset_fonts_file(src: Path, dst: Path) -> bool:
    """Write `src` to `dst` with its fonts cut down to the glyphs it uses.

    The file form of `_with_subset_fonts`, with the same text check, for
    documents too large to hold in memory twice. Returns True only when
    `dst` holds a subset that reads exactly as `src` and is smaller.
    Otherwise `dst` is removed and False returned, and the caller keeps
    `src`. `src` is never changed.
    """
    try:
        doc = fitz.open(str(src), filetype="pdf")
        try:
            before = _page_texts(doc)
            doc.subset_fonts()
            doc.save(str(dst), garbage=3, deflate=True, clean=True)
        finally:
            doc.close()
        check = fitz.open(str(dst), filetype="pdf")
        try:
            same = _same_text(before, check)
        finally:
            check.close()
    except Exception as exc:
        log.warning("Could not shrink the fonts in the output: %r", exc)
        dst.unlink(missing_ok=True)
        return False
    if not same:
        log.warning("Shrinking the fonts changed the text; full fonts kept")
        dst.unlink(missing_ok=True)
        return False
    if dst.stat().st_size >= src.stat().st_size:
        dst.unlink(missing_ok=True)
        return False
    return True


def estimate_paper_background(pdf_page, obj):
    """Recovery-only paper check, excluding known glyphs from the sample.

    A dense bold paragraph can contaminate the ordinary border sample. Only
    propose a patch when the remaining surrounding/inter-word paper is nearly
    uniform and light. Unknown marks, colour, and long rules make us abstain.
    This is evidence for a preview, not proof that hidden artwork is absent.
    """
    import numpy as np
    clip = fitz.Rect(*obj.bbox.expanded(4).as_tuple()) & pdf_page.rect
    pix = pdf_page.get_pixmap(clip=clip, dpi=144, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    paper = np.ones(arr.shape[:2], dtype=bool)
    boxes = [s.bbox for line in obj.lines for s in line.spans if s.text.strip()]
    boxes += obj.cover_boxes
    if not boxes:
        return None
    for box in boxes:
        r = box.expanded(1)
        x0, y0 = max(0, int(r.x0 * 2 - pix.x)), max(0, int(r.y0 * 2 - pix.y))
        x1, y1 = min(pix.width, int(r.x1 * 2 - pix.x + 1)), min(pix.height, int(r.y1 * 2 - pix.y + 1))
        paper[y0:y1, x0:x1] = False
    samples = arr[paper].astype(float)
    if len(samples) < 100 or paper.mean() < .12:
        return None
    median = np.median(samples, axis=0)
    if median.min() < 220 or median.max() - median.min() > 12:
        return None
    if np.mean(np.max(np.abs(samples - median), axis=1) > 24) > .008:
        return None
    dark = arr.max(axis=2) < 120
    # Long rules even entirely inside a word box must remain untouched.
    for rows, limit in ((dark, 48), (dark.T, 48)):
        for row in rows:
            edges = np.diff(np.pad(row.astype(int), (1, 1)))
            if np.any(np.where(edges == -1)[0] - np.where(edges == 1)[0] >= limit):
                return None
    return tuple(float(c / 255) for c in median)
