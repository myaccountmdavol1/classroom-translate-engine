"""How far a line of words in a picture is turned, and whether it is curved.

Tesseract reports upright boxes and nothing about slant ("baseline 0 0" in
its hOCR, on every line). A word printed on a tilted drawing -- "HARBOR" on
a sketched tablet held at 13 degrees, real slide -- comes back as an upright
box twice the word's height that takes in the tablet's frame and the
letter below it. Covering that box paints over the artwork; sizing the
translation from it draws it twice too big; drawing it level looks wrong.

The slant is measured from the letters themselves, with numpy only:

  1. Binarise the word's crop (Otsu, both polarities, so white lettering on
     a dark panel works too).
  2. Find the connected shapes. A shape touching the crop's edge is
     something bigger passing through (a frame, a neighbouring letter), and
     shapes far shorter or taller than the typical one are dots or artwork.
  3. Fit a line through the shapes' centres: every pair is tried and the one
     most shapes lie along wins, so a stray shape cannot tilt the answer.
     Then a least-squares fit over the shapes on it.
  4. A quadratic through the centres says whether the words follow a curve.

Measured on the real slide: 6 letters found, 13.0 degrees, worst letter
0.08 of a letter-height off the line, curve 0.01 (straight). Alternatives
measured and rejected: a principal-axis fit over all dark pixels (the frame
dominates: 178 degrees), a projection sweep (agrees, but ~120 rotations),
and sweeping Tesseract's confidence over angles (noisy, a call per angle).

Rendering density matters: at 150 dpi the frame broke into pieces that
passed as letters and the fit said 2.7 degrees. Crops are measured with the
word box at least ~14 pixels high and at most ~110.

A short level word is the hard case. On a real deck of screenshots, 12 of
182 level labels of 4-9 letters ("+ Add page", "File type", "Grid")
measured 2.7-4.9 degrees with most letters agreeing on an edge: with that
few letters the fit's noise is a few degrees. So a slant is only believed
when it is also *geometrically consistent* (`consistent`):

  * Turning a box to the true slant of a line makes it hug the letters:
    HARBOR's turned box is 47% of its upright box's area. Turning it to a
    false slant does the opposite -- a level word's letters fill an upright
    rectangle, and any tilted box around them is bigger. A slant has to
    shrink the box by AREA_GAIN; every false label above failed this.
  * With fewer than SHORT_LINE letters the slant through the letters'
    centres and the slant along their shared edge must agree within
    FITS_AGREE degrees (HARBOR: 13.0 and 13.5).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

# Below this the words are drawn level, exactly as before.
STRAIGHT_BELOW = 2.5  # degrees
# Steeper than this is not redrawn: Tesseract rarely reads it, and a frame
# inverted from an upright box is ill-conditioned past ~35-45 degrees.
MAX_TILT = 45.0  # degrees
# Bow of the best quadratic over half the line, in letter-heights. Straight
# lines measured 0.01-0.15 (ascenders and descenders move the centres); a
# gentle arc 0.48, a 40-degree arc 0.65.
CURVED_SAGITTA = 0.35
# Fewest letters on one line before a slant is believed without a re-read.
MIN_INLIERS = 4
# A centre this far from the line, in letter-heights, is not on it.
INLIER_DISTANCE = 0.5
# Letters sharing an edge (the baseline, or the x-height) agree on it to
# within this, in letter-heights. A turned letter's lowest point moves with
# its width by ~0.07 of a height at 13 degrees; a descender drops ~0.25.
EDGE_DISTANCE = 0.12
# Work limits: a crop or a shape count past these is not a line of words.
MAX_CROP_PIXELS = 1_500_000
MAX_COMPONENTS = 200
# A word crop is at most ~110 pixels tall: ~40 runs a row. A photo's
# noise is far more, and is declined before the pure-Python pass.
MAX_RUNS = 20_000
# Pairs are tried among at most this many of the biggest shapes (435 pairs).
MAX_FIT_COMPONENTS = 30
# Word box height in pixels the measurement wants: below the floor letters
# are too coarse to trust, above the ceiling the crop is scaled down.
MIN_WORD_PIXELS = 14
MAX_WORD_PIXELS = 110
# A slant is only believed when turning the box to it makes the box hug the
# letters: turned area at most this share of the upright area. A true 3
# degree slant on a 4-letter word gives ~0.87 (so it stays level: safe);
# HARBOR 0.49; the real level labels measured at a false 2.7-4.9 degrees
# 0.93-1.15.
AREA_GAIN = 0.85
# Below this many letters the centre fit and the edge fit must agree.
SHORT_LINE = 10
FITS_AGREE = 1.5  # degrees
# A slant measured on a page translated before slants were measured has no
# re-read to back it and changes a page she may have reviewed, so it needs
# more: this many letters and at least this slant (HARBOR: 6, 13 degrees).
LATE_MIN_LETTERS = 6
LATE_MIN_ANGLE = 5.0


@dataclass
class Orientation:
    """A measured line of letters, in the measured crop's pixels."""

    angle: float  # degrees clockwise
    cx: float
    cy: float
    width: float
    height: float
    glyphs: int  # letters on the fitted line
    candidates: int  # letter-like shapes in the word box
    residual: float  # worst letter centre's distance from the line, in heights
    sagitta: float  # bow of the line, in heights
    # Letters whose bottoms (or tops) lie on one straight edge: what the
    # angle is finally measured from.
    edge_agree: int = 0
    # The slant through the letters' centres, before the edge correction.
    centre_angle: float = 0.0
    # Turned box area over the upright box area of the same letters.
    area_ratio: float = 1.0
    # Letter-sized shapes inside the reader's word box that the turned box
    # does not hold: covering only the box would leave them showing.
    stray: int = 0
    # Page placement of the crop: page point = origin + pixel / scale.
    origin: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    # For the re-read: the crop's shapes and which belong to the words.
    _labels: np.ndarray | None = field(default=None, repr=False)
    _keep: np.ndarray | None = field(default=None, repr=False)

    @property
    def straight(self) -> bool:
        return abs(self.angle) < STRAIGHT_BELOW

    @property
    def curved(self) -> bool:
        return self.sagitta >= CURVED_SAGITTA

    @property
    def steep(self) -> bool:
        return abs(self.angle) > MAX_TILT

    @property
    def strong(self) -> bool:
        """Enough letters sharing one edge to trust with no re-read."""
        return self.edge_agree >= MIN_INLIERS and self.edge_agree >= 0.6 * self.glyphs

    @property
    def consistent(self) -> bool:
        """The slant is one the letters' shape bears out (module notes):
        turning to it shrinks their box, on a short line both fits agree,
        and the turned box holds every letter the reader boxed. Nothing is
        drawn turned without this. ("Process", level and blurred, measured
        16 degrees through 4 of its letters; 5 more were left out.)"""
        if self.area_ratio > AREA_GAIN or self.stray:
            return False
        if self.glyphs < SHORT_LINE and abs(self.centre_angle - self.angle) > FITS_AGREE:
            return False
        return True

    @property
    def tilted(self) -> bool:
        """A slant worth drawing, if something backs it (a re-read, or
        `strong`): not level, not curved, not steep, and consistent."""
        return not (self.straight or self.curved or self.steep) and self.consistent

    @property
    def sure_alone(self) -> bool:
        """A slant believed with nothing else to back it: for pages read
        before slants were measured, where a wrong turn would change words
        she already checked."""
        return (
            self.tilted and self.strong
            and self.glyphs >= LATE_MIN_LETTERS and abs(self.angle) >= LATE_MIN_ANGLE
        )

    def page_box(self):
        """The words' turned box in page points."""
        from ..model.objects import OrientedBox

        return OrientedBox(
            cx=round(self.origin[0] + self.cx / self.scale, 3),
            cy=round(self.origin[1] + self.cy / self.scale, 3),
            width=round(self.width / self.scale, 3),
            height=round(self.height / self.scale, 3),
            angle=round(self.angle, 2),
        )

    def straightened_png(self, *, level: bool = False) -> bytes | None:
        """The words alone, turned level, dark on white: one line for a
        re-read. Everything that is not a letter shape near the line (the
        frame, artwork) is blanked first, or Tesseract reads it too.

        `level=True` gives the same blanked words NOT turned: reading both
        tells a real slant (the turned copy reads clearly better) from a
        level word measured a few degrees off (both read the same)."""
        if self._labels is None or self._keep is None:
            return None
        from PIL import Image

        mask = self._keep[self._labels]
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        pad = 4
        y0, y1 = max(0, ys.min() - pad), ys.max() + pad + 1
        x0, x1 = max(0, xs.min() - pad), xs.max() + pad + 1
        crop = np.where(mask[y0:y1, x0:x1], 0, 255).astype(np.uint8)
        img = Image.fromarray(crop, "L")
        # PIL turns counter-clockwise for a positive angle, which undoes a
        # clockwise slant.
        if not level:
            img = img.rotate(self.angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
        # Tesseract reads best with letters ~30 pixels tall.
        if self.height < 24:
            factor = 32.0 / max(1.0, self.height)
            img = img.resize(
                (max(1, round(img.width * factor)), max(1, round(img.height * factor))),
                Image.BICUBIC,
            )
        framed = Image.new("L", (img.width + 40, img.height + 40), 255)
        framed.paste(img, (20, 20))
        out = io.BytesIO()
        framed.save(out, "PNG")
        return out.getvalue()


def _otsu(values: np.ndarray) -> float:
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 128.0
    levels = np.arange(256, dtype=float)
    weight_b = np.cumsum(hist)
    weight_f = total - weight_b
    sum_b = np.cumsum(hist * levels)
    mean_b = np.divide(sum_b, weight_b, out=np.zeros(256), where=weight_b > 0)
    mean_f = np.divide(
        sum_b[-1] - sum_b, weight_f, out=np.zeros(256), where=weight_f > 0
    )
    between = weight_b * weight_f * (mean_b - mean_f) ** 2
    return float(np.argmax(between)) + 0.5


def _label_runs(mask: np.ndarray):
    """Connected shapes (4-neighbour) of a boolean image, by runs.

    Labelling runs rather than pixels keeps this to a few thousand steps of
    pure Python for a word crop. Returns (labels image, per-run arrays) or
    None when there are too many runs to be a line of words.
    """
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    ys, x0s = np.nonzero(edges == 1)
    _, x1s = np.nonzero(edges == -1)  # exclusive ends, same row order
    n = len(ys)
    if n == 0 or n > MAX_RUNS:
        return None
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Rows start where y changes.
    row_starts = np.flatnonzero(np.diff(ys, prepend=-1)) if n else np.array([], int)
    bounds = list(row_starts) + [n]
    prev_lo = prev_hi = 0
    prev_y = -2
    for r in range(len(bounds) - 1):
        lo, hi = bounds[r], bounds[r + 1]
        y = int(ys[lo])
        if y == prev_y + 1:
            j = prev_lo
            for i in range(lo, hi):
                a0, a1 = x0s[i], x1s[i]
                while j < prev_hi and x1s[j] <= a0:
                    j += 1
                k = j
                while k < prev_hi and x0s[k] < a1:
                    ri, rk = find(i), find(k)
                    if ri != rk:
                        parent[ri] = rk
                    k += 1
        prev_lo, prev_hi, prev_y = lo, hi, y
    roots = np.array([find(i) for i in range(n)])
    _, run_label = np.unique(roots, return_inverse=True)
    labels = np.zeros((h, w), dtype=np.int32)  # 0 = background
    for i in range(n):
        labels[ys[i], x0s[i]:x1s[i]] = run_label[i] + 1
    return labels, ys, x0s, x1s, run_label


def _edge_line(u: np.ndarray, v: np.ndarray, tolerance: float, spacing: float):
    """(letters on it, slope) of the straight edge most of (u, v) lie on
    within `tolerance`, or None. Every pair is tried, then least squares."""
    if len(u) < 3:
        return None
    ii, jj = np.triu_indices(len(u), k=1)
    du = u[jj] - u[ii]
    ok = np.abs(du) >= spacing
    if not ok.any():
        return None
    ii, jj, du = ii[ok], jj[ok], du[ok]
    slopes = (v[jj] - v[ii]) / du
    predicted = v[ii][:, None] + slopes[:, None] * (u[None, :] - u[ii][:, None])
    dist = np.abs(v[None, :] - predicted)
    within = dist < tolerance
    score = within.sum(axis=1) - np.where(within, dist, 0).sum(axis=1) / (tolerance * 1e3)
    best = within[int(np.argmax(score))]
    if best.sum() < 3:
        return None
    slope = float(np.polyfit(u[best], v[best], 1)[0])
    return int(best.sum()), slope


def _fit_polarity(ink: np.ndarray, word_box: tuple[float, float, float, float]):
    labelled = _label_runs(ink)
    if labelled is None:
        return None
    labels, ys, x0s, x1s, run_label = labelled
    h, w = ink.shape
    count = int(run_label.max()) + 1
    if count > MAX_COMPONENTS * 4:
        return None
    lengths = (x1s - x0s).astype(float)
    area = np.bincount(run_label, weights=lengths, minlength=count)
    sum_x = np.bincount(
        run_label, weights=lengths * (x0s + x1s - 1) / 2.0, minlength=count
    )
    sum_y = np.bincount(run_label, weights=lengths * ys, minlength=count)
    min_x = np.full(count, w)
    max_x = np.full(count, -1)
    min_y = np.full(count, h)
    max_y = np.full(count, -1)
    np.minimum.at(min_x, run_label, x0s)
    np.maximum.at(max_x, run_label, x1s - 1)
    np.minimum.at(min_y, run_label, ys)
    np.maximum.at(max_y, run_label, ys)
    cx = sum_x / np.maximum(area, 1)
    cy = sum_y / np.maximum(area, 1)
    heights = (max_y - min_y + 1).astype(float)

    bx0, by0, bx1, by1 = word_box
    touches = (min_x <= 0) | (min_y <= 0) | (max_x >= w - 1) | (max_y >= h - 1)
    inside = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
    candidate = ~touches & inside & (area >= 3)
    if candidate.sum() < 3:
        return None
    median = float(np.median(heights[candidate]))
    # Dots and commas are far shorter than letters; frames and artwork far
    # taller. Neither says where the line runs.
    letters = candidate & (heights >= 0.35 * median) & (heights <= 2.5 * median)
    idx = np.flatnonzero(letters)
    if len(idx) < 3 or len(idx) > MAX_COMPONENTS:
        return None
    median = float(np.median(heights[idx]))
    if median < 3:
        return None

    pts = np.stack([cx[idx], cy[idx]], axis=1)
    fit_idx = np.arange(len(idx))
    if len(idx) > MAX_FIT_COMPONENTS:
        fit_idx = np.argsort(-area[idx])[:MAX_FIT_COMPONENTS]
    fp = pts[fit_idx]
    limit = INLIER_DISTANCE * median
    ii, jj = np.triu_indices(len(fp), k=1)
    d = fp[jj] - fp[ii]
    norm = np.hypot(d[:, 0], d[:, 1])
    ok = norm >= 0.5 * median
    if not ok.any():
        return None
    ii, jj, d, norm = ii[ok], jj[ok], d[ok], norm[ok]
    normals = np.stack([-d[:, 1], d[:, 0]], axis=1) / norm[:, None]
    dist = np.abs(
        np.einsum("mkj,mj->mk", pts[None, :, :] - fp[ii][:, None, :], normals)
    )
    within = dist < limit
    score = within.sum(axis=1) - (np.where(within, dist, 0).sum(axis=1) / (limit * 1e3))
    best = int(np.argmax(score))
    on_line = within[best]

    def line_through(mask_pts: np.ndarray):
        sel = pts[mask_pts]
        centre = sel.mean(axis=0)
        cov = np.cov((sel - centre).T)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
            direction = -direction
        return centre, direction

    centre, direction = line_through(on_line)
    normal = np.array([-direction[1], direction[0]])
    off = np.abs((pts - centre) @ normal)
    on_line = off < limit
    if on_line.sum() < 3:
        return None
    centre, direction = line_through(on_line)
    normal = np.array([-direction[1], direction[0]])
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    centre_angle = angle

    def bow(sel: np.ndarray) -> tuple[float, float]:
        """(sagitta, worst quadratic residual), both in letter-heights."""
        if len(sel) < 5:
            return 0.0, 0.0
        t = (sel - centre) @ direction
        s = (sel - centre) @ normal
        if np.ptp(t) < 1e-6:
            return 0.0, 0.0
        coef = np.polyfit(t, s, 2)
        span = float(np.ptp(t))
        sag = abs(coef[0]) * (span / 2.0) ** 2 / median
        res = float(np.max(np.abs(s - np.polyval(coef, t)))) / median
        return float(sag), res

    sag_line, _ = bow(pts[on_line])
    # Letters around a tight arc fall off any one line, so the curve is
    # also judged over every letter-like shape -- when a curve explains them.
    sag_all, res_all = bow(pts)
    sagitta = sag_line
    if res_all < INLIER_DISTANCE:
        sagitta = max(sagitta, sag_all)

    residual = float(np.max(np.abs((pts[on_line] - centre) @ normal))) / median

    # Centres of mixed letters are not on one line: a capital's centre sits
    # above an x-height letter's, a descender's below. "Temp", printed
    # level, measured 5 degrees from its four centres. Letters standing on
    # the baseline share their bottoms exactly, and x-height letters their
    # tops, so the angle is corrected by whichever edge more letters share.
    ends_x = np.concatenate([x0s, x1s - 1]).astype(float) + 0.5
    ends_y = np.concatenate([ys, ys]).astype(float) + 0.5
    ends_label = np.concatenate([run_label, run_label])
    across = ends_x * normal[0] + ends_y * normal[1]
    lowest = np.full(count, -np.inf)
    highest = np.full(count, np.inf)
    np.maximum.at(lowest, ends_label, across)
    np.minimum.at(highest, ends_label, across)
    letters_on = idx[on_line]
    along = pts[on_line] @ direction
    edge_agree, slope = 0, 0.0
    for edge in (lowest[letters_on], highest[letters_on]):
        found = _edge_line(along, edge, EDGE_DISTANCE * median, median)
        if found is None:
            continue
        if found[0] > edge_agree:
            edge_agree, slope = found
        elif found[0] == edge_agree:
            slope = (slope + found[1]) / 2.0
    if edge_agree >= 3:
        angle += math.degrees(math.atan(slope))
        t = math.radians(angle)
        direction = np.array([math.cos(t), math.sin(t)])
        normal = np.array([-direction[1], direction[0]])

    def box_of(chosen: np.ndarray):
        """(cx, cy, w, h) of the chosen shapes' pixels along the line and
        across it, and (W, H) of the same pixels upright. Run ends are the
        extremes of a linear projection."""
        sel = chosen[run_label]
        ex = np.concatenate([x0s[sel], x1s[sel] - 1]).astype(float) + 0.5
        ey = np.concatenate([ys[sel], ys[sel]]).astype(float) + 0.5
        pu = ex * direction[0] + ey * direction[1]
        pv = ex * normal[0] + ey * normal[1]
        u0, u1 = pu.min() - 0.5, pu.max() + 0.5
        v0, v1 = pv.min() - 0.5, pv.max() + 0.5
        mid_u, mid_v = (u0 + u1) / 2.0, (v0 + v1) / 2.0
        upright = (ex.max() - ex.min() + 1.0, ey.max() - ey.min() + 1.0)
        return (
            mid_u * direction[0] + mid_v * normal[0],
            mid_u * direction[1] + mid_v * normal[1],
            u1 - u0, v1 - v0,
        ), upright

    # The words' box: first the letters on the line...
    members = np.zeros(count, dtype=bool)
    members[idx[on_line]] = True
    (box_cx, box_cy, box_w, box_h), _ = box_of(members)

    # ...then every letter-sized shape on the words' band with them: dots,
    # commas, an icon or a short word the line fit set aside. The cover is
    # this box, so what it leaves out stays showing in English.
    du = (cx - box_cx) * direction[0] + (cy - box_cy) * direction[1]
    dv = (cx - box_cx) * normal[0] + (cy - box_cy) * normal[1]
    band = (
        ~touches
        & (area >= 3)
        & (heights <= 2.5 * median)
        & (np.abs(du) <= box_w / 2.0 + median)
        & (np.abs(dv) <= box_h / 2.0 + 0.35 * median)
    )
    (box_cx, box_cy, box_w, box_h), (up_w, up_h) = box_of(band | members)
    area_ratio = (box_w * box_h) / max(1.0, up_w * up_h)

    # Letter-sized shapes the reader's box holds that the turned box does
    # not: drawing the slant would cover the line and leave these.
    stray_mask = candidate & (heights >= 0.35 * median) & (heights <= 2.5 * median) & ~(band | members)
    stray = int(stray_mask.sum())

    # What a re-read sees: the shapes the box was made from.
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = band | members

    return Orientation(
        angle=float(angle),
        cx=float(box_cx),
        cy=float(box_cy),
        width=float(box_w),
        height=float(box_h),
        glyphs=int(on_line.sum()),
        candidates=int(len(idx)),
        residual=round(residual, 3),
        sagitta=round(sagitta, 3),
        edge_agree=int(edge_agree),
        centre_angle=float(centre_angle),
        area_ratio=round(float(area_ratio), 3),
        stray=stray,
        _labels=labels,
        _keep=keep,
    )


def measure_line(
    rgb: np.ndarray, word_box: tuple[float, float, float, float]
) -> Orientation | None:
    """Measure the line of letters inside `word_box` (crop pixels).

    `rgb` is the crop: the word box with room around it, so shapes passing
    through (which touch its edge) can be told from letters (which do not).
    Returns None when there is no clear line of at least three letters, and
    the caller then treats the words as level, exactly as before.
    """
    if rgb.ndim != 3 or rgb.shape[0] * rgb.shape[1] > MAX_CROP_PIXELS:
        return None
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None
    channels = rgb[:, :, :3].astype(np.float32)
    lum = 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    bx0, by0, bx1, by1 = (
        max(0, int(word_box[0])), max(0, int(word_box[1])),
        min(w, int(math.ceil(word_box[2]))), min(h, int(math.ceil(word_box[3]))),
    )
    inner = lum[by0:by1, bx0:bx1]
    if inner.size < 16:
        return None
    threshold = _otsu(np.clip(inner, 0, 255))
    best: Orientation | None = None
    for ink in (lum < threshold, lum > threshold):
        # Letters are the minority of their own box; the majority is paper.
        if ink[by0:by1, bx0:bx1].mean() > 0.6:
            continue
        found = _fit_polarity(ink, word_box)
        if found is None:
            continue
        if best is None or (found.glyphs, -found.residual) > (best.glyphs, -best.residual):
            best = found
    return best


def measure_region(
    render, bbox: tuple[float, float, float, float], *, dpi: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> Orientation | None:
    """Measure words at `bbox` (page points) through `render(clip, dpi)`,
    which returns a PNG of that page area -- the reader's own renderer, so
    the letters look as they did when read.

    Only a padded crop around the words is rendered (half the box's height
    each side, kept inside `bounds`, the picture), never the whole picture:
    a full-slide picture decoded at the reading density is 15-25 MB, a
    word's crop a few hundred KB. The density is lowered towards a word box
    ~80 pixels tall when it would be taller than MAX_WORD_PIXELS, by at
    most half: letters scaled to ~12 pixels measured fine but no longer
    re-read (Tesseract returned nothing for "Precipitation").
    """
    from PIL import Image

    x0, y0, x1, y1 = bbox
    bh = (y1 - y0) * dpi / 72.0
    if bh < MIN_WORD_PIXELS or (x1 - x0) * dpi / 72.0 < MIN_WORD_PIXELS:
        return None
    if bh > MAX_WORD_PIXELS:
        dpi *= max(0.5, 80.0 / bh)
    pad = 0.5 * (y1 - y0)
    cx0, cy0, cx1, cy1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if bounds is not None:
        cx0, cy0 = max(cx0, bounds[0]), max(cy0, bounds[1])
        cx1, cy1 = min(cx1, bounds[2]), min(cy1, bounds[3])
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    if (cx1 - cx0) * (cy1 - cy0) * (dpi / 72.0) ** 2 > MAX_CROP_PIXELS:
        return None
    png = render((cx0, cy0, cx1, cy1), max(36, int(round(dpi))))
    with Image.open(io.BytesIO(png)) as im:
        arr = np.asarray(im.convert("RGB"))
    # The scale the renderer actually used (its pixel grid rounds).
    sx = arr.shape[1] / (cx1 - cx0)
    sy = arr.shape[0] / (cy1 - cy0)
    found = measure_line(
        arr, ((x0 - cx0) * sx, (y0 - cy0) * sy, (x1 - cx0) * sx, (y1 - cy0) * sy)
    )
    if found is None:
        return None
    found.origin = (cx0, cy0)
    found.scale = (sx + sy) / 2.0
    return found


def measure_on_page(pdf_page, bbox) -> Orientation | None:
    """Measure words at `bbox` (page points) on a PDF page, rendering a
    padded crop at a density where the box is ~80 pixels tall."""
    import fitz
    from PIL import Image

    height = bbox.y1 - bbox.y0
    if height <= 0:
        return None
    dpi = max(72.0, min(300.0, 72.0 * 80.0 / height))
    zoom = dpi / 72.0
    pad = 0.5 * height
    clip = fitz.Rect(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad) & pdf_page.rect
    if clip.is_empty or clip.width * clip.height * zoom * zoom > MAX_CROP_PIXELS:
        return None
    try:
        pix = pdf_page.get_pixmap(
            clip=clip, matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False
        )
    except Exception:
        return None
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    found = measure_line(
        np.asarray(image),
        ((bbox.x0 - clip.x0) * zoom, (bbox.y0 - clip.y0) * zoom,
         (bbox.x1 - clip.x0) * zoom, (bbox.y1 - clip.y0) * zoom),
    )
    if found is None:
        return None
    found.origin = (clip.x0, clip.y0)
    found.scale = zoom
    return found
