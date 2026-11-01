"""Erasing words from a busy picture background, letter by letter.

The flat patch (builder.py) covers a word with the paper's own colour, which
only works when the paper around it is one colour. A word on a shaded label,
a gradient or a photo was left in English instead.

Here only the letters' own pixels are erased -- found as the pixels that
stand out from the background around the word -- and the holes are filled
from their surroundings (OpenCV's Telea inpainting). The holes are thin, so
the fill blends in, and lines and shading beside the words are kept.

The fill borrows from whatever touches the holes. When other ink does -- a
neighbouring word, a pointer line running through -- it would be smeared
into the gap, so the block is refused and left as it was.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import fitz
import numpy as np

from ..model.geometry import Rect

# Detail the letters are found and filled at.
ERASE_DPI = 200
# How far the fill reaches in from the edge of a hole, in pixels.
FILL_RADIUS = 3
# A word's letters stand out from the background by at least this much
# (largest channel difference, 0-255), or the box holds no letters.
MIN_CONTRAST = 40.0
# Share of a word's box its letters cover: below, nothing was found; above,
# the "letters" are really the background.
MIN_COVER, MAX_COVER = 0.02, 0.6
# How far past the reader's word box the letters are looked for, as a share
# of the word's height.
EDGE_SLACK = 0.15


@dataclass(frozen=True)
class LettersErased:
    background: tuple[float, float, float]  # rgb 0-1, around the words
    ink: tuple[float, float, float]  # rgb 0-1, the letters' own colour
    stroke: float  # points, how thick the letters' strokes are (bold or not)


@dataclass
class _Word:
    mask: np.ndarray  # bool, the letters, grown to catch their soft edges
    ink: np.ndarray  # the letters' pixels
    paper: np.ndarray  # the background pixels around them
    stroke: float  # pixels, the letters' stroke thickness


def erase_letters(pdf_page: "fitz.Page", boxes: list[Rect]) -> LettersErased | None:
    """Erase the letters inside `boxes` (one block's words) and fill them in.

    All or nothing: returns None, with the page untouched, when any word's
    letters can't be told from its background or other ink touches them.
    """
    if not boxes:
        return None
    s = ERASE_DPI / 72.0
    heights = [b.height for b in boxes]
    reach = max(heights) * 0.5 + 6.0
    clip = fitz.Rect(
        min(b.x0 for b in boxes) - reach, min(b.y0 for b in boxes) - reach,
        max(b.x1 for b in boxes) + reach, max(b.y1 for b in boxes) + reach,
    ) & pdf_page.rect
    if clip.is_empty:
        return None
    pix = pdf_page.get_pixmap(clip=clip, dpi=ERASE_DPI, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    img = np.ascontiguousarray(img)
    # The pixmap snaps the clip outward to whole pixels: its first pixel is
    # pix.x, pix.y on the page's pixel grid, not clip.x0 * s. Placing the
    # patch from clip.x0 put it up to a pixel off, and a sliver of the old
    # letter showed beside it.
    ox, oy = pix.x, pix.y

    words = []
    for b in boxes:
        x0 = int(np.floor(b.x0 * s)) - ox
        y0 = int(np.floor(b.y0 * s)) - oy
        x1 = int(np.ceil(b.x1 * s)) - ox
        y1 = int(np.ceil(b.y1 * s)) - oy
        word = _find_letters(img, x0, y0, x1, y1, b.height * s)
        if word is None:
            return None
        words.append(word)

    mask = np.zeros(img.shape[:2], np.uint8)
    for w in words:
        mask[w.mask] = 255
    filled = cv2.inpaint(img, mask, FILL_RADIUS, cv2.INPAINT_TELEA)

    # Put back only the part that changed, so the rest of the picture keeps
    # its own pixels.
    ys, xs = np.nonzero(mask)
    px0, py0 = max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1)
    px1, py1 = min(img.shape[1], int(xs.max()) + 2), min(img.shape[0], int(ys.max()) + 2)
    patch = fitz.Pixmap(fitz.csRGB, px1 - px0, py1 - py0,
                        np.ascontiguousarray(filled[py0:py1, px0:px1]).tobytes(), False)
    pdf_page.insert_image(
        fitz.Rect((ox + px0) / s, (oy + py0) / s, (ox + px1) / s, (oy + py1) / s),
        stream=patch.tobytes("png"),
    )

    def rgb(pixels: np.ndarray) -> tuple[float, float, float]:
        m = np.median(pixels, axis=0) / 255.0
        return (float(m[0]), float(m[1]), float(m[2]))

    return LettersErased(
        background=rgb(np.concatenate([w.paper for w in words])),
        ink=rgb(np.concatenate([w.ink for w in words])),
        stroke=float(np.median([w.stroke for w in words])) / s,
    )


def _find_letters(img, x0, y0, x1, y1, height_px) -> _Word | None:
    """The letters in one word's box, or None when they can't be erased safely."""
    H, W = img.shape[:2]
    # The reader's box can clip a letter's edge (the bowl of an S, a serif):
    # look a little past it, or that sliver stays on the page.
    edge = max(1, int(round(height_px * EDGE_SLACK)))
    x0, y0, x1, y1 = max(0, x0 - edge), max(0, y0 - edge), min(W, x1 + edge), min(H, y1 + edge)
    w, h = x1 - x0, y1 - y0
    if w < 3 or h < 3:
        return None
    grow = max(2, int(round(height_px * 0.08)))
    near = grow + FILL_RADIUS + 2
    pad = near + 3
    rx0, ry0, rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)

    yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
    inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    area = img[ry0:ry1, rx0:rx1].astype(float)
    # The background as a plane per channel, fitted to the ring around the
    # box: follows a gradient or shading across the word. Fitted twice, the
    # second time without ring pixels that are ink.
    coords = np.stack([(xx - x0) / w, (yy - y0) / h, np.ones_like(xx, float)], axis=-1)
    ring = ~inside
    A, px = coords[ring], area[ring]
    # errstate: numpy 2.2 on macOS's Accelerate warns about overflow in
    # matmuls whose results are finite and correct.
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, px, rcond=None)
        first = np.abs(px - A @ coef).max(axis=1)
        keep = first <= max(16.0, np.percentile(first, 75) * 2)
        if keep.sum() >= 12:
            coef, *_ = np.linalg.lstsq(A[keep], px[keep], rcond=None)
        resid = np.abs(area - coords @ coef).max(axis=2)
    if not np.isfinite(resid).all():
        return None

    noise = float(np.percentile(resid[ring], 75))
    contrast = float(np.percentile(resid[inside], 98))
    if contrast < MIN_CONTRAST or noise * 3 > contrast:
        return None
    # Low enough to catch the soft anti-aliased and scan-blurred edges.
    thr = max(16.0, noise * 1.5, contrast * 0.22)
    letters = (resid > thr) & inside
    cover = letters.sum() / inside.sum()
    if cover < MIN_COVER or cover > MAX_COVER:
        return None

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    grown = cv2.dilate(letters.astype(np.uint8), k) > 0
    # Other ink within the fill's reach of the holes would be smeared in.
    kn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near + 1,) * 2)
    reached = (cv2.dilate(grown.astype(np.uint8), kn) > 0) & ~grown
    other_ink = (resid > max(thr, contrast * 0.4)) & reached
    if other_ink.sum() > max(3, 0.01 * letters.sum()):
        return None

    mask = np.zeros((H, W), bool)
    mask[ry0:ry1, rx0:rx1] = grown
    core = (resid > max(thr, contrast * 0.6)) & inside
    paper = ring & (resid <= thr)
    return _Word(
        mask=mask,
        ink=area[core] if core.any() else area[letters],
        paper=area[paper] if paper.any() else area[ring],
        stroke=_stroke_width(resid > (thr + contrast) / 2),
    )


# --- Lettering of one ink on one paper -------------------------------------
# A cover's black brush title runs close to its white one ("FROM THE" between
# "NOTES" and "MIDNIGHT"): every pixel that stands out is not the label's, so
# erase_letters above takes the white words for letters too, or refuses. Here
# only the label's own ink colour is erased, and of it only the strokes as
# thick as lettering: the drawing's thin black outlines stay.

# Pixels this far (0-255, any channel) from the paper are ink of some kind.
INK_FROM_PAPER = 80.0
# ... and this near the label's ink colour are its ink.
SAME_INK = 70.0
# Strokes thinner than this share of the line's height are a drawing's
# outlines, not letters.
THIN = 0.04
# A pixel this near (0-255) the line from ink to paper is the letters' soft
# edge (or the brighter JPEG halo just past it), so it is erased too.
ON_EDGE = 45.0
# How far around the letters their edge is looked for, as a share of the
# line's height.
EDGE_REACH = 0.03
# Other artwork (neither ink nor paper) in patches at least this big (side,
# as a share of the line's height): next to it the hole is blended from its
# surroundings; everywhere else it takes the paper's colour, which on flat
# paper leaves no ghost of the letters.
OTHER_ART = 0.05
# Share of the label's box its letters may cover, and the least share that
# must be the paper itself.
MIN_INK_COVER, MAX_INK_COVER, MIN_PAPER = 0.05, 0.75, 0.2
# How far (0-255, mean) the paper's own pixels may stray from its colour. A
# printed cover's flat colour strays 0.6-2.3; misreadings on a photo, where
# the "paper" is a shirt or the sky, 4.8 and up -- erased, they were blotted.
PAPER_SPREAD = 3.5


def erase_ink(pdf_page: "fitz.Page", box: Rect) -> LettersErased | None:
    """Erase one line of lettering by its own ink colour and fill it with
    the paper's. None, with the page untouched, when the box holds no clear
    ink on one flat paper colour."""
    from ..pipeline.word_ends import commonest_colour

    s = ERASE_DPI / 72.0
    reach = box.height * 0.15
    clip = fitz.Rect(box.x0 - reach, box.y0 - reach, box.x1 + reach, box.y1 + reach) & pdf_page.rect
    if clip.is_empty:
        return None
    pix = pdf_page.get_pixmap(clip=clip, dpi=ERASE_DPI, alpha=False)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    )
    ox, oy = pix.x, pix.y
    area = img.astype(np.int16)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(np.floor(box.x0 * s)) - ox), max(0, int(np.floor(box.y0 * s)) - oy)
    x1, y1 = min(W, int(np.ceil(box.x1 * s)) - ox), min(H, int(np.ceil(box.y1 * s)) - oy)
    h = y1 - y0
    if h < 8 or x1 - x0 < 8:
        return None
    inbox = np.zeros((H, W), bool)
    inbox[y0:y1, x0:x1] = True

    flat = area[inbox]
    paper = commonest_colour(flat)
    far = np.abs(flat - paper).max(axis=1) > INK_FROM_PAPER
    if far.mean() < MIN_INK_COVER:
        return None
    ink = commonest_colour(flat[far])

    same = (np.abs(area - ink).max(axis=2) < SAME_INK).astype(np.uint8)
    k = max(1, int(THIN * h))
    thick = cv2.morphologyEx(same, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # Strokes mostly inside the box are the label's; a letter may reach a
    # little past the reader's box (the cover's D did).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thick)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = (inbox & (labels == i)).sum() > 0.5 * stats[i, cv2.CC_STAT_AREA]
    letters = keep[labels]
    if not MIN_INK_COVER <= letters[inbox].mean() <= MAX_INK_COVER:
        return None

    # Where each pixel lies on the line from ink (0) to paper (1), and how
    # far off that line it is.
    d = (paper - ink).astype(float)
    v = (area - ink).astype(float)
    with np.errstate(all="ignore"):
        t = (v @ d) / float(d @ d)
    off = np.linalg.norm(v - t[..., None] * d, axis=2)
    edge = (off < ON_EDGE) & (t < 1.4)
    g = max(2, int(EDGE_REACH * h))
    near = cv2.dilate(letters.astype(np.uint8), np.ones((2 * g + 1,) * 2, np.uint8)) > 0
    hole = letters | (near & edge)
    hole = (cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (hole | edge)

    plain = inbox & ~hole & (np.abs(area - paper).max(axis=2) < 40)
    if plain.sum() < MIN_PAPER * inbox.sum():
        return None
    paper_rgb = np.median(area[plain], axis=0)
    if float(np.abs(area[plain] - paper_rgb).mean()) > PAPER_SPREAD:
        return None

    filled = cv2.inpaint(img, hole.astype(np.uint8) * 255, FILL_RADIUS, cv2.INPAINT_TELEA)
    other = ((off >= ON_EDGE) & ~hole).astype(np.uint8)
    no, olabels, ostats, _ = cv2.connectedComponentsWithStats(other)
    big = np.zeros(no, bool)
    big[1:] = ostats[1:, cv2.CC_STAT_AREA] >= (OTHER_ART * h) ** 2
    dist = cv2.distanceTransform((~big[olabels]).astype(np.uint8), cv2.DIST_L2, 3)
    filled[hole & (dist > 2 * g)] = paper_rgb.astype(np.uint8)

    ys, xs = np.nonzero(hole)
    px0, py0 = max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1)
    px1, py1 = min(W, int(xs.max()) + 2), min(H, int(ys.max()) + 2)
    patch = fitz.Pixmap(fitz.csRGB, px1 - px0, py1 - py0,
                        np.ascontiguousarray(filled[py0:py1, px0:px1]).tobytes(), False)
    pdf_page.insert_image(
        fitz.Rect((ox + px0) / s, (oy + py0) / s, (ox + px1) / s, (oy + py1) / s),
        stream=patch.tobytes("png"),
    )
    core = letters & (np.abs(area - ink).max(axis=2) < SAME_INK / 2)
    return LettersErased(
        background=tuple(float(c) / 255.0 for c in paper_rgb),
        ink=tuple(float(c) / 255.0 for c in (np.median(area[core], axis=0) if core.any() else ink)),
        stroke=_stroke_width(letters) / s,
    )


def _stroke_width(letters: np.ndarray) -> float:
    """Mean stroke thickness in pixels: for thin strokes the ink's area is
    about its length times its width, and its outline twice its length."""
    outlines, _ = cv2.findContours(
        letters.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    perimeter = sum(cv2.arcLength(c, True) for c in outlines)
    return 2.0 * float(letters.sum()) / max(perimeter, 1.0)
