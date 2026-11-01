"""Labels a diagram's pointer lines run through.

On a plant cell worksheet the line pointing at the mitochondrion ran
through the "n" of "Mitochondrion", and the one for the nucleolus through
its "c": the reader saw "Mitochordrion" and "Nudleolus", sure of both, and
"Nudleolus" was translated into nonsense. Reading the label larger did not
help -- the crossing looks like a letter at any size.

So a label a long thin straight line runs through is read but not
trusted: it stays in English for the teacher to check. And a label the
reader missed altogether because a line touched it ("Nucleus") is found by
reading the picture again with the lines painted out; it is kept the same
way, so it is listed rather than silently left in English.
"""

from __future__ import annotations

from ..providers.ocr.base import OCRProvider, OCRRegion

# Ink darker than this (0-255) is lines and letters.
DARK = 170
# A pointer line is at least this long, in label heights ...
LINE_LENGTH = 2.5
# ... and thinner than this, in label heights: a cell's thick outline is
# not one. (The plant cell's lines were 5 px beside 21 px letters.)
LINE_THICK = 0.22
# ... and unbroken: ink along at least this share of its length.
UNBROKEN = 0.95
# A line runs through the letters when it crosses the label's box inside
# this margin (a share of the height) from its edges, not merely touching.
INSIDE = 0.2
# The letter height lines are looked for at when painting them out of a
# whole picture, in pixels (a small label at the reading resolution).
TYPICAL_HEIGHT = 21
# A label a line runs through, or one found only with the lines painted
# out, with at least this many letters, is listed for her to check when
# read this sure. Lower than a label's own bar: a line is why the reading
# is shaky, and it is never translated as read. (In the browser "Nucleus",
# found under a painted-out line, read 0.49, and "Nudleolus" 0.51.)
MIN_LETTERS = 3
MIN_LINE_CONFIDENCE = 0.4


def _lines(grey, h: float):
    """Long thin straight segments in `grey`, as (x0, y0, x1, y1), for
    letters `h` pixels tall."""
    import cv2
    import numpy as np

    dark = (grey < DARK).astype(np.uint8)
    # Thin ink only: strokes thicker than a pointer line are an outline,
    # taken out to their full width (their edges alone are thin too).
    half = max(3.0, LINE_THICK * h)
    dt = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    k = 2 * int(np.ceil(half)) + 3
    thick = cv2.dilate((dt > half).astype(np.uint8), np.ones((k, k), np.uint8))
    thin = dark & (1 - thick)
    segs = cv2.HoughLinesP(
        thin * 255, 1, np.pi / 360, threshold=int(1.5 * h),
        minLineLength=int(LINE_LENGTH * h), maxLineGap=3,
    )
    if segs is None:
        return []
    # Unbroken ink all along: the feet of a row of letters line up too, but
    # with gaps between the letters.
    near = cv2.dilate(thin, np.ones((3, 3), np.uint8))
    out = []
    for x0, y0, x1, y1 in segs[:, 0]:
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        if near[ys, xs].mean() >= UNBROKEN:
            out.append((int(x0), int(y0), int(x1), int(y1)))
    return out


def crossed_by_line(grey, bbox) -> bool:
    """Whether a long thin straight line runs through the letters of the
    label at `bbox` (pixels of `grey`, the picture in grey)."""
    import cv2

    H, W = grey.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    h = y1 - y0
    if h < 6:
        return False
    # Pointer lines are drawn in ink on paper. White words on a dark panel
    # ("Collection Pond") have thin strips of panel round their letters,
    # which look like lines and are not.
    import numpy as np

    if float(np.median(grey[max(0, y0):y1, max(0, x0):x1])) < DARK:
        return False
    m = int(3 * h)
    X0, Y0 = max(0, x0 - m), max(0, y0 - m)
    window = grey[Y0:min(H, y1 + m), X0:min(W, x1 + m)]
    ix0, iy0 = int(x0 - X0 + 0.15 * h), int(y0 - Y0 + INSIDE * h)
    ix1, iy1 = int(x1 - X0 - 0.15 * h), int(y1 - Y0 - INSIDE * h)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    box = (x0 - X0, y0 - Y0, x1 - X0, y1 - Y0)
    for seg in _lines(window, h):
        hit, _, _ = cv2.clipLine((ix0, iy0, ix1 - ix0, iy1 - iy0), seg[:2], seg[2:])
        # A pointer reaches out to what it points at; the feet of the
        # label's own letters, lined up, stay inside its box.
        if hit and _length(seg) - _inside(seg, box) >= h:
            return True
    return False


def hidden_words(
    grey, regions: list[OCRRegion], provider: OCRProvider, language: str | None,
) -> list[OCRRegion]:
    """Words read only once the picture's pointer lines are painted out, and
    nowhere near what was read before. Empty when it has no such lines."""
    import cv2

    segs = _lines(grey, TYPICAL_HEIGHT)
    if not segs:
        return []
    # Not the feet of a label already read, lined up: painting those out
    # would take the bottoms off its letters.
    segs = [
        s for s in segs
        if not any(_inside(s, r.bbox) > 0.5 * _length(s) for r in regions)
    ]
    if not segs:
        return []
    clean = grey.copy()
    for x0, y0, x1, y1 in segs:
        cv2.line(clean, (x0, y0), (x1, y1), 255, 5)
    ok, buf = cv2.imencode(".png", clean)
    if not ok:
        return []
    result = provider.recognize(buf.tobytes(), language=language, sparse=True)
    if result is None or result.error:
        return []
    return [
        r for r in result.regions
        if sum(c.isalpha() for c in r.text) >= MIN_LETTERS
        and r.confidence >= MIN_LINE_CONFIDENCE
        and not any(_overlap(r.bbox, o.bbox) for o in regions)
        # Hidden by a line: one of those painted out reached the word.
        and any(_near(s, r.bbox) for s in segs)
    ]


def _length(seg) -> float:
    return float(((seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2) ** 0.5)


def _inside(seg, box) -> float:
    """How much of the segment lies inside `box` (x0, y0, x1, y1)."""
    import cv2

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    hit, a, b = cv2.clipLine((x0, y0, max(1, x1 - x0), max(1, y1 - y0)), seg[:2], seg[2:])
    return _length((*a, *b)) if hit else 0.0


def _near(seg, box) -> bool:
    """Whether the segment comes within one letter height of `box`."""
    h = box[3] - box[1]
    grown = (box[0] - h, box[1] - h, box[2] + h, box[3] + h)
    return _inside(seg, grown) > 0


def _overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
