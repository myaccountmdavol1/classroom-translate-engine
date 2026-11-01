"""Words in a picture that were read only in part.

Sparse reading of hand-lettered titles stops partway through a word: on a
novel's cover "DRIVER" was read as "RIVER" (its "D" drawn a little apart)
and translated "RÍO"; on its title page "Midnight" came back as "ght" and
became "combate". The reading is confident -- each piece is a real word --
so nothing else catches it.

What gives it away is on the page: lettering of the same colour carries on
past the end of the box, closer than a space between words. Then the line
is widened over those letters and read again as one line. A reading that
takes in what was read before replaces it; otherwise the line is marked
cut, and the caller leaves it in English for the teacher to check.
"""

from __future__ import annotations

from dataclasses import replace

from ..model.document import Page, QAFinding
from ..model.geometry import Rect
from ..providers.ocr.base import OCRProvider, OCRRegion, OCRWord

# A shape this close past a line's end, in line heights, is more of the
# same word. Real word spaces on the novel's pages were 0.3 and wider; the
# cut-off "D" and "i" were 0.07 and 0.08.
CUT_GAP = 0.2
# Ink reaching less than this past the box (in line heights) is the box's
# own letters, a little wider than the reader reported them.
EDGE = 0.1
# A letter is at least this share of the line's height (an "n" beside a
# box that holds a "g"'s tail is about a third) ...
LETTER_HEIGHT = 0.3
# ... its ink at least this share of the height squared (not a speck) ...
LETTER_INK = 0.02
# ... and it stays on the line: no further above its top or below its
# bottom than this, in line heights. A drawing beside the words reaches
# further; chained in, the elf beside "DRIVER" made it unreadable.
ON_LINE = 0.25
# Strokes thinner than this share of the line's height are taken out
# before letters are looked for: the cover's "D" touched the elf's thin
# outline, and the two made one shape running far below the line.
THIN = 0.04
# How far past each end letters are looked for, in line heights.
LOOK = 1.5
# Ink colour: pixels this far (0-255, any channel) from the background
# are ink, and pixels this near the ink's colour are more of it.
INK_FROM_PAPER, SAME_INK = 80.0, 70.0
# A widened reading is kept at this confidence or above. Lower than a
# label's own bar: it must also take in the confident reading it widens, so
# a stray guess cannot pass. Hand-lettered "DRIVER" read 0.54.
MIN_REREAD = 0.5


def finish_cut_words(
    png: bytes, regions: list[OCRRegion], provider: OCRProvider, language: str | None,
) -> list[OCRRegion]:
    """The regions, with each line read only in part read again whole (at
    its widened box), or marked `cut_short` when it cannot be."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return regions
    if img.ndim == 2:
        img = img[:, :, None]
    img = img[:, :, :3].astype(np.int16)

    out: list[OCRRegion | None] = list(regions)
    for i, region in enumerate(regions):
        if out[i] is None or not region.text.strip():
            continue
        whole = _whole_extent(img, region.bbox)
        if whole is None:
            continue
        reread = _reread(img, region, whole, provider, language)
        if reread is region:
            continue
        if reread is None:
            out[i] = replace(region, cut_short=True)
            continue
        out[i] = reread
        # Other pieces of the same words ("Mint" over "Midnight") go.
        for j, other in enumerate(out):
            if j != i and other is not None and _mostly_inside(other.bbox, reread.bbox):
                out[j] = None
    return [r for r in out if r is not None]


def part_of_word_finding(page: Page, object_ids: list[str], bbox: Rect | None) -> QAFinding:
    """The lines on a page that are part of a longer word, left in English."""
    return QAFinding(
        severity="warning",
        code="image_text_part_of_word",
        message=(
            f"Only part of a word in a picture on page {page.number} could be "
            "read, so it was left in English. Check the English and translate it."
        ),
        object_ids=object_ids,
        bbox=bbox,
    )


def commonest_colour(pixels):
    """The commonest colour among `pixels` (n x channels, 0-255), to the
    nearest 16 in each channel: the paper behind a line of lettering.

    Counted, not sorted: np.unique over pixel rows made reading a novel's
    36 pictures take 44 s instead of 9."""
    import numpy as np

    bins = (pixels // 16).astype(np.int64)
    key = np.zeros(len(bins), np.int64)
    for c in range(bins.shape[1]):
        key = key * 16 + bins[:, c]
    best = int(np.bincount(key).argmax())
    out = []
    for _ in range(bins.shape[1]):
        out.append(best % 16)
        best //= 16
    return np.array(out[::-1]) * 16 + 8


def _mostly_inside(a, b) -> bool:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return w * h / area > 0.6


def _whole_extent(img, bbox) -> tuple[float, float] | None:
    """(x0, x1) of the whole word(s) when same-coloured lettering carries on
    past either end of `bbox`; None when the line stands on its own."""
    import cv2
    import numpy as np

    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    y0, y1 = max(0, y0), min(H, y1)
    h = y1 - y0
    if h < 4 or x1 - x0 < 2:
        return None
    inside = img[y0:y1, max(0, x0):min(W, x1)].reshape(-1, img.shape[2])
    # The paper is the commonest colour in the box; the ink, the middle
    # colour of what is far from it.
    paper = commonest_colour(inside)
    far = np.abs(inside - paper).max(axis=1) > INK_FROM_PAPER
    if far.sum() < LETTER_INK * h * h:
        return None
    ink = np.median(inside[far], axis=0)

    # Rows from a little above the line to a little below it, so a shape
    # that runs past the line shows that it does.
    ty0, ty1 = max(0, int(y0 - 2 * ON_LINE * h)), min(H, int(y1 + 2 * ON_LINE * h))
    start, end = x0, x1
    for _ in range(8):
        bx0 = max(0, int(start - LOOK * h))
        bx1 = min(W, int(end + LOOK * h))
        band = img[ty0:ty1, bx0:bx1]
        same = (np.abs(band - ink).max(axis=2) < SAME_INK).astype(np.uint8)
        k = max(1, int(THIN * h))
        same = cv2.morphologyEx(same, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(same)
        grown = False
        for k in range(1, n):
            x, y, w, hh, area = (int(v) for v in stats[k])
            top, bottom = ty0 + y, ty0 + y + hh
            if (hh < LETTER_HEIGHT * h or area < LETTER_INK * h * h
                    or top < y0 - ON_LINE * h or bottom > y1 + ON_LINE * h):
                continue
            left, right = bx0 + x, bx0 + x + w
            # A letter just past an end, or one joined to the word's own
            # first or last letter and reaching well past it.
            if left < start - EDGE * h and start - right <= CUT_GAP * h:
                start, grown = left, True
            if right > end + EDGE * h and left - end <= CUT_GAP * h:
                end, grown = right, True
        if not grown:
            break
    if start == x0 and end == x1:
        return None
    return float(start), float(end)


def _letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _reread(img, region: OCRRegion, whole, provider, language) -> OCRRegion | None:
    """The line read again across its whole extent; `region` itself when
    that reading is the same letters (a last letter reaching a little past
    its box: the logo's "OF EDUCATION"); or None when the reading is not
    sure or does not take in what was read before."""
    import cv2

    x0, y0, x1, y1 = region.bbox
    h = y1 - y0
    pad = 0.25 * h
    H, W = img.shape[:2]
    cx0, cy0 = int(max(0, whole[0] - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(W, whole[1] + pad)), int(min(H, y1 + pad))
    crop = img[cy0:cy1, cx0:cx1].astype("uint8")
    ok, buf = cv2.imencode(".png", crop[:, :, 0] if crop.shape[2] == 1 else crop)
    if not ok:
        return None
    result = provider.recognize_line(buf.tobytes(), language=language)
    if result is None or result.error or not result.regions:
        return None
    text = " ".join(r.text for r in result.regions).strip()
    chars = sum(len(r.text) for r in result.regions) or 1
    confidence = sum(r.confidence * len(r.text) for r in result.regions) / chars
    before, after = _letters(region.text), _letters(text)
    if confidence >= MIN_REREAD and after == before:
        return region
    if confidence < MIN_REREAD or len(after) <= len(before) or before not in after:
        return None

    # The widened reading agrees with the confident one on everything that
    # one read; only the letters added are new. So it is as sure as the
    # better of the two: a label below the bar is left in English.
    confidence = max(confidence, region.confidence)

    def placed(b):
        return (b[0] + cx0, b[1] + cy0, b[2] + cx0, b[3] + cy0)

    words: list[OCRWord] = [
        replace(w, bbox=placed(w.bbox)) for r in result.regions for w in r.words
    ]
    return replace(
        region, text=text, confidence=round(confidence, 4), words=words,
        bbox=(whole[0], y0, whole[1], y1), discarded_boxes=[],
    )
