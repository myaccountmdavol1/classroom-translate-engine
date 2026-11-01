"""Whether a picture's words are hand-lettered.

A translation drawn in a plain face over brush lettering looks pasted on:
"DESDE EL" in Helvetica across a novel's hand-painted cover. Pictures
lettered by hand -- covers, posters, chapter titles -- get their
translations in a hand-lettered face (Amatic SC) instead.

Measures of the strokes alone did not tell them apart (stroke-width
variation was 0.22-0.27 on the cover and 0.24 on a printed diagram). What
does is that the words are known: each label is drawn in a plain face, a
serif face and the hand-lettered one, and compared with its own letters --
how wide the words run for their height, and how much their outlines
overlap. Printed labels match a plain face closely (overlap 0.7-0.8);
hand lettering matches none well, and runs tall and narrow like Amatic.

One label can mislead (a pointer line in its box made a printed "Cell Wall"
look narrow), so the picture decides as a whole: its words are hand-lettered
when more of its labels match the hand face than a printed one.
"""

from __future__ import annotations

from ..fonts import catalog
from ..fonts.catalog import HAND_FACE
from .word_ends import commonest_colour

# Height the letters are compared at, in pixels.
COMPARE_HEIGHT = 48
# A label whose letters overlap a plain face's this much is printed.
PRINTED_OVERLAP = 0.6
# A label votes for the hand face only when the words drawn in it run this
# close to the label's own width for its height (log of the ratio).
HAND_ASPECT = 0.35
# Fewer letters than this say nothing about the lettering.
MIN_LETTERS = 3
# Pixels this far (0-255, any channel) from the paper are ink.
INK_FROM_PAPER = 80


def hand_lettered(png: bytes, labels: list[tuple[str, tuple[float, float, float, float]]]) -> bool:
    """Whether the words in a picture are hand-lettered, from its labels'
    (text, box in the picture's pixels)."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        img = img[:, :, None]
    img = img[:, :, :3].astype(np.int16)
    hand = printed = 0
    for text, box in labels:
        vote = _vote(img, text, box)
        if vote == "hand":
            hand += 1
        elif vote == "printed":
            printed += 1
    return hand > printed


def mark_hand_lettered(objects) -> None:
    """Draw these labels' translations in the hand-lettered face, in its
    bold: its regular is a hairline pen, and lettering on covers and
    worksheets is marker-thick."""
    for obj in objects:
        obj.style = obj.style.model_copy(update={"hand_lettered": True, "bold": True})


def _vote(img, text: str, box) -> str | None:
    import numpy as np

    if sum(c.isalnum() for c in text) < MIN_LETTERS:
        return None
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if crop.size == 0:
        return None
    paper = commonest_colour(crop.reshape(-1, crop.shape[2]))
    own = _tight(_letters_only((np.abs(crop - paper).max(axis=2) > INK_FROM_PAPER).astype(np.uint8)))
    if own is None:
        return None
    scores = {}
    for name, face in (("sans", None), ("serif", None), ("hand", HAND_FACE)):
        drawn = _drawn(text, name, face)
        if drawn is None:
            return None
        scores[name] = _compare(own, drawn)
    best_printed = max(scores["sans"][1], scores["serif"][1])
    if best_printed >= PRINTED_OVERLAP:
        return "printed"
    best = min(scores, key=lambda k: scores[k][0] - scores[k][1])
    if best == "hand":
        return "hand" if scores["hand"][0] <= HAND_ASPECT else None
    return "printed"


# Marks shorter than this share of the box are not letters: the stub of a
# pointer line under a printed "Cell Wall" made its words look tall and
# narrow, like hand lettering. (The dot of an i goes too, and matters not.)
MIN_MARK = 0.3


def _letters_only(mask):
    import cv2
    import numpy as np

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_HEIGHT] >= MIN_MARK * mask.shape[0]
    return keep[labels].astype(np.uint8)


def _tight(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _drawn(text: str, name: str, face: str | None):
    """`text`'s ink drawn in one face, cropped to the letters."""
    import fitz
    import numpy as np

    if face is not None:
        found = catalog.face_path(face, bold=True)
        if found is None:
            return None
        font = fitz.Font(fontfile=str(found[0]))
    else:
        font = fitz.Font("helv" if name == "sans" else "tiro")
    if any(not font.has_glyph(ord(c)) for c in text if not c.isspace()):
        return None
    size = 100
    width = font.text_length(text, fontsize=size) + 40
    doc = fitz.open()
    page = doc.new_page(width=width, height=2.2 * size)
    writer = fitz.TextWriter(page.rect)
    writer.append((20, 1.4 * size), text, font=font, fontsize=size)
    writer.write_text(page, color=(0, 0, 0))
    pix = page.get_pixmap(dpi=72, alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return _tight((arr[:, :, :3].max(axis=2) < 128).astype(np.uint8))


def _compare(own, drawn) -> tuple[float, float]:
    """(how far apart their widths for their height are, as the log of the
    ratio; how much their outlines overlap at the same size, 0-1)."""
    import cv2
    import numpy as np

    own_aspect = own.shape[1] / own.shape[0]
    aspect = abs(float(np.log((drawn.shape[1] / drawn.shape[0]) / own_aspect)))
    size = (max(8, int(COMPARE_HEIGHT * own_aspect)), COMPARE_HEIGHT)
    a = cv2.resize(own * 255, size, interpolation=cv2.INTER_AREA) > 127
    b = cv2.resize(drawn * 255, size, interpolation=cv2.INTER_AREA) > 127
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate(a.astype(np.uint8), k) > 0
    b = cv2.dilate(b.astype(np.uint8), k) > 0
    return aspect, float((a & b).sum() / max(1, (a | b).sum()))
