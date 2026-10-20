"""Prepare a page image for OCR without changing the page.

Tesseract reads a text cell boxed in by ruled lines as part of a picture. On
a scanned table with ordinary 1-2pt rules it recovered as few as 0 of 40
words, at every resolution tried. Hairline rules were fine; it is the weight
of the grid, not the resolution, that defeats it.

So the grid is removed from the copy Tesseract reads. Only that copy: the
stored page keeps its rules, and since covering paints word boxes, the
rebuilt table keeps them too. Coordinates are unchanged -- same image, same
size -- so recognised boxes still land where the words are.

Only lines that meet another line at a right angle are removed: that is what
makes a grid or a box. A lone answer line has nothing crossing it and
reaches OCR untouched, so "Name: ________" is read exactly as before. A page
with no grid gets the original bytes back.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Darker than this is ink. Scanned paper sits well above, scanned ink well
# below; the threshold only has to split them, not measure them.
INK_LEVEL = 128
# A dark run at least this long is a rule, not a glyph stroke. Body text
# strokes are a few points; horizontal rules span a whole table, so the
# horizontal bar can be high. Vertical rules may cross only a row or two.
MIN_HORIZONTAL_RULE = 48.0  # points
MIN_VERTICAL_RULE = 30.0  # points
# How close two lines must come to count as meeting. Scanned corners rarely
# touch exactly.
JOIN_DISTANCE = 2.0  # points
# Rows (or columns) processed at once. Bounds peak memory: the engine runs
# in 512 MB with most of it already spoken for.
BAND = 256


def remove_grid_lines(png: bytes, *, dpi: int) -> bytes | None:
    """The page image with its grid lines whitened, or None if it has none."""
    img = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    dark = img < INK_LEVEL
    if not dark.any():
        return None
    scale = dpi / 72.0
    horizontal = _runs(dark, int(MIN_HORIZONTAL_RULE * scale))
    if not horizontal.any():
        return None
    vertical = _runs(dark.T, int(MIN_VERTICAL_RULE * scale)).T
    if not vertical.any():
        return None
    del dark

    reach = max(1, int(round(JOIN_DISTANCE * scale)))
    near_vertical = _dilate(vertical, reach)
    near_horizontal = _dilate(horizontal, reach)
    erase = _touching(horizontal, near_vertical)
    erase |= _touching(vertical.T, near_horizontal.T).T
    del near_vertical, near_horizontal, horizontal, vertical
    if not erase.any():
        return None

    # Antialiasing leaves a grey fringe either side of a rule, which Tesseract
    # sees as a faint line. One pixel of margin takes it too.
    erase = _dilate(erase, 1)
    sample = img[::8, ::8]
    paper = int(np.median(sample[sample >= INK_LEVEL]))
    out = img.copy()
    out[erase] = paper
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG")
    return buf.getvalue()


def _runs(mask: np.ndarray, length: int) -> np.ndarray:
    """Pixels lying in a horizontal run of True at least `length` long."""
    h, w = mask.shape
    out = np.zeros(mask.shape, dtype=bool)
    if length < 2 or length > w:
        return out
    lower = np.clip(np.arange(w) - length + 1, 0, None)
    for top in range(0, h, BAND):
        band = mask[top : top + BAND]
        # Windowed sums from a running total: a window summing to `length`
        # is entirely dark, and marks where a qualifying run starts.
        total = np.zeros((band.shape[0], w + 1), dtype=np.int32)
        np.cumsum(band, axis=1, out=total[:, 1:])
        starts = np.zeros(band.shape, dtype=np.int32)
        starts[:, : w - length + 1] = (
            total[:, length:] - total[:, :-length]
        ) == length
        # A pixel is covered if a qualifying run started within `length`
        # before it.
        np.cumsum(starts, axis=1, out=total[:, 1:])
        out[top : top + BAND] = (total[:, 1:] - total[:, lower]) > 0
    return out


def _dilate(mask: np.ndarray, reach: int) -> np.ndarray:
    """Grow True regions by `reach` pixels in every direction (a square)."""
    grown = mask.copy()
    for d in range(1, reach + 1):
        grown[d:] |= mask[:-d]
        grown[:-d] |= mask[d:]
    wide = grown.copy()
    for d in range(1, reach + 1):
        wide[:, d:] |= grown[:, :-d]
        wide[:, :-d] |= grown[:, d:]
    return wide


def _touching(runs: np.ndarray, other: np.ndarray) -> np.ndarray:
    """The horizontal runs in `runs` that meet any True pixel of `other`."""
    out = np.zeros(runs.shape, dtype=bool)
    for top in range(0, runs.shape[0], BAND):
        band = runs[top : top + BAND]
        if not band.any():
            continue
        # Number each run: a run starts where True follows False.
        starts = band.copy()
        starts[:, 1:] &= ~band[:, :-1]
        ids = np.cumsum(starts, dtype=np.int32).reshape(band.shape)
        ids[~band] = 0
        met = np.unique(ids[band & other[top : top + BAND]])
        met = met[met > 0]
        if met.size:
            out[top : top + BAND] = np.isin(ids, met)
    return out
