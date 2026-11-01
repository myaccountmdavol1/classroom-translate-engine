"""A Word picture's lettering is erased by its own ink colour.

A novel's cover put its black brush title "FROM THE" between the white words
"NOTES" and "MIDNIGHT", on flat green, beside a drawing with thin black
outlines. No plain ring of paper surrounds the black words, so they were left
in English. `erase_ink` erases only the black strokes as thick as lettering,
fills them with the green, and leaves the white words and the outlines.
"""
from __future__ import annotations

import fitz
import numpy as np

from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.erase import erase_ink

GREEN = (0.22, 0.72, 0.29)
WORDS = Rect(x0=60, y0=110, x1=440, y1=190)


def _cover(*, paper=GREEN) -> fitz.Page:
    """Black bold words between white ones, and a thin black outline beside
    them, flattened to one picture."""
    src = fitz.open()
    page = src.new_page(width=500, height=300)
    page.draw_rect(page.rect, color=None, fill=paper)
    page.insert_text((60, 100), "NOTES", fontsize=90, fontname="hebo", color=(1, 1, 1))
    page.insert_text((60, 180), "FROM", fontsize=90, fontname="hebo", color=(0, 0, 0))
    page.insert_text((60, 270), "NIGHT", fontsize=90, fontname="hebo", color=(1, 1, 1))
    # A drawing's thin outline running into the black word's box.
    page.draw_circle((400, 160), 30, color=(0, 0, 0), width=1.2)
    pix = page.get_pixmap(dpi=150)
    out = fitz.open()
    flat = out.new_page(width=500, height=300)
    flat.insert_image(flat.rect, stream=pix.tobytes("png"))
    return flat


def _pixels(page: fitz.Page, clip: fitz.Rect) -> np.ndarray:
    pix = page.get_pixmap(clip=clip, dpi=150, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(int)


def _black(px: np.ndarray) -> np.ndarray:
    return px.max(axis=2) < 60


def _white(px: np.ndarray) -> np.ndarray:
    return px.min(axis=2) > 200


def test_black_words_between_white_ones_are_erased_to_the_paper():
    page = _cover()
    white_before = _white(_pixels(page, page.rect)).sum()
    word_area = fitz.Rect(60, 110, 330, 190)

    erased = erase_ink(page, WORDS)

    assert erased is not None
    assert _black(_pixels(page, word_area)).sum() == 0
    # The white words are kept whole.
    assert _white(_pixels(page, page.rect)).sum() >= 0.98 * white_before
    # The ink and paper are the drawing's own: black words on green.
    assert max(erased.ink) < 0.2
    assert np.allclose(erased.background, GREEN, atol=0.05)


def test_a_drawings_thin_outline_is_kept():
    page = _cover()
    outline = fitz.Rect(365, 125, 435, 195)
    before = _black(_pixels(page, outline)).sum()

    erase_ink(page, WORDS)

    assert before > 0
    assert _black(_pixels(page, outline)).sum() >= 0.8 * before


def test_a_drawing_a_letter_stood_in_front_of_ends_in_its_own_outline():
    # The cover's D stood over the elf's grey body. Nothing behind it can be
    # known: the body ends where the letter was, closed with a black line,
    # and no grey is smeared into the green.
    src = fitz.open()
    page = src.new_page(width=500, height=300)
    page.draw_rect(page.rect, color=None, fill=GREEN)
    page.draw_rect(fitz.Rect(20, 110, 130, 220), color=(0, 0, 0), fill=(0.7, 0.7, 0.7), width=2)
    page.insert_text((100, 200), "DOOR", fontsize=90, fontname="hebo", color=(0, 0, 0))
    pix = page.get_pixmap(dpi=150)
    flat = fitz.open().new_page(width=500, height=300)
    flat.insert_image(flat.rect, stream=pix.tobytes("png"))

    assert erase_ink(flat, Rect(x0=98, y0=130, x1=330, y1=205)) is not None

    # Where the D was: green, the square's own grey (through the D's bowl)
    # or the black line -- not a blend of them (8% of it, blended).
    px = _pixels(flat, fitz.Rect(100, 105, 165, 225))
    blend = (
        (np.abs(px - np.array(GREEN) * 255).max(axis=2) > 40)
        & (np.abs(px - 178).max(axis=2) > 25)
        & (px.max(axis=2) > 70)
    )
    assert blend.mean() < 0.04


def test_words_on_a_photo_are_refused():
    # Noise everywhere: no one paper colour to fill with.
    src = fitz.open()
    page = src.new_page(width=500, height=300)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (300, 500, 3), dtype=np.uint8)
    page.insert_image(page.rect, pixmap=fitz.Pixmap(fitz.csRGB, 500, 300, noise.tobytes(), False))
    page.insert_text((60, 180), "FROM", fontsize=90, fontname="hebo", color=(0, 0, 0))
    before = _pixels(page, page.rect)

    assert erase_ink(page, WORDS) is None
    assert (_pixels(page, page.rect) == before).all()
