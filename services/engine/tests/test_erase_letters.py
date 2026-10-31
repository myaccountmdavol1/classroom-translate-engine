"""Words on a busy picture background are erased letter by letter.

A flat patch only works on plain paper, so words on a shaded label, a
gradient or a photo used to be left in English. `erase_letters` erases just
the letters' pixels and fills them from the background around them -- and
refuses when the fill would borrow from other ink (a neighbouring word, a
pointer line), which would smear it into the gap.
"""
from __future__ import annotations

import fitz
import numpy as np

from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.erase import erase_letters

LABEL = fitz.Rect(100, 100, 400, 160)


def _picture_page(draw) -> fitz.Page:
    """A page holding one picture, drawn by `draw` and flattened to pixels."""
    src = fitz.open()
    page = src.new_page(width=500, height=300)
    draw(page)
    pix = page.get_pixmap(dpi=150)
    out = fitz.open()
    flat = out.new_page(width=500, height=300)
    flat.insert_image(flat.rect, stream=pix.tobytes("png"))
    return flat


def _gradient(page: fitz.Page) -> None:
    # A navy-to-teal label, in thin vertical strips.
    steps = 60
    for i in range(steps):
        t = i / (steps - 1)
        x0 = LABEL.x0 + LABEL.width * i / steps
        x1 = LABEL.x0 + LABEL.width * (i + 1) / steps + 0.5
        page.draw_rect(fitz.Rect(x0, LABEL.y0, x1, LABEL.y1),
                       color=None, fill=(0.08, 0.12 + 0.3 * t, 0.36 + 0.2 * t))


def _word_box(page: fitz.Page) -> Rect:
    page.insert_text((130, 140), "Collection Pond", fontsize=22, color=(1, 1, 1))
    return Rect(x0=128, y0=122, x1=310, y1=146)


def _pixels(page: fitz.Page, clip: fitz.Rect) -> np.ndarray:
    pix = page.get_pixmap(clip=clip, dpi=150, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(int)


def test_white_words_on_a_gradient_label_are_erased():
    boxes: list[Rect] = []
    page = _picture_page(lambda p: (_gradient(p), boxes.append(_word_box(p))))

    erased = erase_letters(page, boxes)

    assert erased is not None
    # The white letters are gone: nothing in the word box is near white.
    after = _pixels(page, fitz.Rect(128, 122, 310, 146))
    assert after.min(axis=2).max() < 140
    # The ink is the white the words were printed in; the background is the label's.
    assert min(erased.ink) > 0.85
    assert max(erased.background) < 0.7


def test_the_label_outside_the_letters_is_unchanged():
    boxes: list[Rect] = []
    page = _picture_page(lambda p: (_gradient(p), boxes.append(_word_box(p))))
    outside = fitz.Rect(320, 105, 395, 155)
    before = _pixels(page, outside)

    erase_letters(page, boxes)

    assert np.abs(_pixels(page, outside) - before).max() <= 2


def test_words_crowded_by_other_ink_are_refused():
    # A pointer line runs through the word, then on across the label: filling
    # the letters from around them would smear the line into the gap.
    boxes: list[Rect] = []

    def draw(p):
        _gradient(p)
        boxes.append(_word_box(p))
        p.draw_line((105, 132), (395, 132), color=(1, 1, 1), width=2)

    page = _picture_page(draw)
    box = fitz.Rect(128, 122, 310, 146)
    before = _pixels(page, box)

    assert erase_letters(page, boxes) is None
    assert np.array_equal(_pixels(page, box), before)


def test_a_box_with_no_letters_in_it_is_refused():
    page = _picture_page(_gradient)

    assert erase_letters(page, [Rect(x0=128, y0=122, x1=310, y1=146)]) is None


def _word_page(fontname: str) -> tuple[fitz.Page, Rect]:
    src = fitz.open()
    page = src.new_page(width=500, height=300)
    _gradient(page)
    page.insert_text((130, 140), "Producers", fontname=fontname, fontsize=22, color=(1, 1, 1))
    # The word's ink: capitals' tops to the baseline, as the reader reports it.
    ink = fitz.Rect(page.search_for("Producers")[0])
    pix = page.get_pixmap(dpi=150)
    out = fitz.open()
    flat = out.new_page(width=500, height=300)
    flat.insert_image(flat.rect, stream=pix.tobytes("png"))
    return flat, Rect(x0=ink.x0, y0=140 - 22 * 0.72, x1=ink.x1, y1=140)


def test_bold_letters_measure_thicker_than_regular_ones():
    from ct_engine.reconstruct.builder import BOLD_STROKE

    regular = erase_letters(*_one(_word_page("helv")))
    bold = erase_letters(*_one(_word_page("hebo")))

    assert regular is not None and bold is not None
    assert regular.stroke < BOLD_STROKE * 22 < bold.stroke


def _one(found):
    page, box = found
    return page, [box]
