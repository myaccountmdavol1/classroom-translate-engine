"""Colour of redrawn raster text (regression: everything came out black).

OCR reports characters, not styling, so text read from a scan or an image
carries no colour. Defaulting to black destroyed any document whose text was
not black: a flyer with white and yellow text on a navy band came back as
black text, unreadable and with the design ruined.
"""

from __future__ import annotations

import fitz
import pytest

from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.builder import (
    ensure_contrast,
    estimate_background,
    estimate_ink_color,
)


def page_with_band(bg: tuple[float, float, float],
                   ink: tuple[float, float, float]) -> fitz.Document:
    """A page with a coloured band and text drawn on it in `ink`."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(0, 60, 612, 200), color=None, fill=bg, width=0)
    tw = fitz.TextWriter(page.rect)
    tw.append(fitz.Point(72, 130), "LEARN MORE ABOUT PACK 305", fontsize=28)
    tw.write_text(page, color=ink)
    return doc


NAVY = (0.09, 0.16, 0.42)
YELLOW = (0.98, 0.78, 0.09)
WHITE = (1.0, 1.0, 1.0)


def test_white_text_on_navy_is_sampled_as_light():
    doc = page_with_band(NAVY, WHITE)
    try:
        page = doc[0]
        box = Rect(x0=70, y0=105, x1=520, y1=140)
        background, _ = estimate_background(page, box)
        assert background is not None
        ink = estimate_ink_color(page, box, background)
        assert ink is not None, "no ink detected"
        # Must come back light, not black.
        assert sum(ink) / 3 > 0.6, f"sampled {ink}, expected a light colour"
    finally:
        doc.close()


def test_yellow_text_on_navy_keeps_its_hue():
    doc = page_with_band(NAVY, YELLOW)
    try:
        page = doc[0]
        box = Rect(x0=70, y0=105, x1=520, y1=140)
        background, _ = estimate_background(page, box)
        ink = estimate_ink_color(page, box, background)
        assert ink is not None
        r, g, b = ink
        # Yellow: red and green high, blue low.
        assert r > 0.5 and g > 0.4 and b < r
    finally:
        doc.close()


def test_black_text_on_white_is_sampled_as_dark():
    doc = page_with_band(WHITE, (0.0, 0.0, 0.0))
    try:
        page = doc[0]
        box = Rect(x0=70, y0=105, x1=520, y1=140)
        background, _ = estimate_background(page, box)
        ink = estimate_ink_color(page, box, background)
        assert ink is not None
        assert sum(ink) / 3 < 0.4
    finally:
        doc.close()


def test_blank_region_yields_no_ink():
    """No glyphs means no colour to infer; the caller must fall back."""
    doc = page_with_band(WHITE, (0.0, 0.0, 0.0))
    try:
        page = doc[0]
        empty = Rect(x0=70, y0=500, x1=520, y1=540)
        background, _ = estimate_background(page, empty)
        assert estimate_ink_color(page, empty, background or (1.0, 1.0, 1.0)) is None
    finally:
        doc.close()


# --- Contrast guarantee ---------------------------------------------------
def test_contrast_is_enforced_against_a_light_background():
    """Sampling can land too close to the background; never draw invisibly."""
    result = ensure_contrast((0.95, 0.95, 0.95), (1.0, 1.0, 1.0))
    assert result == (0.0, 0.0, 0.0)


def test_contrast_is_enforced_against_a_dark_background():
    result = ensure_contrast((0.05, 0.05, 0.08), NAVY)
    assert result == (1.0, 1.0, 1.0)


def test_sufficient_contrast_is_left_alone():
    assert ensure_contrast(WHITE, NAVY) == WHITE
    assert ensure_contrast((0.0, 0.0, 0.0), WHITE) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "ink,background",
    [(WHITE, NAVY), (YELLOW, NAVY), ((0.0, 0.0, 0.0), WHITE)],
)
def test_contrast_output_is_always_visible(ink, background):
    from ct_engine.reconstruct.builder import _relative_luminance

    chosen = ensure_contrast(ink, background)
    delta = abs(_relative_luminance(chosen) - _relative_luminance(background))
    assert delta >= 0.28


def test_each_colour_gets_its_own_text_writer():
    """A shared writer applies one colour to the whole page.

    Regression: rebuild_page used a single TextWriter, so every block was
    painted in whichever colour happened to be last.
    """
    import inspect

    from ct_engine.reconstruct import builder

    source = inspect.getsource(builder.rebuild_page)
    assert "writer_for" in source, "per-colour writers were removed"
    assert "writers.items()" in source
