"""Words on a panel stay on it.

A line widened into what looked like empty space ran past a slide's green
panel ("Revisión de la Academia Magma") and into the next one: the panel
was underneath the words, so nothing stopped them. Its edge is a wall.
"""
from __future__ import annotations

from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.fitting import INSIDE_MARGIN, available_growth, available_growth_right

PAGE = Rect(x0=0, y0=0, x1=810, y1=1012)
PANEL = Rect(x0=20, y0=263, x1=398, y1=609)
LINE = Rect(x0=60, y0=380, x1=330, y1=410)


def test_a_line_widens_only_to_its_panel_edge():
    room = available_growth_right(LINE, PAGE, [PANEL])
    assert LINE.x1 + room == PANEL.x1 - INSIDE_MARGIN


def test_a_block_grows_down_only_to_its_panel_edge():
    low = Rect(x0=60, y0=560, x1=330, y1=590)
    room = available_growth(low, PAGE, [PANEL])
    assert low.y1 + room == PANEL.y1 - INSIDE_MARGIN


def test_a_panel_beside_the_words_is_not_around_them():
    beside = Rect(x0=415, y0=263, x1=793, y1=609)
    assert available_growth_right(LINE, PAGE, [beside]) == beside.x0 - 2.0 - LINE.x1
