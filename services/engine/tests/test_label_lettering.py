"""A picture label is redrawn at its printed size, on its printed baseline.

The reader's box is the label's ink. For a word with no descenders that is
the capitals' height, about three quarters of the font size, so sizing from
it drew "Producers" a quarter too small and above its line.
"""
from __future__ import annotations

import pytest

from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextStyle
from ct_engine.reconstruct.builder import ASCENT_RATIO, _label_em_box, _label_lettering


@pytest.mark.parametrize(
    "text, ink_share",
    [
        ("Producers", 0.73),  # capital, no descender
        ("Sunlight gives energy", 0.94),  # capital and descenders
        ("ocean", 0.52),  # x-height only
        ("same", 0.52),
        ("agua", 0.73),  # x-height and a descender
        ("Time (hours)", 0.94),  # brackets reach below the line like a g
    ],
)
def test_size_is_the_ink_height_over_its_share_of_the_font(text, ink_share):
    size, _ = _label_lettering(text, 15.0)

    assert size == pytest.approx(15.0 / ink_share, rel=0.01)


class _Label:
    def __init__(self, text, bbox):
        self.flat_text, self.bbox, self.style = text, bbox, TextStyle()


def test_the_translation_sits_on_the_original_baseline():
    ink = Rect(x0=100, y0=200, x1=180, y1=216)  # "Producers": baseline at 216
    size, box = _label_em_box(_Label("Producers", ink))

    assert box.y0 + ASCENT_RATIO * size == pytest.approx(216.0)
    assert box.height >= size  # room for a whole line at that size
    assert (box.x0, box.x1) == (ink.x0, ink.x1)


def test_a_descender_moves_the_baseline_up_from_the_box_bottom():
    ink = Rect(x0=100, y0=200, x1=180, y1=219)  # "energy": the y's tail at 219
    size, box = _label_em_box(_Label("energy", ink))

    assert box.y0 + ASCENT_RATIO * size == pytest.approx(219.0 - 0.21 * size)


class _Tilted:
    def __init__(self, text, oriented):
        self.flat_text, self.oriented, self.style = text, oriented, TextStyle()


@pytest.mark.parametrize("angle", [0.0, 12.0, -20.0])
def test_a_tilted_label_is_sized_and_placed_like_a_level_one(angle):
    import math

    from ct_engine.model.objects import OrientedBox
    from ct_engine.reconstruct.builder import _label_em_frame

    ob = OrientedBox(cx=300, cy=400, width=120, height=16, angle=angle)
    size, frame = _label_em_frame(_Tilted("Collection Pond", ob), angle)

    level, box = _label_em_box(_Label("Collection Pond", Rect(x0=240, y0=392, x1=360, y1=408)))
    assert size == pytest.approx(level)
    assert frame.height == pytest.approx(box.height)
    # The frame moved across the line, never along it, and by as much as
    # the level box's centre moved down.
    t = math.radians(angle)
    dx, dy = frame.cx - ob.cx, frame.cy - ob.cy
    assert dx * math.cos(t) + dy * math.sin(t) == pytest.approx(0.0, abs=1e-9)
    assert -dx * math.sin(t) + dy * math.cos(t) == pytest.approx(box.center[1] - 400.0)
