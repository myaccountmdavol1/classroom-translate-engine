"""Fix layout's fitting: the box that holds the words at the largest size."""

from __future__ import annotations

from ct_engine.fonts.registry import REGISTRY
from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.fitting import fit_text, fit_with_room

TEXT = "Los detectores de humo están instalados en cada piso de la casa. " * 3
PAGE = Rect(x0=0, y0=0, x1=612, y1=792)
BOX = Rect(x0=72, y0=100, x1=272, y1=113)


def _font():
    return REGISTRY.resolve(TEXT)


def test_room_below_keeps_the_full_size():
    tight = fit_text(TEXT, BOX, _font(), base_size=11.0)
    assert tight.shrink_ratio < 1.0
    found = fit_with_room(TEXT, BOX, _font(), base_size=11.0, page_bounds=PAGE,
                          obstacles=[], room_right=0.0)
    assert found is not None
    box, fit = found
    assert fit.fits and fit.font_size == 11.0
    assert box.x1 == BOX.x1 and box.y1 > BOX.y1
    # Snug: no more than one spare line below the words.
    assert box.height <= fit.used_height + fit.leading


def test_room_beside_is_used_before_growing_tall():
    narrow = fit_with_room(TEXT, BOX, _font(), base_size=11.0, page_bounds=PAGE,
                           obstacles=[], room_right=0.0)
    wide = fit_with_room(TEXT, BOX, _font(), base_size=11.0, page_bounds=PAGE,
                         obstacles=[], room_right=300.0)
    assert wide[1].font_size == 11.0
    assert wide[0].x1 > BOX.x1
    assert wide[0].height < narrow[0].height


def test_never_grows_into_a_block_below():
    below = Rect(x0=72, y0=130, x1=272, y1=143)
    found = fit_with_room(TEXT, BOX, _font(), base_size=11.0, page_bounds=PAGE,
                          obstacles=[below], room_right=0.0)
    if found is not None:
        box, _ = found
        assert box.y1 <= below.y0


def test_no_room_gives_none():
    walls = [Rect(x0=72, y0=114, x1=272, y1=120)]
    assert fit_with_room("x " * 400, BOX, _font(), base_size=11.0, page_bounds=PAGE,
                         obstacles=walls, room_right=0.0) is None
