"""The shape geometry, held to the golden file the editor's copy is held to.

reconstruct/shapes.py and apps/web/src/lib/shapes.ts compute the same
outlines; both are checked against apps/web/src/lib/shapes.golden.json
(this file, and src/lib/shapes.test.mjs under `node --test`), so a change
to one copy without the other fails one suite. Run with
CT_UPDATE_GOLDEN=1 to write the file afresh from this copy after a
deliberate change -- then make the same change in shapes.ts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from ct_engine.reconstruct import shapes

GOLDEN = (
    Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "lib" / "shapes.golden.json"
)
SIZES = [(72.0, 72.0), (200.0, 50.0), (30.0, 120.0)]
# Each preset at its defaults and at two other adjusts (where it has any).
ADJUSTS: dict[str, list[list[float] | None]] = {
    "rect": [None],
    "roundRect": [None, [5000.0], [50000.0]],
    "ellipse": [None],
}
WEIGHTS = [0.5, 1.0, 6.0]


def _cases() -> list[dict]:
    out = []
    for name in shapes.SHAPES:
        for w, h in SIZES:
            for adjust in ADJUSTS[name]:
                out.append({
                    "shape": name,
                    "w": w,
                    "h": h,
                    "adjust": adjust,
                    "outline": [list(c) for c in shapes.outline(name, w, h, adjust)],
                    "text_area": list(shapes.text_area(name, w, h, adjust)),
                    "ink_reach": [shapes.ink_reach(name, w, h, adjust, wt) for wt in WEIGHTS],
                })
    # An unknown name (a later build's shape) is drawn as a rectangle.
    out.append({
        "shape": "star99",
        "w": 72.0,
        "h": 72.0,
        "adjust": None,
        "outline": [list(c) for c in shapes.outline("star99", 72.0, 72.0)],
        "text_area": list(shapes.text_area("star99", 72.0, 72.0)),
        "ink_reach": [shapes.ink_reach("star99", 72.0, 72.0, None, wt) for wt in WEIGHTS],
    })
    return out


def _close(a, b, path="") -> None:
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: {a!r} != {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)), f"{path}: {a!r} != {b!r}"
        assert math.isclose(a, b, abs_tol=1e-6), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_golden_file_matches_this_copy() -> None:
    cases = _cases()
    if os.environ.get("CT_UPDATE_GOLDEN") == "1":
        rows = ",\n  ".join(json.dumps(c) for c in cases)
        GOLDEN.write_text(f'{{\n "weights": {json.dumps(WEIGHTS)},\n "cases": [\n  {rows}\n ]\n}}\n')
    golden = json.loads(GOLDEN.read_text())
    assert golden["weights"] == WEIGHTS
    assert len(golden["cases"]) == len(cases)
    for want, got in zip(golden["cases"], cases):
        _close(want, got, f"{got['shape']} {got['w']}x{got['h']} {got['adjust']}")


def test_every_closed_outline_ends_with_z_and_no_segment_back() -> None:
    for name in shapes.SHAPES:
        cmds = shapes.outline(name, 100.0, 60.0)
        assert cmds[0][0] == "M" and cmds[-1] == ("Z",)
        # The last drawn point is not the start again by a straight line.
        assert cmds[-2][0] != "L" or cmds[-2][1:] != cmds[0][1:]


def test_rect_starts_top_left_and_runs_clockwise() -> None:
    assert shapes.outline("rect", 10, 20) == [
        ("M", 0.0, 0.0), ("L", 10.0, 0.0), ("L", 10.0, 20.0), ("L", 0.0, 20.0), ("Z",)
    ]


def test_ellipse_passes_through_the_side_middles() -> None:
    cmds = shapes.outline("ellipse", 200.0, 50.0)
    ends = [c[-2:] for c in cmds if c[0] in ("M", "C")]
    for want in [(0.0, 25.0), (100.0, 0.0), (200.0, 25.0), (100.0, 50.0), (0.0, 25.0)]:
        assert any(math.isclose(x, want[0], abs_tol=1e-9) and math.isclose(y, want[1], abs_tol=1e-9)
                   for x, y in ends), want


@pytest.mark.parametrize("name", ["rect", "roundRect", "ellipse"])
def test_text_area_lies_inside_the_frame(name) -> None:
    l, t, r, b = shapes.text_area(name, 120.0, 80.0)
    assert 0 <= l < r <= 120 and 0 <= t < b <= 80


def test_ink_reach() -> None:
    assert math.isclose(shapes.ink_reach("rect", 10, 10, None, 6.0), 3.0 * math.sqrt(2))
    assert shapes.ink_reach("ellipse", 10, 10, None, 6.0) == 3.0
    assert shapes.ink_reach("roundRect", 72, 72, None, 4.0) == 2.0
    # A rounded rectangle with no rounding has sharp corners again.
    assert math.isclose(shapes.ink_reach("roundRect", 72, 72, [0.0], 4.0), 2.0 * math.sqrt(2))
    assert shapes.ink_reach("rect", 10, 10, None, 0.0) == 0.0
