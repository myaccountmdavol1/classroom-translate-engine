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


# Lines: a band from each start to each end (every direction, and a
# zero-length one), each end kind at each weight, straight and curved.
LINE_POINTS = [
    ((10.0, 20.0), (110.0, 20.0)),
    ((110.0, 20.0), (10.0, 20.0)),
    ((50.0, 90.0), (50.0, 10.0)),
    ((50.0, 10.0), (50.0, 90.0)),
    ((0.0, 0.0), (30.0, 40.0)),
    ((30.0, 40.0), (0.0, 0.0)),
    ((5.0, 5.0), (5.0, 5.0)),
]
LINE_BENDS = [0.0, 0.25, -0.4]
LINE_KINDS = [None, *shapes.LINE_ENDS]


def _cmds(cmds) -> list:
    return [list(c) for c in cmds]


def _line_cases() -> dict:
    frames = []
    for start, end in LINE_POINTS:
        frame, angle = shapes.line_frame(start, end)
        s, e = shapes.line_ends(frame, angle)
        frames.append({"start": list(start), "end": list(end), "frame": list(frame),
                       "angle": angle, "ends": [list(s), list(e)]})
    parts = []
    for bend in LINE_BENDS:
        for weight in WEIGHTS:
            for i, kind in enumerate(LINE_KINDS):
                # Each kind at the end, with the next one at the start.
                start = LINE_KINDS[(i + 1) % len(LINE_KINDS)]
                stroke, heads = shapes.line_parts(120.0, weight, start, kind, bend)
                parts.append({
                    "length": 120.0, "weight": weight, "start": start, "end": kind, "bend": bend,
                    "stroke": _cmds(stroke), "heads": [[_cmds(c), f] for c, f in heads],
                })
    # Ends that meet on a short line: no stroke left between them.
    stroke, heads = shapes.line_parts(10.0, 1.0, "arrow", "arrow", 0.0)
    parts.append({"length": 10.0, "weight": 1.0, "start": "arrow", "end": "arrow", "bend": 0.0,
                  "stroke": _cmds(stroke), "heads": [[_cmds(c), f] for c, f in heads]})
    reach = []
    for shape, adjust in (("line", None), ("curve", None), ("curve", [-40000.0]), ("curve", [90000.0])):
        for kind in LINE_KINDS:
            reach.append({
                "shape": shape, "w": 120.0, "adjust": adjust, "ends": [kind, kind],
                "ink_reach": [shapes.ink_reach(shape, 120.0, shapes.LINE_BAND, adjust, wt, (kind, kind))
                              for wt in WEIGHTS],
            })
    return {"line_frames": frames, "line_parts": parts, "line_reach": reach}


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
    lines = _line_cases()
    if os.environ.get("CT_UPDATE_GOLDEN") == "1":
        def block(items):
            return "[\n  " + ",\n  ".join(json.dumps(c) for c in items) + "\n ]"
        body = ",\n ".join(f'"{k}": {block(v)}' for k, v in lines.items())
        GOLDEN.write_text(
            f'{{\n "weights": {json.dumps(WEIGHTS)},\n "cases": {block(cases)},\n {body}\n}}\n'
        )
    golden = json.loads(GOLDEN.read_text())
    assert golden["weights"] == WEIGHTS
    assert len(golden["cases"]) == len(cases)
    for want, got in zip(golden["cases"], cases):
        _close(want, got, f"{got['shape']} {got['w']}x{got['h']} {got['adjust']}")
    for key, items in lines.items():
        assert len(golden[key]) == len(items), key
        for i, (want, got) in enumerate(zip(golden[key], items)):
            _close(want, json.loads(json.dumps(got)), f"{key}[{i}]")


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


# --- Lines ------------------------------------------------------------------
def test_line_frame_and_ends_round_trip() -> None:
    for start, end in LINE_POINTS[:-1]:
        frame, angle = shapes.line_frame(start, end)
        assert math.isclose(frame[3] - frame[1], shapes.LINE_BAND)
        s, e = shapes.line_ends(frame, angle)
        for got, want in ((s, start), (e, end)):
            assert math.isclose(got[0], want[0], abs_tol=1e-9)
            assert math.isclose(got[1], want[1], abs_tol=1e-9)


def test_line_direction_is_its_angle() -> None:
    assert shapes.line_frame((0, 0), (10, 0))[1] is None
    assert shapes.line_frame((10, 0), (0, 0))[1] == 180.0
    # Up the page (y runs down) is -90.
    assert shapes.line_frame((0, 10), (0, 0))[1] == -90.0
    assert shapes.line_frame((0, 0), (0, 10))[1] == 90.0


@pytest.mark.parametrize("weight", WEIGHTS)
def test_a_filled_arrow_tip_is_the_end_and_the_line_stops_at_its_base(weight) -> None:
    stroke, (head,) = shapes.line_parts(100.0, weight, None, "arrow")
    s = shapes.head_size(weight)
    cmds, filled = head
    assert filled and cmds[0] == ("M", 100.0, 4.0) and cmds[-1] == ("Z",)
    assert stroke == [("M", 0.0, 4.0), ("L", 100.0 - s, 4.0)]
    assert all(c[1] <= 100.0 for c in cmds if c[0] != "Z")


def test_open_ends_are_stroked_and_the_line_stops_short() -> None:
    stroke, (vee,) = shapes.line_parts(100.0, 2.0, None, "open_arrow")
    assert vee[1] is False and vee[0][-1] != ("Z",) and vee[0][1] == ("L", 100.0, 4.0)
    assert stroke[-1] == ("L", 99.0, 4.0)
    stroke, (ring,) = shapes.line_parts(100.0, 2.0, "open_circle", None)
    assert ring[1] is False and ring[0][-1] == ("Z",)
    assert stroke[0] == ("M", 3.0, 4.0)


def test_a_curve_bulges_and_its_arrow_follows_it() -> None:
    stroke, (head,) = shapes.line_parts(100.0, 1.0, None, "arrow", 0.25)
    assert stroke[1][0] == "C"
    # Its middle sits a quarter of its length toward the band's top.
    mx, my = shapes.bend_point(100.0, 0.25)
    assert (mx, my) == (50.0, 4.0 - 25.0)
    cmds, _ = head
    tip, a, b = (cmds[0][1:], cmds[1][1:], cmds[2][1:])
    assert tip == (100.0, 4.0)
    # The base's middle is where the stroke stops, on the curve.
    base = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    assert math.isclose(base[0], stroke[1][5], abs_tol=1e-9)
    assert math.isclose(base[1], stroke[1][6], abs_tol=1e-9)
    # Coming down into its end, the arrow points down and right.
    assert base[1] < 4.0 and base[0] < 100.0


def test_line_ink_reach() -> None:
    assert shapes.ink_reach("line", 100, 8, None, 2.0) == 1.0
    assert shapes.ink_reach("line", 100, 8, None, 2.0, (None, "arrow")) == 3.0
    assert math.isclose(shapes.ink_reach("line", 100, 8, None, 1.0, ("square", None)), 6 * math.sqrt(2) / 2)
    assert math.isclose(shapes.ink_reach("curve", 100, 8, [25000.0], 2.0), 26.0)
