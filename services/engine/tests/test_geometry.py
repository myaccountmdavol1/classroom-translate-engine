"""Geometry is the foundation of coordinate fidelity; test it hard."""

from __future__ import annotations

import pytest

from ct_engine.model.geometry import Rect


def test_rect_basic_measures():
    r = Rect(x0=10, y0=20, x1=110, y1=70)
    assert r.width == 100 and r.height == 50
    assert r.area == 5000
    assert r.center == (60.0, 45.0)


@pytest.mark.parametrize(
    "coords",
    [(100, 200, 0, 0), (100, 0, 0, 200), (0, 200, 100, 0)],
)
def test_inverted_rects_are_normalized(coords):
    """Producers emit corners in any order; we must not store an inverted box."""
    r = Rect(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])
    assert r.as_tuple() == (0.0, 0.0, 100.0, 200.0)
    assert r.width > 0 and r.height > 0


def test_intersection_and_iou():
    a = Rect(x0=0, y0=0, x1=100, y1=100)
    b = Rect(x0=50, y0=50, x1=150, y1=150)
    assert a.intersection_area(b) == 2500
    assert a.iou(b) == pytest.approx(2500 / 17500)
    assert a.intersection(Rect(x0=200, y0=200, x1=300, y1=300)) is None
    assert a.iou(a) == 1.0


def test_overlap_fraction_is_asymmetric():
    """A small label fully covered by a big diagram is a 100% hit for the label."""
    small = Rect(x0=10, y0=10, x1=20, y1=20)
    big = Rect(x0=0, y0=0, x1=100, y1=100)
    assert small.overlap_fraction_of_self(big) == pytest.approx(1.0)
    assert big.overlap_fraction_of_self(small) == pytest.approx(0.01)


def test_contains_and_union():
    outer = Rect(x0=0, y0=0, x1=100, y1=100)
    inner = Rect(x0=10, y0=10, x1=20, y1=20)
    assert outer.contains(inner)
    assert not inner.contains(outer)
    assert outer.union(inner).as_tuple() == outer.as_tuple()


def test_normalized_round_trip():
    r = Rect(x0=61.2, y0=79.2, x1=306.0, y1=396.0)
    n = r.normalized(612, 792)
    assert n.x0 == pytest.approx(0.1)
    assert n.y0 == pytest.approx(0.1)
    assert n.x1 == pytest.approx(0.5)
    assert n.y1 == pytest.approx(0.5)


def test_bounding_of_many():
    rects = [Rect(x0=10, y0=10, x1=20, y1=20), Rect(x0=50, y0=5, x1=60, y1=40)]
    assert Rect.bounding(rects).as_tuple() == (10.0, 5.0, 60.0, 40.0)
    assert Rect.bounding([]) is None


def test_clamped_to_page_bounds():
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    overflowing = Rect(x0=-20, y0=700, x1=700, y1=900)
    c = overflowing.clamped_to(page)
    assert c.x0 == 0 and c.x1 == 612 and c.y1 == 792


def test_degenerate_detection():
    assert Rect(x0=10, y0=10, x1=10, y1=50).is_degenerate()
    assert not Rect(x0=10, y0=10, x1=40, y1=50).is_degenerate()
