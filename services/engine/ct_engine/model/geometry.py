"""Geometry primitives for the page object model.

Coordinate system contract (this is load-bearing -- read before changing):

  * Units are PDF points (1/72 inch), matching the source document. We never
    store pixel coordinates in the model; pixels are a rendering concern.
  * Origin is TOP-LEFT of the page, x increasing right, y increasing down.
    This matches PyMuPDF's default coordinate space and CSS/browser space, so
    overlays in the web viewer are a pure scale -- no axis flip, no drift.
    (Raw PDF content streams use bottom-left origin; the parser converts once,
    at the boundary, and nothing downstream needs to know.)
  * Coordinates are pre-rotation in the sense that they already reflect the
    page's visual appearance: PyMuPDF applies /Rotate for us. `Page.rotation`
    is retained for export fidelity, not for transforming these rects.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field, model_validator


class Rect(BaseModel):
    """An axis-aligned bounding box in page points, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _normalize_ordering(self) -> "Rect":
        # PDF producers emit rects in arbitrary corner order. Normalize so that
        # (x0, y0) is always the top-left corner; downstream geometry (overlap,
        # containment, area) assumes this and would silently return garbage on
        # an inverted rect rather than raising.
        if self.x1 < self.x0:
            lo, hi = self.x1, self.x0
            object.__setattr__(self, "x0", lo)
            object.__setattr__(self, "x1", hi)
        if self.y1 < self.y0:
            lo, hi = self.y1, self.y0
            object.__setattr__(self, "y0", lo)
            object.__setattr__(self, "y1", hi)
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def is_degenerate(self, epsilon: float = 0.01) -> bool:
        return self.width <= epsilon or self.height <= epsilon

    def intersection(self, other: "Rect") -> "Rect | None":
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return Rect(x0=x0, y0=y0, x1=x1, y1=y1)

    def intersection_area(self, other: "Rect") -> float:
        hit = self.intersection(other)
        return hit.area if hit else 0.0

    def iou(self, other: "Rect") -> float:
        """Intersection over union. 0.0 when disjoint, 1.0 when identical."""
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction_of_self(self, other: "Rect") -> float:
        """How much of *this* rect is covered by `other` (0.0 - 1.0).

        Asymmetric on purpose: a tiny label sitting on a large diagram is a
        100% collision for the label and a 2% collision for the diagram, and
        the QA layer cares about the former.
        """
        if self.area <= 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def contains(self, other: "Rect", tolerance: float = 0.5) -> bool:
        return (
            self.x0 - tolerance <= other.x0
            and self.y0 - tolerance <= other.y0
            and self.x1 + tolerance >= other.x1
            and self.y1 + tolerance >= other.y1
        )

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def expanded(self, dx: float, dy: float | None = None) -> "Rect":
        dy = dx if dy is None else dy
        return Rect(x0=self.x0 - dx, y0=self.y0 - dy, x1=self.x1 + dx, y1=self.y1 + dy)

    def clamped_to(self, bounds: "Rect") -> "Rect":
        return Rect(
            x0=min(max(self.x0, bounds.x0), bounds.x1),
            y0=min(max(self.y0, bounds.y0), bounds.y1),
            x1=min(max(self.x1, bounds.x0), bounds.x1),
            y1=min(max(self.y1, bounds.y0), bounds.y1),
        )

    def normalized(self, page_width: float, page_height: float) -> "NormRect":
        """Express this rect as page-relative fractions (0.0 - 1.0).

        Normalized coordinates survive rescaling, so the web viewer can place
        overlays at any zoom level without knowing the point dimensions.
        """
        pw = page_width or 1.0
        ph = page_height or 1.0
        return NormRect(
            x0=self.x0 / pw, y0=self.y0 / ph, x1=self.x1 / pw, y1=self.y1 / ph
        )

    def scaled(self, factor: float) -> "Rect":
        return Rect(
            x0=self.x0 * factor,
            y0=self.y0 * factor,
            x1=self.x1 * factor,
            y1=self.y1 * factor,
        )

    @classmethod
    def from_tuple(cls, t: Iterable[float]) -> "Rect":
        x0, y0, x1, y1 = list(t)[:4]
        return cls(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def bounding(cls, rects: Iterable["Rect"]) -> "Rect | None":
        acc: Rect | None = None
        for r in rects:
            acc = r if acc is None else acc.union(r)
        return acc


class NormRect(BaseModel):
    """Page-relative bounding box, each component in [0, 1]."""

    x0: float = Field(ge=-0.5, le=1.5)
    y0: float = Field(ge=-0.5, le=1.5)
    x1: float = Field(ge=-0.5, le=1.5)
    y1: float = Field(ge=-0.5, le=1.5)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class Point(BaseModel):
    x: float
    y: float
