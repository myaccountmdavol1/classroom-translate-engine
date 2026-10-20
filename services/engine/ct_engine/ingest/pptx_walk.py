"""The single authoritative walk over a PPTX package's text.

Same job as `docx_walk`, and the same reasons for existing, but DrawingML is
a different dialect and the differences are not cosmetic:

  * Run formatting lives in ATTRIBUTES of `a:rPr` (`b="1" sz="1800"`), not in
    child elements. A fingerprint built the WordprocessingML way would see
    every run as identical and merge text that looks nothing alike.
  * A hyperlink is `a:hlinkClick` INSIDE the run's properties, not a wrapper
    around the run. Emptying such a run therefore leaves a link with no
    clickable text, so link runs are kept as their own unit -- the same
    conclusion as DOCX, reached from the opposite structure.
  * `a:fld` holds generated text: slide numbers, dates. It has an `a:t` like
    a run does, and rewriting it would replace a live field with a frozen
    string that stops updating. Fields are read but never written.

ANCHOR FORMAT

    <part name>#<paragraph ordinal in that part>#<run-group ordinal>

as in `docx_walk`, and for the same reason: the parser and the writer run
this identical walk, so an ordinal cannot mean two different things.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from lxml import etree

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# Attributes of a:rPr that say nothing about appearance. Spell-check and
# language bookkeeping churn constantly and would fragment every paragraph.
_IGNORED_RPR_ATTRS = frozenset(
    {"lang", "altLang", "dirty", "smtClean", "smtId", "err", "noProof"}
)


def run_text(r: etree._Element) -> str:
    """Characters this run contributes."""
    t = r.find(f"{A}t")
    return t.text or "" if t is not None else ""


def _is_field(el: etree._Element) -> bool:
    return el.tag == f"{A}fld"


def format_key(r: etree._Element) -> tuple:
    """A fingerprint of a run's visible formatting.

    Attributes carry bold, italic, underline and size; children carry fill,
    typeface and the hyperlink. Both matter, so both are included -- the
    children as their serialized form, which is exact and cheap.
    """
    rpr = r.find(f"{A}rPr")
    if rpr is None:
        return ()
    attrs = tuple(
        sorted(
            (k, v) for k, v in rpr.attrib.items() if k not in _IGNORED_RPR_ATTRS
        )
    )
    kids = tuple(
        etree.tostring(child)
        for child in rpr
        if child.tag != f"{A}hlinkClick"
    )
    return (attrs, kids)


def is_link_run(r: etree._Element) -> bool:
    rpr = r.find(f"{A}rPr")
    return rpr is not None and rpr.find(f"{A}hlinkClick") is not None


@dataclass
class RunGroup:
    runs: list[etree._Element]
    key: tuple
    in_link: bool = False

    @property
    def text(self) -> str:
        return "".join(run_text(r) for r in self.runs)

    @property
    def lead(self) -> etree._Element:
        return self.runs[0]


def own_runs(p: etree._Element) -> list[etree._Element]:
    """This paragraph's `a:r` children, in order.

    Direct children only. DrawingML does not nest paragraphs inside runs the
    way WordprocessingML does, so there is nothing to descend into -- but
    `a:fld` and `a:br` sit alongside the runs and are deliberately excluded:
    a field is generated text, and a break carries no characters.
    """
    return [child for child in p if child.tag == f"{A}r"]


def group_runs(runs: list[etree._Element]) -> list[RunGroup]:
    groups: list[RunGroup] = []
    for r in runs:
        if r.find(f"{A}t") is None:
            continue
        link = is_link_run(r)
        key = format_key(r)
        if groups and groups[-1].key == key and groups[-1].in_link == link and not link:
            groups[-1].runs.append(r)
        else:
            groups.append(RunGroup(runs=[r], key=key, in_link=link))
    return groups


_LABEL_RE = re.compile(
    r"^\s*(?:\d{1,3}\s*[.)]|\(?[A-Da-d]\s*[.)]|[A-Z][A-Za-z ]{0,20}:)\s*$"
)
MAX_LABEL_CHARS = 24


def _is_label(text: str) -> bool:
    return len(text) <= MAX_LABEL_CHARS and bool(_LABEL_RE.match(text))


@dataclass
class Unit:
    groups: list[RunGroup]
    collapsed: bool = False
    is_link: bool = False

    @property
    def text(self) -> str:
        return "".join(g.text for g in self.groups)


def _split_plain(groups: list[RunGroup]) -> list[Unit]:
    if not groups:
        return []
    if len(groups) == 1:
        return [Unit(groups=[groups[0]])]
    if len(groups) == 2 and _is_label(groups[0].text) and groups[1].text.strip():
        return [Unit(groups=[groups[0]]), Unit(groups=[groups[1]])]
    return [Unit(groups=list(groups), collapsed=True)]


def split_paragraph(groups: list[RunGroup]) -> list[Unit]:
    """Map a paragraph's runs onto translation units.

    Identical policy to `docx_walk.split_paragraph`, and identically
    evidence-based: marking run boundaries with placeholders so the
    translator carries them was measured and rejected, because the words
    beside a marker stop being translated.
    """
    if not groups:
        return []
    if any(g.in_link for g in groups):
        units: list[Unit] = []
        span: list[RunGroup] = []
        for g in groups:
            if g.in_link:
                units.extend(_split_plain(span))
                span = []
                units.append(Unit(groups=[g], is_link=True))
            else:
                span.append(g)
        units.extend(_split_plain(span))
        return [u for u in units if u.groups]
    return _split_plain(groups)


@dataclass
class ParaRef:
    part_name: str
    ordinal: int
    element: etree._Element
    # Shape this paragraph is drawn in, for geometry and reading order.
    shape: etree._Element | None = None
    origin: str = "body"
    units: list[Unit] = field(default_factory=list)

    def anchor(self, unit_index: int) -> str:
        return f"{self.part_name}#{self.ordinal}#{unit_index}"


def iter_part_paragraphs(
    root: etree._Element, part_name: str, *, origin_override: str | None = None
) -> Iterator[ParaRef]:
    """Every `a:p` in one part, in document order, numbered from zero."""
    for ordinal, p in enumerate(root.iter(f"{A}p")):
        ref = ParaRef(
            part_name=part_name,
            ordinal=ordinal,
            element=p,
            shape=owning_shape(p),
            origin=origin_override or _origin_of(p),
        )
        ref.units = split_paragraph(group_runs(own_runs(p)))
        yield ref


def owning_shape(p: etree._Element) -> etree._Element | None:
    """The shape that draws this paragraph, found by walking up.

    Deliberately not a paragraph-to-shape index built by iterating downward:
    lxml creates element proxies on demand, so the object identity of a node
    is not stable between two traversals of the same tree. An index keyed on
    `id()` silently missed every lookup.
    """
    node = p.getparent()
    while node is not None:
        if node.tag in (f"{P}sp", f"{P}graphicFrame", f"{P}pic"):
            return node
        node = node.getparent()
    return None


def _origin_of(p: etree._Element) -> str:
    node = p.getparent()
    while node is not None:
        if node.tag == f"{A}tc":
            return "table"
        node = node.getparent()
    return "body"


def shape_bounds(shape: etree._Element | None) -> tuple[float, float, float, float] | None:
    """Shape position and size in POINTS, or None when it is inherited.

    PowerPoint omits `a:xfrm` on a placeholder that takes its position from
    the slide layout. Inventing a box in that case would be a guess, so this
    returns None and the caller falls back to a flow position it labels as
    such.
    """
    if shape is None:
        return None
    # A `p:sp` carries `a:xfrm` inside `a:spPr`; a `p:graphicFrame` (which is
    # what a table is) carries `p:xfrm` in the presentation namespace. Both
    # are real geometry, and looking for only one of them silently lost every
    # table on every slide.
    xfrm = shape.find(f".//{A}xfrm")
    if xfrm is None:
        xfrm = shape.find(f"{P}xfrm")
    if xfrm is None:
        return None
    off = xfrm.find(f"{A}off")
    ext = xfrm.find(f"{A}ext")
    if off is None or ext is None:
        return None
    try:
        # 914400 EMU to the inch, 72 points to the inch.
        x = float(off.get("x", 0)) / 12700.0
        y = float(off.get("y", 0)) / 12700.0
        cx = float(ext.get("cx", 0)) / 12700.0
        cy = float(ext.get("cy", 0)) / 12700.0
    except (TypeError, ValueError):
        return None
    if cx <= 0 or cy <= 0:
        return None
    return (x, y, x + cx, y + cy)


def _ph_element(shape: etree._Element | None) -> etree._Element | None:
    if shape is None:
        return None
    return shape.find(f".//{P}nvSpPr/{P}nvPr/{P}ph")


def placeholder_type(shape: etree._Element | None) -> str | None:
    """"title", "body", "ctrTitle" etc., or None for an ordinary shape."""
    ph = _ph_element(shape)
    return None if ph is None else (ph.get("type") or "body")


def placeholder_key(shape: etree._Element | None) -> tuple[str, str] | None:
    """(type, idx) identifying which layout placeholder a shape fills.

    PowerPoint omits geometry on a placeholder that has not been moved, so
    this is how its real position is looked up on the slide layout. `idx` is
    what distinguishes two body placeholders on a two-column layout; type
    alone would put both halves in the same box.
    """
    ph = _ph_element(shape)
    if ph is None:
        return None
    return (ph.get("type") or "body", ph.get("idx") or "0")


def layout_placeholder_bounds(
    layout_root: etree._Element,
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Every placeholder box defined by a slide layout or master."""
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for shape in layout_root.iter(f"{P}sp"):
        key = placeholder_key(shape)
        bounds = shape_bounds(shape)
        if key and bounds:
            out[key] = bounds
    return out


def paragraph_size_pt(p: etree._Element, default: float) -> float:
    """Point size of a paragraph's first sized run, else the default."""
    for r in p.iter(f"{A}rPr"):
        sz = r.get("sz")
        if sz:
            try:
                return float(sz) / 100.0  # hundredths of a point
            except ValueError:
                pass
    return default


def paragraph_alignment(p: etree._Element) -> str | None:
    ppr = p.find(f"{A}pPr")
    return ppr.get("algn") if ppr is not None else None


def parse_xml(data: bytes) -> etree._Element:
    """Parse one part of the package. See `docx_walk.parse_xml` for why
    entity resolution and network access are both off."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser)


def table_cell_bounds(
    shape: etree._Element | None, p: etree._Element
) -> tuple[float, float, float, float] | None:
    """The exact box of the table cell this paragraph is in, in points.

    A table's geometry is fully stated: the frame gives the origin, the grid
    gives the column widths and each row gives its height. So cells get real
    coordinates rather than a share of the frame -- which matters, because a
    grid of cells stacked vertically would be wrong on every slide that has
    a table.
    """
    frame = shape_bounds(shape)
    if shape is None or frame is None:
        return None
    tc = p.getparent()
    while tc is not None and tc.tag != f"{A}tc":
        tc = tc.getparent()
    if tc is None:
        return None
    row = tc.getparent()
    tbl = row.getparent() if row is not None else None
    if row is None or tbl is None or tbl.tag != f"{A}tbl":
        return None

    grid = tbl.find(f"{A}tblGrid")
    if grid is None:
        return None
    widths = [_emu_pt(c.get("w")) for c in grid.findall(f"{A}gridCol")]
    rows = tbl.findall(f"{A}tr")
    heights = [_emu_pt(r.get("h")) for r in rows]
    cells = row.findall(f"{A}tc")
    try:
        col_index = cells.index(tc)
        row_index = rows.index(row)
    except ValueError:
        return None
    if col_index >= len(widths) or row_index >= len(heights):
        return None

    # Declared sizes are minimums; PowerPoint scales the table to the frame.
    total_w = sum(widths) or 1.0
    total_h = sum(heights) or 1.0
    fx = (frame[2] - frame[0]) / total_w
    fy = (frame[3] - frame[1]) / total_h
    x0 = frame[0] + sum(widths[:col_index]) * fx
    y0 = frame[1] + sum(heights[:row_index]) * fy
    return (x0, y0, x0 + widths[col_index] * fx, y0 + heights[row_index] * fy)


def _emu_pt(value: str | None) -> float:
    try:
        return max(0.0, float(value) / 12700.0) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
