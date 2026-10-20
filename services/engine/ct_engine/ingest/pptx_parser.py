"""PPTX ingest (spec sec. 8, phase 8).

WHERE THIS SITS BETWEEN THE OTHER TWO PARSERS

A PDF states where every glyph is. A DOCX states almost nothing. A PPTX is
in between, and the split matters:

  * Position IS real. Every shape has a box, in EMU, and PowerPoint honours
    it exactly. So slides get true coordinates -- `has_fixed_layout` is
    True -- and one slide is one page, which makes the complete-document
    guarantee exact rather than approximate.
  * Line layout is NOT. PowerPoint wraps and positions text inside a shape
    when it opens the file, using its own metrics. So text is written back
    into the original runs, exactly as in DOCX, and never redrawn.

That combination creates the one problem PPTX has and DOCX does not.
PowerPoint will not grow a shape or push the next one down, so a translation
that runs 25% longer than its English source simply overflows its box. That
is handled at write-back time by `reconstruct.pptx_builder`, which asks
PowerPoint to shrink the text the way PowerPoint itself would.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from ..model.document import DocumentMetadata, Page
from ..model.enums import (
    ObjectKind,
    Orientation,
    PageTextMode,
    SemanticRole,
    TextAlignment,
    TextSource,
)
from ..model.geometry import NormRect, Rect
from ..model.objects import TextLine, TextObject, TextSpan, TextStyle
from ..util.logging import get_logger
from .base import DocumentParser, OpenDocument, PageParseError, ParseError
from .pptx_walk import (
    A,
    P,
    R,
    ParaRef,
    iter_part_paragraphs,
    layout_placeholder_bounds,
    paragraph_alignment,
    paragraph_size_pt,
    parse_xml,
    placeholder_key,
    placeholder_type,
    shape_bounds,
    table_cell_bounds,
)

log = get_logger(__name__)

PKG_R = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_REL_SLIDE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
_REL_LAYOUT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"
)
_REL_MASTER = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster"
)
_REL_NOTES = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
)

# 4:3 at 72pt/inch, PowerPoint's fallback when the file says nothing.
_DEFAULT_W = 720.0
_DEFAULT_H = 540.0
_DEFAULT_FONT_PT = 18.0

_PH_ROLES = {
    "title": SemanticRole.TITLE,
    "ctrTitle": SemanticRole.TITLE,
    "subTitle": SemanticRole.HEADING,
    "ftr": SemanticRole.FOOTER,
    "hdr": SemanticRole.HEADER,
    "sldNum": SemanticRole.PAGE_NUMBER,
    "dt": SemanticRole.FOOTER,
}

_ALIGNMENTS = {
    "l": TextAlignment.LEFT,
    "ctr": TextAlignment.CENTER,
    "r": TextAlignment.RIGHT,
    "just": TextAlignment.JUSTIFY,
    "justLow": TextAlignment.JUSTIFY,
    "dist": TextAlignment.JUSTIFY,
}


@dataclass
class _Slide:
    index: int
    part: str
    notes_part: str | None
    # Placeholder boxes inherited from the layout, then the master.
    inherited: dict[tuple[str, str], tuple[float, float, float, float]]
    # Point size the master sets for title / body / other text, used when a
    # run states no size of its own -- which is the normal case.
    default_sizes: dict[str, float]


class OpenPptx(OpenDocument):
    """An opened .pptx package, presented one slide at a time."""

    # Rendering a slide means laying out and drawing DrawingML, which is
    # PowerPoint's job. Declared false rather than attempted badly.
    supports_preview = False

    def __init__(self, data: bytes) -> None:
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ParseError(
                "That PowerPoint file could not be opened. It may be damaged.",
                diagnostic=repr(exc),
            ) from exc
        self._names = set(self._zip.namelist())
        if "ppt/presentation.xml" not in self._names:
            raise ParseError(
                "That file is a ZIP archive but not a PowerPoint presentation.",
                diagnostic="ppt/presentation.xml missing",
            )
        self._cache: dict[str, etree._Element] = {}
        try:
            self._presentation = parse_xml(self._zip.read("ppt/presentation.xml"))
        except etree.XMLSyntaxError as exc:
            raise ParseError(
                "That presentation could not be read. It may be damaged.",
                diagnostic=f"XMLSyntaxError: {exc}",
            ) from exc
        self._width, self._height = self._slide_size()
        self._slides = self._load_slides()

    # --- package plumbing -------------------------------------------------

    def _rels_for(self, part: str) -> dict[str, tuple[str, str]]:
        """rId -> (type, part name) for one part's relationships."""
        base, _, name = part.rpartition("/")
        rels_part = f"{base}/_rels/{name}.rels"
        if rels_part not in self._names:
            return {}
        root = parse_xml(self._zip.read(rels_part))
        out: dict[str, tuple[str, str]] = {}
        for rel in root.iter(f"{PKG_R}Relationship"):
            target = rel.get("Target") or ""
            if rel.get("TargetMode") == "External" or target.startswith("http"):
                continue
            out[rel.get("Id") or ""] = (rel.get("Type") or "", _resolve(base, target))
        return out

    def _xml(self, part: str) -> etree._Element | None:
        if part not in self._names:
            return None
        if part not in self._cache:
            self._cache[part] = parse_xml(self._zip.read(part))
        return self._cache[part]

    def _slide_size(self) -> tuple[float, float]:
        sz = self._presentation.find(f"{P}sldSz")
        if sz is None:
            return _DEFAULT_W, _DEFAULT_H
        try:
            return float(sz.get("cx", 0)) / 12700.0, float(sz.get("cy", 0)) / 12700.0
        except (TypeError, ValueError):
            return _DEFAULT_W, _DEFAULT_H

    def _load_slides(self) -> list[_Slide]:
        """Slides in presentation order, with their inherited geometry.

        Order comes from `p:sldIdLst`, not from the part names: slide7.xml is
        not necessarily the seventh slide, and sorting by filename reorders a
        deck whose slides were rearranged after they were created.
        """
        rels = self._rels_for("ppt/presentation.xml")
        slides: list[_Slide] = []
        lst = self._presentation.find(f"{P}sldIdLst")
        ids = list(lst) if lst is not None else []
        for index, sld in enumerate(ids):
            rid = sld.get(f"{R}id")
            rel = rels.get(rid or "")
            if not rel or rel[0] != _REL_SLIDE or rel[1] not in self._names:
                continue
            part = rel[1]
            slide_rels = self._rels_for(part)
            notes = next(
                (t for typ, t in slide_rels.values() if typ == _REL_NOTES), None
            )
            slides.append(
                _Slide(
                    index=len(slides),
                    part=part,
                    notes_part=notes if notes in self._names else None,
                    inherited=self._inherited_bounds(part, slide_rels),
                    default_sizes=self._default_sizes(slide_rels),
                )
            )
        return slides

    def _inherited_bounds(
        self, slide_part: str, slide_rels: dict[str, tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[float, float, float, float]]:
        """Placeholder boxes from this slide's layout, then its master.

        PowerPoint omits geometry on a placeholder the author never moved, so
        without this every title and body on a deck would have no position at
        all. The layout wins over the master, which is the order PowerPoint
        resolves them in.
        """
        out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        layout_part = next(
            (t for typ, t in slide_rels.values() if typ == _REL_LAYOUT), None
        )
        if not layout_part:
            return out
        master_part = next(
            (
                t
                for typ, t in self._rels_for(layout_part).values()
                if typ == _REL_MASTER
            ),
            None,
        )
        for part in (master_part, layout_part):  # later wins
            root = self._xml(part) if part else None
            if root is not None:
                out.update(layout_placeholder_bounds(root))
        return out

    def _default_sizes(self, slide_rels: dict[str, tuple[str, str]]) -> dict[str, float]:
        """Point sizes the slide master sets for title, body and other text.

        Most runs state no size at all, inheriting it from here. Falling back
        to one fixed number instead would size a 44pt title like 18pt body
        text -- which matters, because the overflow estimate at write-back
        time is built on it.
        """
        sizes = {"title": 44.0, "body": _DEFAULT_FONT_PT, "other": _DEFAULT_FONT_PT}
        layout_part = next(
            (t for typ, t in slide_rels.values() if typ == _REL_LAYOUT), None
        )
        if not layout_part:
            return sizes
        master_part = next(
            (t for typ, t in self._rels_for(layout_part).values() if typ == _REL_MASTER),
            None,
        )
        root = self._xml(master_part) if master_part else None
        if root is None:
            return sizes
        for key, tag in (("title", "titleStyle"), ("body", "bodyStyle"),
                         ("other", "otherStyle")):
            el = root.find(f"{P}txStyles/{P}{tag}/{A}lvl1pPr/{A}defRPr")
            if el is not None and el.get("sz"):
                try:
                    sizes[key] = float(el.get("sz")) / 100.0
                except ValueError:
                    pass
        return sizes

    # --- OpenDocument interface -------------------------------------------

    @property
    def page_count(self) -> int:
        return len(self._slides)

    @property
    def paragraph_count(self) -> int:
        total = 0
        for slide in self._slides:
            for part in (slide.part, slide.notes_part):
                root = self._xml(part) if part else None
                if root is None:
                    continue
                total += sum(
                    1 for ref in iter_part_paragraphs(root, part) if ref.units
                )
        return total

    def metadata(self) -> DocumentMetadata:
        md = DocumentMetadata()
        root = self._xml("docProps/core.xml")
        if root is None:
            return md
        DC = "{http://purl.org/dc/elements/1.1/}"
        DCT = "{http://purl.org/dc/terms/}"
        CP = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"

        def text(tag: str) -> str | None:
            el = root.find(tag)
            return el.text if el is not None and el.text else None

        md.title = text(f"{DC}title")
        md.author = text(f"{DC}creator")
        md.subject = text(f"{DC}subject")
        md.keywords = text(f"{CP}keywords")
        md.creation_date = text(f"{DCT}created")
        md.modification_date = text(f"{DCT}modified")
        return md

    def _bounds_for(self, slide: _Slide, ref: ParaRef) -> tuple[float, float, float, float] | None:
        if ref.origin == "notes":
            # Speaker notes live in their own part and are not drawn on the
            # slide. Giving them the slide's body box would put a teacher's
            # private notes on top of what students see, in the review UI
            # and in every overlap check.
            return None
        cell = table_cell_bounds(ref.shape, ref.element)
        if cell:
            return cell
        direct = shape_bounds(ref.shape)
        if direct:
            return direct
        key = placeholder_key(ref.shape)
        if key is None:
            return None
        # Exact (type, idx) first; then type alone, because a master numbers
        # its footer placeholders differently from the layouts that use them.
        if key in slide.inherited:
            return slide.inherited[key]
        for (ph_type, _), box in slide.inherited.items():
            if ph_type == key[0]:
                return box
        return None

    def parse_page(self, index: int) -> Page:
        if not 0 <= index < len(self._slides):
            raise PageParseError(index, f"Slide {index + 1} does not exist.")
        slide = self._slides[index]

        try:
            objects: list[TextObject] = []
            collapsed: list[str] = []
            links: list[str] = []
            # Text with no resolvable box: stacked below the slide so it is
            # still reviewable and still translated, never dropped.
            unplaced_y = self._height + 12.0
            unplaced_any = False

            for part, origin in ((slide.part, None), (slide.notes_part, "notes")):
                root = self._xml(part) if part else None
                if root is None:
                    continue
                refs = [
                    r
                    for r in iter_part_paragraphs(root, part, origin_override=origin)
                    if any(u.text.strip() for u in r.units)
                ]
                # Grouped by shape so each paragraph can be given its own
                # slice of the shape's box. Every paragraph of a placeholder
                # otherwise reports the placeholder's whole box, and two
                # blocks with identical boxes look to QA like two blocks
                # printing on top of each other -- a warning on every slide,
                # about nothing.
                for shape_refs in _by_shape(refs):
                    box = self._bounds_for(slide, shape_refs[0])
                    sized = [
                        (
                            r,
                            paragraph_size_pt(
                                r.element,
                                slide.default_sizes.get(
                                    _size_bucket(placeholder_type(r.shape)),
                                    _DEFAULT_FONT_PT,
                                ),
                            ),
                        )
                        for r in shape_refs
                    ]
                    if box is None:
                        boxes = []
                        for _, size in sized:
                            boxes.append(
                                (36.0, unplaced_y, self._width - 36.0,
                                 unplaced_y + size * 1.3)
                            )
                            unplaced_y = boxes[-1][3] + 4.0
                            unplaced_any = True
                    else:
                        boxes = _stack_within(box, sized)

                    for (ref, size), sub in zip(sized, boxes):
                        # Units of one paragraph run along a line, so they
                        # share its height and divide its width. Giving them
                        # the same box would read to QA as two blocks
                        # printing over each other.
                        slices = _side_by_side(sub, ref)
                        for unit_index, unit in enumerate(ref.units):
                            text = unit.text
                            if not text.strip():
                                continue
                            part_box = slices[unit_index]
                            obj = _make_object(
                                ref, unit_index, text, size,
                                Rect(x0=part_box[0], y0=part_box[1],
                                     x1=part_box[2], y1=part_box[3]),
                            )
                            objects.append(obj)
                            if unit.collapsed:
                                collapsed.append(obj.id)
                            if unit.is_link:
                                links.append(obj.id)
        except Exception as exc:
            raise PageParseError(
                index,
                f"Slide {index + 1} could not be read.",
                diagnostic=f"{type(exc).__name__}: {exc}",
            ) from exc

        # Only grown when something genuinely had no box. A slide whose
        # shapes all resolved is exactly its own height.
        height = max(self._height, unplaced_y) if unplaced_any else self._height
        for obj in objects:
            obj.norm_bbox = _norm(obj.bbox, self._width, height)

        page = Page(
            index=index,
            number=index + 1,
            width=self._width,
            height=height,
            # A slide's shapes have real, honoured coordinates, so the review
            # UI can position overlays truthfully -- unlike a Word section.
            has_fixed_layout=True,
            orientation=(
                Orientation.LANDSCAPE if self._width > self._height
                else Orientation.PORTRAIT
            ),
            text_mode=PageTextMode.NATIVE if objects else PageTextMode.EMPTY,
            text_coverage=1.0 if objects else 0.0,
            text_objects=objects,
        )
        for i, obj in enumerate(objects):
            obj.reading_index = i
        page.recompute_diagnostics()
        if collapsed:
            page.diagnostics.extra["pptx_collapsed_formatting"] = collapsed
        if links:
            page.diagnostics.extra["pptx_link_fragments"] = links
        return page

    def render_page_png(
        self, index: int, *, dpi: int = 144, gray: bool = False
    ) -> bytes:
        raise ParseError(
            "PowerPoint slides do not have page images to preview.",
            diagnostic="rendering DrawingML is PowerPoint's job",
        )

    def render_region_png(
        self, index: int, bbox: tuple[float, float, float, float], *, dpi: int = 300
    ) -> bytes:
        raise ParseError(
            "PowerPoint slides do not have page images to preview.",
            diagnostic="rendering DrawingML is PowerPoint's job",
        )

    def extract_image_asset(self, xref: int) -> tuple[bytes, str] | None:
        """Never extracted: the whole package is carried through untouched."""
        return None

    @property
    def embedded_media_count(self) -> int:
        return sum(1 for n in self._names if n.startswith("ppt/media/"))

    def close(self) -> None:
        try:
            self._zip.close()
        except Exception:
            pass


def _side_by_side(
    box: tuple[float, float, float, float], ref: ParaRef
) -> list[tuple[float, float, float, float]]:
    """Divide a paragraph's box horizontally among its units."""
    if len(ref.units) <= 1:
        return [box]
    x0, y0, x1, y1 = box
    lengths = [max(1, len(u.text)) for u in ref.units]
    total = sum(lengths)
    out: list[tuple[float, float, float, float]] = []
    cursor = x0
    for n in lengths:
        w = (x1 - x0) * (n / total)
        out.append((cursor, y0, min(x1, cursor + w), y1))
        cursor += w
    return out


def _size_bucket(ph: str | None) -> str:
    return "title" if ph in ("title", "ctrTitle") else "body" if ph else "other"


def _by_shape(refs: list[ParaRef]) -> list[list[ParaRef]]:
    """Consecutive paragraphs drawn by the same shape.

    Compared by identity of the underlying node, not with `==`: lxml hands
    out a fresh proxy object per traversal, so two references to one shape
    are not the same Python object.
    """
    groups: list[list[ParaRef]] = []
    for ref in refs:
        # A table cell has its own exact box, so it is never pooled with a
        # sibling cell to be divided up -- that would stack a grid into a
        # column.
        if (
            groups
            and ref.origin != "table"
            and groups[-1][0].origin != "table"
            and _same_node(groups[-1][0].shape, ref.shape)
        ):
            groups[-1].append(ref)
        else:
            groups.append([ref])
    return groups


def _same_node(a: etree._Element | None, b: etree._Element | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.getroottree().getpath(a) == b.getroottree().getpath(b)


def _stack_within(
    box: tuple[float, float, float, float],
    sized: list[tuple[ParaRef, float]],
) -> list[tuple[float, float, float, float]]:
    """Divide a shape's box among its paragraphs, top to bottom.

    The ORDER is certain: PowerPoint stacks a text frame's paragraphs in
    document order. The heights are an estimate, from a line count at an
    average glyph width of 0.5 em. That is enough to give each block a
    distinct, roughly right region to select in the review UI, and it is not
    used to place anything -- write-back never touches coordinates.
    """
    x0, y0, x1, y1 = box
    width = max(24.0, x1 - x0)
    weights: list[float] = []
    for ref, size in sized:
        text = "".join(u.text for u in ref.units)
        per_line = max(1, int(width / (size * 0.5)))
        lines = max(1, -(-len(text) // per_line))
        weights.append(lines * size * 1.2)
    total = sum(weights) or 1.0
    available = max(size * 1.2, y1 - y0)
    out: list[tuple[float, float, float, float]] = []
    cursor = y0
    for w in weights:
        h = available * (w / total)
        out.append((x0, cursor, x1, min(y1, cursor + h)))
        cursor += h
    return out


def _resolve(base: str, target: str) -> str:
    """Resolve a relationship target against the part's own directory."""
    if target.startswith("/"):
        return target.lstrip("/")
    parts = [p for p in base.split("/") if p]
    for segment in target.split("/"):
        if segment == "..":
            if parts:
                parts.pop()
        elif segment not in ("", "."):
            parts.append(segment)
    return "/".join(parts)


def _norm(box: Rect, width: float, height: float) -> NormRect:
    return NormRect(
        x0=min(1.0, max(0.0, box.x0 / width)),
        y0=min(1.0, max(0.0, box.y0 / height)),
        x1=min(1.0, max(0.0, box.x1 / width)),
        y1=min(1.0, max(0.0, box.y1 / height)),
    )


def _make_object(
    ref: ParaRef, unit_index: int, text: str, size: float, box: Rect
) -> TextObject:
    ph = placeholder_type(ref.shape)
    role = _PH_ROLES.get(ph or "")
    if role is None and ref.origin == "table":
        role = SemanticRole.TABLE_CELL
    elif role is None and ref.origin == "notes":
        # Speaker notes are translated -- dropping them would lose a
        # teacher's own words -- but they are not what a student reads.
        role = SemanticRole.CAPTION

    style = TextStyle(font_size=size)
    return TextObject(
        id=f"pptx_{abs(hash(ref.anchor(unit_index))) & 0xFFFFFFFFFFFF:012x}",
        kind=ObjectKind.TEXT,
        bbox=box,
        norm_bbox=NormRect(x0=0, y0=0, x1=0, y1=0),  # filled once page height is known
        anchor=ref.anchor(unit_index),
        source=TextSource.NATIVE,
        role=role or SemanticRole.OTHER,
        role_confidence=0.9 if role else 0.0,
        alignment=_ALIGNMENTS.get(
            paragraph_alignment(ref.element) or "", TextAlignment.UNKNOWN
        ),
        style=style,
        lines=[TextLine(bbox=box, spans=[TextSpan(text=text, bbox=box, style=style)])],
    )


class PptxParser(DocumentParser):
    name = "pptx"
    supported_content_types = frozenset(
        {
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        }
    )
    supported_extensions = frozenset({"pptx"})

    def can_parse(self, content_type: str | None, filename: str | None) -> bool:
        ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
        if ext:
            return ext == "pptx"
        return (content_type or "").lower() in self.supported_content_types

    def open_path(self, path: Path) -> OpenPptx:
        return OpenPptx(path.read_bytes())

    def open_bytes(self, data: bytes) -> OpenPptx:
        return OpenPptx(data)
