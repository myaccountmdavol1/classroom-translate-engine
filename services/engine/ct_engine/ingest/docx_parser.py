"""DOCX ingest (spec sec. 8, phase 8).

HOW THIS DIFFERS FROM THE PDF PARSER, AND WHY IT HAS TO

A PDF states where every glyph sits. A DOCX states almost nothing about
position: Word lays the document out when it opens the file, using the fonts
and printer metrics of that machine. There is therefore no page geometry to
recover and no sense in which this parser could report one. Two consequences
run through the whole DOCX path:

  * Reconstruction edits the original package in place instead of redrawing
    at coordinates. Nothing is rasterized, no font has to be matched, and the
    exported file stays a real Word document a teacher can keep editing. The
    fitting, reflow and overlap machinery in `reconstruct/` is not merely
    unnecessary here -- it is inapplicable, and pages are marked
    `has_fixed_layout=False` so it never runs.
  * A "page" is a Word SECTION, not a sheet of paper. Word decides page
    breaks at open time and they change with the target language, so a
    page-for-page promise would be a fiction. The completeness guarantee that
    replaces it is exact and checkable: every paragraph in the source appears
    in the output, counted in `docx_paragraph_count`.

Object boxes are a synthesised reading-order flow (see `_FlowCursor`), not
measurements. They exist so the review UI can order and select blocks; they
are never used to place anything.
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
from .docx_walk import (
    W,
    ParaRef,
    assign_sections,
    iter_part_paragraphs,
    paragraph_alignment,
    paragraph_style,
    parse_xml,
)

log = get_logger(__name__)

R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_R = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_REL_HEADER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
_REL_FOOTER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"
_REL_FOOTNOTES = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
)
_REL_ENDNOTES = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
)

# Word's own fallback when a document specifies nothing: Letter, 1" margins,
# 11pt Calibri. Used only to give the flow something to measure against.
_DEFAULT_PAGE_W = 612.0
_DEFAULT_PAGE_H = 792.0
_DEFAULT_MARGIN = 72.0
_DEFAULT_FONT_PT = 11.0

# Twentieths of a point -- the unit of w:pgSz and w:pgMar.
def _twips(value: str | None, fallback: float) -> float:
    try:
        return abs(float(value)) / 20.0 if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


_STYLE_ROLES = {
    "Title": SemanticRole.TITLE,
    "Subtitle": SemanticRole.TITLE,
    "Heading1": SemanticRole.HEADING,
    "Heading2": SemanticRole.HEADING,
    "Heading3": SemanticRole.HEADING,
    "Heading4": SemanticRole.HEADING,
    "Heading5": SemanticRole.HEADING,
    "Heading6": SemanticRole.HEADING,
    "Caption": SemanticRole.CAPTION,
    "Quote": SemanticRole.READING_PASSAGE,
    "IntenseQuote": SemanticRole.READING_PASSAGE,
}

_ORIGIN_ROLES = {
    "header": SemanticRole.HEADER,
    "footer": SemanticRole.FOOTER,
    "table": SemanticRole.TABLE_CELL,
    "footnote": SemanticRole.CAPTION,
    "endnote": SemanticRole.CAPTION,
}

_ALIGNMENTS = {
    "left": TextAlignment.LEFT,
    "start": TextAlignment.LEFT,
    "center": TextAlignment.CENTER,
    "right": TextAlignment.RIGHT,
    "end": TextAlignment.RIGHT,
    "both": TextAlignment.JUSTIFY,
    "justify": TextAlignment.JUSTIFY,
}


@dataclass
class _Section:
    """One Word section: the unit this parser presents as a page."""

    index: int
    width: float
    height: float
    left: float
    right: float
    top: float
    bottom: float
    header_parts: list[str]
    footer_parts: list[str]

    @property
    def column_width(self) -> float:
        return max(72.0, self.width - self.left - self.right)


class _FlowCursor:
    """Assigns each block a box by stacking it down the text column.

    This is a reading-order layout, not a prediction of what Word will do.
    Line count is estimated from an average glyph width of 0.5 em, which is
    close enough to keep long paragraphs visually longer than short ones in
    the review list and is used for nothing else.
    """

    def __init__(self, section: _Section) -> None:
        self.section = section
        self.y = section.top

    def take(self, text: str, font_pt: float) -> Rect:
        size = font_pt if font_pt > 0 else _DEFAULT_FONT_PT
        width = max(36.0, self.section.column_width)
        per_line = max(1, int(width / (size * 0.5)))
        lines = max(1, -(-len(text) // per_line)) if text else 1
        leading = size * 1.2
        height = max(leading, lines * leading)
        x0 = self.section.left
        box = Rect(x0=x0, y0=self.y, x1=x0 + width, y1=self.y + height)
        # A blank line between blocks, matching Word's default paragraph
        # spacing closely enough to keep the review list readable.
        self.y = box.y1 + size * 0.35
        return box

    @property
    def used_height(self) -> float:
        return self.y + self.section.bottom


def _norm(box: Rect, width: float, height: float) -> NormRect:
    return NormRect(
        x0=min(1.0, max(0.0, box.x0 / width)),
        y0=min(1.0, max(0.0, box.y0 / height)),
        x1=min(1.0, max(0.0, box.x1 / width)),
        y1=min(1.0, max(0.0, box.y1 / height)),
    )


class OpenDocx(OpenDocument):
    """An opened .docx package, presented one section at a time."""

    # Word decides pagination when the file is opened, so there is nothing
    # here to rasterize. Declared rather than discovered so the analyze stage
    # can skip previews cleanly instead of catching an exception per page.
    supports_preview = False

    def __init__(self, data: bytes) -> None:
        self._data = data
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ParseError(
                "That Word file could not be opened. It may be damaged.",
                diagnostic=repr(exc),
            ) from exc
        self._names = set(self._zip.namelist())
        if "word/document.xml" not in self._names:
            raise ParseError(
                "That file is a ZIP archive but not a Word document.",
                diagnostic="word/document.xml missing",
            )
        self._rels = self._load_rels()
        try:
            self._body = parse_xml(self._zip.read("word/document.xml"))
        except etree.XMLSyntaxError as exc:
            raise ParseError(
                "That Word document could not be read. It may be damaged.",
                diagnostic=f"XMLSyntaxError: {exc}",
            ) from exc
        self._default_font_pt, self._style_sizes = self._load_style_sizes()
        self._sections = self._load_sections()
        self._paras_by_section = self._collect_paragraphs()

    # --- package plumbing -------------------------------------------------

    def _load_rels(self) -> dict[str, tuple[str, str]]:
        """rId -> (relationship type, part name) for document.xml."""
        name = "word/_rels/document.xml.rels"
        if name not in self._names:
            return {}
        root = parse_xml(self._zip.read(name))
        out: dict[str, tuple[str, str]] = {}
        for rel in root.iter(f"{PKG_R}Relationship"):
            target = rel.get("Target") or ""
            if rel.get("TargetMode") == "External" or target.startswith("http"):
                continue
            out[rel.get("Id") or ""] = (rel.get("Type") or "", f"word/{target.lstrip('/')}")
        return out

    def _load_style_sizes(self) -> tuple[float, dict[str, float]]:
        """Default point size, and the size each named style sets.

        Only one level is resolved: a style's own `w:sz`, not its inherited
        chain. Font size here only feeds the flow estimate and the review
        UI's type scale, so an unresolved `basedOn` costs nothing real.
        """
        default = _DEFAULT_FONT_PT
        sizes: dict[str, float] = {}
        if "word/styles.xml" not in self._names:
            return default, sizes
        root = parse_xml(self._zip.read("word/styles.xml"))
        dflt = root.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr/{W}sz")
        if dflt is not None and dflt.get(f"{W}val"):
            try:
                default = float(dflt.get(f"{W}val")) / 2.0  # half-points
            except ValueError:
                pass
        for st in root.iter(f"{W}style"):
            sid = st.get(f"{W}styleId")
            sz = st.find(f"{W}rPr/{W}sz")
            if sid and sz is not None and sz.get(f"{W}val"):
                try:
                    sizes[sid] = float(sz.get(f"{W}val")) / 2.0
                except ValueError:
                    pass
        return default, sizes

    def _load_sections(self) -> list[_Section]:
        body = self._body.find(f"{W}body")
        sect_prs: list[etree._Element] = []
        if body is not None:
            for child in body:
                if child.tag == f"{W}p":
                    sp = child.find(f"{W}pPr/{W}sectPr")
                    if sp is not None:
                        sect_prs.append(sp)
                elif child.tag == f"{W}sectPr":
                    sect_prs.append(child)
        if not sect_prs:
            sect_prs = [None]  # a document with no explicit sectPr is one section

        sections: list[_Section] = []
        for i, sp in enumerate(sect_prs):
            pg = sp.find(f"{W}pgSz") if sp is not None else None
            mar = sp.find(f"{W}pgMar") if sp is not None else None
            w = _twips(pg.get(f"{W}w") if pg is not None else None, _DEFAULT_PAGE_W)
            h = _twips(pg.get(f"{W}h") if pg is not None else None, _DEFAULT_PAGE_H)
            headers, footers = [], []
            if sp is not None:
                for ref in sp.iter(f"{W}headerReference"):
                    rel = self._rels.get(ref.get(f"{R}id") or "")
                    if rel and rel[0] == _REL_HEADER:
                        headers.append(rel[1])
                for ref in sp.iter(f"{W}footerReference"):
                    rel = self._rels.get(ref.get(f"{R}id") or "")
                    if rel and rel[0] == _REL_FOOTER:
                        footers.append(rel[1])
            sections.append(
                _Section(
                    index=i,
                    width=w,
                    height=h,
                    left=_twips(mar.get(f"{W}left") if mar is not None else None, _DEFAULT_MARGIN),
                    right=_twips(mar.get(f"{W}right") if mar is not None else None, _DEFAULT_MARGIN),
                    top=_twips(mar.get(f"{W}top") if mar is not None else None, _DEFAULT_MARGIN),
                    bottom=_twips(mar.get(f"{W}bottom") if mar is not None else None, _DEFAULT_MARGIN),
                    header_parts=headers,
                    footer_parts=footers,
                )
            )
        return sections

    def _collect_paragraphs(self) -> dict[int, list[ParaRef]]:
        """Every text-bearing paragraph in the package, bucketed by section.

        Ordered header -> body -> notes -> footer within a section, which is
        the order a reader meets them on the page.
        """
        body_refs = list(iter_part_paragraphs(self._body, "word/document.xml"))
        assign_sections(body_refs)

        buckets: dict[int, list[ParaRef]] = {s.index: [] for s in self._sections}

        def add(ref: ParaRef, section: int) -> None:
            buckets.setdefault(section, []).append(ref)

        for sec in self._sections:
            for part in sec.header_parts:
                for ref in self._part_refs(part, "header"):
                    add(ref, sec.index)

        for ref in body_refs:
            add(ref, min(ref.section, len(self._sections) - 1))

        # Notes are not section-bound in the file format. They are attached to
        # the first section so that they are translated and reviewed; nothing
        # in the output depends on which page they are listed under, because
        # the writer puts each one back in its own part.
        for rel_type, origin in ((_REL_FOOTNOTES, "footnote"), (_REL_ENDNOTES, "endnote")):
            for rid, (typ, part) in self._rels.items():
                if typ != rel_type or part not in self._names:
                    continue
                for ref in self._note_refs(part, origin):
                    add(ref, 0)

        for sec in self._sections:
            for part in sec.footer_parts:
                for ref in self._part_refs(part, "footer"):
                    add(ref, sec.index)

        return buckets

    def _part_refs(self, part: str, origin: str) -> list[ParaRef]:
        if part not in self._names:
            return []
        root = parse_xml(self._zip.read(part))
        return list(iter_part_paragraphs(root, part, origin_override=origin))

    def _note_refs(self, part: str, origin: str) -> list[ParaRef]:
        """Footnote/endnote paragraphs, minus Word's separator boilerplate."""
        if part not in self._names:
            return []
        root = parse_xml(self._zip.read(part))
        skip: set[etree._Element] = set()
        tag = f"{W}footnote" if origin == "footnote" else f"{W}endnote"
        for note in root.iter(tag):
            if note.get(f"{W}type") in ("separator", "continuationSeparator"):
                skip.update(note.iter(f"{W}p"))
        return [
            ref
            for ref in iter_part_paragraphs(root, part, origin_override=origin)
            if ref.element not in skip
        ]

    # --- OpenDocument interface -------------------------------------------

    @property
    def page_count(self) -> int:
        return len(self._sections)

    @property
    def paragraph_count(self) -> int:
        """Text-bearing paragraphs across the package.

        This is the number the completeness gate checks for DOCX, standing in
        for the page count a fixed-layout format would use.
        """
        return sum(
            1 for refs in self._paras_by_section.values() for r in refs if r.units
        )

    @property
    def embedded_media_count(self) -> int:
        """Pictures stored in the package.

        Reported so the translation report can state how many there are.
        They are never parsed into `image_objects`, because there is no
        layout to give them a position in -- and never extracted, because
        reconstruction leaves the whole package untouched apart from the
        characters inside runs. Preservation is therefore not something this
        engine achieves; it is something it cannot break.
        """
        return sum(1 for n in self._names if n.startswith("word/media/"))

    def metadata(self) -> DocumentMetadata:
        md = DocumentMetadata()
        name = "docProps/core.xml"
        if name not in self._names:
            return md
        try:
            root = parse_xml(self._zip.read(name))
        except etree.XMLSyntaxError:
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
        md.creator = text(f"{CP}lastModifiedBy")
        md.creation_date = text(f"{DCT}created")
        md.modification_date = text(f"{DCT}modified")
        return md

    def _style_for(self, ref: ParaRef, unit) -> TextStyle:
        """Typography of a unit, from its lead run and its paragraph style."""
        style_id = paragraph_style(ref.element)
        size = self._style_sizes.get(style_id or "", self._default_font_pt)
        lead = unit.groups[0].lead if unit.groups else None
        bold = italic = False
        color = None
        family = None
        if lead is not None:
            rpr = lead.find(f"{W}rPr")
            if rpr is not None:
                sz = rpr.find(f"{W}sz")
                if sz is not None and sz.get(f"{W}val"):
                    try:
                        size = float(sz.get(f"{W}val")) / 2.0
                    except ValueError:
                        pass
                bold = _toggle(rpr.find(f"{W}b"))
                italic = _toggle(rpr.find(f"{W}i"))
                col = rpr.find(f"{W}color")
                if col is not None and col.get(f"{W}val") not in (None, "auto"):
                    color = "#" + str(col.get(f"{W}val")).lstrip("#").lower()
                fonts = rpr.find(f"{W}rFonts")
                if fonts is not None:
                    family = fonts.get(f"{W}ascii") or fonts.get(f"{W}hAnsi")
        if (style_id or "").startswith("Heading") or style_id in ("Title", "Subtitle"):
            bold = True
        return TextStyle(
            font_family=family,
            font_name_raw=family,
            font_size=size,
            bold=bold,
            italic=italic,
            color=color,
        )

    def parse_page(self, index: int) -> Page:
        if not 0 <= index < len(self._sections):
            raise PageParseError(index, f"Section {index + 1} does not exist.")
        section = self._sections[index]
        refs = self._paras_by_section.get(index, [])

        try:
            cursor = _FlowCursor(section)
            objects: list[TextObject] = []
            collapsed: list[str] = []
            links: list[str] = []
            for ref in refs:
                for unit_index, unit in enumerate(ref.units):
                    text = unit.text
                    if not text.strip():
                        continue
                    style = self._style_for(ref, unit)
                    box = cursor.take(text, style.font_size)
                    obj = _make_object(
                        ref, unit_index, unit, text, style, box, section
                    )
                    objects.append(obj)
                    if unit.collapsed:
                        collapsed.append(obj.id)
                    if unit.is_link:
                        links.append(obj.id)
        except Exception as exc:  # parser contract: one section, one failure
            raise PageParseError(
                index,
                f"Section {index + 1} of that Word document could not be read.",
                diagnostic=f"{type(exc).__name__}: {exc}",
            ) from exc

        # The page is as tall as the flow needs. A Word section is not one
        # sheet of paper, so clipping to Letter height would push most blocks
        # off the bottom and make every normalised box meaningless.
        height = max(section.height, cursor.used_height)
        for obj in objects:
            obj.norm_bbox = _norm(obj.bbox, section.width, height)

        page = Page(
            index=index,
            number=index + 1,
            width=section.width,
            height=height,
            has_fixed_layout=False,
            orientation=(
                Orientation.LANDSCAPE if section.width > section.height
                else Orientation.PORTRAIT
            ),
            text_mode=PageTextMode.NATIVE if objects else PageTextMode.EMPTY,
            text_coverage=1.0 if objects else 0.0,
            is_scanned=False,
            needs_ocr=False,
            text_objects=objects,
        )
        for i, obj in enumerate(objects):
            obj.reading_index = i
        # Both of these are honest losses, so they are recorded rather than
        # hidden: QA turns them into reviewer-visible findings.
        if collapsed:
            page.diagnostics.extra["docx_collapsed_formatting"] = collapsed
        if links:
            page.diagnostics.extra["docx_link_fragments"] = links
        page.diagnostics.extra["docx_paragraphs"] = len(
            [r for r in refs if r.units]
        )
        page.recompute_diagnostics()
        return page

    def render_page_png(
        self, index: int, *, dpi: int = 144, gray: bool = False
    ) -> bytes:
        raise ParseError(
            "Word documents do not have page images to preview.",
            diagnostic="DOCX pagination is decided by Word at open time",
        )

    def render_region_png(
        self, index: int, bbox: tuple[float, float, float, float], *, dpi: int = 300
    ) -> bytes:
        raise ParseError(
            "Word documents do not have page images to preview.",
            diagnostic="DOCX pagination is decided by Word at open time",
        )

    def extract_image_asset(self, xref: int) -> tuple[bytes, str] | None:
        """Images are never extracted from a DOCX.

        Reconstruction rewrites text inside the original package, so every
        picture stays exactly where and as it was -- byte-identical, with no
        extraction or re-encode step to get wrong (spec sec. 14).
        """
        return None

    def close(self) -> None:
        try:
            self._zip.close()
        except Exception:  # already closed
            pass


def _toggle(el: etree._Element | None) -> bool:
    """A w:b / w:i toggle: present means on unless explicitly val="0"."""
    if el is None:
        return False
    return str(el.get(f"{W}val", "1")).lower() not in ("0", "false", "off")


def _make_object(
    ref: ParaRef,
    unit_index: int,
    unit,
    text: str,
    style: TextStyle,
    box: Rect,
    section: _Section,
) -> TextObject:
    align = _ALIGNMENTS.get(paragraph_alignment(ref.element) or "", TextAlignment.UNKNOWN)
    style_id = paragraph_style(ref.element)
    role = _STYLE_ROLES.get(style_id or "") or _ORIGIN_ROLES.get(ref.origin)

    obj = TextObject(
        id=f"docx_{abs(hash(ref.anchor(unit_index))) & 0xFFFFFFFFFFFF:012x}",
        kind=ObjectKind.TEXT,
        bbox=box,
        norm_bbox=_norm(box, section.width, section.height),
        anchor=ref.anchor(unit_index),
        source=TextSource.NATIVE,
        role=role or SemanticRole.OTHER,
        # A style name is a statement by the document's author, not an
        # inference, so it carries real confidence. An origin-derived role
        # (a table cell, a footer) is equally certain about where the text
        # is, if not what it means.
        role_confidence=0.9 if role else 0.0,
        alignment=align,
        style=style,
        lines=[
            TextLine(
                bbox=box,
                spans=[TextSpan(text=text, bbox=box, style=style)],
            )
        ],
    )
    return obj


class DocxParser(DocumentParser):
    name = "docx"
    supported_content_types = frozenset(
        {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            # Some browsers and LMS exports send these instead.
            "application/msword",
            "application/zip",
        }
    )
    supported_extensions = frozenset({"docx"})

    def can_parse(self, content_type: str | None, filename: str | None) -> bool:
        # The generic types above are claimed only alongside a .docx name:
        # application/zip on its own says nothing, and .doc (the old binary
        # format) is a different file this parser cannot read.
        ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
        if ext:
            return ext == "docx"
        return (content_type or "").lower() == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def open_path(self, path: Path) -> OpenDocx:
        return OpenDocx(path.read_bytes())

    def open_bytes(self, data: bytes) -> OpenDocx:
        return OpenDocx(data)
