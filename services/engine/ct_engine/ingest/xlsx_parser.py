"""XLSX ingest.

A spreadsheet is the third Office format, and the one where most of the file
is NOT text to translate. A worksheet is mostly numbers, dates, formulas and
codes, with words in a minority of cells -- headings, directions, answer
choices, a column of vocabulary. So the rule here is the opposite way round
from Word: a cell is only taken when it plainly holds words, and everything
else is left exactly as it was.

WHAT IS TRANSLATED

Shared strings and inline strings that read as words (see
`xlsx_walk.looks_translatable`). One block per distinct shared string, not
per cell, because Excel stores "Name" once and shows it in forty cells; the
block is listed on the first sheet that shows it, in tab order, and its
translation reaches every cell that uses it. A string the workbook's own
logic compares against (a formula literal, a dropdown item, a table column
name) is left alone, because translating it would break the formula.

WHAT IS NOT, IN THIS PHASE

Sheet names, comments and notes, headers and footers, chart titles and
labels, text boxes and shapes, data-validation messages and pivot caches.
They are all copied through untouched. Formula results (`t="str"`) are
never touched: Excel recomputes them.

POSITIONS

A cell has a grid position, not a printed box, so pages are marked
`has_fixed_layout=False` like a Word section. Each block is given a
synthesised box on a compact grid -- only rows and columns that hold text
are kept, in their original order -- so the review list reads row by row,
left to right. The boxes are for ordering and selection only.
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
from .xlsx_walk import (
    SheetRef,
    item_is_rich,
    item_text,
    iter_string_cells,
    load_sheets,
    looks_translatable,
    object_id,
    parse_xml,
    protected_literals,
    shared_items,
    shared_strings_part,
)

log = get_logger(__name__)

# The synthesised grid. Wider than Excel's 64px default column so a review
# list of words stays readable; the numbers are used for nothing else.
_MARGIN = 36.0
_COL_W = 144.0
_ROW_H = 20.0
_MIN_W = 612.0
_MIN_H = 792.0
_DEFAULT_FONT_PT = 11.0


@dataclass
class _Entry:
    """One block awaiting a page: where it lives and where it is first shown."""

    anchor: str
    text: str
    row: int
    col: int
    rich: bool


class OpenXlsx(OpenDocument):
    """An opened .xlsx package, presented one worksheet at a time."""

    # Excel lays the grid out when it opens the file; there is nothing here
    # to rasterize. Declared so the analyze stage skips previews cleanly.
    supports_preview = False

    def __init__(self, data: bytes) -> None:
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ParseError(
                "That Excel file could not be opened. It may be damaged.",
                diagnostic=repr(exc),
            ) from exc
        self._names = set(self._zip.namelist())
        if "xl/workbook.xml" not in self._names:
            raise ParseError(
                "That file is a ZIP archive but not an Excel workbook.",
                diagnostic="xl/workbook.xml missing",
            )
        try:
            self._sheets = load_sheets(self._zip, self._names)
            self._entries, self._cell_count = self._collect()
        except etree.XMLSyntaxError as exc:
            raise ParseError(
                "That Excel workbook could not be read. It may be damaged.",
                diagnostic=f"XMLSyntaxError: {exc}",
            ) from exc

    # --- package plumbing -------------------------------------------------

    def _collect(self) -> tuple[dict[int, list[_Entry]], int]:
        """Every translatable block, bucketed by the sheet that first shows it.

        Read once, at open, because a shared string's home sheet depends on
        every sheet before it.
        """
        sst_part = shared_strings_part(self._zip, self._names)
        items: list[etree._Element] = []
        if sst_part:
            items = shared_items(parse_xml(self._zip.read(sst_part)))
        protected = protected_literals(self._zip, self._names)

        def wanted(text: str) -> bool:
            return (
                bool(text.strip())
                and looks_translatable(text)
                and text.strip().casefold() not in protected
            )

        buckets: dict[int, list[_Entry]] = {s.index: [] for s in self._sheets}
        claimed: set[int] = set()
        string_cells = 0
        for sheet in self._sheets:
            root = parse_xml(self._zip.read(sheet.part))
            for cell in iter_string_cells(root):
                string_cells += 1
                row, col = cell.position
                if cell.kind == "s":
                    index = cell.shared
                    if index is None or index >= len(items) or index in claimed:
                        continue
                    text = item_text(items[index])
                    if not wanted(text):
                        continue
                    claimed.add(index)
                    buckets[sheet.index].append(
                        _Entry(
                            anchor=f"{sst_part}#si{index}",
                            text=text,
                            row=row,
                            col=col,
                            rich=item_is_rich(items[index]),
                        )
                    )
                else:
                    text = item_text(cell.inline)
                    if not wanted(text):
                        continue
                    buckets[sheet.index].append(
                        _Entry(
                            anchor=f"{sheet.part}#{cell.ref}",
                            text=text,
                            row=row,
                            col=col,
                            rich=item_is_rich(cell.inline),
                        )
                    )
        return buckets, string_cells

    # --- OpenDocument interface -------------------------------------------

    @property
    def page_count(self) -> int:
        return len(self._sheets)

    @property
    def sheets(self) -> list[SheetRef]:
        return list(self._sheets)

    @property
    def cell_count(self) -> int:
        """Cells holding a string, across every worksheet.

        The completeness check for XLSX, standing in for a page count. It is
        structural -- it counts string cells whatever they say -- so a
        translation cannot change it, but a write-back that dropped or
        retyped a cell would.
        """
        return self._cell_count

    @property
    def translatable_count(self) -> int:
        """Blocks that will be sent for translation; what the size limit bounds."""
        return sum(len(v) for v in self._entries.values())

    @property
    def embedded_media_count(self) -> int:
        """Pictures in the package, reported and never opened (see docx_parser)."""
        return sum(1 for n in self._names if n.startswith("xl/media/"))

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

    def parse_page(self, index: int) -> Page:
        if not 0 <= index < len(self._sheets):
            raise PageParseError(index, f"Sheet {index + 1} does not exist.")
        sheet = self._sheets[index]
        entries = sorted(self._entries.get(index, []), key=lambda e: (e.row, e.col))

        try:
            # A compact grid: only the rows and columns that hold text, in
            # their original order. Using the real indices would put a label
            # in row 900 eighteen thousand points down the page.
            row_rank = {r: i for i, r in enumerate(sorted({e.row for e in entries}))}
            col_rank = {c: i for i, c in enumerate(sorted({e.col for e in entries}))}
            width = max(_MIN_W, 2 * _MARGIN + len(col_rank) * _COL_W)
            height = max(_MIN_H, 2 * _MARGIN + len(row_rank) * _ROW_H)
            objects: list[TextObject] = []
            collapsed: list[str] = []
            for entry in entries:
                x0 = _MARGIN + col_rank[entry.col] * _COL_W
                y0 = _MARGIN + row_rank[entry.row] * _ROW_H
                box = Rect(x0=x0, y0=y0, x1=x0 + _COL_W - 4.0, y1=y0 + _ROW_H - 2.0)
                obj = _make_object(entry, box, width, height)
                objects.append(obj)
                if entry.rich:
                    collapsed.append(obj.id)
        except Exception as exc:  # parser contract: one sheet, one failure
            raise PageParseError(
                index,
                f"Sheet {index + 1} of that workbook could not be read.",
                diagnostic=f"{type(exc).__name__}: {exc}",
            ) from exc

        page = Page(
            index=index,
            number=index + 1,
            width=width,
            height=height,
            has_fixed_layout=False,
            orientation=(
                Orientation.LANDSCAPE if width > height else Orientation.PORTRAIT
            ),
            text_mode=PageTextMode.NATIVE if objects else PageTextMode.EMPTY,
            text_coverage=1.0 if objects else 0.0,
            is_scanned=False,
            needs_ocr=False,
            text_objects=objects,
        )
        for i, obj in enumerate(objects):
            obj.reading_index = i
        page.diagnostics.extra["xlsx_sheet_name"] = sheet.name
        page.diagnostics.extra["xlsx_sheet_part"] = sheet.part
        # An honest loss, recorded rather than hidden (as for DOCX).
        if collapsed:
            page.diagnostics.extra["xlsx_collapsed_formatting"] = collapsed
        page.recompute_diagnostics()
        return page

    def render_page_png(
        self, index: int, *, dpi: int = 144, gray: bool = False
    ) -> bytes:
        raise ParseError(
            "Excel sheets do not have page images to preview.",
            diagnostic="the grid is laid out by Excel at open time",
        )

    def render_region_png(
        self, index: int, bbox: tuple[float, float, float, float], *, dpi: int = 300
    ) -> bytes:
        raise ParseError(
            "Excel sheets do not have page images to preview.",
            diagnostic="the grid is laid out by Excel at open time",
        )

    def extract_image_asset(self, xref: int) -> tuple[bytes, str] | None:
        """Never extracted: the whole package is carried through untouched."""
        return None

    def close(self) -> None:
        try:
            self._zip.close()
        except Exception:  # already closed
            pass


def _make_object(entry: _Entry, box: Rect, width: float, height: float) -> TextObject:
    style = TextStyle(font_size=_DEFAULT_FONT_PT)
    return TextObject(
        id=object_id(entry.anchor),
        kind=ObjectKind.TEXT,
        bbox=box,
        norm_bbox=NormRect(
            x0=min(1.0, box.x0 / width),
            y0=min(1.0, box.y0 / height),
            x1=min(1.0, box.x1 / width),
            y1=min(1.0, box.y1 / height),
        ),
        anchor=entry.anchor,
        source=TextSource.NATIVE,
        # Where the text is is certain -- a cell -- even if what it means is
        # not, which is the same footing as a Word table cell.
        role=SemanticRole.TABLE_CELL,
        role_confidence=0.9,
        alignment=TextAlignment.UNKNOWN,
        style=style,
        lines=[
            TextLine(bbox=box, spans=[TextSpan(text=entry.text, bbox=box, style=style)])
        ],
    )


class XlsxParser(DocumentParser):
    name = "xlsx"
    supported_content_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    )
    supported_extensions = frozenset({"xlsx"})

    def can_parse(self, content_type: str | None, filename: str | None) -> bool:
        ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
        if ext:
            return ext == "xlsx"
        return (content_type or "").lower() in self.supported_content_types

    def open_path(self, path: Path) -> OpenXlsx:
        return OpenXlsx(path.read_bytes())

    def open_bytes(self, data: bytes) -> OpenXlsx:
        return OpenXlsx(data)
