"""Walking an .xlsx package: shared by the parser and the write-back.

Both sides must agree exactly on which cells exist and what each anchor
names, so the walk lives here once -- the same arrangement as `docx_walk`
and `pptx_walk`.

WHERE A SPREADSHEET KEEPS ITS TEXT

A cell does not usually hold its own characters. A cell with `t="s"` holds
an index into `xl/sharedStrings.xml`, and one `<si>` there may be shown by
any number of cells on any number of sheets. A cell with `t="inlineStr"`
holds its text in an `<is>` element of its own. A cell with `t="str"` holds
the cached result of a formula, which Excel recomputes on open, so it is
never text this engine owns. Everything else is a number, a boolean, a date
(a number with a date format), or an error.

ANCHORS

  * `xl/sharedStrings.xml#si12` -- the thirteenth shared string. One block
    per distinct string actually used by a translatable cell, written back
    once, so every cell showing it changes together.
  * `xl/worksheets/sheet1.xml#B4` -- an inline string, owned by that cell.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from typing import Iterator

from lxml import etree

from ..translate.protection import has_translatable_content, is_math_dominant
from .docx_walk import parse_xml

S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_R = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_REL_WORKSHEET = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
)
_REL_SHARED_STRINGS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
)
SHARED_STRINGS_DEFAULT = "xl/sharedStrings.xml"

_CELL_REF = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d+)$")
# A string literal inside a formula: "...", with "" as an escaped quote.
_FORMULA_LITERAL = re.compile(r'"((?:[^"]|"")*)"')

__all__ = [
    "S",
    "SheetRef",
    "parse_xml",
    "load_sheets",
    "shared_strings_part",
    "shared_items",
    "item_text",
    "item_text_elements",
    "item_is_rich",
    "StringCell",
    "iter_string_cells",
    "cell_position",
    "protected_literals",
    "looks_translatable",
    "object_id",
]


@dataclass
class SheetRef:
    """One worksheet, in workbook (tab) order."""

    index: int
    name: str
    part: str


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


def _workbook_rels(z: zipfile.ZipFile, names: set[str]) -> dict[str, tuple[str, str]]:
    """rId -> (type, part name) for xl/workbook.xml."""
    rels_part = "xl/_rels/workbook.xml.rels"
    if rels_part not in names:
        return {}
    root = parse_xml(z.read(rels_part))
    out: dict[str, tuple[str, str]] = {}
    for rel in root.iter(f"{PKG_R}Relationship"):
        target = rel.get("Target") or ""
        if rel.get("TargetMode") == "External" or target.startswith("http"):
            continue
        out[rel.get("Id") or ""] = (rel.get("Type") or "", _resolve("xl", target))
    return out


def load_sheets(z: zipfile.ZipFile, names: set[str]) -> list[SheetRef]:
    """Worksheets in tab order.

    Order comes from `<sheets>` in the workbook, not from part names:
    sheet3.xml is not necessarily the third tab. Chart sheets and dialog
    sheets have no cells and are skipped. A hidden sheet IS included:
    hidden lookup lists are often what a visible dropdown shows, and they
    share their strings with the cells that show them.
    """
    workbook = parse_xml(z.read("xl/workbook.xml"))
    rels = _workbook_rels(z, names)
    out: list[SheetRef] = []
    sheets = workbook.find(f"{S}sheets")
    for sheet in sheets if sheets is not None else []:
        rel = rels.get(sheet.get(f"{R}id") or "")
        if not rel or rel[0] != _REL_WORKSHEET or rel[1] not in names:
            continue
        out.append(SheetRef(index=len(out), name=sheet.get("name") or "", part=rel[1]))
    return out


def shared_strings_part(z: zipfile.ZipFile, names: set[str]) -> str | None:
    """Where the shared string table lives, if the workbook has one."""
    for typ, part in _workbook_rels(z, names).values():
        if typ == _REL_SHARED_STRINGS and part in names:
            return part
    return SHARED_STRINGS_DEFAULT if SHARED_STRINGS_DEFAULT in names else None


def shared_items(root: etree._Element) -> list[etree._Element]:
    return root.findall(f"{S}si")


def item_text_elements(item: etree._Element) -> list[etree._Element]:
    """The `<t>` elements that make up a string item, in order.

    An `<si>` (or an inline `<is>`) is either one direct `<t>`, or a
    sequence of rich-text runs `<r><t>`. Phonetic guides (`<rPh>`) carry
    their own `<t>` and are deliberately NOT included: they annotate the
    text, they are not part of it.
    """
    direct = item.find(f"{S}t")
    if direct is not None:
        return [direct]
    out = []
    for run in item.findall(f"{S}r"):
        t = run.find(f"{S}t")
        if t is not None:
            out.append(t)
    return out


def item_text(item: etree._Element) -> str:
    return "".join(t.text or "" for t in item_text_elements(item))


def item_is_rich(item: etree._Element) -> bool:
    """True when more than one formatted run carries characters."""
    return item.find(f"{S}t") is None and sum(
        1 for t in item_text_elements(item) if t.text
    ) > 1


def _col_letters(n: int) -> str:
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _col_number(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def cell_position(ref: str) -> tuple[int, int]:
    """(row, column), both 1-based, for a reference such as "B4"."""
    m = _CELL_REF.match(ref)
    if not m:
        return (0, 0)
    return int(m.group(2)), _col_number(m.group(1))


@dataclass
class StringCell:
    """A cell whose value is a string this engine may own."""

    ref: str
    element: etree._Element
    # "s" -> `shared` is the index into the shared string table.
    # "inlineStr" -> `inline` is the cell's own `<is>` element.
    kind: str
    shared: int | None = None
    inline: etree._Element | None = None

    @property
    def position(self) -> tuple[int, int]:
        return cell_position(self.ref)


def iter_string_cells(sheet_root: etree._Element) -> Iterator[StringCell]:
    """Every shared- or inline-string cell, row by row, left to right.

    A cell with a formula is never yielded, whatever its type: its value is
    computed, and the formula -- not the cached text -- is what Excel keeps.
    The `r` attribute is optional in the format; where a writer left it out
    the position is counted, exactly as Excel does.
    """
    data = sheet_root.find(f"{S}sheetData")
    if data is None:
        return
    row_no = 0
    for row in data.findall(f"{S}row"):
        try:
            row_no = int(row.get("r") or row_no + 1)
        except ValueError:
            row_no += 1
        col_no = 0
        for cell in row.findall(f"{S}c"):
            ref = cell.get("r")
            pos = cell_position(ref) if ref else (0, 0)
            if pos[1]:
                col_no = pos[1]
            else:
                col_no += 1
                ref = f"{_col_letters(col_no)}{row_no}"
            if cell.find(f"{S}f") is not None:
                continue
            kind = cell.get("t")
            if kind == "s":
                v = cell.find(f"{S}v")
                try:
                    index = int((v.text or "").strip()) if v is not None else -1
                except ValueError:
                    continue
                if index >= 0:
                    yield StringCell(ref=ref, element=cell, kind="s", shared=index)
            elif kind == "inlineStr":
                inline = cell.find(f"{S}is")
                if inline is not None:
                    yield StringCell(
                        ref=ref, element=cell, kind="inlineStr", inline=inline
                    )


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def protected_literals(z: zipfile.ZipFile, names: set[str]) -> set[str]:
    """Text the workbook's own logic compares against, casefolded.

    `=IF(B2="Yes", 1, 0)` stops working the moment B2 says "Sí", and a
    dropdown listing "Yes,No" rejects a cell that now says "Sí". Excel
    compares text case-insensitively, so the comparison here does too. A
    string that matches one of these is left exactly as it is: an untranslated
    word is a small loss, a gradebook whose formulas silently return zero is
    a large one.

    Collected from formulas, data-validation lists, conditional formatting,
    autofilter values and table column names (a table header cell must
    match its column name, or Excel "repairs" the file on open).
    """
    out: set[str] = set()

    def add(text: str | None) -> None:
        if text and text.strip():
            out.add(text.strip().casefold())

    for name in names:
        if not name.endswith(".xml"):
            continue
        if name.startswith("xl/worksheets/") or name == "xl/workbook.xml":
            root = parse_xml(z.read(name))
            for el in root.iter():
                local = _local(el.tag)
                if local in ("f", "formula", "formula1", "formula2", "definedName"):
                    for m in _FORMULA_LITERAL.finditer(el.text or ""):
                        literal = m.group(1).replace('""', '"')
                        add(literal)
                        if local in ("formula1", "formula2", "f"):
                            # A validation list is one literal of items.
                            for item in literal.split(","):
                                add(item)
                elif local == "cfRule":
                    add(el.get("text"))
                elif local == "filter":
                    add(el.get("val"))
        elif name.startswith("xl/tables/"):
            root = parse_xml(z.read(name))
            for el in root.iter(f"{S}tableColumn"):
                add(el.get("name"))
    return out


def looks_translatable(text: str) -> bool:
    """True when a cell's text is words, not a value that happens to be text.

    Uses the same judgement as the rest of the engine: URLs, emails, codes,
    measurements and formulas are masked, and what is left must contain
    real letters. "A-12", "3.5", "x = 2y" and "kim@school.org" all fail.
    """
    return has_translatable_content(text) and not is_math_dominant(text)


def object_id(anchor: str) -> str:
    """A block id that is the same in every process.

    Derived with hashlib rather than `hash()`, which Python salts per
    process: a block id has to survive a restart, because the review UI and
    every stored edit refer to it.
    """
    return "xlsx_" + hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:12]
