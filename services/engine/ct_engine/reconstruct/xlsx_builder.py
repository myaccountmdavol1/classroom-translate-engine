"""XLSX reconstruction: edit the original package, never rebuild it.

Same principle as `docx_builder`: translated characters are substituted into
the `<t>` elements they came from, and every other part of the package --
styles, formulas, numbers, merged cells, column widths, charts, pictures,
defined names, the calculation chain -- is copied through byte-for-byte,
because it is never parsed in the first place.

Only two kinds of part are ever rewritten:

  * the shared string table, once per distinct translated string, so every
    cell that shows "Name" changes together and the cells themselves are
    not touched at all;
  * a worksheet, only when one of its own inline-string cells was
    translated. Inside it, nothing but that cell's `<is>` text changes.

The string table's `count` and `uniqueCount` stay correct by construction:
no string is added or removed, and no cell is retyped.

What is NOT carried: the review editor's colour and size choices. A cell's
look lives in a shared style record in `styles.xml`; changing one cell
would mean minting a new style, which is a different and riskier edit than
replacing characters. Excel also needs no overflow handling: a cell that
grows simply shows more, or wraps if the author set it to.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

from lxml import etree

from ..ingest.xlsx_walk import (
    S,
    item_is_rich,
    item_text,
    item_text_elements,
    iter_string_cells,
    parse_xml,
    shared_items,
)
from ..model.document import Page, QAFinding
from ..model.objects import TextObject
from ..util.logging import get_logger
from .builder import PageBuildStats
from .docx_builder import _output_text

log = get_logger(__name__)

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# Characters XML 1.0 cannot hold at all. A translation provider can emit
# one (a stray vertical tab from a copy-paste), and lxml refuses to store
# it -- which would fail the whole export over one invisible character.
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")

# Excel's own ceiling on the characters in one cell.
_MAX_CELL_CHARS = 32767


@dataclass
class _Target:
    obj: TextObject
    page_index: int


@dataclass
class XlsxBuildResult:
    data: bytes
    stats: dict[int, PageBuildStats]
    # Never silently ignored, for the same reason as in docx_builder: a cell
    # that kept its source text while the reviewer was shown a translation.
    unresolved: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return _XML_INVALID.sub("", text)[:_MAX_CELL_CHARS]


def _set_t(t: etree._Element, text: str) -> None:
    t.text = text
    # Without this Excel trims leading and trailing spaces, and line breaks
    # inside a wrapped cell are only kept reliably when it is set.
    if text != text.strip() or "\n" in text or "\t" in text:
        t.set(XML_SPACE, "preserve")


def _write_item(item: etree._Element, text: str) -> bool:
    """Put `text` into a string item; True when it was rich text.

    The translation goes into the first run and every other run is emptied,
    as for a Word paragraph whose formatting could not be kept: the word
    order of a translation does not line up with the source's bold word.
    Phonetic guides (`<rPh>`) are removed, because they annotate characters
    that are no longer there.
    """
    rich = item_is_rich(item)
    for rph in item.findall(f"{S}rPh"):
        item.remove(rph)
    elements = item_text_elements(item)
    if not elements:
        t = etree.SubElement(item, f"{S}t")
        elements = [t]
    _set_t(elements[0], text)
    for t in elements[1:]:
        t.text = ""
        if XML_SPACE in t.attrib:
            del t.attrib[XML_SPACE]
    return rich


def _anchor_lost(t: _Target, anchor: str) -> QAFinding:
    return QAFinding(
        severity="error",
        code="xlsx_anchor_lost",
        message=(
            "A cell kept its original wording because it could not be "
            "located in the Excel file."
        ),
        object_ids=[t.obj.id],
        details={"anchor": anchor},
    )


def _collapsed(t: _Target) -> QAFinding:
    return QAFinding(
        severity="warning",
        code="xlsx_formatting_collapsed",
        message=(
            "Bold, italic or colour inside this cell was replaced by the "
            "cell's main style, because the translated wording no longer "
            "lines up with the original words."
        ),
        object_ids=[t.obj.id],
    )


def rebuild_xlsx(source_bytes: bytes, pages: list[Page]) -> XlsxBuildResult:
    """Write every sheet's translations back into the source package."""
    targets: dict[str, list[_Target]] = {}
    for page in pages:
        for obj in page.text_objects:
            if not obj.anchor:
                continue
            part = obj.anchor.split("#", 1)[0]
            targets.setdefault(part, []).append(_Target(obj=obj, page_index=page.index))

    stats: dict[int, PageBuildStats] = {p.index: PageBuildStats() for p in pages}
    unresolved: list[str] = []
    rewritten: dict[str, bytes] = {}

    with zipfile.ZipFile(io.BytesIO(source_bytes)) as zin:
        names = set(zin.namelist())

        for part, part_targets in targets.items():
            if part not in names:
                for t in part_targets:
                    unresolved.append(t.obj.anchor or "")
                    stats[t.page_index].findings.append(_anchor_lost(t, t.obj.anchor or ""))
                continue

            root = parse_xml(zin.read(part))
            # Resolve every anchor before modifying anything; the same rule
            # as docx_builder, although here no edit can renumber another.
            by_key: dict[str, etree._Element] = {}
            if root.tag == f"{S}sst":
                for i, item in enumerate(shared_items(root)):
                    by_key[f"si{i}"] = item
            else:
                for cell in iter_string_cells(root):
                    if cell.kind == "inlineStr" and cell.inline is not None:
                        by_key[cell.ref] = cell.inline

            touched = False
            for t in part_targets:
                anchor = t.obj.anchor or ""
                key = anchor.split("#", 1)[1] if "#" in anchor else ""
                item = by_key.get(key)
                s = stats[t.page_index]
                if item is None:
                    unresolved.append(anchor)
                    s.findings.append(_anchor_lost(t, anchor))
                    continue

                text = _output_text(t.obj, item_text(item))
                if text is None:
                    continue
                rich = _write_item(item, _clean(text))
                touched = True
                if t.obj.suppressed:
                    s.suppressed += 1
                else:
                    s.written += 1
                if rich:
                    s.findings.append(_collapsed(t))

            if touched:
                rewritten[part] = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )

        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            # Original order, original entry metadata; [Content_Types].xml
            # stays first.
            for item in zin.infolist():
                data = rewritten.get(item.filename)
                if data is None:
                    data = zin.read(item.filename)
                zout.writestr(item, data)

    if unresolved:
        log.error("%d XLSX anchors did not resolve", len(unresolved))
    return XlsxBuildResult(data=out.getvalue(), stats=stats, unresolved=unresolved)
