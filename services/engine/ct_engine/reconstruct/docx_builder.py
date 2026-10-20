"""DOCX reconstruction: edit the original package, never rebuild it.

The output is the uploaded .docx with translated characters substituted into
the runs they came from. Everything the pipeline never touched -- styles,
numbering, images, tables, headers, section breaks, theme, the lot -- is
copied through byte-for-byte, because it is never parsed in the first place.
That is what makes this safe in a way redrawing a PDF can never be: there is
no font to match, no box to fit, and no artwork to re-encode (spec sec. 14).

Two ordering rules make it correct:

  * Every anchor is resolved BEFORE any element is modified. Anchors are
    ordinals in a document-order walk, so a single insertion partway through
    would renumber everything after it.
  * A run is only ever rewritten, never removed, unless it was matched as
    part of a unit. Runs holding pictures, text boxes or field codes are not
    matched at all (see `docx_walk.group_runs`), so they cannot be lost.

Bilingual output needs no special handling here, and this is worth stating
plainly: the entire overflow problem that bilingual mode causes in PDFs --
text colliding because a box is a fixed size -- does not exist. Word reflows
when the file is opened, so a paragraph that doubles in length simply pushes
the next one down.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

from lxml import etree

from ..ingest.docx_walk import W, Unit, iter_part_paragraphs, parse_xml
from ..model.document import Page, QAFinding
from ..model.enums import TranslationStatus
from ..model.objects import TextObject
from ..util.logging import get_logger
from . import office_lists
from .builder import PageBuildStats

log = get_logger(__name__)

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

_TEXT_CHILD_TAGS = (
    f"{W}t",
    f"{W}tab",
    f"{W}br",
    f"{W}cr",
    f"{W}noBreakHyphen",
)


@dataclass
class _Target:
    """A block awaiting write-back, grouped by the part it lives in."""

    obj: TextObject
    page_index: int


@dataclass
class DocxBuildResult:
    data: bytes
    stats: dict[int, PageBuildStats]
    # Anchors that no longer resolve. Never silently ignored: an unresolvable
    # anchor means a paragraph would keep its source text while the reviewer
    # was shown a translation, which is the failure mode section 46 exists to
    # prevent.
    unresolved: list[str] = field(default_factory=list)


def _set_run_text(run: etree._Element, text: str) -> None:
    """Replace a run's characters, re-emitting tabs and breaks as elements.

    Tabs matter in worksheets: `Name: ____\\tDate: ____` is two fields, and
    flattening the tab to a space merges them into one. Word only honours a
    tab as `w:tab`, never as a character inside `w:t`.
    """
    for child in list(run):
        if child.tag in _TEXT_CHILD_TAGS:
            run.remove(child)

    if not text:
        el = etree.SubElement(run, f"{W}t")
        el.text = ""
        el.set(XML_SPACE, "preserve")
        return

    # Split into runs of plain text separated by tab/newline markers, keeping
    # the markers.
    buf = ""
    for ch in text:
        if ch in ("\t", "\n"):
            if buf:
                _append_text(run, buf)
                buf = ""
            etree.SubElement(run, f"{W}tab" if ch == "\t" else f"{W}br")
        else:
            buf += ch
    if buf:
        _append_text(run, buf)


def _append_text(run: etree._Element, text: str) -> None:
    el = etree.SubElement(run, f"{W}t")
    el.text = text
    # Without this Word strips leading and trailing spaces, which silently
    # joins "Directions: " to the sentence that follows it.
    if text != text.strip():
        el.set(XML_SPACE, "preserve")


def _rpr(run: etree._Element) -> etree._Element:
    """This run's properties, created (in the required position) if absent."""
    rpr = run.find(f"{W}rPr")
    if rpr is None:
        rpr = etree.Element(f"{W}rPr")
        run.insert(0, rpr)  # w:rPr must be the first child of w:r
    return rpr


# The order w:rPr's children must appear in (CT_RPr in the schema). Word
# can refuse to open a file whose run properties are out of order, so a
# property is inserted before the first sibling that must follow it.
_RPR_ORDER = (
    "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike",
    "dstrike", "outline", "shadow", "emboss", "imprint", "noProof",
    "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern",
    "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd",
    "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
    "specVanish", "oMath",
)


def _prop(rpr: etree._Element, name: str) -> etree._Element:
    """rPr's `name` child, created in its schema position if absent."""
    el = rpr.find(f"{W}{name}")
    if el is not None:
        return el
    el = etree.Element(f"{W}{name}")
    later = set(_RPR_ORDER[_RPR_ORDER.index(name) + 1:])
    for i, child in enumerate(rpr):
        if etree.QName(child).localname in later:
            rpr.insert(i, el)
            return el
    rpr.append(el)
    return el


def _set_prop(rpr: etree._Element, name: str, value: str) -> None:
    _prop(rpr, name).set(f"{W}val", value)


# Her border, inside space and turning are deliberately not carried: the
# words are rewritten in place inside the source's own text boxes or flow,
# one source shape can hold several blocks, and a Word paragraph has no box
# at all, so none of the three can be set for one block without
# restructuring the file. They change the pages and the PDF, and the editor
# says so beside each of them.
def _apply_manual_style(run: etree._Element, obj: TextObject) -> bool:
    """Carry the review editor's colour, size, underline and highlight
    choices into Word.

    Position and box size are deliberately not carried: they describe a
    fixed-layout page, and this document does not have one. Colour and size
    are properties of the text itself, so they transfer exactly; so are the
    underline and the highlight. The words' alignment within the box, line
    spacing and justify are not carried (see word_export): a Word paragraph
    keeps its own.
    """
    changed = False
    if obj.manual_color:
        _set_prop(_rpr(run), "color", obj.manual_color.lstrip("#").upper())
        changed = True
    if obj.manual_font_size:
        # w:sz is in half-points.
        _set_prop(_rpr(run), "sz", str(int(round(obj.manual_font_size * 2))))
        changed = True
    if obj.manual_underline:
        # False cannot take away an underline the paragraph already had: the
        # page editor never showed the source's underline either.
        _set_prop(_rpr(run), "u", "single")
        changed = True
    face = office_lists.font_name(getattr(obj, "manual_font_family", None))
    if face:
        # Named, not embedded (see office_lists). A theme font on the run
        # would win over the name, so it is taken off.
        fonts = _prop(_rpr(run), "rFonts")
        for theme in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(f"{W}{theme}", None)
        for slot in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(f"{W}{slot}", face)
        changed = True
    if obj.manual_highlight:
        # As shading, not w:highlight, which allows only 16 named colours.
        # Shading shows any colour, in Word, LibreOffice and Google Docs.
        # A highlighter colour the source run already had is taken off:
        # Word paints w:highlight over the shading, hiding the one she chose.
        rpr = _rpr(run)
        for old in rpr.findall(f"{W}highlight"):
            rpr.remove(old)
        shd = _prop(rpr, "shd")
        shd.set(f"{W}val", "clear")
        shd.set(f"{W}color", "auto")
        shd.set(f"{W}fill", obj.manual_highlight.lstrip("#").upper())
        changed = True
    return changed


def _preserve_edges(source: str, translated: str) -> str:
    """Give the translation back the spaces the translator trimmed.

    Translation providers strip leading and trailing whitespace, and so does
    the pipeline when it composes output. On a PDF that is harmless: each
    block is drawn at its own coordinates. Inside a Word paragraph it is not,
    because the blocks are adjacent characters -- a trimmed "Directions: "
    emitted next to its body produced "Instrucciones:Lee cuidadosamente",
    and a mid-sentence hyperlink produced "Visitaguia de seguridadantes".

    The source's own edges are authoritative, so they are simply restored.
    """
    if not translated.strip():
        return translated
    lead = source[: len(source) - len(source.lstrip())]
    trail = source[len(source.rstrip()) :]
    return f"{lead}{translated.strip()}{trail}"


def _output_text(obj: TextObject, source: str) -> str | None:
    """What this block should say, or None to leave the source alone."""
    if obj.suppressed:
        return ""
    if obj.translation_status in (
        TranslationStatus.PROTECTED,
        TranslationStatus.NOT_REQUIRED,
        TranslationStatus.FAILED,
    ):
        return None
    if obj.translated_text is None:
        return None
    return _preserve_edges(source, obj.translated_text)


# The order w:pPr's children must appear in (CT_PPr), for w:numPr.
_PPR_ORDER = (
    "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr",
    "widowControl", "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs",
    "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct",
    "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
    "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
    "suppressOverlap", "jc", "textDirection", "textAlignment",
    "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr",
    "pPrChange",
)
NUMBERING_PART = "word/numbering.xml"
NUMBERING_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"
NUMBERING_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering"
_RELS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_TYPES = "{http://schemas.openxmlformats.org/package/2006/content-types}"


class _Numbering:
    """The package's numbering part, opened when a list first needs it.

    One w:abstractNum per list look (kind and style, four levels), and one
    w:num per box: two boxes side by side then count 1 and 1, as the pages
    show them, rather than Word running them on as 1 and 2.
    """

    def __init__(self, zin: zipfile.ZipFile, names: set[str]) -> None:
        self.created = NUMBERING_PART not in names
        if self.created:
            self.root = etree.Element(f"{W}numbering", nsmap={"w": W[1:-1]})
        else:
            self.root = parse_xml(zin.read(NUMBERING_PART))
        self._looks: dict[tuple[str, str], int] = {}
        self._next_abstract = 1 + max(
            (int(a.get(f"{W}abstractNumId", "0")) for a in self.root.findall(f"{W}abstractNum")),
            default=-1,
        )
        self._next_num = 1 + max(
            (int(n.get(f"{W}numId", "0")) for n in self.root.findall(f"{W}num")),
            default=0,
        )

    def _insert(self, el: etree._Element, before: tuple[str, ...]) -> None:
        for i, child in enumerate(self.root):
            if etree.QName(child).localname in before:
                self.root.insert(i, el)
                return
        self.root.append(el)

    def _abstract(self, kind: str, style: str | None) -> int:
        key = (kind, style or "")
        if key in self._looks:
            return self._looks[key]
        aid = self._next_abstract
        self._next_abstract += 1
        abstract = etree.Element(f"{W}abstractNum")
        abstract.set(f"{W}abstractNumId", str(aid))
        etree.SubElement(abstract, f"{W}multiLevelType").set(f"{W}val", "multilevel")
        for level in range(4):
            fmt, text = office_lists.word_level(kind, style, level)
            lvl = etree.SubElement(abstract, f"{W}lvl")
            lvl.set(f"{W}ilvl", str(level))
            etree.SubElement(lvl, f"{W}start").set(f"{W}val", "1")
            etree.SubElement(lvl, f"{W}numFmt").set(f"{W}val", fmt)
            etree.SubElement(lvl, f"{W}lvlText").set(f"{W}val", text)
            etree.SubElement(lvl, f"{W}lvlJc").set(f"{W}val", "left")
            ind = etree.SubElement(etree.SubElement(lvl, f"{W}pPr"), f"{W}ind")
            ind.set(f"{W}left", str(720 * (level + 1)))
            ind.set(f"{W}hanging", "360")
            if text in office_lists.SYMBOL_BULLETS:
                fonts = etree.SubElement(etree.SubElement(lvl, f"{W}rPr"), f"{W}rFonts")
                for slot in ("ascii", "hAnsi"):
                    fonts.set(f"{W}{slot}", office_lists.SYMBOL_FONT)
        # Every abstractNum before the first num (CT_Numbering).
        self._insert(abstract, ("num", "numIdMacAtCleanup"))
        self._looks[key] = aid
        return aid

    def num_for(self, kind: str, style: str | None, start: int | None) -> int:
        nid = self._next_num
        self._next_num += 1
        num = etree.Element(f"{W}num")
        num.set(f"{W}numId", str(nid))
        etree.SubElement(num, f"{W}abstractNumId").set(f"{W}val", str(self._abstract(kind, style)))
        override = etree.SubElement(num, f"{W}lvlOverride")
        override.set(f"{W}ilvl", "0")
        etree.SubElement(override, f"{W}startOverride").set(f"{W}val", str(start or 1))
        self._insert(num, ("numIdMacAtCleanup",))
        return nid

    def finish(self, zin: zipfile.ZipFile, rewritten: dict[str, bytes]) -> dict[str, bytes]:
        """The numbering part, and for a new one its relationship and
        content type. Returns parts the source did not have."""
        data = etree.tostring(self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        if not self.created:
            rewritten[NUMBERING_PART] = data
            return {}
        rels_name = "word/_rels/document.xml.rels"
        rels = parse_xml(rewritten.get(rels_name) or zin.read(rels_name))
        taken = {r.get("Id") for r in rels}
        rid = next(f"rIdCtNumbering{i}" for i in range(1, 1000) if f"rIdCtNumbering{i}" not in taken)
        rel = etree.SubElement(rels, f"{_RELS}Relationship")
        rel.set("Id", rid)
        rel.set("Type", NUMBERING_REL)
        rel.set("Target", "numbering.xml")
        rewritten[rels_name] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        types = parse_xml(rewritten.get("[Content_Types].xml") or zin.read("[Content_Types].xml"))
        if not any(o.get("PartName") == f"/{NUMBERING_PART}" for o in types.findall(f"{_TYPES}Override")):
            override = etree.SubElement(types, f"{_TYPES}Override")
            override.set("PartName", f"/{NUMBERING_PART}")
            override.set("ContentType", NUMBERING_TYPE)
        rewritten["[Content_Types].xml"] = etree.tostring(
            types, xml_declaration=True, encoding="UTF-8", standalone=True
        )
        return {NUMBERING_PART: data}


def _set_num_pr(p: etree._Element, level: int, num_id: int) -> None:
    ppr = p.find(f"{W}pPr")
    if ppr is None:
        ppr = etree.Element(f"{W}pPr")
        p.insert(0, ppr)  # w:pPr must be the first child of w:p
    num_pr = etree.Element(f"{W}numPr")
    etree.SubElement(num_pr, f"{W}ilvl").set(f"{W}val", str(level))
    etree.SubElement(num_pr, f"{W}numId").set(f"{W}val", str(num_id))
    later = set(_PPR_ORDER[_PPR_ORDER.index("numPr") + 1:])
    for i, child in enumerate(ppr):
        if etree.QName(child).localname in later:
            ppr.insert(i, num_pr)
            return
    ppr.append(num_pr)


def _list_text(obj: TextObject, ref, unit_text: str, text: str, numbering) -> str:
    """The box's words with her list, as Word can hold it (see
    office_lists): a real numbered or bulleted paragraph when the box is
    this paragraph's only words and one line, else each line starting
    with its marker. A paragraph Word numbers already keeps Word's own
    numbers (word_export took the page's typed one off)."""
    paras = office_lists.plan(obj)
    if not paras or obj.suppressed:
        return text
    ppr = ref.element.find(f"{W}pPr")
    if ppr is not None and ppr.find(f"{W}numPr") is not None:
        return text
    item = office_lists.single_item(paras)
    if item is not None and len(ref.units) == 1:
        num_id = numbering().num_for(obj.manual_list, obj.manual_list_style, obj.manual_list_start)
        _set_num_pr(ref.element, item.level, num_id)
        return _preserve_edges(unit_text, item.body)
    return _preserve_edges(unit_text, office_lists.marker_text(paras))


def rebuild_docx(
    source_bytes: bytes, pages: list[Page]
) -> DocxBuildResult:
    """Write every page's translations back into the source package."""
    targets: dict[str, list[_Target]] = {}
    for page in pages:
        for obj in page.text_objects:
            if not obj.anchor:
                continue
            part = obj.anchor.split("#", 1)[0]
            targets.setdefault(part, []).append(
                _Target(obj=obj, page_index=page.index)
            )

    stats: dict[int, PageBuildStats] = {p.index: PageBuildStats() for p in pages}
    unresolved: list[str] = []
    rewritten: dict[str, bytes] = {}

    with zipfile.ZipFile(io.BytesIO(source_bytes)) as zin:
        names = set(zin.namelist())
        opened: list[_Numbering] = []

        def numbering() -> _Numbering:
            if not opened:
                opened.append(_Numbering(zin, names))
            return opened[0]

        for part, part_targets in targets.items():
            if part not in names:
                unresolved.extend(t.obj.anchor or "" for t in part_targets)
                continue

            root = parse_xml(zin.read(part))
            # The same walk the parser used, so ordinals line up exactly.
            by_anchor: dict[str, Unit] = {}
            refs: dict[str, object] = {}
            for ref in iter_part_paragraphs(root, part):
                for i, unit in enumerate(ref.units):
                    by_anchor[ref.anchor(i)] = unit
                    refs[ref.anchor(i)] = ref

            touched = False
            for t in part_targets:
                anchor = t.obj.anchor or ""
                unit = by_anchor.get(anchor)
                s = stats[t.page_index]
                if unit is None or not unit.groups:
                    unresolved.append(anchor)
                    s.findings.append(
                        QAFinding(
                            severity="error",
                            code="docx_anchor_lost",
                            message=(
                                "A paragraph kept its original wording because "
                                "it could not be located in the Word file."
                            ),
                            object_ids=[t.obj.id],
                            details={"anchor": anchor},
                        )
                    )
                    continue

                text = _output_text(t.obj, unit.text)
                if text is None:
                    continue
                text = _list_text(t.obj, refs[anchor], unit.text, text, numbering)

                lead = unit.groups[0].lead
                _set_run_text(lead, text)
                # Every other run in the unit is emptied. For a single-group
                # unit these are just the split halves of one stretch of
                # identical text; for a collapsed unit they are the runs whose
                # distinct formatting could not be preserved.
                for gi, group in enumerate(unit.groups):
                    for ri, run in enumerate(group.runs):
                        if gi == 0 and ri == 0:
                            continue
                        _set_run_text(run, "")
                _apply_manual_style(lead, t.obj)

                touched = True
                if t.obj.suppressed:
                    s.suppressed += 1
                else:
                    s.written += 1
                if len(unit.groups) > 1:
                    s.findings.append(
                        QAFinding(
                            severity="warning",
                            code="docx_formatting_collapsed",
                            message=(
                                "Bold or italic inside this paragraph was "
                                "replaced by the paragraph's main style, "
                                "because the translated wording no longer "
                                "lines up with the original words."
                            ),
                            object_ids=[t.obj.id],
                        )
                    )

            if touched:
                rewritten[part] = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )

        added = opened[0].finish(zin, rewritten) if opened else {}
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            # Copy in the original order. [Content_Types].xml must stay first
            # for the package to be readable.
            for item in zin.infolist():
                data = rewritten.get(item.filename)
                if data is None:
                    data = zin.read(item.filename)
                zout.writestr(item, data)
            for name, data in added.items():
                zout.writestr(name, data)

    return DocxBuildResult(data=out.getvalue(), stats=stats, unresolved=unresolved)
