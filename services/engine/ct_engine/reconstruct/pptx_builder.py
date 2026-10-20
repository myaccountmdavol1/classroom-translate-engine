"""PPTX reconstruction: edit the original package, never rebuild it.

Same principle as `docx_builder` -- translated characters are substituted
into the runs they came from, and every other part of the package is copied
through untouched -- with one problem DOCX does not have.

THE OVERFLOW PROBLEM

Word reflows: a paragraph that doubles in length pushes the next one down.
PowerPoint does not. A shape is a fixed box, and text longer than the box
either spills over whatever is beneath it or is clipped. Spanish runs
roughly 20-25% longer than English, so this is the normal case, not an edge
one, and it is invisible until someone opens the deck.

What is done about it: the text is measured against its shape, and where it
will not fit, PowerPoint is asked to shrink it -- by writing the same
`a:normAutofit` element PowerPoint itself writes for "Shrink text on
overflow". That keeps the rendering PowerPoint's job rather than this
engine's guess at it.

What is NOT claimed: the measurement is an estimate, from an average glyph
width, because the real answer depends on the font metrics on the machine
that opens the file. So the shrink is conservative, it is floored at a size
that stays readable, and every shape that needed one is reported. A slide
that still cannot fit is flagged for a person rather than quietly squashed.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, field

from lxml import etree

from ..ingest.pptx_walk import (
    A,
    P,
    Unit,
    iter_part_paragraphs,
    owning_shape,
    parse_xml,
)
from ..model.document import Page, QAFinding
from ..model.enums import TranslationStatus
from ..model.objects import TextObject
from ..util.logging import get_logger
from . import lists, office_lists
from .builder import PageBuildStats

log = get_logger(__name__)

# PowerPoint's default text-frame insets, in points (91440 / 45720 EMU).
_INSET_X = 7.2
_INSET_Y = 3.6

# Below this the text stops being readable from the back of a classroom, so
# the shrink stops here and the slide is flagged instead.
MIN_FONT_SCALE = 0.62


@dataclass
class _Target:
    obj: TextObject
    page_index: int


@dataclass
class _ShapeFit:
    """Accumulates one shape's final text and its resolved box."""

    shape: etree._Element
    box: list[float] | None = None
    # Paragraph key -> (translated text, source text, largest point size).
    paragraphs: dict[str, tuple[str, str, float]] = field(default_factory=dict)

    def union(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self.box is None:
            self.box = [x0, y0, x1, y1]
            return
        b = self.box
        b[0], b[1] = min(b[0], x0), min(b[1], y0)
        b[2], b[3] = max(b[2], x1), max(b[3], y1)

    def add_text(self, para_key: str, text: str, source: str, size: float) -> None:
        prev, prev_src, prev_size = self.paragraphs.get(para_key, ("", "", 0.0))
        self.paragraphs[para_key] = (
            prev + text,
            prev_src + source,
            max(prev_size, size),
        )

    def lines(self) -> list[tuple[str, float]]:
        return [(t, size) for t, _, size in self.paragraphs.values()]

    def source_lines(self) -> list[tuple[str, float]]:
        return [(src, size) for _, src, size in self.paragraphs.values()]

    @property
    def width(self) -> float:
        return (self.box[2] - self.box[0]) if self.box else 0.0

    @property
    def height(self) -> float:
        return (self.box[3] - self.box[1]) if self.box else 0.0


@dataclass
class PptxBuildResult:
    data: bytes
    stats: dict[int, PageBuildStats]
    unresolved: list[str] = field(default_factory=list)


def _set_run_text(run: etree._Element, text: str) -> None:
    """Replace a run's characters.

    DrawingML has no tab element inside a run and no `xml:space` convention
    to worry about: `a:t` preserves what it is given. A tab character is
    written through as-is, which is what PowerPoint stores for one.
    """
    t = run.find(f"{A}t")
    if t is None:
        t = etree.SubElement(run, f"{A}t")
    t.text = text


def _rpr(run: etree._Element) -> etree._Element:
    rpr = run.find(f"{A}rPr")
    if rpr is None:
        rpr = etree.Element(f"{A}rPr")
        run.insert(0, rpr)  # a:rPr must come first inside a:r
    return rpr


# The order a:rPr's children must appear in (CT_TextCharacterProperties).
# PowerPoint reports a file with them out of order as needing repair.
_RPR_ORDER = (
    "ln", "noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill",
    "effectLst", "effectDag", "highlight", "uLnTx", "uLn", "uFillTx", "uFill",
    "latin", "ea", "cs", "sym", "hlinkClick", "hlinkMouseOver", "rtl", "extLst",
)


def _insert_ordered(rpr: etree._Element, el: etree._Element) -> None:
    """Put `el` into rPr before the first child that must follow it."""
    name = etree.QName(el).localname
    later = set(_RPR_ORDER[_RPR_ORDER.index(name) + 1:])
    for i, child in enumerate(rpr):
        if etree.QName(child).localname in later:
            rpr.insert(i, el)
            return
    rpr.append(el)


def _replace_colour_child(rpr: etree._Element, name: str, hex_color: str) -> None:
    for existing in rpr.findall(f"{A}{name}"):
        rpr.remove(existing)
    el = etree.Element(f"{A}{name}")
    etree.SubElement(el, f"{A}srgbClr").set("val", hex_color.lstrip("#").upper())
    _insert_ordered(rpr, el)


# Her border, inside space and turning are deliberately not carried: the
# words are rewritten in place inside the source's own text boxes or flow,
# one source shape can hold several blocks, and a Word paragraph has no box
# at all, so none of the three can be set for one block without
# restructuring the file. They change the pages and the PDF, and the editor
# says so beside each of them.
def _apply_manual_style(run: etree._Element, obj: TextObject) -> None:
    """Carry the review editor's colour, size, underline and highlight
    choices into the slide.

    The words' place up and down in the box is not carried: in PowerPoint
    that anchor belongs to the whole shape, and changing it would move her
    other paragraphs. Alignment, justify and line spacing are not carried
    either; the paragraph keeps its own (see word_export).
    """
    if obj.manual_color:
        _replace_colour_child(_rpr(run), "solidFill", obj.manual_color)
    if obj.manual_font_size:
        # a:rPr/@sz is in hundredths of a point.
        _rpr(run).set("sz", str(int(round(obj.manual_font_size * 100))))
    if obj.manual_underline:
        # False cannot take away an underline the paragraph already had.
        _rpr(run).set("u", "sng")
    if obj.manual_highlight:
        # PowerPoint 2019 and 365 show a:highlight; older versions ignore it
        # and show the words without it.
        _replace_colour_child(_rpr(run), "highlight", obj.manual_highlight)
    face = office_lists.font_name(getattr(obj, "manual_font_family", None))
    if face:
        # Named, not embedded (see office_lists); word_export names it only
        # where it has the language's letters, so every script slot gets it.
        rpr = _rpr(run)
        for slot in ("latin", "ea", "cs"):
            for existing in rpr.findall(f"{A}{slot}"):
                rpr.remove(existing)
            el = etree.Element(f"{A}{slot}")
            el.set("typeface", face)
            _insert_ordered(rpr, el)


# The order a:pPr's children must appear in (CT_TextParagraphProperties).
_PPR_ORDER = (
    "lnSpc", "spcBef", "spcAft", "buClrTx", "buClr", "buSzTx", "buSzPct",
    "buSzPts", "buFontTx", "buFont", "buNone", "buAutoNum", "buChar", "buBlip",
    "tabLst", "defRPr", "extLst",
)
_BULLET_TAGS = ("buNone", "buAutoNum", "buChar", "buBlip")
EMU_PER_POINT = 12700


def _insert_ppr(ppr: etree._Element, el: etree._Element) -> None:
    later = set(_PPR_ORDER[_PPR_ORDER.index(etree.QName(el).localname) + 1:])
    for i, child in enumerate(ppr):
        if etree.QName(child).localname in later:
            ppr.insert(i, el)
            return
    ppr.append(el)


_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_LAYOUT_REL = "/slideLayout"
_MASTER_REL = "/slideMaster"


class _Inherited:
    """Whether a slide paragraph gets bullets it does not say it has, from
    its shape's list style, its layout's placeholder, its master's
    placeholder or the master's text styles (the order PowerPoint looks).
    Read from the package once per part."""

    def __init__(self, zin: zipfile.ZipFile) -> None:
        self._zin = zin
        self._names = set(zin.namelist())
        self._xml: dict[str, etree._Element | None] = {}

    def _part(self, name: str | None) -> etree._Element | None:
        if name is None or name not in self._names:
            return None
        if name not in self._xml:
            try:
                self._xml[name] = parse_xml(self._zin.read(name))
            except Exception:
                self._xml[name] = None
        return self._xml[name]

    def _related(self, part: str, kind: str) -> str | None:
        folder, _, base = part.rpartition("/")
        rels = self._part(f"{folder}/_rels/{base}.rels")
        if rels is None:
            return None
        for rel in rels.iter(f"{_REL_NS}Relationship"):
            if rel.get("Type", "").endswith(kind):
                target = rel.get("Target", "")
                parts = folder.split("/") if not target.startswith("/") else []
                for piece in target.lstrip("/").split("/"):
                    if piece == "..":
                        parts = parts[:-1]
                    elif piece and piece != ".":
                        parts.append(piece)
                return "/".join(parts)
        return None

    @staticmethod
    def _bullet(lst: etree._Element | None, level: int) -> bool | None:
        """True (bullets), False (buNone) or None (says nothing)."""
        if lst is None:
            return None
        lvl = lst.find(f"{A}lvl{level + 1}pPr")
        if lvl is None:
            return None
        for child in lvl:
            name = etree.QName(child).localname
            if name in _BULLET_TAGS:
                return name != "buNone"
        return None

    @staticmethod
    def _placeholder(root: etree._Element, ph_type: str | None, idx: str | None,
                     by_idx: bool) -> etree._Element | None:
        for ph in root.iter(f"{P}ph"):
            if by_idx and idx is not None and ph.get("idx") == idx:
                return owning_shape(ph)
        for ph in root.iter(f"{P}ph"):
            if (ph.get("type") or "body") == (ph_type or "body"):
                return owning_shape(ph)
        return None

    def bullets(self, part: str, p: etree._Element) -> bool:
        ppr = p.find(f"{A}pPr")
        level = int(ppr.get("lvl", "0")) if ppr is not None and ppr.get("lvl", "0").isdigit() else 0
        shape = owning_shape(p)
        if shape is None:
            return False
        own = self._bullet(shape.find(f"{P}txBody/{A}lstStyle"), level)
        if own is not None:
            return own
        ph = shape.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
        if ph is None:
            # A plain text box: the master's "other" style and the deck's
            # defaults give text boxes no bullets in every deck we have.
            return False
        ph_type, idx = ph.get("type"), ph.get("idx")
        layout_part = self._related(part, _LAYOUT_REL)
        master_part = self._related(layout_part, _MASTER_REL) if layout_part else None
        for name, by_idx in ((layout_part, True), (master_part, False)):
            root = self._part(name)
            sp = self._placeholder(root, ph_type, idx, by_idx) if root is not None else None
            if sp is not None:
                found = self._bullet(sp.find(f"{P}txBody/{A}lstStyle"), level)
                if found is not None:
                    return found
        master = self._part(master_part)
        if master is None:
            return False
        style = (
            "titleStyle" if ph_type in ("title", "ctrTitle")
            else "bodyStyle" if ph_type in (None, "body", "obj", "subTitle")
            else "otherStyle"
        )
        return bool(self._bullet(master.find(f"{P}txStyles/{P}{style}"), level))


def _list_text(obj: TextObject, p: etree._Element, units: int, unit: Unit, text: str,
               default_pt: float, inherits_bullets: bool = False) -> str:
    """The box's words with her list, as a slide can hold it (see
    office_lists): real bullets or numbers when the box is this paragraph's
    only words and one line, else each line starting with its marker.

    Only added: a paragraph that already says how its bullets look keeps
    that, and nothing is ever taken away. One with bullets of its own
    draws them, so its text is left as it is. One that inherits bullets
    (a body placeholder, usually) gets her real list over them, or, where
    lines start with their markers, an explicit no-bullet, so PowerPoint's
    own bullet is not drawn in front of her "a." as well.
    """
    paras = office_lists.plan(obj)
    if not paras or obj.suppressed:
        return text
    ppr = p.find(f"{A}pPr")
    own = [] if ppr is None else [c for c in ppr if etree.QName(c).localname in _BULLET_TAGS]
    if any(etree.QName(c).localname != "buNone" for c in own):
        return text
    item = office_lists.single_item(paras)
    if item is None or units != 1 or own:
        body = office_lists.marker_text(paras)
        if inherits_bullets and not own:
            if ppr is None:
                ppr = etree.Element(f"{A}pPr")
                p.insert(0, ppr)  # a:pPr must be the first child of a:p
            _insert_ppr(ppr, etree.Element(f"{A}buNone"))
    else:
        if ppr is None:
            ppr = etree.Element(f"{A}pPr")
            p.insert(0, ppr)  # a:pPr must be the first child of a:p
        rpr = unit.groups[0].lead.find(f"{A}rPr")
        size = (int(rpr.get("sz")) / 100.0) if rpr is not None and rpr.get("sz") else default_pt
        offset, hang = office_lists.hang_points(item, size)
        ppr.set("lvl", str(item.level))
        ppr.set("marL", str(int(round((offset + hang) * EMU_PER_POINT))))
        ppr.set("indent", str(-int(round(hang * EMU_PER_POINT))))
        if obj.manual_list == "bullet":
            if item.marker in office_lists.SYMBOL_BULLETS:
                font = etree.Element(f"{A}buFont")
                font.set("typeface", office_lists.SYMBOL_FONT)
                _insert_ppr(ppr, font)
            bullet = etree.Element(f"{A}buChar")
            bullet.set("char", item.marker)
        else:
            bullet = etree.Element(f"{A}buAutoNum")
            bullet.set("type", office_lists.PPTX_AUTONUM[
                lists.level_style("number", obj.manual_list_style, item.level)
            ])
            if item.level == 0 and (obj.manual_list_start or 1) != 1:
                bullet.set("startAt", str(obj.manual_list_start))
        _insert_ppr(ppr, bullet)
        body = item.body
    if not body.strip():
        return body
    # The source's own spacing at the edges, as _output_text keeps it.
    source = unit.text
    left = source[: len(source) - len(source.lstrip())]
    right = source[len(source.rstrip()):]
    return f"{left}{body.strip()}{right}"


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
    # The source's spacing at a unit boundary is authoritative; see the same
    # reasoning in docx_builder._preserve_edges.
    translated = obj.translated_text
    if not translated.strip():
        return translated
    lead = source[: len(source) - len(source.lstrip())]
    trail = source[len(source.rstrip()) :]
    return f"{lead}{translated.strip()}{trail}"


# --- Overflow -------------------------------------------------------------


def estimate_text_height(
    lines_of: list[tuple[str, float]], width_pt: float
) -> float:
    """Estimated rendered height of a text frame's contents, in points.

    An average glyph is taken as 0.5 em and leading as 1.2. Both are
    approximations -- the true values depend on the font on the machine that
    opens the deck -- so this is used only to decide whether to ASK
    PowerPoint to shrink, never to position anything.
    """
    usable = max(12.0, width_pt - 2 * _INSET_X)
    total = 0.0
    for text, size in lines_of:
        per_line = max(1, int(usable / (size * 0.5)))
        lines = max(1, math.ceil(len(text) / per_line))
        total += lines * size * 1.2
    return total


def _body_pr(shape: etree._Element) -> etree._Element | None:
    tx = shape.find(f"{P}txBody")
    return tx.find(f"{A}bodyPr") if tx is not None else None


def _apply_autofit(shape: etree._Element, scale: float) -> bool:
    """Ask PowerPoint to shrink this shape's text to `scale`.

    Written as `a:normAutofit`, the element PowerPoint writes itself for
    "Shrink text on overflow", so the deck renders identically in PowerPoint,
    Keynote and Google Slides. A shape already set to grow to fit its text is
    left alone -- it will handle the longer translation on its own.
    """
    body = _body_pr(shape)
    if body is None:
        return False
    if body.find(f"{A}spAutoFit") is not None:
        return False
    for tag in (f"{A}noAutofit", f"{A}normAutofit"):
        for existing in body.findall(tag):
            body.remove(existing)
    fit = etree.SubElement(body, f"{A}normAutofit")
    # Thousandths of a percent, as PowerPoint writes them.
    fit.set("fontScale", str(int(round(scale * 100_000))))
    if scale < 0.9:
        fit.set("lnSpcReduction", "10000")
    return True


def rebuild_pptx(
    source_bytes: bytes, pages: list[Page], *, default_font_pt: float = 18.0
) -> PptxBuildResult:
    """Write every slide's translations back into the source package."""
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
        inherited = _Inherited(zin)

        for part, part_targets in targets.items():
            if part not in names:
                unresolved.extend(t.obj.anchor or "" for t in part_targets)
                continue

            root = parse_xml(zin.read(part))
            by_anchor: dict[str, Unit] = {}
            para_of: dict[str, etree._Element] = {}
            units_in: dict[str, int] = {}
            for ref in iter_part_paragraphs(root, part):
                for i, unit in enumerate(ref.units):
                    by_anchor[ref.anchor(i)] = unit
                    para_of[ref.anchor(i)] = ref.element
                    units_in[ref.anchor(i)] = len(ref.units)

            touched = False
            changed_paragraphs: set[str] = set()
            for t in part_targets:
                anchor = t.obj.anchor or ""
                unit = by_anchor.get(anchor)
                s = stats[t.page_index]
                if unit is None or not unit.groups:
                    unresolved.append(anchor)
                    s.findings.append(
                        QAFinding(
                            severity="error",
                            code="pptx_anchor_lost",
                            message=(
                                "Text on this slide kept its original wording "
                                "because it could not be located in the file."
                            ),
                            object_ids=[t.obj.id],
                            details={"anchor": anchor},
                        )
                    )
                    continue

                text = _output_text(t.obj, unit.text)
                if text is None:
                    continue
                text = _list_text(
                    t.obj, para_of[anchor], units_in[anchor], unit, text, default_font_pt,
                    inherits_bullets=bool(t.obj.manual_list) and inherited.bullets(
                        part, para_of[anchor]
                    ),
                )

                lead = unit.groups[0].lead
                _set_run_text(lead, text)
                for gi, group in enumerate(unit.groups):
                    for ri, run in enumerate(group.runs):
                        if gi == 0 and ri == 0:
                            continue
                        _set_run_text(run, "")
                _apply_manual_style(lead, t.obj)
                changed_paragraphs.add(anchor)

                touched = True
                if t.obj.suppressed:
                    s.suppressed += 1
                else:
                    s.written += 1
                if len(unit.groups) > 1:
                    s.findings.append(
                        QAFinding(
                            severity="warning",
                            code="pptx_formatting_collapsed",
                            message=(
                                "Bold or italic inside this text was replaced "
                                "by the surrounding style, because the "
                                "translated wording no longer lines up with "
                                "the original words."
                            ),
                            object_ids=[t.obj.id],
                        )
                    )

            if touched:
                _fit_shapes(
                    part_targets,
                    para_of,
                    changed_paragraphs,
                    stats,
                    default_font_pt,
                )
                rewritten[part] = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )

        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = rewritten.get(item.filename)
                if data is None:
                    data = zin.read(item.filename)
                zout.writestr(item, data)

    return PptxBuildResult(data=out.getvalue(), stats=stats, unresolved=unresolved)


def _fit_shapes(
    part_targets: list[_Target],
    para_of: dict[str, etree._Element],
    changed: set[str],
    stats: dict[int, PageBuildStats],
    default_font_pt: float,
) -> None:
    """Shrink the text of any shape the translation no longer fits.

    Done once per shape, after every run in the part has been rewritten, so
    the measurement is of the final text and not of a half-translated
    intermediate.

    The shape's box comes from the parsed blocks rather than from the XML
    here. PowerPoint omits geometry on any placeholder the author did not
    move, so reading the slide alone would find no box for most of a deck's
    text -- and silently skip the shapes most likely to overflow. The parser
    has already resolved those through the layout and master, and its
    per-paragraph boxes tile the shape, so their union is the shape.
    """
    page_index = part_targets[0].page_index if part_targets else 0
    s = stats.get(page_index)
    if s is None:
        return

    by_anchor = {t.obj.anchor: t.obj for t in part_targets if t.obj.anchor}
    shapes: dict[str, _ShapeFit] = {}
    for anchor in changed:
        p = para_of.get(anchor)
        shape = owning_shape(p) if p is not None else None
        obj = by_anchor.get(anchor)
        if shape is None or obj is None:
            continue
        path = shape.getroottree().getpath(shape)
        fit = shapes.get(path)
        if fit is None:
            fit = shapes[path] = _ShapeFit(shape=shape)
        b = obj.bbox
        fit.union(b.x0, b.y0, b.x1, b.y1)
        # Point size comes from the parsed block, which resolved it through
        # the slide layout and master. Reading `a:rPr/@sz` off the run finds
        # nothing on most decks -- the size is inherited, not stated -- and
        # a 44pt title measured as 18pt never looks like it overflows.
        size = obj.style.font_size or default_font_pt
        # Units of one paragraph share a line, so they are measured as one.
        para_key = anchor.rsplit("#", 1)[0]
        text = obj.translated_text if obj.translated_text is not None else obj.text
        fit.add_text(para_key, text or "", obj.text or "", size)

    for fit in shapes.values():
        width = fit.width
        height = fit.height
        if width <= 0 or height <= 0:
            continue

        lines = fit.lines()
        if not lines:
            continue
        shape = fit.shape
        needed = estimate_text_height(lines, width)
        # The ORIGINAL text is the reference, not the box.
        #
        # The estimator is only approximate in absolute terms -- it assumes
        # an average glyph of 0.5 em, while the real width depends on the
        # font on the machine that opens the deck. Measured against the box
        # alone it called a 36-character title an overflow, which would have
        # shrunk text on slides where nothing had changed. But whatever the
        # estimator's bias is, it applies equally to both strings, so the
        # RATIO between them is sound. The author saw the original fit, so
        # anything that is not longer than the original needs nothing.
        baseline = estimate_text_height(fit.source_lines(), width)
        available = max(baseline, height - 2 * _INSET_Y)
        if needed <= available:
            continue

        # Height scales roughly with the square of a font scale, because
        # smaller text both shortens each line and fits more per line.
        scale = math.sqrt(available / needed)
        clamped = max(MIN_FONT_SCALE, scale)
        if not _apply_autofit(shape, clamped):
            continue
        s.shrunk += 1

        if scale < MIN_FONT_SCALE:
            s.overflowed += 1
            s.findings.append(
                QAFinding(
                    severity="warning",
                    code="pptx_text_may_not_fit",
                    message=(
                        "Translated text on this slide is much longer than "
                        "the original and may not fit its box, even at a "
                        "smaller size. Check this slide before using it."
                    ),
                    details={"estimated_scale": round(scale, 2)},
                )
            )
