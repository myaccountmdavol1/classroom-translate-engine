"""The single authoritative walk over a DOCX package's text.

WHY THIS EXISTS INSTEAD OF python-docx's `Document.paragraphs`

Two measured gaps make the convenience API unsafe for this pipeline, and
both fail in the silent direction the spec forbids (sec. 3):

  * `paragraph.runs` EXCLUDES runs nested in a `w:hyperlink`, while
    `paragraph.text` INCLUDES their characters. Translating `.text` and
    writing it back over `.runs` therefore leaves the hyperlink's original
    words behind -- the sentence is emitted twice, half of it untranslated.
  * `Document.paragraphs` yields body-level paragraphs only. Text boxes
    (`w:txbxContent`), table cells, headers, footers, footnotes and endnotes
    are simply absent. Worksheets use text boxes constantly, so "translate
    the document" would quietly skip the callout boxes.

So the walk is done over the XML, over every part of the package, and it is
shared verbatim by the parser and the writer. That sharing is the point: the
writer resolves an anchor by running the *same* walk, so an anchor can never
mean one paragraph on the way in and a different one on the way out.

ANCHOR FORMAT

    <part name>#<paragraph ordinal in that part>#<run-group ordinal>

Ordinals are positions in this walk's deterministic document order. They stay
valid because reconstruction only ever rewrites the text inside existing
runs; it never inserts or deletes a paragraph before the walk has finished
resolving every anchor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Elements inside a run that contribute characters to the paragraph.
_TEXT_TAGS = {f"{W}t", f"{W}tab", f"{W}br", f"{W}cr", f"{W}noBreakHyphen"}
# A run holding one of these owns embedded content (a picture, a text box, a
# field result). Its text is never rewritten and it is never deleted, because
# deleting it would take the drawing or the nested paragraph with it.
_OPAQUE_TAGS = {f"{W}drawing", f"{W}pict", f"{W}object", f"{W}fldChar", f"{W}instrText"}


def _run_is_textual(r: etree._Element) -> bool:
    """True when this run is plain text we may rewrite.

    Any character-producing element counts, not just `w:t`. Word routinely
    puts a tab in a run of its own, and such a run has no `w:t` at all --
    skipping it dropped the tab in `Name: ____<tab>Date: ____`, merging two
    fields into one without a trace.
    """
    if not any(r.find(tag) is not None for tag in _TEXT_TAGS):
        return False
    for el in r.iter():
        if el.tag in _OPAQUE_TAGS or el.tag == f"{W}p":
            return False
    return True


def run_text(r: etree._Element) -> str:
    """Characters this run contributes, ignoring anything nested deeper.

    Only *direct* children are read. A run that contains a text box also
    contains that box's `w:t` elements, and those belong to their own
    paragraph, not to this one.
    """
    out: list[str] = []
    for el in r:
        if el.tag == f"{W}t":
            out.append(el.text or "")
        elif el.tag in (f"{W}tab",):
            out.append("\t")
        elif el.tag in (f"{W}br", f"{W}cr"):
            out.append("\n")
        elif el.tag == f"{W}noBreakHyphen":
            out.append("‑")
    return "".join(out)


def own_runs(p: etree._Element) -> list[tuple[etree._Element, bool]]:
    """Runs belonging to this paragraph, in document order.

    Descends through wrappers that do not start a new paragraph -- `w:hyperlink`
    (the gap described above), plus revision and smart-tag wrappers -- and
    stops at any nested `w:p`, whose runs belong to that paragraph instead.

    Each run is flagged with whether it sits inside a `w:hyperlink`. That flag
    is load-bearing: the link's address lives on the wrapper, not the run, so
    emptying the run to make room for a translation elsewhere would leave a
    link pointing at nothing. Section 20 protects URLs, so link text is never
    merged into a neighbouring stretch.
    """
    found: list[tuple[etree._Element, bool]] = []

    def visit(node: etree._Element, in_link: bool) -> None:
        for child in node:
            if child.tag == f"{W}p":
                continue  # a nested paragraph owns its own runs
            if child.tag == f"{W}r":
                found.append((child, in_link))
            elif child.tag == f"{W}hyperlink":
                visit(child, True)
            elif child.tag in (f"{W}ins", f"{W}smartTag", f"{W}sdt",
                               f"{W}sdtContent", f"{W}bdo", f"{W}dir"):
                visit(child, in_link)

    visit(p, False)
    return found


# --- Run grouping ---------------------------------------------------------
# Word splits runs for reasons that have nothing to do with appearance:
# revision-save ids, proofing marks, language tagging. Those splits must be
# merged away before any decision is made about formatting, or almost every
# paragraph would look "mixed" and lose its inline styling for no reason.

# Properties that actually change how a run looks. Anything outside this set
# (w:rsid*, w:lang, w:noProof, ...) is deliberately ignored.
_VISIBLE_PROPS = (
    "b", "bCs", "i", "iCs", "u", "strike", "dstrike", "caps", "smallCaps",
    "vertAlign", "color", "sz", "szCs", "rFonts", "highlight", "shd",
    "spacing", "position", "em", "outline", "vanish", "rStyle",
)


def format_key(r: etree._Element) -> tuple:
    """A fingerprint of a run's visible formatting.

    Two runs with equal keys are indistinguishable on the page, so they can be
    merged and treated as one.
    """
    rpr = r.find(f"{W}rPr")
    if rpr is None:
        return ()
    parts: list[tuple] = []
    for name in _VISIBLE_PROPS:
        el = rpr.find(f"{W}{name}")
        if el is None:
            continue
        attrs = tuple(sorted((k, v) for k, v in el.attrib.items()))
        parts.append((name, attrs))
    return tuple(parts)


@dataclass
class RunGroup:
    """Adjacent runs that look identical, treated as one stretch of text."""

    runs: list[etree._Element]
    key: tuple
    # True when these runs are the clickable text of a hyperlink.
    in_link: bool = False

    @property
    def text(self) -> str:
        return "".join(run_text(r) for r in self.runs)

    @property
    def lead(self) -> etree._Element:
        """The run that keeps the text when the group is rewritten."""
        return self.runs[0]


def group_runs(runs: list[tuple[etree._Element, bool]]) -> list[RunGroup]:
    """Collapse adjacent same-looking runs. Opaque runs are dropped.

    Dropping opaque runs here is what keeps pictures and text boxes intact:
    they are never part of a group, so they are never rewritten or deleted.
    Link text never merges with surrounding text, for the reason in
    `own_runs`.
    """
    groups: list[RunGroup] = []
    for r, in_link in runs:
        if not _run_is_textual(r):
            continue
        key = format_key(r)
        if groups and groups[-1].key == key and groups[-1].in_link == in_link \
                and not in_link:
            groups[-1].runs.append(r)
        elif groups and in_link and groups[-1].in_link \
                and groups[-1].runs[0].getparent() is r.getparent():
            # Same link, split into runs by proofing: still one stretch.
            groups[-1].runs.append(r)
        else:
            groups.append(RunGroup(runs=[r], key=key, in_link=in_link))
    return groups


# --- Splitting a paragraph into translation units -------------------------

# A formatted lead-in that is safe to translate on its own: a question number,
# an answer letter, or a short bold label ending in a colon. These are the
# overwhelmingly common reason a classroom paragraph has mixed formatting,
# and they are self-contained phrases, so splitting them costs no meaning.
_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,3}\s*[.)]"              # 1.  12)
    r"|\(?[A-Da-d]\s*[.)]"         # A.  (b)
    r"|[A-Z][A-Za-z ]{0,20}:"      # Directions:  Warning:
    r")\s*$"
)

MAX_LABEL_CHARS = 24


def _is_label(text: str) -> bool:
    return len(text) <= MAX_LABEL_CHARS and bool(_LABEL_RE.match(text))


@dataclass
class Unit:
    """One translatable stretch of a paragraph.

    `groups` is what gets rewritten. When more than one group is listed the
    paragraph could not keep its inline formatting: the translation is a
    single new string and there is no honest way to know which of its words
    belonged to which original run, so the groups are merged into the first
    and `collapsed` records that a distinction was lost.
    """

    groups: list[RunGroup]
    collapsed: bool = False
    # True when this unit is a hyperlink's clickable text, split out to keep
    # the link intact rather than because it reads well alone.
    is_link: bool = False

    @property
    def text(self) -> str:
        return "".join(g.text for g in self.groups)


def split_paragraph(groups: list[RunGroup]) -> list[Unit]:
    """Decide how a paragraph's runs map onto translation units.

    The rule, in order:

      1. One group  -> one unit. Formatting is preserved exactly.
      2. A short formatted label followed by one differently-formatted body
         -> two units. Both keep their formatting, and neither is a sentence
         fragment, so translating them apart is safe.
      3. Anything else -> one unit spanning the paragraph, with inline
         formatting collapsed to the first group's and the loss recorded.

    Case 3 is the honest answer to a question with no good one. The obvious
    alternative -- marking run boundaries with placeholders so the translator
    carries them along -- was measured against the local NMT engine and
    rejected: the markers survive, but the words beside them stop being
    translated ("the cell" came back as "the cell" in Spanish and French,
    "Solve" transliterated to "Сольве" in Russian). A visible loss of bold is
    a far smaller harm than a silently untranslated word in a worksheet.
    """
    if not groups:
        return []

    # Hyperlinks cut the paragraph first: their text has to stay inside the
    # link element, so it can never be folded into a neighbour.
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


def _split_plain(groups: list[RunGroup]) -> list[Unit]:
    """Apply rules 1-3 to a stretch of groups containing no hyperlink."""
    if not groups:
        return []
    if len(groups) == 1:
        return [Unit(groups=[groups[0]])]
    if len(groups) == 2 and _is_label(groups[0].text) and groups[1].text.strip():
        return [Unit(groups=[groups[0]]), Unit(groups=[groups[1]])]
    return [Unit(groups=list(groups), collapsed=True)]


# --- Walking the package --------------------------------------------------


@dataclass
class ParaRef:
    """One paragraph, located."""

    part_name: str
    ordinal: int
    element: etree._Element
    # Which section of the document this paragraph is displayed in. Used only
    # to decide which Page it lands on.
    section: int = 0
    # Where it came from, for the reviewer: "body", "table", "textbox",
    # "header", "footer", "footnote", "endnote".
    origin: str = "body"
    units: list[Unit] = field(default_factory=list)

    def anchor(self, unit_index: int) -> str:
        return f"{self.part_name}#{self.ordinal}#{unit_index}"


def _origin_of(p: etree._Element) -> str:
    """Classify a paragraph by what encloses it."""
    node = p.getparent()
    while node is not None:
        tag = node.tag
        if tag == f"{W}txbxContent":
            return "textbox"
        if tag == f"{W}tc":
            return "table"
        node = node.getparent()
    return "body"


def iter_part_paragraphs(
    root: etree._Element, part_name: str, *, origin_override: str | None = None
) -> Iterator[ParaRef]:
    """Every `w:p` in one part, in document order, numbered from zero.

    Document order is `root.iter()` order, which is exactly the order the
    writer will see, so ordinals agree on both sides by construction.
    """
    for ordinal, p in enumerate(el for el in root.iter(f"{W}p")):
        ref = ParaRef(
            part_name=part_name,
            ordinal=ordinal,
            element=p,
            origin=origin_override or _origin_of(p),
        )
        ref.units = split_paragraph(group_runs(own_runs(p)))
        yield ref


def section_of_body_paragraph(p: etree._Element) -> bool:
    """True when this paragraph ends a section (it carries a `w:sectPr`)."""
    ppr = p.find(f"{W}pPr")
    return ppr is not None and ppr.find(f"{W}sectPr") is not None


def assign_sections(refs: list[ParaRef]) -> int:
    """Number each body paragraph's section, returning the section count.

    In WordprocessingML a section break lives in the `w:sectPr` of the LAST
    paragraph of the section it closes; the final section's `sectPr` hangs off
    `w:body` instead. So the paragraph carrying the break belongs to the
    section it terminates, and the next one starts the following section.
    """
    section = 0
    for ref in refs:
        # Nested paragraphs (table cells, text boxes) simply inherit whatever
        # section is open around them, which is what this assignment already
        # does; only a body paragraph can close one.
        ref.section = section
        if ref.origin == "body" and section_of_body_paragraph(ref.element):
            section += 1
    return section + 1


def paragraph_style(p: etree._Element) -> str | None:
    ppr = p.find(f"{W}pPr")
    if ppr is None:
        return None
    st = ppr.find(f"{W}pStyle")
    return st.get(f"{W}val") if st is not None else None


def paragraph_alignment(p: etree._Element) -> str | None:
    ppr = p.find(f"{W}pPr")
    if ppr is None:
        return None
    jc = ppr.find(f"{W}jc")
    return jc.get(f"{W}val") if jc is not None else None


def parse_xml(data: bytes) -> etree._Element:
    """Parse one part of the package, with entity expansion disabled.

    These files are uploaded by the public. `resolve_entities=False` is what
    stops a crafted document declaring `<!ENTITY xxe SYSTEM "file:///etc/
    passwd">` and having its contents read into a paragraph the uploader
    then gets back as a "translation"; `no_network` stops the same trick
    over HTTP, and `huge_tree=False` keeps entity-expansion bombs bounded.
    A document using an undefined entity fails to parse, which is the
    intended outcome.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser)
