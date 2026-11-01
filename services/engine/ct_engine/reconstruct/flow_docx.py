"""Export a translated PDF as a Word file that is easy to edit.

The other Word download (layered_docx) keeps the look: every line a text box
on a picture of the page. This one keeps the words: paragraphs that flow and
wrap as she types, in the order they are read, so it opens cleanly in Word
and in Google Docs. The page is rebuilt from what the engine knows about it:

  * each block's paragraphs as they were wrapped and drawn (DrawnLayout
    rows), with their list markers and hanging indents, face, size, colour;
  * blocks side by side on one line (Name / Date / Period) as one paragraph
    with tab stops, and side-by-side columns as a table without borders;
  * answer lines as lines to write on, an empty box as a box to write in, a
    box with words in it as a bordered box she can edit them in, and a grid
    of ruled lines as a Word table;
  * pictures and drawings cut from the translated PDF, so their translated
    labels come with them -- with the editable words taken out first, so
    nothing appears twice.

Each PDF page starts a new Word page (its own section, at its own size), so
page 3 of the worksheet is still page 3 unless she adds to it. A scanned
page's words come through; its drawings, part of the scan, do not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO

import fitz
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt, RGBColor

from ..model.enums import SemanticRole, TextAlignment, TextSource
from .layered_docx import MAX_PAGE, MIN_PAGE, family

# Something covering this share of the page is its background, not a picture.
BACKDROP = 0.85
# Ruled lines shorter than this share of the page width underline a word
# ("Tema", a heading): one under a whole block underlines it, others are
# not rebuilt.
SHORT_RULE = 0.3
# A box smaller than this (points) both ways is a tick box or a mark: kept
# as a small picture rather than made a box to write in.
SMALL_BOX = 30
# A picture no bigger than this (points) beside a one-line heading is an
# icon, set at the end of the heading's line.
ICON = 2 * SMALL_BOX
# The space between two things is kept as on the page -- it is often room
# to write the answer in -- up to most of a page.
MAX_GAP = 600
PICTURE_DPI = 200
_BUILTINS = {
    "helv": ("Arial", False, False), "hebo": ("Arial", True, False),
    "heit": ("Arial", False, True), "hebi": ("Arial", True, True),
    "tiro": ("Times New Roman", False, False), "tibo": ("Times New Roman", True, False),
    "tiit": ("Times New Roman", False, True), "tibi": ("Times New Roman", True, True),
    "cour": ("Courier New", False, False), "cobo": ("Courier New", True, False),
    "coit": ("Courier New", False, True), "cobi": ("Courier New", True, True),
}
_ALIGN = {
    TextAlignment.CENTER: WD_ALIGN_PARAGRAPH.CENTER,
    TextAlignment.RIGHT: WD_ALIGN_PARAGRAPH.RIGHT,
    TextAlignment.JUSTIFY: WD_ALIGN_PARAGRAPH.JUSTIFY,
}
# Children that come after w:pBdr in w:pPr, and after w:tblInd in w:tblPr,
# in schema order: Word calls a file with them out of order damaged.
_AFTER_PBDR = (
    "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
    "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
    "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
    "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
    "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
    "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange",
)
_AFTER_TBLIND = ("w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
_CJK = r"[　-鿿가-힯]"

Box = tuple[float, float, float, float]


@dataclass
class Item:
    """One thing on the page to rebuild, in page points."""

    kind: str  # text | picture | box | lines | rule | grid
    box: Box
    text: object | None = None  # the TextObject, for "text"
    inner: list["Item"] = field(default_factory=list)  # a box's contents
    count: int = 0  # answer lines
    xs: list[float] = field(default_factory=list)  # a grid's column edges
    ys: list[float] = field(default_factory=list)  # ... and row edges
    cells: dict = field(default_factory=dict)  # (row, col) -> [Item]
    stroke: str | None = None
    fill: str | None = None
    underline: bool = False  # a ruled line under the whole block

    @property
    def x0(self): return self.box[0]
    @property
    def y0(self): return self.box[1]
    @property
    def x1(self): return self.box[2]
    @property
    def y1(self): return self.box[3]

    def single_line(self) -> bool:
        if self.kind != "text":
            return False
        layout = self.text.drawn_layout
        if layout is not None:
            return len([r for r in layout.rows if r.text]) <= 1
        return "\n" not in _plain_text(self.text)


def _rect(r) -> Box:
    return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))


def _ink(t) -> Box:
    """Where a block's words are, which can be narrower than its frame: a
    centred heading's frame reaches the picture beside it, its words do
    not."""
    x0, y0, x1, y1 = _rect(t.bbox)
    rows = [r for r in t.drawn_layout.rows if r.text] if t.drawn_layout is not None else []
    if not rows or t.drawn_layout.angle:
        return x0, y0, x1, y1
    left = min([r.x0 for r in rows] + [r.marker_x0 for r in rows if r.marker_x0 is not None])
    return max(x0, left - 1), y0, min(x1, max(r.x1 for r in rows) + 1), y1


def _inside(point, box: Box, pad: float = 0.0) -> bool:
    x, y = point
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def _centre(box: Box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _hex(colour) -> str | None:
    if colour is None:
        return None
    if isinstance(colour, str):
        value = colour.lstrip("#").upper()[:6]
        return value if re.fullmatch(r"[0-9A-F]{6}", value) else None
    return "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in colour[:3])


def _plain_text(t) -> str:
    """The words the PDF shows for a block that has no drawn layout."""
    if t.translated_text and not t.keep_original:
        return t.translated_text
    return t.flat_text


def _words(t) -> str:
    """A one-line block's words, as drawn."""
    if t.drawn_layout is not None and t.drawn_layout.rows:
        return _join([r.text for r in t.drawn_layout.rows if r.text])
    return _plain_text(t).replace("​", "")


def _join(rows: list[str]) -> str:
    """Wrapped rows back into one paragraph: a space between them, except
    between Chinese, Japanese or Korean characters."""
    out = ""
    for text in rows:
        if out and not (re.match(_CJK, text[:1]) or re.search(_CJK + "$", out)):
            out += " "
        out += text
    return out


def _page_number(t, height: float) -> bool:
    """A page number, which Word numbers itself. Only at the page's top or
    bottom edge: a number in a table can be given the role too."""
    return (t.role is SemanticRole.PAGE_NUMBER and not t.suppressed
            and (t.bbox.y0 > 0.88 * height or t.bbox.y1 < 0.08 * height))


def _visible(t) -> bool:
    if t.suppressed:
        return False
    if t.drawn_layout is not None:
        return any(r.text.strip() for r in t.drawn_layout.rows) or any(
            s.strip() for _, _, s in t.drawn_layout.lines)
    return bool(_plain_text(t).replace("​", "").strip())


# --- Reading the page -----------------------------------------------------------
def _grids(pdf_page) -> list[Item]:
    """Tables drawn with lines or boxes, by their cells' edges (PyMuPDF's
    table finder, from the drawing alone: the words are placed after)."""
    def edges(values):
        values = sorted(values)
        return [v for k, v in enumerate(values) if k == 0 or v - values[k - 1] > 3]

    grids = []
    for table in pdf_page.find_tables(strategy="lines").tables:
        # One row or one column is a box or a pair of the patches that
        # cover replaced words on a scan, not a table.
        if table.row_count < 2 or table.col_count < 2:
            continue
        cells = [c for row in table.rows for c in row.cells if c]
        xs = edges({c[0] for c in cells} | {c[2] for c in cells})
        ys = edges({c[1] for c in cells} | {c[3] for c in cells})
        if len(xs) >= 2 and len(ys) >= 2:
            grids.append(Item("grid", (xs[0], ys[0], xs[-1], ys[-1]), xs=xs, ys=ys))
    return grids


def _flat(pdf_page, box: Box) -> tuple[bool, str | None]:
    """Whether a picture is one colour inside its edge (a frame drawn as an
    image, not a photo), and that colour when it is not white."""
    import numpy as np

    w, h = box[2] - box[0], box[3] - box[1]
    inset = fitz.Rect(box[0] + 0.08 * w, box[1] + 0.08 * h, box[2] - 0.08 * w, box[3] - 0.08 * h)
    if inset.width < 4 or inset.height < 4:
        return False, None
    pix = pdf_page.get_pixmap(clip=inset, dpi=36, alpha=False)
    px = np.frombuffer(pix.samples, np.uint8).reshape(-1, pix.n)[:, :3].astype(int)
    median = np.median(px, axis=0)
    flat = (np.abs(px - median).max(axis=1) <= 16).mean() >= 0.97
    colour = "".join(f"{int(c):02X}" for c in median)
    return flat, (None if median.min() >= 235 else colour)


def _same_box(a: Box, b: Box, pad: float = 2.0) -> bool:
    return all(abs(p - q) <= pad for p, q in zip(a, b))


def _shows(pdf_page, box: Box, colour: str, on_top: list[Box]) -> bool:
    """Whether the page shows `colour` over most of the box, the words and
    pictures on it aside. A filled path can be a mask or a clip drawn under
    the real one (a Canva panel comes with a black one of its size)."""
    import numpy as np

    # Its edge aside too: a rounded outline, and the page in its corners.
    w, h = box[2] - box[0], box[3] - box[1]
    box = (box[0] + 0.06 * w, box[1] + 0.06 * h, box[2] - 0.06 * w, box[3] - 0.06 * h)
    pix = pdf_page.get_pixmap(clip=fitz.Rect(box), dpi=36, alpha=False)
    px = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    keep = np.ones(px.shape[:2], bool)
    sx, sy = pix.width / max(1.0, box[2] - box[0]), pix.height / max(1.0, box[3] - box[1])
    for r in on_top:
        x0, x1 = int((r[0] - box[0]) * sx), int((r[2] - box[0]) * sx) + 1
        y0, y1 = int((r[1] - box[1]) * sy), int((r[3] - box[1]) * sy) + 1
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    if keep.sum() < 16:
        return False
    want = np.array([int(colour[k:k + 2], 16) for k in (0, 2, 4)])
    return (np.abs(px[keep] - want).max(axis=1) <= 24).mean() >= 0.8


def _panels(vectors: list, texts: list, images: list, pdf_page,
            used: set[int]) -> tuple[list[Item], set[int]]:
    """Coloured panels with words on them, as boxes to hold the words.

    A filled rectangle is not an answer box (the parser leaves filled ones
    out of looks_like_box), so it was a picture, and the words on a Canva
    slide's panel came out after it, white on a white page. One with words
    on it that the page really shows in its colour is a box of that colour;
    the outline and masks drawn at its size go with it, its border the
    outline's colour (else its own, never black).
    """
    frames: list[Item] = []
    taken: set[int] = set()
    on_top = [_rect(t.drawn_bbox or t.bbox) for t in texts] + [_rect(m.bbox) for m in images]
    for i, v in enumerate(vectors):
        b = _rect(v.bbox)
        fill = _hex(v.fill_color)
        if (i in used or i in taken or not v.is_filled or not fill or v.rect_count < 1
                or v.curve_count or min(b[2] - b[0], b[3] - b[1]) < SMALL_BOX):
            continue
        on = [t for t in texts if t.source in (TextSource.NATIVE, TextSource.MANUAL)
              and _inside(_centre(_rect(t.bbox)), b)]
        # Icons drawn on it are on top of it too.
        drawn = [_rect(w.bbox) for k, w in enumerate(vectors)
                 if k != i and _inside(_centre(_rect(w.bbox)), b) and _area(_rect(w.bbox)) < 0.25 * _area(b)]
        if not on or not _shows(pdf_page, b, fill, on_top + drawn):
            continue
        twins = {k for k, w in enumerate(vectors) if k not in used and _same_box(_rect(w.bbox), b)}
        outline = next((_hex(vectors[k].stroke_color) for k in twins
                        if not vectors[k].is_filled and vectors[k].stroke_color), None)
        frames.append(Item("box", b, stroke=outline or fill, fill=fill))
        taken |= twins | {i}
    return frames, taken


def _merge(pictures: list[Item]) -> list[Item]:
    """Drawings that touch are one picture."""
    out = list(pictures)
    merged = True
    while merged:
        merged = False
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                p, q = out[a].box, out[b].box
                if p[0] - 4 <= q[2] and q[0] - 4 <= p[2] and p[1] - 4 <= q[3] and q[1] - 4 <= p[3]:
                    out[a] = Item("picture", (min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3])))
                    del out[b]
                    merged = True
                    break
            if merged:
                break
    return out


def _read_page(page, pdf_page) -> tuple[list[Item], list]:
    """The page's items to rebuild, and its footer blocks."""
    W, H = page.width, page.height

    def backdrop(b: Box) -> bool:
        return _area(b) >= BACKDROP * W * H

    texts = [t for t in page.text_objects if _visible(t) and not _page_number(t, H)]
    footer =[t for t in texts if t.role is SemanticRole.FOOTER and t.bbox.y0 > 0.88 * H]
    texts = [t for t in texts if t not in footer]
    claimed: set[str] = set()
    items: list[Item] = []

    vectors = [v for v in page.vector_objects if not backdrop(_rect(v.bbox))]
    rules = [(i, v) for i, v in enumerate(vectors) if v.looks_like_rule]
    grids = _grids(pdf_page)
    used = {i for i, v in rules if any(_inside(_centre(_rect(v.bbox)), g.box, 2) for g in grids)}

    # Answer lines, and the rules they are drawn with.
    for area in page.response_areas:
        a = _rect(area.bbox)
        for i, v in rules:
            if _inside((v.bbox.x0, v.bbox.y0), a, 2) and _inside((v.bbox.x1, v.bbox.y1), a, 2):
                used.add(i)
        if area.kind == "writing_lines" and area.line_count:
            items.append(Item("lines", a, count=area.line_count))

    # Frames (boxes, and images of one flat colour) hold the words in them;
    # other images and drawings are pictures.
    frames, taken = _panels(vectors, texts, page.image_objects, pdf_page, used)
    panels = list(frames)
    used |= taken
    pictures: list[Item] = []
    for i, v in enumerate(vectors):
        b = _rect(v.bbox)
        if i in used or v.looks_like_rule or any(_inside(_centre(b), g.box) for g in grids):
            continue
        if v.looks_like_box and min(b[2] - b[0], b[3] - b[1]) >= SMALL_BOX:
            frames.append(Item("box", b, stroke=v.stroke_color, fill=v.fill_color if v.is_filled else None))
        else:
            pictures.append(Item("picture", b))
    for image in page.image_objects:
        b = _rect(image.bbox)
        if backdrop(b) or _area(b) < 4:
            continue
        flat, colour = _flat(pdf_page, b) if min(b[2] - b[0], b[3] - b[1]) >= SMALL_BOX else (False, None)
        if flat:
            frames.append(Item("box", b, fill=colour))
        else:
            pictures.append(Item("picture", b))
    # Drawings on a panel are merged with each other only, and go in it:
    # merged with a logo overlapping the panel's edge, the form icon on a
    # slide's panel made one picture across both, which took the panel.
    for f in panels:
        on = [p for p in pictures if _inside(_centre(p.box), f.box) and _area(p.box) < _area(f.box)]
        pictures = [p for p in pictures if p not in on]
        f.inner += _merge(on)
    pictures = _merge(pictures)
    # A picture whose top a panel covers (a logo set into a panel's bottom
    # edge) starts below the panel, or it is cut with a strip of the panel
    # and the icons on it.
    for p in pictures:
        for f in panels:
            if f.y0 <= p.y0 < f.y1 < p.y1 and f.x0 - 2 <= p.x0 and p.x1 <= f.x1 + 2:
                p.box = (p.x0, f.y1, p.x1, p.y1)
    # A frame inside a drawing is part of the drawing; a panel, its colour
    # seen on the page, is not.
    frames = [f for f in frames if f in panels
              or not any(_inside(_centre(f.box), p.box) and _area(p.box) > _area(f.box) for p in pictures)]

    # Words in a picture are part of it when they were read from it (and
    # translated into it, or from a scan) or label a drawing; words typed
    # over a picture stay words.
    for p in pictures:
        for t in texts:
            if (t.source in (TextSource.IMAGE_OCR, TextSource.OCR) or t.role is SemanticRole.IMAGE_LABEL) \
                    and _inside(_centre(_rect(t.bbox)), p.box):
                claimed.add(t.id)
    # Words in a grid's cell, or a frame, go in it.
    for g in grids:
        for t in texts:
            c = _centre(_rect(t.bbox))
            if t.id in claimed or not _inside(c, g.box):
                continue
            row = next((r for r in range(len(g.ys) - 1) if g.ys[r] <= c[1] <= g.ys[r + 1]), None)
            col = next((k for k in range(len(g.xs) - 1) if g.xs[k] <= c[0] <= g.xs[k + 1]), None)
            if row is not None and col is not None:
                g.cells.setdefault((row, col), []).append(Item("text", _ink(t), text=t))
                claimed.add(t.id)
    frames.sort(key=lambda f: _area(f.box))
    for f in frames:
        for t in texts:
            if t.id not in claimed and _inside(_centre(_rect(t.bbox)), f.box):
                f.inner.append(Item("text", _ink(t), text=t))
                claimed.add(t.id)
        # A picture or answer lines drawn in the frame go in it too.
        for other in [p for p in pictures if _area(p.box) < _area(f.box)] + [
                i for i in items if i.kind == "lines"]:
            if _inside(_centre(other.box), f.box):
                f.inner.append(other)
                (pictures if other.kind == "picture" else items).remove(other)
    # Frames inside a bigger frame go in it, smallest first.
    for k, f in enumerate(frames):
        outer = next((g for g in frames[k + 1:] if _inside(_centre(f.box), g.box)), None)
        if outer is not None:
            outer.inner.append(f)
    frames = [f for f in frames if not any(f in g.inner for g in frames)]

    underlined: set[str] = set()
    for i, v in rules:
        b = _rect(v.bbox)
        if i in used or b[3] - b[1] >= 2:
            continue
        if b[2] - b[0] >= SHORT_RULE * W:
            items.append(Item("rule", b))
            continue
        under = next((t for t in texts if abs(b[1] - t.bbox.y1) <= 4
                      and b[0] >= t.bbox.x0 - 6 and b[2] <= t.bbox.x1 + 6
                      and b[2] - b[0] >= 0.8 * (t.bbox.x1 - t.bbox.x0)), None)
        if under is not None:
            underlined.add(under.id)
    items += grids + frames + pictures
    items += [Item("text", _ink(t), text=t) for t in texts if t.id not in claimed]
    for item in _texts(items):
        item.underline = item.text.id in underlined
    return items, footer


def _bands(items: list[Item]) -> list[list[Item]]:
    """Items that share some height, top to bottom."""
    bands: list[list[Item]] = []
    span = None
    for item in sorted(items, key=lambda i: (i.y0, i.x0)):
        h = max(1.0, item.y1 - item.y0)
        if span is not None and min(span[1], item.y1) - max(span[0], item.y0) > 0.4 * min(h, span[1] - span[0]):
            bands[-1].append(item)
            span = (span[0], max(span[1], item.y1))
        else:
            bands.append([item])
            span = (item.y0, item.y1)
    return bands


def _columns(band: list[Item]) -> list[list[Item]]:
    """A band's items split where nothing crosses, left to right."""
    columns: list[list[Item]] = []
    reach = None
    for item in sorted(band, key=lambda i: i.x0):
        if reach is not None and item.x0 < reach - 1:
            columns[-1].append(item)
            reach = max(reach, item.x1)
        else:
            columns.append([item])
            reach = item.x1
    return columns


def _heading_with_icons(columns: list[list[Item]]) -> tuple[Item, list[Item]] | None:
    """A one-line heading with small pictures beside it ("Aprender" and its
    link symbol). Made columns, the heading ran into the icon, its column
    was narrower than the word, and Word broke it ("Aprende / r"). The
    heading takes the line's width and the icons follow it in the line."""
    texts = [c[0] for c in columns if len(c) == 1 and c[0].single_line()]
    icons = [c[0] for c in columns if len(c) == 1 and c[0].kind == "picture"
             and max(c[0].x1 - c[0].x0, c[0].y1 - c[0].y0) <= ICON]
    if len(texts) != 1 or len(texts) + len(icons) != len(columns):
        return None
    return texts[0], sorted(icons, key=lambda i: i.x0)


def _texts(items: list[Item]):
    for item in items:
        if item.kind == "text":
            yield item
        yield from _texts(item.inner)
        for cell in item.cells.values():
            yield from _texts(cell)


# --- Writing Word -----------------------------------------------------------------
def _face(t) -> tuple[str, bool, bool]:
    layout = t.drawn_layout
    if layout is not None and layout.font_builtin in _BUILTINS:
        return _BUILTINS[layout.font_builtin]
    if layout is not None and layout.font_path:
        stem = layout.font_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return family(stem), "Bold" in stem, "Italic" in stem
    style = t.style
    name = family(style.font_name_raw) if style.font_name_raw else (style.font_family or "Arial")
    return name, style.bold, style.italic


def _run(p, text: str, t, underline: bool = False):
    name, bold, italic = _face(t)
    run = p.add_run(text)
    font = run.font
    font.name = name
    fonts = run._r.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:cs"), name)
    fonts.set(qn("w:eastAsia"), name)
    layout = t.drawn_layout
    size = layout.size if layout is not None else (t.translated_font_size or t.style.font_size or 11)
    font.size = Pt(max(4.0, round(size * 2) / 2))
    font.bold = bool(t.manual_bold if t.manual_bold is not None else bold)
    font.italic = bool(t.manual_italic if t.manual_italic is not None else italic)
    colour = _hex(layout.color if layout is not None else (t.manual_color or t.style.color))
    if colour and colour != "000000":
        font.color.rgb = RGBColor.from_string(colour)
    if underline or (layout is not None and layout.underline) or t.manual_underline:
        font.underline = True
    return run


def _border(p, colour: str):
    """A line under the paragraph. Word joins paragraphs with the same
    borders into one bordered group, drawing only the group's last bottom
    line, so the line between them is set too."""
    line = f'w:val="single" w:sz="6" w:space="1" w:color="{colour}"'
    p._p.get_or_add_pPr().insert_element_before(
        parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom {line}/><w:between {line}/></w:pBdr>'),
        *_AFTER_PBDR)


class _Writer:
    def __init__(self, picture_page):
        # The translated page with the words rebuilt as text taken out.
        self.picture_page = picture_page

    @staticmethod
    def paragraph(container, gap: float):
        p = container.add_paragraph()
        f = p.paragraph_format
        f.space_before = Pt(max(0.0, min(MAX_GAP, gap)))
        f.space_after = Pt(0)
        return p

    # Words ------------------------------------------------------------------------
    @staticmethod
    def _alignment(t, left: float, right: float):
        align = t.manual_alignment or t.alignment
        if align in _ALIGN:
            return _ALIGN[align]
        # Not known: centred when it sits in the middle of its column.
        x0, _, x1, _ = _rect(t.bbox)
        if abs((x0 + x1) / 2 - (left + right) / 2) < 6 and x0 - left > 24:
            return WD_ALIGN_PARAGRAPH.CENTER
        return None

    def text(self, container, item: Item, left: float, right: float, gap: float):
        t = item.text
        layout = t.drawn_layout
        align = self._alignment(t, left, right)
        bx0 = item.x0
        if layout is not None and layout.rows:
            paragraphs, current = [], []
            for row in layout.rows:
                current.append(row)
                if row.end:
                    paragraphs.append(current)
                    current = []
            if current:
                paragraphs.append(current)
        else:
            paragraphs = [line for line in _plain_text(t).split("\n")]
        for k, rows in enumerate(paragraphs):
            p = self.paragraph(container, gap if k == 0 else 0)
            f = p.paragraph_format
            marker = None
            if isinstance(rows, str):
                words, indent = rows, bx0 - left
            else:
                words = _join([r.text for r in rows if r.text])
                head = rows[0]
                starts = [r.x0 for r in rows if r.text]
                indent = (min(starts) if starts else bx0 + head.indent) - left
                if head.marker and head.marker_x0 is not None:
                    marker = head.marker
                    f.first_line_indent = Pt(-max(0.0, min(starts or [head.x0]) - head.marker_x0))
            if align is not None:
                f.alignment = align
                if align is WD_ALIGN_PARAGRAPH.CENTER:
                    indent = 0.0
            f.left_indent = Pt(max(0.0, indent))
            if layout is not None and layout.leading:
                f.line_spacing = Pt(layout.leading)
                f.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
            if marker:
                _run(p, marker + "\t", t)
            if words:
                _run(p, words, t, item.underline)

    def tab_row(self, container, items: list[Item], left: float, gap: float):
        """Blocks of one line each, side by side: one paragraph, tab stops."""
        items = sorted(items, key=lambda i: i.x0)
        p = self.paragraph(container, gap)
        f = p.paragraph_format
        f.left_indent = Pt(max(0.0, items[0].x0 - left))
        for k, item in enumerate(items):
            if k:
                f.tab_stops.add_tab_stop(Pt(max(0.0, item.x0 - left)))
            _run(p, ("\t" if k else "") + _words(item.text), item.text, item.underline)

    # Lines and pictures --------------------------------------------------------------
    def lines(self, container, item: Item, left: float, right: float, gap: float):
        """Lines to write on: empty paragraphs ruled underneath."""
        step = (item.y1 - item.y0) / max(1, item.count)
        for n in range(item.count):
            p = self.paragraph(container, gap if n == 0 else 0)
            _border(p, "808080")
            f = p.paragraph_format
            f.left_indent = Pt(max(0.0, item.x0 - left))
            f.right_indent = Pt(max(0.0, right - item.x1))
            f.line_spacing = Pt(max(14.0, step))
            f.line_spacing_rule = WD_LINE_SPACING.EXACTLY

    def rule(self, container, item: Item, left: float, right: float, gap: float):
        p = self.paragraph(container, gap)
        _border(p, "808080")
        f = p.paragraph_format
        f.left_indent = Pt(max(0.0, item.x0 - left))
        f.right_indent = Pt(max(0.0, right - item.x1))
        f.line_spacing = Pt(2)
        f.line_spacing_rule = WD_LINE_SPACING.EXACTLY

    def picture(self, container, item: Item, left: float, right: float, gap: float):
        clip = fitz.Rect(item.box) & self.picture_page.rect
        if clip.is_empty:
            return
        png = self.picture_page.get_pixmap(clip=clip, dpi=PICTURE_DPI, alpha=False).tobytes("png")
        p = self.paragraph(container, gap)
        width = min(clip.width, max(24.0, right - left))
        p.paragraph_format.left_indent = Pt(max(0.0, min(item.x0 - left, right - left - width)))
        p.add_run().add_picture(BytesIO(png), width=Pt(width))

    def icon(self, paragraph, item: Item):
        """A small picture at the end of a line of words."""
        clip = fitz.Rect(item.box) & self.picture_page.rect
        if clip.is_empty:
            return
        png = self.picture_page.get_pixmap(clip=clip, dpi=PICTURE_DPI, alpha=False).tobytes("png")
        paragraph.add_run("  ")
        paragraph.add_run().add_picture(BytesIO(png), width=Pt(clip.width))

    # Tables ----------------------------------------------------------------------------
    @staticmethod
    def _table(container, rows: int, widths: list[float], indent: float, border: str | None, margin: float):
        table = container.add_table(rows=rows, cols=len(widths))
        table.autofit = False
        tblPr = table._tbl.tblPr
        for el in tblPr.findall(qn("w:tblW")):
            el.set(qn("w:w"), str(round(sum(widths) * 20)))
            el.set(qn("w:type"), "dxa")
        tblPr.insert_element_before(
            parse_xml(f'<w:tblInd {nsdecls("w")} w:w="{round(max(0.0, indent) * 20)}" w:type="dxa"/>'),
            *_AFTER_TBLIND)
        if border:
            edges = "".join(f'<w:{e} w:val="single" w:sz="6" w:space="0" w:color="{border}"/>'
                            for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
            tblPr.insert_element_before(parse_xml(f'<w:tblBorders {nsdecls("w")}>{edges}</w:tblBorders>'),
                                        *_AFTER_TBLIND[1:])
        tblPr.insert_element_before(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'),
                                    "w:tblCellMar", "w:tblLook")
        m = round(margin * 20)
        tblPr.insert_element_before(parse_xml(
            f'<w:tblCellMar {nsdecls("w")}><w:top w:w="{m // 2}" w:type="dxa"/><w:left w:w="{m}" w:type="dxa"/>'
            f'<w:bottom w:w="{m // 2}" w:type="dxa"/><w:right w:w="{m}" w:type="dxa"/></w:tblCellMar>'),
            "w:tblLook")
        for col, w in zip(table._tbl.tblGrid.iterchildren(), widths):
            col.set(qn("w:w"), str(round(w * 20)))
        for row in table.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Pt(w)
        return table

    def _cell(self, cell, items: list[Item], left: float, right: float, top: float):
        """Fill a table cell, replacing the empty paragraph it starts with."""
        if not items:
            return
        empty = cell.paragraphs[0]
        self.layout(cell, items, left, right, top)
        if len(cell.paragraphs) > 1 and not empty.text:
            empty._p.getparent().remove(empty._p)
        first = cell.paragraphs[0].paragraph_format
        first.space_before = Pt(0)

    def spacer(self, container, gap: float):
        """Room above a table, which has no space-before of its own."""
        if gap >= 3:
            p = self.paragraph(container, 0)
            p.paragraph_format.line_spacing = Pt(min(MAX_GAP, gap))
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY

    def grid(self, container, item: Item, left: float, gap: float):
        self.spacer(container, gap)
        widths = [b - a for a, b in zip(item.xs, item.xs[1:])]
        table = self._table(container, len(item.ys) - 1, widths, item.x0 - left, "000000", 4)
        for r, row in enumerate(table.rows):
            row.height = Pt(item.ys[r + 1] - item.ys[r])
            row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
            for c, cell in enumerate(row.cells):
                self._cell(cell, item.cells.get((r, c), []), item.xs[c] + 4, item.xs[c + 1] - 4, item.ys[r])

    def box(self, container, item: Item, left: float, gap: float):
        self.spacer(container, gap)
        table = self._table(container, 1, [item.x1 - item.x0], item.x0 - left, _hex(item.stroke) or "000000", 6)
        row = table.rows[0]
        row.height = Pt(item.y1 - item.y0)
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        fill = _hex(item.fill)
        if fill and fill != "FFFFFF":
            row.cells[0]._tc.get_or_add_tcPr().append(
                parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill}"/>'))
        self._cell(row.cells[0], item.inner, item.x0 + 6, item.x1 - 6, item.y0 + 3)

    def columns(self, container, columns: list[list[Item]], left: float, gap: float):
        """Side-by-side columns: a table without borders."""
        self.spacer(container, gap)
        edges = [min(i.x0 for i in col) for col in columns]
        ends = [max(i.x1 for i in col) for col in columns]
        widths = [b - a for a, b in zip(edges, edges[1:])] + [max(24.0, ends[-1] - edges[-1] + 6)]
        table = self._table(container, 1, widths, edges[0] - left, None, 0)
        top = min(i.y0 for col in columns for i in col)
        for c, (cell, col) in enumerate(zip(table.rows[0].cells, columns)):
            self._cell(cell, col, edges[c], edges[c] + widths[c] - 6, top)

    # Layout -----------------------------------------------------------------------------
    def layout(self, container, items: list[Item], left: float, right: float, top: float):
        bottom = top
        for band in _bands(items):
            gap = min(i.y0 for i in band) - bottom
            columns = _columns(band)
            if len(columns) == 1:
                # One thing, or things on top of each other: one after another.
                for k, item in enumerate(sorted(band, key=lambda i: (i.y0, i.x0))):
                    self.one(container, item, left, right, gap if k == 0 else 4)
            elif all(len(c) == 1 and c[0].single_line() for c in columns):
                self.tab_row(container, [c[0] for c in columns], left, gap)
            elif (headed := _heading_with_icons(columns)) is not None:
                heading, icons = headed
                self.text(container, heading, left, right, gap)
                line = container.paragraphs[-1]
                for icon in icons:
                    self.icon(line, icon)
            else:
                self.columns(container, columns, left, gap)
            bottom = max(i.y1 for i in band)

    def one(self, container, item: Item, left: float, right: float, gap: float):
        if item.kind == "text":
            self.text(container, item, left, right, gap)
        elif item.kind == "picture":
            self.picture(container, item, left, right, gap)
        elif item.kind == "lines":
            self.lines(container, item, left, right, gap)
        elif item.kind == "rule":
            self.rule(container, item, left, right, gap)
        elif item.kind == "grid":
            self.grid(container, item, left, gap)
        elif item.kind == "box":
            self.box(container, item, left, gap)


def _footer(section, footer: list, number_at: float | None, left: float):
    """The page's footer words, and its page number as Word's own."""
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    tabs = p.paragraph_format.tab_stops
    pieces = [(t.bbox.x0, t) for t in footer] + ([(number_at, None)] if number_at is not None else [])
    pieces.sort(key=lambda piece: piece[0])
    for k, (x, t) in enumerate(pieces):
        if k == 0:
            p.paragraph_format.left_indent = Pt(max(0.0, x - left))
        else:
            tabs.add_tab_stop(Pt(max(0.0, x - left)))
            p.add_run("\t")
        if t is not None:
            _run(p, _words(t), t)
        else:
            for xml in ('<w:fldChar {ns} w:fldCharType="begin"/>',
                        '<w:instrText {ns} xml:space="preserve"> PAGE </w:instrText>',
                        '<w:fldChar {ns} w:fldCharType="end"/>'):
                p.add_run()._r.append(parse_xml(xml.format(ns=nsdecls("w"))))


def export_flow_docx(pdf_bytes: bytes, pages: list) -> bytes:
    """`pages`: the manifest's pages, in order, matching the PDF's."""
    document = Document()
    body = document.element.body
    for p in list(body.iterchildren(qn("w:p"))):
        body.remove(p)
    normal = document.styles["Normal"].paragraph_format
    normal.space_before = normal.space_after = Pt(0)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf, \
            fitz.open(stream=pdf_bytes, filetype="pdf") as pictures:
        if not len(pdf):
            raise ValueError("This document has no pages.")
        if len(pdf) != len(pages):
            raise ValueError("The translated file does not match its pages. Download the PDF instead.")
        for pdf_page in pdf:
            w, h = pdf_page.rect.width, pdf_page.rect.height
            if min(w, h) < MIN_PAGE or max(w, h) > MAX_PAGE:
                raise ValueError("This page size is outside Word's supported range. Use the PDF download.")
        for index, (page, pdf_page) in enumerate(zip(pages, pdf)):
            items, footer = _read_page(page, pdf_page)
            # Pictures are cut from the page without the words rebuilt as
            # text, so a word on a picture is not there twice.
            picture_page = pictures[index]
            for item in _texts(items):
                picture_page.add_redact_annot(fitz.Rect(_rect(item.text.drawn_bbox or item.text.bbox)), fill=None)
            picture_page.apply_redactions(images=0, graphics=0, text=0)

            w, h = pdf_page.rect.width, pdf_page.rect.height
            section = document.sections[0] if index == 0 else document.add_section()
            section.orientation = WD_ORIENT.LANDSCAPE if w > h else WD_ORIENT.PORTRAIT
            section.page_width, section.page_height = Pt(w), Pt(h)
            left = max(18.0, min(72.0, min((i.x0 for i in items), default=72.0)))
            right = w - max(18.0, min(72.0, w - max((i.x1 for i in items), default=w - 72.0)))
            top = max(18.0, min(72.0, min((i.y0 for i in items), default=72.0)))
            section.left_margin, section.right_margin = Pt(left), Pt(w - right)
            # The page's words end where its content does, so a table drawn
            # nearly to the edge is not pushed onto a page of its own; the
            # footer sits in what is left.
            bottom = max(14.0, min(36.0, h - max((i.y1 for i in items), default=h - 36.0) - 2))
            section.top_margin, section.bottom_margin = Pt(top), Pt(bottom)
            section.header_distance = Pt(18)
            section.footer_distance = Pt(min(18.0, bottom / 2))
            number = next((t for t in page.text_objects if _page_number(t, h) and t.bbox.y0 > h / 2), None)
            if footer or number is not None:
                _footer(section, footer, number.bbox.x0 if number is not None else None, left)
            elif index:
                section.footer.is_linked_to_previous = False
            _Writer(picture_page).layout(document, items, left, right, top)
            if not items:
                document.add_paragraph()
    output = BytesIO()
    document.save(output)
    return output.getvalue()
