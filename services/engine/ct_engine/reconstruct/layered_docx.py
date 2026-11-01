"""Export a translated PDF as a Word file that looks the same.

Each page is a Word section of the PDF's own size, with no margins: the page
drawn as a picture behind the text, and each upright line of text a text box
on top, exactly where the PDF put it (the same layers as the PowerPoint
export, see `layered_pptx.page_layers`). Unlike a slide deck, each section
keeps its own page size, so mixed sizes need no refusal.

It is for the look. The text boxes do not flow into each other, so a longer
sentence is typed into its own box; the easy-to-edit Word download is the
other one.
"""
from __future__ import annotations

import re
from io import BytesIO

import fitz
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import parse_xml
from docx.shared import Pt

from .layered_pptx import page_layers

EMU_PER_PT = 12700
# Word's page size limits, in points (0.1 to 22 inches).
MIN_PAGE, MAX_PAGE = 7.2, 1584

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
)


def _emu(points: float) -> int:
    return int(round(points * EMU_PER_PT))


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _anchor(shape_id: int, name: str, x: float, y: float, w: float, h: float,
            graphic: str, *, behind: bool) -> str:
    """A drawing fixed to the page at (x, y) points, size w x h, with no
    text wrapping round it."""
    return (
        f'<w:r {NS}><w:drawing><wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" '
        f'relativeHeight="{shape_id}" behindDoc="{int(behind)}" locked="0" layoutInCell="1" allowOverlap="1">'
        '<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{_emu(x)}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{_emu(y)}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{_emu(w)}" cy="{_emu(h)}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:wrapNone/><wp:docPr id="{shape_id}" name="{_esc(name)}"/><wp:cNvGraphicFramePr/>'
        f'<a:graphic>{graphic}</a:graphic></wp:anchor></w:drawing></w:r>'
    )


def _picture(rid: str, shape_id: int, w: float, h: float) -> str:
    return (
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
        f'<pic:nvPicPr><pic:cNvPr id="{shape_id}" name="Worksheet artwork"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{_emu(w)}" cy="{_emu(h)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData>'
    )


# The PDF's stand-ins for fonts every computer has under another name.
STAND_INS = {
    "Nimbus Sans": "Arial", "Helvetica": "Arial", "Liberation Sans": "Arial",
    "Nimbus Roman": "Times New Roman", "Times": "Times New Roman",
    "Liberation Serif": "Times New Roman", "Nimbus Mono": "Courier New",
    "Courier": "Courier New",
}


def family(font: str) -> str:
    """The family name Word knows a PDF font by: "ABCDEF+NotoSans-BoldItalic"
    is "Noto Sans", "Arial-BoldMT" is "Arial"."""
    name = font.split("+")[-1].split(",")[0]
    name = re.sub(r"(PSMT|MT|PS)$", "", name)
    name = re.sub(r"[-_](Regular|Bold|Italic|BoldItalic|Oblique|BoldOblique|Medium|Light|Semibold|SemiBold)$", "", name)
    # CamelCase to words, keeping runs of capitals ("SC", "MS") together.
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip() or "Calibri"
    return STAND_INS.get(name, name)


def _run(span: dict) -> str:
    size = max(2, int(round(span["size"] * 2)))
    face = _esc(family(span["font"]))
    colour = f'{span["color"] & 0xffffff:06X}'
    props = (
        f'<w:rFonts w:ascii="{face}" w:hAnsi="{face}" w:cs="{face}" w:eastAsia="{face}"/>'
        + ("<w:b/>" if span["flags"] & 16 else "")
        + ("<w:i/>" if span["flags"] & 2 else "")
        + f'<w:color w:val="{colour}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    )
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{_esc(span["text"])}</w:t></w:r>'


def _text_box(w: float, h: float, line: dict) -> str:
    # One line, set at exactly the line's height so Word puts it where the
    # PDF did; the box does not wrap or grow.
    line_twips = max(20, int(round(h * 20)))
    paragraph = (
        f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="{line_twips}" w:lineRule="exact"/></w:pPr>'
        + "".join(_run(s) for s in line["spans"] if s["text"])
        + "</w:p>"
    )
    return (
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><wps:wsp>'
        '<wps:cNvSpPr txBox="1"/>'
        f'<wps:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{_emu(w)}" cy="{_emu(h)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></wps:spPr>'
        f'<wps:txbx><w:txbxContent>{paragraph}</w:txbxContent></wps:txbx>'
        '<wps:bodyPr rot="0" vert="horz" wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t" anchorCtr="0">'
        '<a:noAutofit/></wps:bodyPr></wps:wsp></a:graphicData>'
    )


def export_layered_docx(pdf_bytes: bytes) -> bytes:
    document = Document()
    body = document.element.body
    # The template's one empty paragraph; each page brings its own.
    for p in list(body.iterchildren("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")):
        body.remove(p)
    shape_id = 1
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if not len(pdf):
            raise ValueError("This document has no pages.")
        for page in pdf:
            w, h = page.rect.width, page.rect.height
            if min(w, h) < MIN_PAGE or max(w, h) > MAX_PAGE:
                raise ValueError("This page size is outside Word's supported range. Use the PDF download.")
        for index, page in enumerate(pdf):
            w, h = page.rect.width, page.rect.height
            section = document.sections[0] if index == 0 else document.add_section()
            section.orientation = WD_ORIENT.LANDSCAPE if w > h else WD_ORIENT.PORTRAIT
            section.page_width, section.page_height = Pt(w), Pt(h)
            for side in ("left_margin", "right_margin", "top_margin", "bottom_margin",
                         "header_distance", "footer_distance"):
                setattr(section, side, 0)
            png, lines, _ = page_layers(page)
            rid, _ = document.part.get_or_add_image(BytesIO(png))
            paragraph = document.add_paragraph()
            paragraph._p.append(parse_xml(_anchor(
                shape_id, f"Page {index + 1} artwork", 0, 0, w, h,
                _picture(rid, shape_id, w, h), behind=True)))
            shape_id += 1
            for n, line in enumerate(lines, start=1):
                x0, y0, x1, y1 = line["bbox"]
                bw, bh = max(1.0, x1 - x0) + 2, max(1.0, y1 - y0)
                paragraph._p.append(parse_xml(_anchor(
                    shape_id, f"Page {index + 1} text {n}", x0, y0, bw, bh,
                    _text_box(bw, bh, line), behind=False)))
                shape_id += 1
    output = BytesIO()
    document.save(output)
    return output.getvalue()
