"""Generate realistic Word documents for testing.

The PDF corpus targets rendering failures. This one targets the failures that
are specific to a reflowable format, and every fixture here exists because
something measured went wrong without it:

  * runs split by Word for reasons that are invisible on the page, which make
    an ordinary paragraph look like it has mixed formatting
  * a bold lead-in label followed by body text -- the one mixed-formatting
    case that can be split safely
  * genuinely interleaved bold inside a sentence, which cannot
  * a hyperlink mid-sentence, whose address lives on the wrapper element and
    disappears if its run is emptied
  * a text box, which `Document.paragraphs` does not report at all
  * headers, footers, footnotes and table cells, likewise
  * two sections, because a Word "page" in this engine is a section

Run:  python corpus/build_docx_corpus.py
Out:  corpus/out/*.docx
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree
from PIL import Image

OUT = Path(__file__).parent / "out"


def _hyperlink(paragraph, url: str, text: str) -> None:
    """Add a real external hyperlink; python-docx has no API for this."""
    rid = paragraph.part.relate_to(
        url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True
    )
    link = paragraph._p.makeelement(qn("w:hyperlink"), {qn("r:id"): rid})
    run = paragraph._p.makeelement(qn("w:r"), {})
    rpr = paragraph._p.makeelement(qn("w:rPr"), {})
    style = paragraph._p.makeelement(qn("w:rStyle"), {qn("w:val"): "Hyperlink"})
    rpr.append(style)
    run.append(rpr)
    t = paragraph._p.makeelement(qn("w:t"), {})
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def _textbox(paragraph, text: str) -> None:
    """Add a VML text box, the shape Word 2007+ still writes for these."""
    run = paragraph.add_run()
    xml = f"""<w:pict
        xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:v="urn:schemas-microsoft-com:vml">
      <v:shape style="width:300pt;height:40pt"><v:textbox><w:txbxContent>
        <w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{text}</w:t></w:r></w:p>
      </w:txbxContent></v:textbox></v:shape></w:pict>"""
    run._r.append(etree.fromstring(xml))


def _split_runs(paragraph, text: str, at: int) -> None:
    """Write `text` as two identically formatted runs.

    This is what Word does constantly -- a proofing or revision boundary
    splits a run with no visible effect. If the engine treated these as mixed
    formatting, nearly every paragraph would lose its styling for nothing.
    """
    paragraph.add_run(text[:at])
    paragraph.add_run(text[at:])


def doc_worksheet(path: Path) -> None:
    d = Document()

    section = d.sections[0]
    section.header.paragraphs[0].text = "Grade 7 Life Science — Ms. Alvarez"
    section.footer.paragraphs[0].text = "Riverside Middle School | Page 1"

    d.add_heading("Cells and Energy: Review Packet", level=1)

    p = d.add_paragraph()
    p.add_run("Directions: ").bold = True
    _split_runs(
        p,
        "Read each question carefully. Show all of your work in the space "
        "provided. You may use a calculator for question 4.",
        48,
    )

    p = d.add_paragraph()
    p.add_run("1. ").bold = True
    p.add_run("What is the main function of the ")
    p.add_run("mitochondria").bold = True
    p.add_run(" in a plant cell?")
    d.add_paragraph("_______________________________________________")

    p = d.add_paragraph()
    p.add_run("2. ").bold = True
    p.add_run("Balance the equation ")
    p.add_run("6CO2 + 6H2O → C6H12O6 + 6O2").italic = True
    p.add_run(" and name the process.")

    p = d.add_paragraph()
    p.add_run("3. ").bold = True
    p.add_run("Solve for x: ")
    p.add_run("3x + 7 = 22").italic = True

    p = d.add_paragraph()
    p.add_run("4. ").bold = True
    p.add_run("A sample loses 12.5 g of water. Visit ")
    _hyperlink(p, "https://example.org/lab-safety", "the lab safety guide")
    p.add_run(" before you begin.")

    d.add_paragraph("Answer choices:", style="Normal")
    for letter, choice in (
        ("A.", "Photosynthesis"),
        ("B.", "Cellular respiration"),
        ("C.", "Osmosis"),
        ("D.", "Mitosis"),
    ):
        p = d.add_paragraph()
        p.add_run(f"{letter} ").bold = True
        p.add_run(choice)

    table = d.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    data = [
        ["Trial", "Mass before (g)", "Mass after (g)"],
        ["1", "45.2", "32.7"],
        ["2", "45.0", "33.1"],
    ]
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            if r == 0:
                cell.paragraphs[0].runs[0].bold = True

    d.add_paragraph()
    _textbox(d.add_paragraph(), "Remember: label every axis on your graph.")

    img = OUT / "_cell_diagram.png"
    Image.new("RGB", (240, 160), (232, 240, 248)).save(img)
    d.add_picture(str(img), width=Inches(2.0))
    caption = d.add_paragraph("Figure 1. A plant cell.", style="Caption")
    caption.alignment = 1

    p = d.add_paragraph()
    p.add_run("Name: ______________________")
    p.add_run("\t")
    p.add_run("Date: ______________")

    coloured = d.add_paragraph()
    run = coloured.add_run("Do not write below this line.")
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    run.font.size = Pt(9)

    d.save(path)
    img.unlink(missing_ok=True)


def doc_two_sections(path: Path) -> None:
    """Two sections, the second landscape: two 'pages' in this engine."""
    d = Document()
    d.add_heading("Part One: Vocabulary", level=1)
    for term, meaning in (
        ("Chloroplast", "The organelle where photosynthesis takes place."),
        ("Stomata", "Openings in a leaf that let gases in and out."),
    ):
        p = d.add_paragraph()
        p.add_run(f"{term}. ").bold = True
        p.add_run(meaning)

    new = d.add_section(WD_SECTION.NEW_PAGE)
    new.orientation = WD_ORIENT.LANDSCAPE
    new.page_width, new.page_height = new.page_height, new.page_width
    d.add_heading("Part Two: Data Table", level=1)
    d.add_paragraph("Record your measurements in the table below.")

    d.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing Word corpus to {OUT}")
    for name, fn in (
        ("20_worksheet.docx", doc_worksheet),
        ("21_two_sections.docx", doc_two_sections),
    ):
        path = OUT / name
        fn(path)
        print(f"  {name:32s} {path.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
