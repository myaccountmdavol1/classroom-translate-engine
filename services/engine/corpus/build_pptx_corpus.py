"""Generate realistic PowerPoint decks for testing.

Targets the failures specific to a slide deck, each of which was measured
rather than imagined:

  * a placeholder with NO geometry of its own, which is the normal case --
    PowerPoint stores the box on the layout or master, and a parser that
    reads only the slide finds no position for most of a deck's text
  * a shape deliberately too small for its text once translated, which is
    what PowerPoint will not reflow
  * a table, whose geometry is a grid and not a stack
  * a hyperlink, whose address lives on the run's properties
  * a text box the author set to grow with its text, which must be left alone
  * speaker notes, which live in a separate part

Run:  python corpus/build_pptx_corpus.py
Out:  corpus/out/*.pptx
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt

OUT = Path(__file__).parent / "out"


def _hyperlink_run(paragraph, text: str, url: str) -> None:
    run = paragraph.add_run()
    run.text = text
    run.hyperlink.address = url


def doc_lesson(path: Path) -> None:
    prs = Presentation()

    # Slide 1: title + body placeholders, neither given explicit geometry.
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Photosynthesis: How Plants Make Food"
    body = s.shapes.placeholders[1].text_frame
    body.text = "Plants use sunlight, water and carbon dioxide."
    p = body.add_paragraph()
    p.add_run().text = "Key term: "
    p.runs[0].font.bold = True
    p.add_run().text = "chlorophyll"
    p2 = body.add_paragraph()
    p2.text = "Balance the equation 6CO2 + 6H2O → C6H12O6 + 6O2."
    s.notes_slide.notes_text_frame.text = (
        "Remind students that the equation must balance on both sides."
    )

    # Slide 2: a box that is only just big enough in English.
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Directions"
    tight = s2.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(0.6))
    tight.text_frame.word_wrap = True
    tight.text_frame.auto_size = MSO_AUTO_SIZE.NONE
    tight.text_frame.text = (
        "Read each question carefully and write your answer on the lines."
    )
    roomy = s2.shapes.add_textbox(Inches(5), Inches(2), Inches(4), Inches(3))
    roomy.text_frame.word_wrap = True
    roomy.text_frame.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
    roomy.text_frame.text = "This box grows to fit its text."

    link_box = s2.shapes.add_textbox(Inches(1), Inches(4), Inches(5), Inches(0.5))
    lp = link_box.text_frame.paragraphs[0]
    lp.add_run().text = "Review "
    _hyperlink_run(lp, "the study guide", "https://example.org/study-guide")
    lp.add_run().text = " before Friday."

    # Slide 3: a table and a picture.
    s3 = prs.slides.add_slide(prs.slide_layouts[5])
    s3.shapes.title.text = "Results"
    table = s3.shapes.add_table(3, 3, Inches(0.7), Inches(2), Inches(6), Inches(2)).table
    data = [
        ["Trial", "Mass before (g)", "Mass after (g)"],
        ["1", "45.2", "32.7"],
        ["2", "45.0", "33.1"],
    ]
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    img = OUT / "_leaf.png"
    Image.new("RGB", (200, 140), (224, 240, 224)).save(img)
    s3.shapes.add_picture(str(img), Inches(7), Inches(2), width=Inches(2))

    prs.save(path)
    img.unlink(missing_ok=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing PowerPoint corpus to {OUT}")
    path = OUT / "22_lesson.pptx"
    doc_lesson(path)
    print(f"  {'22_lesson.pptx':32s} {path.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
