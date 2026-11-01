"""A translated PDF downloaded as a Word file that looks the same."""
import io
import zipfile

import fitz
import pytest
from docx import Document
from lxml import etree
from PIL import Image

from ct_engine.reconstruct.layered_docx import export_layered_docx, family

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


def _pdf(*sizes):
    with fitz.open() as pdf:
        for i, (w, h) in enumerate(sizes):
            page = pdf.new_page(width=w, height=h)
            page.insert_text((72, 90), f"Texto editable {i + 1}", fontsize=16)
            page.draw_line((72, 130), (300, 130))
        return pdf.tobytes()


def _boxes(data):
    """Each text box's words and its place on the page, in points."""
    body = Document(io.BytesIO(data)).element.body
    out = []
    for anchor in body.iter(f"{WP}anchor"):
        texts = ["".join(t.text for t in p.iter(f"{W}t")) for p in anchor.iter(f"{W}p")]
        if texts:
            x = int(anchor.find(f"{WP}positionH/{WP}posOffset").text) / 12700
            y = int(anchor.find(f"{WP}positionV/{WP}posOffset").text) / 12700
            out.append((texts[0], round(x), round(y)))
    return out


def test_each_line_is_a_text_box_where_the_pdf_put_it():
    data = export_layered_docx(_pdf((612, 792)))
    [(text, x, y)] = _boxes(data)
    assert text == "Texto editable 1"
    # insert_text's point is the baseline; the box starts at the line's top.
    assert x == 72 and 70 <= y <= 78


def test_the_picture_behind_has_the_lines_but_not_the_words():
    data = export_layered_docx(_pdf((612, 792)))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        [name] = [n for n in z.namelist() if n.startswith("word/media/")]
        picture = Image.open(io.BytesIO(z.read(name))).convert("L")
    # At 144 dpi: the words' place is blank, the rule under them is not.
    assert picture.crop((150, 150, 400, 180)).getextrema()[0] == 255
    assert picture.crop((150, 258, 590, 262)).getextrema()[0] < 128


def test_each_page_keeps_its_own_size():
    data = export_layered_docx(_pdf((612, 792), (792, 612)))
    sections = Document(io.BytesIO(data)).sections
    assert [(round(s.page_width.pt), round(s.page_height.pt)) for s in sections] == [(612, 792), (792, 612)]
    assert all(s.left_margin == 0 and s.top_margin == 0 for s in sections)
    assert [t for t, _, _ in _boxes(data)] == ["Texto editable 1", "Texto editable 2"]


def test_shape_ids_are_unique():
    body = Document(io.BytesIO(export_layered_docx(_pdf((612, 792), (612, 792))))).element.body
    ids = [d.get("id") for d in body.iter(f"{WP}docPr")]
    assert len(ids) == len(set(ids)) == 4


def test_a_page_word_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="Word"):
        export_layered_docx(_pdf((612, 2000)))


@pytest.mark.parametrize("font, name", [
    ("ABCDEF+NotoSans-BoldItalic", "Noto Sans"),
    ("Arial-BoldMT", "Arial"),
    ("TimesNewRomanPSMT", "Times New Roman"),
    ("ComicSansMS-Bold", "Comic Sans MS"),
    ("NimbusSans-Regular", "Arial"),
    ("AmaticSC-Bold", "Amatic SC"),
])
def test_fonts_are_named_as_word_knows_them(font, name):
    assert family(font) == name


def test_the_file_is_well_formed_word():
    data = export_layered_docx(_pdf((612, 792)))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        etree.fromstring(z.read("word/document.xml"))
