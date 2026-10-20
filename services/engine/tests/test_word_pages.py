"""A Word file shown as pages and handed back as a Word file.

The page editor works on the PDF LibreOffice lays the .docx out as; the
download writes each Word paragraph's translation -- with her edits -- back
into the .docx. These pin the matching between the two.
"""

from __future__ import annotations

import shutil

import pytest

from ct_engine.model.enums import TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
from ct_engine.pipeline.word_export import _strip_marker, match_paragraphs


def _obj(text: str, *, source=TextSource.NATIVE) -> TextObject:
    box = Rect(x0=0, y0=0, x1=100, y1=10)
    style = TextStyle(font_size=11)
    return TextObject(
        id=f"t{abs(hash(text)) % 10**8}",
        bbox=box, norm_bbox=box.normalized(612, 792),
        lines=[TextLine(bbox=box, spans=[TextSpan(text=line, bbox=box, style=style)])
               for line in text.split("\n")],
        source=source, style=style,
    )


def test_each_paragraph_finds_the_blocks_its_words_became():
    units = [_obj("Select Agree to grant the permissions."),
             _obj("In the window, open the menu and select From a custom URL.")]
    blocks = [_obj("3. Select Agree to grant the permissions."),
              _obj("In the window, open the menu and"),
              _obj("select From a custom URL.")]
    matched = match_paragraphs(units, blocks)
    assert matched[0] == [blocks[0]]
    assert matched[1] == [blocks[1], blocks[2]]


def test_a_block_holding_two_paragraphs_matches_neither():
    """The page reader joined them; they are translated on their own."""
    units = [_obj("Installation instructions"), _obj("Open the Admin console.")]
    blocks = [_obj("Installation instructions Open the Admin console.")]
    assert match_paragraphs(units, blocks) == {}


def test_the_page_list_number_is_not_written_twice():
    unit = _obj("Select Agree to grant the permissions.")
    block = _obj("3. Select Agree to grant the permissions.")
    assert _strip_marker(unit.text, block, "3. Selecciona «Aceptar».") == "Selecciona «Aceptar»."
    typed = _obj("3. Select Agree")
    assert _strip_marker(typed.text, block, "3. Selecciona") == "3. Selecciona"


@pytest.mark.skipif(
    not __import__("ct_engine.ingest.word_render", fromlist=["x"]).can_render_word(),
    reason="LibreOffice not installed",
)
def test_a_word_file_is_laid_out_as_pages():
    import fitz
    from docx import Document  # python-docx, already a dependency
    import io

    d = Document()
    d.add_heading("Unit 1 test", level=1)
    d.add_paragraph("Write your name at the top of the page.")
    buf = io.BytesIO()
    d.save(buf)
    from ct_engine.ingest.word_render import docx_to_pdf
    pdf = fitz.open(stream=docx_to_pdf(buf.getvalue()), filetype="pdf")
    assert pdf.page_count == 1
    assert "Write your name" in pdf[0].get_text()


def test_an_emoji_with_no_font_is_left_out_not_the_whole_heading():
    from ct_engine.reconstruct.builder import _without_symbols

    assert _without_symbols("📘 Instrucciones de instalación", "📘") == "Instrucciones de instalación"
    # Real letters with no font are a real failure: nothing is dropped.
    assert _without_symbols("日本語の説明", "日本語") is None


def test_only_pictures_drawn_on_the_page_are_counted():
    """LibreOffice lists every picture of the file on every page."""
    import fitz
    from ct_engine.pipeline.reconstruct_stage import count_artwork

    doc = fitz.open()
    png = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), 0).tobytes("png")
    doc.new_page().insert_image(fitz.Rect(10, 10, 50, 50), stream=png)
    doc.new_page()
    # Same resources (so the picture is listed), but nothing drawn.
    _, resources = doc.xref_get_key(doc[0].xref, "Resources")
    doc.xref_set_key(doc[1].xref, "Resources", resources)
    drawn, listed = doc[0], doc[1]
    assert len(listed.get_images()) == 1
    assert count_artwork(drawn) == 1
    assert count_artwork(listed) == 0


def test_a_translated_picture_replaces_the_word_picture_it_came_from():
    """Matched by appearance: LibreOffice re-encodes pictures, so bytes differ."""
    import io
    import zipfile

    from PIL import Image, ImageDraw
    from ct_engine.pipeline.word_pictures import replace_pictures

    def picture(colour, mark):
        img = Image.new("RGB", (200, 100), colour)
        ImageDraw.Draw(img).rectangle(mark, fill="black")
        return img

    def png(img):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    dialog = picture("white", [10, 10, 60, 30])
    panel = picture("lightblue", [120, 50, 190, 90])
    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/image1.tmp", png(dialog))
        z.writestr("word/media/image2.tmp", png(panel))

    # The page's copy of the panel, slightly different (re-encoded), and its
    # translation, larger.
    seen = panel.copy()
    translated = picture("lightblue", [120, 50, 190, 90]).resize((400, 200))
    out, stats = replace_pictures(docx.getvalue(), [(seen, translated)])

    assert stats.translated == 1 and stats.names == ["image2.tmp"]
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert Image.open(io.BytesIO(z.read("word/media/image2.tmp"))).size == (400, 200)
        assert z.read("word/media/image1.tmp") == png(dialog)
