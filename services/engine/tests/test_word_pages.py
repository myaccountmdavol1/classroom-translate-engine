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


def _book(count):
    """A Word file with `count` pictures, the page's copy of each, and a
    translated PDF with one picture per page -- a long book, small."""
    import io
    import zipfile
    from types import SimpleNamespace

    import fitz
    from PIL import Image, ImageDraw

    def picture(i):
        img = Image.new("RGB", (200, 100), (40 * i % 256, 90, 200 - 30 * i % 200))
        ImageDraw.Draw(img).rectangle([10 + 15 * i, 10, 40 + 15 * i, 60], fill="black")
        return img

    def png(img):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        for i in range(count):
            z.writestr(f"word/media/image{i + 1}.png", png(picture(i)))
    pdf = fitz.open()
    pages = []
    for i in range(count):
        box = Rect(x0=100, y0=100, x1=300, y1=200)
        pdf.new_page().insert_image(fitz.Rect(*box.as_tuple()), stream=png(picture(i)))
        image = SimpleNamespace(bbox=box, asset_key=f"pictures/{i}.png")
        pages.append(SimpleNamespace(number=i + 1, index=i, image_objects=[image],
                                     text_objects=[]))
    asked: list[str] = []
    store = SimpleNamespace(
        get_bytes=lambda key: (asked.append(key), png(picture(int(key[9:-4]))))[1],
    )
    return docx.getvalue(), pdf.tobytes(), SimpleNamespace(pages=pages), store, asked


def test_a_long_book_makes_its_translated_pictures_one_at_a_time(monkeypatch):
    """The browser engine has little memory: a 36-picture book crashed it
    when every picture was made before any went into the file."""
    from ct_engine.pipeline import word_pictures

    _, pdf, manifest, store, asked = _book(3)
    monkeypatch.setattr(word_pictures, "_labels_in", lambda page, image: [image.bbox])
    pictures = word_pictures.translated_pictures(manifest, pdf, store)
    assert asked == []
    page_number, original, translated = next(pictures)
    assert (page_number, asked) == (1, ["pictures/0.png"])
    assert translated.width > original.width
    assert [p for p, _, _ in pictures] == [2, 3]
    assert asked == [f"pictures/{i}.png" for i in range(3)]


def test_pictures_go_into_the_file_as_they_come(monkeypatch):
    import io
    import zipfile

    from ct_engine.pipeline import word_pictures

    docx, pdf, manifest, store, _ = _book(3)
    monkeypatch.setattr(word_pictures, "_labels_in", lambda page, image: [image.bbox])
    out, stats = word_pictures.replace_pictures(
        docx, word_pictures.translated_pictures(manifest, pdf, store),
    )
    assert stats.translated == 3 and stats.unmatched == 0 and stats.kept_pages == []
    assert stats.names == ["image1.png", "image2.png", "image3.png"]
    with zipfile.ZipFile(io.BytesIO(docx)) as before, zipfile.ZipFile(io.BytesIO(out)) as after:
        assert after.namelist() == before.namelist()
        assert after.read("word/document.xml") == before.read("word/document.xml")
        for name in stats.names:
            from PIL import Image

            new = Image.open(io.BytesIO(after.read(f"word/media/{name}")))
            assert new.format == "PNG" and new.size == (556, 278)
