"""A Word file's pictures as pages, for the browser, where nothing lays a
Word file out (ingest/word_pictures_pdf.py)."""

from __future__ import annotations

import io

import fitz
from docx import Document  # python-docx, already a dependency
from docx.shared import Inches
from PIL import Image

from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.ingest.word_pictures_pdf import MARGIN, pictures_pdf
from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.builder import estimate_background


def _picture(fmt: str, size: tuple[int, int], colour: str) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, fmt)
    buf.seek(0)
    return buf


def _docx(*pictures: io.BytesIO) -> bytes:
    d = Document()
    d.add_paragraph("Chapter one")
    for p in pictures:
        p.seek(0)
        d.add_picture(p, width=Inches(2))
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


def test_each_distinct_picture_gets_a_page_of_its_own_shape():
    wide = _picture("PNG", (400, 100), "red")
    tall = _picture("JPEG", (100, 300), "blue")
    # The same picture shown twice is one picture in the file: one page.
    result = pictures_pdf(_docx(wide, tall, wide))
    assert result.shown == 2
    assert result.paged == 2
    pdf = fitz.open(stream=result.pdf, filetype="pdf")
    assert pdf.page_count == 2
    # The picture's own placed rect, not the page (a plain picture has a margin).
    ratios = []
    for p in pdf:
        (xref,) = [i[0] for i in p.get_images(full=True)]
        r = p.get_image_rects(xref)[0]
        ratios.append(r.width / r.height)
    ratios.sort()
    assert abs(ratios[0] - 100 / 300) < 0.01
    assert abs(ratios[1] - 400 / 100) < 0.01


def test_a_picture_that_is_not_png_or_jpeg_is_counted_but_given_no_page():
    result = pictures_pdf(_docx(_picture("GIF", (200, 200), "green")))
    assert result.shown == 1
    assert result.paged == 0
    assert result.pdf is None


def test_a_word_file_without_pictures_has_no_pictures_pdf():
    result = pictures_pdf(_docx())
    assert (result.pdf, result.shown, result.paged) == (None, 0, 0)


def _bar_picture() -> io.BytesIO:
    # White, with a black bar touching the top and bottom edges.
    img = Image.new("RGB", (400, 100), "white")
    img.paste("black", (150, 0, 250, 100))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf


def test_a_plain_edged_picture_gets_a_margin_of_its_own_background():
    result = pictures_pdf(_docx(_bar_picture()))
    pdf = fitz.open(stream=result.pdf, filetype="pdf")
    page = pdf[0]
    w_pt, h_pt = 400 * 72 / 150, 100 * 72 / 150
    assert abs(page.rect.width - (w_pt + 2 * MARGIN)) < 0.5
    assert abs(page.rect.height - (h_pt + 2 * MARGIN)) < 0.5
    pix = page.get_pixmap(dpi=72, alpha=False)
    assert pix.pixel(0, 0) == (255, 255, 255)
    # Still exactly one picture object, placed inside the margin: the
    # engine's own parser finds it.
    parsed = PdfParser().open_bytes(result.pdf).parse_page(0)
    assert len(parsed.image_objects) == 1
    b = parsed.image_objects[0].bbox
    assert abs(b.x0 - MARGIN) < 0.5 and abs(b.y0 - MARGIN) < 0.5
    assert abs(b.x1 - (MARGIN + w_pt)) < 0.5 and abs(b.y1 - (MARGIN + h_pt)) < 0.5


def test_a_noisy_picture_gets_no_margin():
    import os

    img = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    result = pictures_pdf(_docx(buf))
    page = fitz.open(stream=result.pdf, filetype="pdf")[0]
    assert abs(page.rect.width - 200 * 72 / 150) < 0.5
    assert abs(page.rect.height - 200 * 72 / 150) < 0.5


def test_a_label_across_the_whole_picture_has_a_background_to_paint_over():
    result = pictures_pdf(_docx(_bar_picture()))
    page = fitz.open(stream=result.pdf, filetype="pdf")[0]
    (xref,) = [i[0] for i in page.get_images(full=True)]
    placed = page.get_image_rects(xref)[0]
    whole = Rect(x0=placed.x0, y0=placed.y0, x1=placed.x1, y1=placed.y1)
    colour, _ = estimate_background(page, whole)
    assert colour is not None
