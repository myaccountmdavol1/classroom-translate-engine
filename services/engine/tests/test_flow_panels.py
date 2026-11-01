"""Words on a coloured panel, in the easy-to-edit Word file.

A Canva slide's panels are filled rectangles (a rounded outline drawn over
each) with white words on them. The Word file made each panel a picture
and put its words after it, white on the white page: a teacher opened it
to empty green boxes and nothing to read. A panel made as a one-colour
image went the same way, the words on it making it look like a photo.
"""
from __future__ import annotations

import io

import fitz
from docx import Document

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest
from ct_engine.pipeline.reconstruct_stage import reconstruct_and_export
from ct_engine.pipeline.translate import translate_page
from ct_engine.reconstruct.flow_docx import export_flow_docx
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
GREEN = (0.0, 0.302, 0.145)  # #004D25
PANEL = fitz.Rect(40, 120, 360, 420)
WORDS = "Learn about plants and how they grow"


def _pdf(draw) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 80), "Plants Unit Review", fontsize=20)
    draw(page)
    data = doc.tobytes()
    doc.close()
    return data


def _vector_panel(page):
    page.draw_rect(PANEL, color=None, fill=GREEN)
    # The pale rounded outline Canva draws over it.
    page.draw_rect(PANEL, color=(0.82, 0.91, 0.87), fill=None, width=2, radius=0.05)
    # A small white icon in its corner.
    page.draw_circle((340, 140), 8, color=None, fill=(1, 1, 1))
    page.insert_text((60, 200), WORDS, fontsize=14, color=(1, 1, 1))


def _image_panel(page):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64), False)
    pix.set_rect(pix.irect, (0, 77, 37))
    page.insert_image(PANEL, pixmap=pix)
    page.insert_text((60, 200), WORDS, fontsize=14, color=(1, 1, 1))


def _word(store, data: bytes) -> Document:
    store.put_bytes("documents/p/source.pdf", data)
    manifest = DocumentManifest(
        document_id="p", filename="p.pdf", content_type="application/pdf",
        byte_size=len(data), sha256=sha256_bytes(data), source_key="documents/p/source.pdf",
    )
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
        manifest.pages.append(page)
    manifest.analysis.page_count = 1
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    return Document(io.BytesIO(export_flow_docx(store.get_bytes(export.export_key), manifest.pages)))


def _panel_cell(doc):
    for table in doc.tables:
        for cell in table._cells:
            if "plants and how" in cell.text:
                return cell
    return None


def _fill(cell) -> str | None:
    shd = cell._tc.find(f"{W}tcPr/{W}shd")
    return None if shd is None else shd.get(f"{W}fill")


def test_words_on_a_drawn_panel_are_in_it_on_its_colour(store):
    doc = _word(store, _pdf(_vector_panel))

    cell = _panel_cell(doc)
    assert cell is not None, "the words came out after the panel, white on the page"
    assert _fill(cell) == "004D25"
    # The panel is a box to type in, not a picture; its icon goes in it.
    assert len(cell._tc.xpath(".//w:drawing")) == 1
    outside = [p.text for p in doc.paragraphs if "plants and how" in p.text]
    assert not outside


def test_the_panel_has_no_black_edge(store):
    doc = _word(store, _pdf(_vector_panel))
    table = next(t for t in doc.tables if "plants and how" in t._cells[0].text)
    edges = table._tbl.tblPr.find(f"{W}tblBorders")
    colours = {e.get(f"{W}color") for e in edges} if edges is not None else set()
    assert "000000" not in colours


def test_words_on_a_one_colour_image_are_in_it(store):
    doc = _word(store, _pdf(_image_panel))

    cell = _panel_cell(doc)
    assert cell is not None, "the words on the image made it look like a photo"
    assert _fill(cell) == "004D25"


def test_a_heading_running_into_its_icon_stays_one_word(store):
    """On a slide the heading reaches its link symbol; as two columns the
    heading's was narrower than the word and Word broke it in two."""
    def draw(page):
        page.draw_rect(PANEL, color=None, fill=GREEN)
        page.insert_text((150, 170), "Explore", fontsize=40, color=(0.8, 1.0, 0.4))
        page.draw_circle((300, 157), 12, color=None, fill=(1, 1, 1))
        page.insert_text((60, 260), WORDS, fontsize=14, color=(1, 1, 1))

    doc = _word(store, _pdf(draw))
    cell = _panel_cell(doc)
    assert cell is not None
    heading = next(p for p in cell.paragraphs if "Explore" in p.text)
    assert heading._p.xpath(".//w:drawing"), "the icon follows the heading in its line"
    assert not cell.tables, "no columns inside the panel"


def test_a_logo_set_into_the_panel_edge_is_cut_below_it(store):
    """The slide's logo overlaps the panel's bottom; cut whole, the picture
    brought a strip of the panel and its icon with it."""
    def draw(page):
        _vector_panel(page)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 32), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        pix.set_rect(fitz.IRect(0, 16, 32, 32), (240, 220, 60))  # two colours: a picture
        page.insert_image(fitz.Rect(60, 380, 340, 560), pixmap=pix)

    doc = _word(store, _pdf(draw))
    outside = [s for s in doc.inline_shapes if s._inline.getparent().getparent().getparent().tag != f"{W}tc"]
    heights = sorted(round(s.height.pt) for s in doc.inline_shapes)
    assert any(abs(h - 140) <= 2 for h in heights), heights
    assert outside


def test_a_panel_with_no_words_on_it_is_left_alone(store):
    def draw(page):
        page.draw_rect(PANEL, color=None, fill=GREEN)
        page.insert_text((60, 460), WORDS, fontsize=14)

    doc = _word(store, _pdf(draw))
    assert _panel_cell(doc) is None
    assert any("plants and how" in p.text for p in doc.paragraphs)
