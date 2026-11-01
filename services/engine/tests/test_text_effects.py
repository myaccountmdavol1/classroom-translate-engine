"""A Canva title effect: the words drawn twice, and their shadow a picture.

A slide's title ("MAGMA MATH ACADEMY") was its text in black, the same text
in green over it at the same place, and a grey picture of the words just
behind both, the shadow. The engine read the title twice ("MAGMA MATH
ACADEMY MAGMA MATH ACADEMY"), drew it in the hidden copy's black, and left
the English shadow showing behind the Spanish.
"""
from __future__ import annotations

import fitz

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.dedupe import drop_duplicate_text
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest
from ct_engine.pipeline.reconstruct_stage import reconstruct_and_export
from ct_engine.pipeline.translate import translate_page
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider

TITLE = "PLANT SCIENCE ACADEMY"
GREEN = (0.04, 0.33, 0.15)
SPOT = (60, 80)


def _slide(shadow: bool = True) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if shadow:
        # The shadow: a grey picture of the words, just behind them.
        page.insert_text((SPOT[0] + 2, SPOT[1] + 2), TITLE, fontsize=30, fontname="hebo", color=(0.7, 0.7, 0.7))
        # Cut to the words, as Canva cuts it.
        width = fitz.get_text_length(TITLE, fontname="hebo", fontsize=30)
        clip = fitz.Rect(SPOT[0] - 1, SPOT[1] - 28, SPOT[0] + width + 3, SPOT[1] + 8)
        pix = page.get_pixmap(clip=clip, dpi=144)
        page.add_redact_annot(clip, fill=None)
        page.apply_redactions(images=0, graphics=0, text=0)
        page.insert_image(clip, pixmap=pix)
    page.insert_text(SPOT, TITLE, fontsize=30, fontname="hebo", color=(0, 0, 0))
    page.insert_text(SPOT, TITLE, fontsize=30, fontname="hebo", color=GREEN)
    page.insert_text((60, 200), "Plants need light to grow.", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _page(data: bytes):
    with PdfParser().open_bytes(data) as doc:
        return doc.parse_page(0)


def _title_blocks(page):
    return [t for t in page.text_objects if "SCIENCE" in t.flat_text]


def test_the_title_drawn_twice_is_read_once_in_its_visible_colour():
    page = _page(_slide(shadow=False))
    assert sum(t.flat_text.count("SCIENCE") for t in _title_blocks(page)) == 2, "the PDF repeats it"

    drop_duplicate_text(page)

    (title,) = _title_blocks(page)
    assert title.flat_text == TITLE
    assert title.style.color == "#0a5426", "the green copy on top, not the black one under it"


def test_words_that_merely_repeat_elsewhere_are_kept():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 80), "Name: ________", fontsize=12)
    page.insert_text((60, 400), "Name: ________", fontsize=12)
    data = doc.tobytes()
    parsed = _page(data)
    drop_duplicate_text(parsed)
    assert sum(t.flat_text.count("Name") for t in parsed.text_objects) == 2


def _translated_pdf(store, data: bytes) -> fitz.Document:
    store.put_bytes("documents/e/source.pdf", data)
    manifest = DocumentManifest(
        document_id="e", filename="e.pdf", content_type="application/pdf",
        byte_size=len(data), sha256=sha256_bytes(data), source_key="documents/e/source.pdf",
    )
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        drop_duplicate_text(page)
        split_page_blocks(page)
        apply_reading_order(page)
        classify_page(page, is_first_page=True)
        translate_page(page, EchoProvider(), source_lang="en", target_lang="es")
        manifest.pages.append(page)
    manifest.analysis.page_count = 1
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    return fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")


def test_the_shadow_picture_of_the_english_goes_with_it(store):
    with _translated_pdf(store, _slide()) as out:
        page = out[0]
        title_area = fitz.Rect(SPOT[0] - 2, SPOT[1] - 30, SPOT[0] + 400, SPOT[1] + 10)
        # A deleted picture is left as an empty 1x1 stand-in; any real one is a shadow.
        shadows = [i for i in page.get_image_info() if fitz.Rect(i["bbox"]).intersects(title_area)
                   and i["width"] > 1]
        assert not shadows, "the grey English shadow still shows behind the translation"
        assert "<" + TITLE + ">" in " ".join(page.get_text().split())


def test_each_line_of_a_two_line_title_loses_its_shadow(store):
    """A two-line title has a shadow picture per line, which fits its line
    but not the whole block."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    lines = [((60, 80), "PLANT SCIENCE"), ((60, 116), "STRATEGY LIBRARY")]
    for (x, y), words in lines:
        page.insert_text((x + 2, y + 2), words, fontsize=30, fontname="hebo", color=(0.7, 0.7, 0.7))
        width = fitz.get_text_length(words, fontname="hebo", fontsize=30)
        clip = fitz.Rect(x - 1, y - 28, x + width + 3, y + 8)
        pix = page.get_pixmap(clip=clip, dpi=144)
        page.add_redact_annot(clip, fill=None)
        page.apply_redactions(images=0, graphics=0, text=0)
        page.insert_image(clip, pixmap=pix)
    for (x, y), words in lines:
        page.insert_text((x, y), words, fontsize=30, fontname="hebo", color=GREEN)
    data = doc.tobytes()
    with _translated_pdf(store, data) as out:
        real = [i for i in out[0].get_image_info() if i["width"] > 1]
        assert not real


def test_a_photo_beside_the_words_is_kept(store):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    page.insert_image(fitz.Rect(60, 100, 360, 400), pixmap=pix)
    page.insert_text((70, 120), "A short caption", fontsize=12)
    data = doc.tobytes()
    with _translated_pdf(store, data) as out:
        assert len(out[0].get_image_info()) == 1
