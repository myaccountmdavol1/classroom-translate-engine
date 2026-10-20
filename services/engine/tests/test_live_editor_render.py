"""The live editor shows a page as a clean background plus one picture per
translated block, so a block can move with no server round trip.

That only works if the pieces put back together are the page the teacher
downloads. These tests hold the editor to that: background + sprites,
composited at the drawn positions, must match the rebuilt PDF page.
"""

from __future__ import annotations

import io

import fitz
import numpy as np
from PIL import Image

from ct_engine.reconstruct.builder import rebuild_document
from ct_engine.reconstruct.sprite import PAD, render_sprite
from test_reconstruct import prepared_pages
from test_translate_pipeline import EchoProvider

DPI = 96
SCALE = DPI / 72


def _png(pix) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")


def _rebuild_with_plates(name):
    data, pages = prepared_pages(name, provider=EchoProvider(lambda t: f"{t} ok"))
    plates: dict[int, Image.Image] = {}

    def on_plate(index, pdf_page):
        plates[index] = _png(pdf_page.get_pixmap(dpi=DPI, alpha=False))

    out, _ = rebuild_document(data, pages, target_lang="es", on_plate=on_plate)
    return out, pages, plates


def test_every_drawn_block_records_how_it_was_drawn():
    _, pages, _ = _rebuild_with_plates("01_clean_worksheet.pdf")
    drawn = [t for t in pages[0].text_objects if t.drawn_bbox is not None]
    assert drawn
    for t in drawn:
        assert t.drawn_layout is not None and t.drawn_layout.lines


def test_plate_has_no_translated_text_but_keeps_everything_else():
    out, pages, plates = _rebuild_with_plates("01_clean_worksheet.pdf")
    plate = np.asarray(plates[0].convert("L"), dtype=np.int16)
    with fitz.open(stream=out, filetype="pdf") as doc:
        full = np.asarray(_png(doc[0].get_pixmap(dpi=DPI, alpha=False)).convert("L"), dtype=np.int16)
    # The plate is lighter where translations sit, identical elsewhere.
    assert plate.shape == full.shape
    assert (plate >= full - 2).mean() > 0.999
    assert (full < 128).sum() > (plate < 128).sum()


def test_plate_plus_sprites_is_the_downloaded_page():
    out, pages, plates = _rebuild_with_plates("01_clean_worksheet.pdf")
    canvas = plates[0].copy()
    for t in pages[0].text_objects:
        if t.drawn_bbox is None:
            continue
        sprite = Image.open(io.BytesIO(render_sprite(t, scale=SCALE))).convert("RGBA")
        canvas.alpha_composite(
            sprite,
            (round((t.drawn_bbox.x0 - PAD) * SCALE), round((t.drawn_bbox.y0 - PAD) * SCALE)),
        )
    with fitz.open(stream=out, filetype="pdf") as doc:
        full = _png(doc[0].get_pixmap(dpi=DPI, alpha=False))
    # Placing a sprite at a whole pixel moves it by up to half a pixel, which
    # shows up only along glyph edges (a browser places it fractionally).
    # A one-pixel blur forgives exactly that; a missing word, a wrong font
    # or a misplaced line still differs by far more.
    from PIL import ImageFilter

    def soft(img):
        return np.asarray(
            img.convert("L").filter(ImageFilter.GaussianBlur(1)), dtype=np.int16
        )

    a, b = soft(canvas), soft(full)
    assert (np.abs(a - b) <= 40).mean() > 0.999


def test_api_serves_plate_and_sprites(client):
    """End to end through the routes a browser calls."""
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    assert _translate(client, doc_id).status_code in (200, 202)
    assert _wait_terminal(client, doc_id)["stage"] == "COMPLETE"

    page = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    assert page["has_plate"]
    plate = client.get(f"/api/documents/{doc_id}/pages/1/plate")
    assert plate.status_code == 200 and plate.headers["content-type"] == "image/png"
    drawn = [b for b in page["blocks"] if b["sprite_bbox"]]
    assert drawn
    r = client.get(
        f"/api/documents/{doc_id}/pages/1/blocks/{drawn[0]['id']}/sprite?scale=2"
    )
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
