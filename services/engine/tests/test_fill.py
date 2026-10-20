"""A box can have a background colour that covers what is beneath it.

For text that would not translate -- "HW" in a starburst, a stylised
heading -- a teacher adds a box with a white fill over it and types the
replacement. The fill must be in the PDF, under the text, and in the
box's picture in the editor.
"""

from __future__ import annotations

import io

import fitz
import numpy as np
import pytest
from PIL import Image

from ct_engine.pipeline.edit import EditError, add_block, apply_layout
from ct_engine.reconstruct.sprite import PAD, render_sprite
from test_edit import first_editable, prepared


def _render(store, manifest, clip):
    key = manifest.pages[0].renders.translated_preview_key
    img = Image.open(io.BytesIO(store.get_bytes(key))).convert("L")
    return img


def test_a_new_box_with_a_white_fill_covers_what_is_under_it(store):
    manifest = prepared(store)
    # Over the title, which is dark text on white.
    title = next(t for t in manifest.pages[0].text_objects if "Photosynthesis" in t.flat_text)
    box = title.bbox
    _, created = add_block(
        manifest, 1,
        bbox={"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
        text="X", fill="#ffffff", font_size=6,
        store=store, target_lang="es",
    )
    assert created.manual_fill == "#ffffff"
    assert created.drawn_layout and created.drawn_layout.fill == (1.0, 1.0, 1.0)

    # On the rendered page, the box is white but for the one small "X".
    page = manifest.pages[0]
    img = _render(store, manifest, None)
    sx = img.width / page.width
    region = np.asarray(img.crop((
        int((box.x0 + 2) * sx), int((box.y0 + 1) * sx),
        int((box.x1 - 2) * sx), int((box.y1 - 1) * sx),
    )))
    assert (region < 128).mean() < 0.06


def test_the_fill_is_in_the_sprite(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    apply_layout(manifest, 1, obj.id, fill="#ffe066", store=store, target_lang="es")
    png = Image.open(io.BytesIO(render_sprite(obj, scale=2))).convert("RGBA")
    px = np.asarray(png)
    # A corner inside the frame is the fill colour and opaque.
    inner = px[int(PAD * 2) + 1, int(PAD * 2) + 1]
    assert inner[3] == 255 and tuple(inner[:3]) == (255, 224, 102)


def test_none_clears_the_fill(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    apply_layout(manifest, 1, obj.id, fill="#ffffff", store=store, target_lang="es")
    apply_layout(manifest, 1, obj.id, fill="none", store=store, target_lang="es")
    assert obj.manual_fill is None and obj.drawn_layout.fill is None


def test_a_bad_colour_is_refused(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    with pytest.raises(EditError):
        apply_layout(manifest, 1, obj.id, fill="purple", store=store, target_lang="es")


# --- Cover: a filled box with no words, to white-out what did not translate ---
def test_a_cover_needs_no_text_and_hides_what_is_under_it(store):
    manifest = prepared(store)
    title = next(t for t in manifest.pages[0].text_objects if "Photosynthesis" in t.flat_text)
    box = title.bbox
    _, cover = add_block(
        manifest, 1,
        bbox={"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
        text="", fill="#ffffff", store=store, target_lang="es",
    )
    assert cover.drawn_bbox is not None and cover.drawn_layout.fill == (1.0, 1.0, 1.0)
    page = manifest.pages[0]
    img = _render(store, manifest, None)
    sx = img.width / page.width
    region = np.asarray(img.crop((
        int((box.x0 + 1) * sx), int((box.y0 + 1) * sx),
        int((box.x1 - 1) * sx), int((box.y1 - 1) * sx),
    )))
    assert (region < 128).mean() < 0.01
    # Its picture is the fill alone.
    png = np.asarray(Image.open(io.BytesIO(render_sprite(cover, scale=2))).convert("RGBA"))
    assert png[int(PAD * 2) + 2, int(PAD * 2) + 2][3] == 255


def test_an_empty_box_without_a_cover_is_still_refused(store):
    manifest = prepared(store)
    with pytest.raises(EditError):
        add_block(manifest, 1, bbox={"x0": 100, "y0": 100, "x1": 200, "y1": 130},
                  text="", store=store, target_lang="es")


def test_a_cover_is_listed_so_it_can_be_selected(client):
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id)
    _wait_terminal(client, doc_id)
    r = client.post(f"/api/documents/{doc_id}/pages/1/blocks",
                    json={"text": "", "fill": "#ffffff", "bbox": {"x0": 400, "y0": 600, "x1": 500, "y1": 650}})
    assert r.status_code == 200, r.text
    blocks = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]
    covers = [b for b in blocks if b["fill"] == "#ffffff" and not b["original"].strip()]
    assert covers and covers[0]["sprite_rev"]
