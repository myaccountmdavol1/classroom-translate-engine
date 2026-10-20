"""Bring to front, send to back.

A teacher put 家庭作业 on a starburst and then a white box to hide the
"HW" under it; the white box landed on top and hid her words. Text above a
box with a background must be drawn above it, and either can be moved to
the front or the back.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ct_engine.pipeline.edit import add_block, apply_layout, undo
from test_edit import prepared

BOX = {"x0": 300, "y0": 600, "x1": 420, "y1": 640}


def _ink(store, manifest) -> int:
    page = manifest.pages[0]
    img = Image.open(io.BytesIO(store.get_bytes(page.renders.translated_preview_key))).convert("L")
    s = img.width / page.width
    crop = np.asarray(img)[int(BOX["y0"] * s):int(BOX["y1"] * s), int(BOX["x0"] * s):int(BOX["x1"] * s)]
    return int((crop < 128).sum())


def test_text_brought_in_front_of_a_white_box_shows(store):
    manifest = prepared(store)
    _, words = add_block(manifest, 1, bbox=BOX, text="WORDS", font_size=20,
                         store=store, target_lang="es")
    _, cover = add_block(manifest, 1, bbox=BOX, text="", fill="#ffffff",
                         store=store, target_lang="es")
    assert _ink(store, manifest) == 0  # the later white box hides the words

    apply_layout(manifest, 1, words.id, layer="front", store=store, target_lang="es")
    assert words.z_order > cover.z_order
    assert _ink(store, manifest) > 50

    apply_layout(manifest, 1, words.id, layer="back", store=store, target_lang="es")
    assert _ink(store, manifest) == 0


def test_restacking_can_be_undone(store):
    manifest = prepared(store)
    _, words = add_block(manifest, 1, bbox=BOX, text="WORDS", font_size=20,
                         store=store, target_lang="es")
    _, cover = add_block(manifest, 1, bbox=BOX, text="", fill="#ffffff",
                         store=store, target_lang="es")
    before = words.z_order
    apply_layout(manifest, 1, words.id, layer="front", store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")
    restored = manifest.pages[0].find_text_object(words.id)
    assert restored.z_order == before
