"""A transparency mask is not a picture.

Scanners that save pages as layered images (a low-resolution background plus
a sharp foreground held by a full-resolution mask) often list the mask as a
page resource of its own. Counted naively, the source had three pictures and
the rebuilt page two -- the mask was still attached to its image, just no
longer listed separately -- and every page of a real scanned letter was
reported as missing a picture.
"""
from __future__ import annotations

import io

import fitz
from PIL import Image

from ct_engine.pipeline.reconstruct_stage import count_artwork


def _masked_image_page(list_mask_as_resource: bool) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    rgba = Image.new("RGBA", (40, 40), (200, 30, 30, 128))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    page.insert_image(fitz.Rect(20, 20, 120, 120), stream=buf.getvalue())
    if list_mask_as_resource:
        xref, smask = page.get_images(full=True)[0][:2]
        assert smask, "the test image should carry a soft mask"
        resources = doc.xref_get_key(page.xref, "Resources")
        if resources[0] == "xref":
            target = int(resources[1].split()[0])
            doc.xref_set_key(target, "XObject/Mask0", f"{smask} 0 R")
        else:
            doc.xref_set_key(page.xref, "Resources/XObject/Mask0", f"{smask} 0 R")
    return doc


def test_a_mask_listed_as_a_resource_is_not_counted_as_a_picture():
    doc = _masked_image_page(list_mask_as_resource=True)
    assert len(doc[0].get_images(full=True)) == 2, "the mask is listed separately"
    assert count_artwork(doc[0]) == 1


def test_an_ordinary_masked_image_counts_once():
    doc = _masked_image_page(list_mask_as_resource=False)
    assert count_artwork(doc[0]) == 1


def test_real_pictures_are_all_counted():
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    for x in (10, 110):
        img = Image.new("RGB", (20, 20), (x, 50, 50))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        page.insert_image(fitz.Rect(x, 10, x + 50, 60), stream=buf.getvalue())
    assert count_artwork(page) == 2
