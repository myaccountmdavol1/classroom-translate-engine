"""Labels on shaded or textured pictures are read from a flattened copy too.

Tesseract turns a picture black-and-white with one threshold, so dark words
on a gradient or a leafy texture vanish with the background and were never
read -- nothing was translated and nothing was reported. A second read of a
copy whose background is evened out to white finds them.
"""
from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from ct_engine.pipeline.image_text import _with_second_read
from ct_engine.providers.ocr.base import OCRResult

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def _poster() -> np.ndarray:
    """Dark words on a warm gradient and on a green texture."""
    h, w = 800, 1200
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = (255 - int(y * 0.08), 245 - int(y * 0.12), 225 - int(y * 0.2))
    rng = np.random.default_rng(1)
    leaf = np.clip(np.array([150, 190, 140]) + rng.normal(0, 9, (300, 500, 3)), 0, 255)
    img[300:600, 620:1120] = cv2.GaussianBlur(leaf.astype(np.uint8), (5, 5), 0)
    for text, org in (("Producers", (680, 420)), ("Consumers", (680, 520)),
                      ("Sunlight gives energy", (80, 700))):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (40, 30, 20), 4, cv2.LINE_AA)
    return img


class _Counting:
    """Wraps a provider and counts the reads."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def recognize(self, png, **kw):
        self.calls += 1
        return self.inner.recognize(png, **kw)


@needs_tesseract
def test_words_on_a_gradient_and_a_texture_are_found():
    from ct_engine.providers.ocr.local_tesseract import LocalTesseractOCRProvider

    provider = LocalTesseractOCRProvider()
    png = _png(_poster())
    first = provider.recognize(png, language="en", sparse=True)

    merged = _with_second_read(first, png, provider, "en")

    text = " ".join(r.text for r in merged.regions)
    for word in ("Producers", "Consumers", "Sunlight"):
        assert word in text
    # Nothing the first read found is lost or read twice.
    assert merged.regions[: len(first.regions)] == first.regions
    assert len({r.text for r in merged.regions}) == len(merged.regions)


def test_a_plain_white_picture_is_read_once():
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.putText(img, "Nucleus", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    provider = _Counting(None)
    first = OCRResult()

    assert _with_second_read(first, _png(img), provider, "en") is first
    assert provider.calls == 0


@needs_tesseract
def test_white_words_on_a_tilted_navy_panel_are_found():
    from PIL import Image, ImageDraw, ImageFont

    from ct_engine.providers.ocr.local_tesseract import LocalTesseractOCRProvider

    img = Image.new("RGB", (1400, 900), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
    except OSError:
        font = ImageFont.load_default(size=50)
    l, t, r, b = font.getbbox("Collection Pond")
    tile = Image.new("RGBA", (r - l + 60, b - t + 60), "#14215a")
    ImageDraw.Draw(tile).text((30 - l, 30 - t), "Collection Pond", font=font, fill="white")
    tile = tile.rotate(-12, expand=True, resample=Image.BICUBIC)
    img.paste(tile, (700, 500), tile)
    png = _png(np.array(img))
    provider = LocalTesseractOCRProvider()
    first = provider.recognize(png, language="en", sparse=True)

    merged = _with_second_read(first, png, provider, "en")

    found = [r for r in merged.regions if "Collection" in r.text]
    assert found
    # The box is upright and in the picture's own pixels, over the panel.
    x0, y0, x1, y1 = found[0].bbox
    assert 700 <= x0 < x1 <= 700 + tile.width
    assert 500 <= y0 < y1 <= 500 + tile.height


def test_a_frame_with_no_lettering_is_not_read_as_a_panel():
    from ct_engine.pipeline.image_text import _lettered_panels

    img = np.full((600, 900), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (700, 450), 0, 30)  # a tablet's frame
    cv2.rectangle(img, (750, 100), (880, 300), 0, -1)  # a solid block

    assert _lettered_panels(img, 300) == []
