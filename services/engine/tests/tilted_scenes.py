"""Synthetic pictures with words printed at a slant, for the tilt tests.

Drawn with MuPDF's own Helvetica so the tests need no system fonts. A scene
is a small page; `picture_pdf` rasterises it and places it on a letter page
as an ordinary embedded picture, which is what a pasted screenshot or a
slide's background image is.
"""

from __future__ import annotations

import math

import fitz

from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult, OCRWord

SCENE_W, SCENE_H = 400.0, 200.0
# Where the picture sits on the letter page.
PICTURE_AT = (100.0, 120.0)
PICTURE_DPI = 300


def draw_words(page, text, angle, *, cx, cy, size=24.0, color=(0, 0, 0)):
    font = fitz.Font("helv")
    width = font.text_length(text, fontsize=size)
    writer = fitz.TextWriter(page.rect)
    writer.append(fitz.Point(cx - width / 2, cy + 0.35 * size), text, font=font, fontsize=size)
    writer.write_text(page, color=color, morph=(fitz.Point(cx, cy), fitz.Matrix(-angle)))


def turned_quad(cx, cy, w, h, angle):
    t = math.radians(angle)
    u = (math.cos(t), math.sin(t))
    v = (-math.sin(t), math.cos(t))

    def at(a, b):
        return fitz.Point(cx + a * u[0] + b * v[0], cy + a * u[1] + b * v[1])

    return fitz.Quad(at(-w / 2, -h / 2), at(w / 2, -h / 2), at(-w / 2, h / 2), at(w / 2, h / 2))


def scene(
    text="Photosynthesis Lab",
    angle=13.0,
    *,
    size=24.0,
    frame=False,
    big_letter=False,
    light=False,
    arc=None,
) -> fitz.Document:
    """A one-page document holding the scene. `arc` = (radius, degrees)
    places the letters one by one around a circle instead."""
    doc = fitz.open()
    page = doc.new_page(width=SCENE_W, height=SCENE_H)
    cx, cy = SCENE_W / 2, SCENE_H / 2
    ink = (0, 0, 0)
    if light:
        page.draw_rect(page.rect, color=None, fill=(0.06, 0.1, 0.32))
        ink = (1, 1, 1)
    if frame:
        # A thick bar along the line, a few points above the letters and
        # running off both ends of any crop around them: a tablet's frame.
        page.draw_quad(
            turned_quad(cx, cy - 0.95 * size, 3 * SCENE_W, 0.35 * size, angle),
            color=None, fill=(0, 0, 0),
        )
    if big_letter:
        # A big red letter just below the words' left end.
        draw_words(page, "M", angle, cx=cx - 2.2 * size, cy=cy + 1.9 * size,
                   size=3 * size, color=(0.7, 0.1, 0.1))
    if arc is None:
        draw_words(page, text, angle, cx=cx, cy=cy, size=size, color=ink)
    else:
        radius, degrees = arc
        centre_y = cy + radius
        n = len(text)
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            a = math.radians(-degrees / 2 + degrees * i / (n - 1))
            x = cx + radius * math.sin(a)
            y = centre_y - radius * math.cos(a)
            draw_words(page, ch, math.degrees(a), cx=x, cy=y, size=size, color=ink)
    return doc


def words_box(doc) -> fitz.Rect:
    """Upright box of all the scene's words, as MuPDF reports it."""
    box = fitz.Rect()
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if any(s["text"].strip() and s["text"] != "M" for s in line["spans"]):
                box |= fitz.Rect(line["bbox"])
    return box


def picture_pdf(doc) -> tuple[bytes, fitz.Rect]:
    """The scene as a picture on a letter page; returns (pdf, picture rect)."""
    png = doc[0].get_pixmap(dpi=PICTURE_DPI, alpha=False).tobytes("png")
    out = fitz.open()
    page = out.new_page(width=612, height=792)
    rect = fitz.Rect(PICTURE_AT[0], PICTURE_AT[1],
                     PICTURE_AT[0] + SCENE_W, PICTURE_AT[1] + SCENE_H)
    page.insert_image(rect, stream=png)
    return out.tobytes(), rect


def region_for(doc, text, confidence, *, box=None) -> OCRRegion:
    """What sparse reading reports for the scene's words: one upright box,
    in pixels of the picture rendered at the label-reading density."""
    box = box or words_box(doc)
    scale = PICTURE_DPI / 72.0
    px = (box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale)
    return OCRRegion(
        text=text, bbox=px, confidence=confidence,
        words=[OCRWord(text=w, bbox=px, confidence=confidence) for w in text.split()],
    )


class ScriptedTilt(OCRProvider):
    """Sparse reading gives `regions`. A single-line read gives `line` as
    (text, confidence) when the words it is shown are level -- straightened,
    or printed level -- and `as_is` when they are still at a slant (by
    default nothing readable). `line=None` is a provider with no such mode.

    Which one is decided by measuring the picture it is given, as a real
    reader would succeed or fail by what it sees, not by call order."""

    name = "scripted-tilt"
    sends_data_externally = False

    def __init__(self, regions, line=None, as_is=None):
        self.regions = regions
        self.line = line
        self.as_is = as_is
        self.calls = 0
        self.line_calls = 0
        self.slanted_calls = 0

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, *, language=None, detect_handwriting=False, sparse=False):
        self.calls += 1
        return OCRResult(regions=list(self.regions), provider=self.name)

    @staticmethod
    def _slanted(image_png) -> bool:
        import io

        import numpy as np
        from PIL import Image

        from ct_engine.analysis.text_angle import measure_line

        with Image.open(io.BytesIO(image_png)) as im:
            arr = np.asarray(im.convert("RGB"))
        found = measure_line(arr, (0, 0, arr.shape[1], arr.shape[0]))
        return found is not None and abs(found.angle) >= 2.5

    def recognize_line(self, image_png, *, language=None):
        self.line_calls += 1
        if self.line is None:
            return None
        if self._slanted(image_png):
            self.slanted_calls += 1
            if self.as_is is None:
                return OCRResult(regions=[], provider=self.name)
            text, conf = self.as_is
        else:
            text, conf = self.line
        return OCRResult(
            regions=[OCRRegion(
                text=text, bbox=(0, 0, 1, 1), confidence=conf,
                words=[OCRWord(text=w, bbox=(0, 0, 1, 1), confidence=conf) for w in text.split()],
            )],
            provider=self.name,
        )
