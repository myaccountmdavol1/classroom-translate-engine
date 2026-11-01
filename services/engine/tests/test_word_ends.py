"""A word in a picture read only in part is read again whole, or kept.

On a novel's hand-lettered cover "DRIVER" was read as "RIVER" (translated
"RÍO") and "Midnight" as "ght". Lettering of the same colour carries on
past the box closer than a word space: that is what gives it away.
"""
from __future__ import annotations

import cv2
import numpy as np

from ct_engine.pipeline.word_ends import finish_cut_words
from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult, OCRWord

FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 4, 12


def _width(text):
    return cv2.getTextSize(text, FONT, SCALE, THICK)[0][0]


def _picture(*words: tuple[str, int]) -> tuple[bytes, dict[str, tuple]]:
    """Black words on white at the given x positions, and each one's box."""
    img = np.full((300, 1600, 3), 255, np.uint8)
    boxes = {}
    for text, x in words:
        (w, h), _ = cv2.getTextSize(text, FONT, SCALE, THICK)
        cv2.putText(img, text, (x, 200), FONT, SCALE, (0, 0, 0), THICK, cv2.LINE_AA)
        boxes[text] = (x - 4, 200 - h - 6, x + w + 4, 200 + 8)
    return cv2.imencode(".png", img)[1].tobytes(), boxes


def _midnight():
    """"MID" and "NIGHT" a letter's gap apart (0.13 of their height): one word."""
    return _picture(("MID", 100), ("NIGHT", 100 + _width("MID") - 14))


class _Line(OCRProvider):
    """A reader whose one-line reading is `text` at `confidence`."""

    name = "line"
    sends_data_externally = False

    def __init__(self, text, confidence):
        self.text, self.confidence, self.calls = text, confidence, 0

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, **kw):
        return OCRResult()

    def recognize_line(self, image_png, *, language=None):
        self.calls += 1
        if self.text is None:
            return OCRResult()
        return OCRResult(regions=[OCRRegion(
            text=self.text, bbox=(0, 0, 10, 10), confidence=self.confidence,
            words=[OCRWord(text=self.text, bbox=(0, 0, 10, 10), confidence=self.confidence)],
        )])


def _region(text, box, confidence=0.9):
    return OCRRegion(text=text, bbox=box, confidence=confidence)


def test_a_word_read_in_part_is_read_again_whole():
    png, boxes = _midnight()

    (out,) = finish_cut_words(png, [_region("NIGHT", boxes["NIGHT"])], _Line("MIDNIGHT", 0.9), "en")

    assert out.text == "MIDNIGHT" and not out.cut_short
    # The box now runs over the whole word (from the M's ink, a little
    # inside the box drawn around it).
    assert out.bbox[0] <= boxes["MID"][0] + 20
    assert out.bbox[2] >= boxes["NIGHT"][2] - 20


def test_a_reread_that_does_not_take_in_the_reading_is_refused():
    png, boxes = _midnight()

    # A different word, an unsure one, and nothing at all.
    for line in (("MIDWAY", 0.95), ("MIDNIGHT", 0.3), (None, 0.0)):
        (out,) = finish_cut_words(png, [_region("NIGHT", boxes["NIGHT"])], _Line(*line), "en")
        assert out.cut_short and out.text == "NIGHT", line


def test_a_word_beside_a_word_space_is_left_alone():
    png, boxes = _picture(("CELL", 100), ("WALL", 100 + _width("CELL") + 70))
    provider = _Line("CELLWALL", 0.99)

    (out,) = finish_cut_words(png, [_region("WALL", boxes["WALL"])], provider, "en")

    assert out.text == "WALL" and not out.cut_short
    assert provider.calls == 0


def test_a_word_read_the_same_across_its_overhang_is_whole():
    # The logo's "OF EDUCATION": its N reached 2 px past the box, and read
    # again across that it was "OF EDUCATION" still. Nothing was missing.
    png, boxes = _picture(("CELL", 100))
    x0, y0, x1, y1 = boxes["CELL"]
    region = _region("CELL", (x0, y0, x1 - 30, y1))

    (out,) = finish_cut_words(png, [region], _Line("CELL", 0.9), "en")

    assert out == region


def test_pieces_inside_the_whole_word_are_dropped():
    png, boxes = _midnight()
    regions = [_region("NIGHT", boxes["NIGHT"]), _region("Mi", boxes["MID"], 0.2)]

    out = finish_cut_words(png, regions, _Line("MIDNIGHT", 0.9), "en")

    assert [r.text for r in out] == ["MIDNIGHT"]
