"""Labels a diagram's pointer lines run through are not trusted.

On a plant cell worksheet a pointer line ran through the "c" of
"Nucleolus": read "Nudleolus", it was translated into nonsense. Another
line hid "Nucleus" from the reader altogether.
"""
from __future__ import annotations

import cv2
import numpy as np

from ct_engine.pipeline.pointer_lines import crossed_by_line, hidden_words
from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult

FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2


def _label(text="Nucleolus", at=(300, 200)):
    """A white picture with one small label; its box."""
    img = np.full((400, 700), 255, np.uint8)
    (w, h), base = cv2.getTextSize(text, FONT, SCALE, THICK)
    cv2.putText(img, text, at, FONT, SCALE, 0, THICK, cv2.LINE_AA)
    x, y = at
    return img, (x - 2, y - h - 2, x + w + 2, y + base)


def test_a_thin_line_through_the_letters_crosses_the_label():
    img, box = _label()
    # From the structure it points at, up through the middle of the word.
    cv2.line(img, (250, 330), (360, 150), 90, 3)

    assert crossed_by_line(img, box)


def test_a_line_that_ends_beside_the_label_does_not():
    img, box = _label()
    # The usual pointer: from the label's end, away to the drawing.
    cv2.line(img, (box[2] + 4, (box[1] + box[3]) // 2), (box[2] + 200, 330), 90, 3)

    assert not crossed_by_line(img, box)


def test_a_thick_outline_is_not_a_pointer_line():
    img, box = _label()
    cv2.line(img, (250, 330), (360, 150), 60, 14)

    assert not crossed_by_line(img, box)


class _Reader(OCRProvider):
    """Reads `found` from whatever picture it is given."""

    name = "fixed"
    sends_data_externally = False

    def __init__(self, found):
        self.found, self.calls = found, 0

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, **kw):
        self.calls += 1
        return OCRResult(regions=self.found)


def _region(text, box, confidence=0.8):
    return OCRRegion(text=text, bbox=box, confidence=confidence)


def test_a_word_found_once_the_lines_are_painted_out_is_returned():
    img, box = _label("Cell Wall", at=(40, 60))
    cv2.line(img, (300, 380), (420, 150), 90, 3)
    known = [_region("Cell Wall", box)]
    reader = _Reader([_region("Cell Wall", box), _region("Nucleus", (380, 200, 470, 222)), _region("—", (10, 10, 30, 14))])

    found = hidden_words(img, known, reader, "en")

    assert [r.text for r in found] == ["Nucleus"]


def test_a_picture_without_pointer_lines_is_not_read_again():
    img, box = _label()
    reader = _Reader([_region("Nucleus", (380, 200, 470, 222))])

    assert hidden_words(img, [_region("Nucleolus", box)], reader, "en") == []
    assert reader.calls == 0


def test_a_word_no_line_reached_is_not_called_hidden():
    # Found only on the second reading, but far from every painted-out line:
    # a word the first reading missed for some other reason, not this one.
    img, box = _label("Cell Wall", at=(40, 60))
    cv2.line(img, (300, 380), (420, 150), 90, 3)
    reader = _Reader([_region("Vacuole", (560, 330, 660, 352))])

    assert hidden_words(img, [_region("Cell Wall", box)], reader, "en") == []
