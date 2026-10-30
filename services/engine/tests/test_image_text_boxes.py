"""Words in a picture of side-by-side boxes, such as a course flowchart.

On a high-school planning guide, Tesseract read a row of six boxes as one
line, "English 7 English 8 English 9 English 10 English 11 English 12". It
was translated as one label and drawn across all six boxes, leaving the
last ones empty. Inside a box the words were 0.4 line-heights apart; from
one box to the next, 2.5 to 5.
"""

from __future__ import annotations

from ct_engine.pipeline.image_text import _join_line_pieces, _split_at_wide_gaps
from ct_engine.providers.ocr.base import OCRRegion, OCRWord

HEIGHT = 46.0


def _row(*words: tuple[str, float, float], confidence: float = 0.95) -> OCRRegion:
    """One OCR line from (text, x0, x1) words, as Tesseract reports it."""
    ocr = [OCRWord(text=t, bbox=(x0, 200, x1, 200 + HEIGHT), confidence=confidence)
           for t, x0, x1 in words]
    return OCRRegion(text=" ".join(t for t, _, _ in words),
                     bbox=(words[0][1], 200, words[-1][2], 200 + HEIGHT),
                     confidence=confidence, words=ocr)


def test_a_row_of_boxes_is_read_as_one_label_per_box():
    # Gaps measured on the guide: 19px inside a box, 116-156px between.
    row = _row(("English", 88, 230), ("7", 249, 270),
               ("English", 386, 528), ("8", 545, 566),
               ("English", 694, 836), ("9", 853, 874))
    pieces = _join_line_pieces(_split_at_wide_gaps([row]))
    assert sorted(p.text for p in pieces) == ["English 7", "English 8", "English 9"]
    first = next(p for p in pieces if p.text == "English 7")
    assert first.bbox == (88, 200, 270, 200 + HEIGHT)
    assert [w.text for w in first.words] == ["English", "7"]


def test_a_sentence_with_wide_spacing_stays_one_line():
    # Justified text can stretch a space to a whole line-height.
    line = _row(("Students", 0, 180), ("must", 226, 320), ("take", 366, 450))
    assert [r.text for r in _split_at_wide_gaps([line])] == ["Students must take"]


def test_a_piece_keeps_the_unreadable_glyphs_inside_it():
    row = _row(("English", 88, 230), ("7", 249, 270), ("English", 386, 528), ("8", 545, 566))
    row.discarded_boxes = [(540, 200, 570, 246)]
    pieces = {p.text: p for p in _split_at_wide_gaps([row])}
    assert pieces["English 8"].discarded_boxes == [(540, 200, 570, 246)]
    assert pieces["English 7"].discarded_boxes == []
