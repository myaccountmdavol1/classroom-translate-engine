"""Hand-lettered pictures get their translations in a hand-lettered face.

"DESDE EL" drawn in Helvetica across a novel's brush-lettered cover looked
pasted on. Each label is drawn in the plain, serif and hand-lettered faces
and compared with its own letters; the picture as a whole decides.
"""
from __future__ import annotations

import fitz
import pytest

from ct_engine.fonts import catalog
from ct_engine.model.objects import TextObject, TextStyle
from ct_engine.pipeline.lettering import hand_lettered, mark_hand_lettered
from ct_engine.reconstruct.builder import HAND_TALL, LABEL_TALL, _label_lettering


def _picture(lines: list[str], font: fitz.Font) -> tuple[bytes, list]:
    """The lines drawn one under another, and each one's ink box in pixels."""
    doc = fitz.open()
    page = doc.new_page(width=700, height=110 * len(lines) + 40)
    writer = fitz.TextWriter(page.rect)
    for i, text in enumerate(lines):
        writer.append((30, 100 + 110 * i), text, font=font, fontsize=64)
    writer.write_text(page, color=(0, 0, 0))
    pix = page.get_pixmap(dpi=72, alpha=False)
    labels = []
    for text in lines:
        box = page.search_for(text)[0]
        labels.append((text, (box.x0, box.y0, box.x1, box.y1)))
    return pix.tobytes("png"), labels


LINES = ["THE WAKE-UP", "MY DAY IN COURT", "Notes from the"]


def test_words_lettered_like_the_hand_face_are_hand_lettered():
    hand = fitz.Font(fontfile=str(catalog.face_path(catalog.HAND_FACE, bold=True)[0]))
    assert hand_lettered(*_picture(LINES, hand))


@pytest.mark.parametrize("name", ["helv", "tiro"])
def test_printed_words_are_not(name):
    assert not hand_lettered(*_picture(LINES, fitz.Font(name)))


def test_marked_labels_are_drawn_in_the_hand_face_in_bold_unless_she_chose_one():
    obj = TextObject(
        id="imgtxt_1", kind="TEXT", bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
        norm_bbox={"x0": 0, "y0": 0, "x1": 0.1, "y1": 0.1}, style=TextStyle(),
    )
    assert obj.render_family is None

    mark_hand_lettered([obj])

    assert obj.render_family == catalog.HAND_FACE and obj.render_bold
    obj.manual_font_family = "andika"
    assert obj.render_family == "andika"


def test_the_hand_face_is_sized_so_its_capitals_match_the_original():
    plain, _ = _label_lettering("PLAN B", 20.0)
    hand, _ = _label_lettering("PLAN B", 20.0, hand=True)

    assert hand * HAND_TALL == pytest.approx(plain * LABEL_TALL)


def test_the_baseline_stays_where_the_original_had_it():
    # "Notes from the": no descender; "energy": the y's tail below the line.
    for text in ("Notes from the", "energy"):
        plain, descent = _label_lettering(text, 20.0)
        hand, hand_descent = _label_lettering(text, 20.0, hand=True)
        assert hand * hand_descent == pytest.approx(plain * descent)
