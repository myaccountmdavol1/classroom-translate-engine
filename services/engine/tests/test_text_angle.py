"""Measuring the slant of words in a picture (analysis/text_angle.py).

The measurement decides whether a translation is drawn turned, so it must
be right when it answers and must decline when it cannot tell.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

from ct_engine.analysis.text_angle import measure_line, measure_on_page
from ct_engine.model.geometry import Rect
from tilted_scenes import scene, words_box


def _measure(doc):
    box = words_box(doc)
    return measure_on_page(doc[0], Rect(x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1))


@pytest.mark.parametrize("angle", [0.0, 5.0, 13.0, 25.0, 40.0])
def test_the_slant_of_a_printed_line_is_measured(angle):
    found = _measure(scene("Photosynthesis Lab", angle))
    assert found is not None
    assert abs(found.angle - angle) <= 1.0
    assert found.glyphs >= 10
    assert found.sagitta < 0.2
    assert not found.curved


def test_a_frame_and_a_big_letter_beside_the_words_do_not_tilt_the_answer():
    """The real slide: a tablet's frame just above "HARBOR" and a big red
    letter below it, both inside the word's upright box."""
    found = _measure(scene("HARBOR SCHOOL", 13.0, frame=True, big_letter=True))
    assert found is not None
    assert abs(found.angle - 13.0) <= 1.0
    # The box is the letters' own: about the font's height, not the
    # upright box's.
    assert found.page_box().height < 24 * 1.2


def test_the_turned_box_is_the_letters_own_size():
    found = _measure(scene("Photosynthesis Lab", 25.0, size=20.0))
    box = found.page_box()
    # Helvetica: cap height to descender is roughly 0.93 of the size.
    assert 15.0 < box.height < 22.0
    assert box.width > 4 * box.height


def test_words_around_an_arc_are_called_curved():
    found = _measure(scene("WELCOME STUDENTS", 0.0, size=20.0, arc=(230.0, 40.0)))
    assert found is not None and found.curved


def test_white_lettering_on_a_dark_panel_is_measured():
    found = _measure(scene("Photosynthesis Lab", 13.0, light=True))
    assert found is not None
    assert abs(found.angle - 13.0) <= 1.0


def test_two_letters_are_not_a_line():
    assert _measure(scene("Hi", 13.0)) is None


def test_a_huge_crop_is_declined_quickly():
    start = time.perf_counter()
    assert measure_line(np.zeros((3000, 3000, 3), dtype=np.uint8), (0, 0, 3000, 3000)) is None
    assert time.perf_counter() - start < 0.5


def test_a_blank_crop_is_declined():
    blank = np.full((80, 300, 3), 255, dtype=np.uint8)
    assert measure_line(blank, (20, 20, 280, 60)) is None


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
def test_the_straightened_line_reads_where_the_tilted_one_did_not():
    import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
    from ct_engine.providers import get_ocr_provider

    found = _measure(scene("Photosynthesis Lab", 25.0))
    result = get_ocr_provider("local").recognize_line(found.straightened_png(), language="en")
    assert result is not None and not result.error
    text = " ".join(r.text for r in result.regions)
    assert "Photosynthesis" in text
    assert result.mean_confidence >= 0.8
