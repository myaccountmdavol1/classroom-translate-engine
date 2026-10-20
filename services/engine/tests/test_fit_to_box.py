"""The box sets the text size: drag it bigger and the words grow.

A teacher put "Homework" over a small "HW" starburst. The form's fixed
19pt wrapped 家庭作业 one character per line out of the star, and finding
a size that fit took several guesses. New text now fills the box she
drew, and any box can be set to fit and follows its own resizes.
"""

from __future__ import annotations

from ct_engine.fonts.registry import REGISTRY
from ct_engine.model.geometry import Rect
from ct_engine.pipeline.edit import add_block, apply_layout
from ct_engine.reconstruct.fitting import fit_to_box
from test_edit import first_editable, prepared


def _font():
    return REGISTRY.resolve("家庭作业", script="Han")


def test_text_fills_a_small_box_on_one_line():
    fit = fit_to_box("家庭作业", Rect(x0=0, y0=0, x1=51, y1=27), _font())
    assert fit.fits and fit.lines == ["家庭作业"]
    assert 11 <= fit.font_size <= 13


def test_a_bigger_box_gives_bigger_text():
    small = fit_to_box("Homework", Rect(x0=0, y0=0, x1=60, y1=20), REGISTRY.resolve("Homework"))
    big = fit_to_box("Homework", Rect(x0=0, y0=0, x1=180, y1=60), REGISTRY.resolve("Homework"))
    assert big.font_size > small.font_size * 2


def test_an_english_word_is_never_broken_to_fit():
    fit = fit_to_box("Photosynthesis happens", Rect(x0=0, y0=0, x1=80, y1=200),
                     REGISTRY.resolve("Photosynthesis"))
    assert "Photosynthesis" in fit.lines


def test_new_text_with_no_size_fills_its_box(store):
    manifest = prepared(store)
    _, created = add_block(
        manifest, 1, bbox={"x0": 400, "y0": 600, "x1": 451, "y1": 627},
        text="家庭作业", store=store, target_lang="zh-Hans",
    )
    assert created.manual_fit
    assert created.drawn_layout.size > 10
    assert len(created.drawn_layout.lines) == 1


def test_fit_follows_a_resize_and_a_chosen_size_ends_it(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    apply_layout(manifest, 1, obj.id, fit=True, store=store, target_lang="es")
    before = obj.drawn_layout.size
    b = obj.render_bbox
    apply_layout(manifest, 1, obj.id,
                 bbox={"x0": b.x0, "y0": b.y0, "x1": b.x0 + b.width * 2, "y1": b.y0 + b.height * 2},
                 store=store, target_lang="es")
    assert obj.drawn_layout.size > before
    apply_layout(manifest, 1, obj.id, font_size=9, store=store, target_lang="es")
    assert not obj.manual_fit and obj.drawn_layout.size == 9
