"""Font resolution and text fitting (spec sec. 25/26/27)."""

from __future__ import annotations

import pytest

from ct_engine.fonts.registry import REGISTRY, FontResolutionError
from ct_engine.fonts.shaping import shape_logical, shaping_available, to_visual
from ct_engine.model.geometry import Rect
from ct_engine.reconstruct.fitting import (
    MIN_FONT_SIZE,
    available_growth,
    available_growth_right,
    fit_text,
    wrap_text,
)

SAMPLES = {
    "Latin": ("Escribe tu respuesta en las líneas.", "Latin"),
    "Cyrillic": ("Напишите свой ответ ниже.", "Latin"),
    "Greek": ("Γράψε την απάντησή σου.", "Latin"),
    "Chinese": ("请在下面的横线上写出你的答案。", "Latin"),
    "Japanese": ("下の行に答えを書きなさい。", "Latin"),
    "Korean": ("아래 줄에 답을 쓰십시오.", "Latin"),
    "Arabic": ("اكتب إجابتك على السطور أدناه", "Arabic"),
    "Hebrew": ("כתוב את התשובה שלך", "Hebrew"),
    "Bengali": ("নিচের লাইনে উত্তর লিখুন", "Bengali"),
}


@pytest.mark.parametrize("label", sorted(SAMPLES))
def test_every_supported_script_resolves_a_font(label):
    """Never render a missing-character box (spec sec. 26)."""
    text, script = SAMPLES[label]
    resolved = REGISTRY.resolve(text, script=script)
    font = REGISTRY.font_for(resolved)
    assert font is not None
    assert REGISTRY.missing_characters(font, text) == ""


def test_lastresort_font_is_never_selected():
    """LastResort claims every glyph and draws labelled boxes."""
    for path in REGISTRY._system_fonts():
        assert "lastresort" not in str(path).lower()


def test_unrenderable_text_raises_instead_of_drawing_boxes():
    # Unassigned plane-15 private use: no real font covers this.
    with pytest.raises(FontResolutionError):
        REGISTRY.resolve("\U000f0000\U000f0001\U000f0002", script="Latin")


def test_measure_scales_with_size_and_length():
    f = REGISTRY.resolve("Hello world")
    assert REGISTRY.measure(f, "Hello world", 20) > REGISTRY.measure(f, "Hello world", 10)
    assert REGISTRY.measure(f, "Hello world world", 10) > REGISTRY.measure(f, "Hello", 10)


# --- Wrapping -------------------------------------------------------------
def test_wrapped_lines_respect_the_width():
    f = REGISTRY.resolve("Lorem ipsum dolor sit amet consectetur")
    lines = wrap_text("Lorem ipsum dolor sit amet consectetur adipiscing elit", f, 11, 90)
    assert len(lines) > 1
    for line in lines:
        assert REGISTRY.measure(f, line, 11) <= 90 + 0.5


def test_cjk_wraps_even_without_spaces():
    """Space-based wrapping alone silently overflows for Chinese."""
    text = "请在下面的横线上写出你的完整答案并使用文中的证据。"
    f = REGISTRY.resolve(text)
    lines = wrap_text(text, f, 11, 120)
    assert len(lines) > 1
    for line in lines:
        assert REGISTRY.measure(f, line, 11) <= 120 + 0.5


# --- Fitting --------------------------------------------------------------
def test_text_that_fits_keeps_the_original_size():
    f = REGISTRY.resolve("Escribe tu respuesta.")
    box = Rect(x0=72, y0=100, x1=372, y1=140)
    result = fit_text("Escribe tu respuesta.", box, f, base_size=11)
    assert result.fits and result.font_size == 11 and result.shrink_ratio == 1.0


def test_long_text_shrinks_rather_than_overflowing():
    f = REGISTRY.resolve("x")
    box = Rect(x0=72, y0=100, x1=200, y1=118)
    long_text = "Lea cada pregunta con mucho cuidado y escriba su respuesta completa."
    result = fit_text(long_text, box, f, base_size=11)
    assert result.font_size <= 11
    if result.fits:
        assert result.used_height <= box.height + 0.01


def test_fitting_never_goes_below_the_legibility_floor():
    f = REGISTRY.resolve("x")
    box = Rect(x0=0, y0=0, x1=40, y1=10)
    result = fit_text("word " * 200, box, f, base_size=11)
    assert result.font_size >= MIN_FONT_SIZE
    assert not result.fits
    assert result.overflowed_by > 0


def test_growth_stops_at_an_obstacle():
    box = Rect(x0=72, y0=100, x1=372, y1=140)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    blocker = Rect(x0=72, y0=170, x1=372, y1=200)
    assert available_growth(box, page, [blocker]) == 30.0
    assert available_growth(box, page, []) > 0


def test_growth_ignores_objects_that_do_not_overlap_horizontally():
    box = Rect(x0=72, y0=100, x1=200, y1=140)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    beside = Rect(x0=400, y0=150, x1=500, y1=200)
    assert available_growth(box, page, [beside]) > 30


def test_narrow_boxes_get_an_absolute_width_allowance():
    """Regression: a ratio-only cap left 'Lab 7' unable to fit 'Laboratorio 7'."""
    tiny = Rect(x0=72, y0=740, x1=93, y1=752)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    assert available_growth_right(tiny, page, []) > 60


def test_growth_right_stops_at_an_obstacle():
    box = Rect(x0=72, y0=100, x1=200, y1=140)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    blocker = Rect(x0=250, y0=110, x1=300, y1=130)
    assert available_growth_right(box, page, [blocker]) == pytest.approx(48.0, abs=1)


def test_growth_right_stops_at_the_documents_own_margin():
    """A paragraph already at the text margin must not spread into it."""
    paragraph = Rect(x0=73, y0=80, x1=539, y1=160)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    assert available_growth_right(paragraph, page, [], right_margin=539) == 0


def test_short_label_still_widens_inside_the_margin():
    tiny = Rect(x0=72, y0=740, x1=93, y1=752)
    page = Rect(x0=0, y0=0, x1=612, y1=792)
    assert available_growth_right(tiny, page, [], right_margin=540) > 60


# --- RTL shaping ----------------------------------------------------------
def test_arabic_shaping_produces_presentation_forms():
    """Without shaping, Arabic renders as disconnected isolated letters."""
    raw = "اكتب إجابتك"
    shaped = shape_logical(raw, "Arabic")
    if not shaping_available("Arabic"):
        pytest.skip("arabic_reshaper not installed")
    assert shaped != raw
    # Presentation forms live in the Arabic Presentation Forms blocks.
    assert any("ﭐ" <= c <= "﻿" for c in shaped)


def test_bidi_reorders_arabic_for_display():
    if not shaping_available("Arabic"):
        pytest.skip("python-bidi not installed")
    shaped = shape_logical("اكتب إجابتك", "Arabic")
    visual = to_visual(shaped, "Arabic")
    assert visual != shaped
    assert len(visual) == len(shaped)


def test_shaping_is_a_noop_for_latin():
    assert shape_logical("Hello", "Latin") == "Hello"
    assert to_visual("Hello", "Latin") == "Hello"


def test_shaped_arabic_is_still_fully_renderable():
    """The font must cover the presentation forms, not just the base letters."""
    shaped = shape_logical("اكتب إجابتك على السطور", "Arabic")
    resolved = REGISTRY.resolve(shaped, script="Arabic")
    font = REGISTRY.font_for(resolved)
    assert REGISTRY.missing_characters(font, shaped) == ""


# --- Weight-aware resolution (spec sec. 26 extension) ----------------------
def test_resolve_returns_a_bold_face_when_one_exists():
    """Bold is not rendered at all today, in any document.

    `resolve()` took only text and script, so a source block marked bold
    came back regular in every PDF this engine produced.
    """
    from ct_engine.fonts.registry import REGISTRY

    regular = REGISTRY.resolve("Directions", bold=False)
    bold = REGISTRY.resolve("Directions", bold=True)
    assert bold.weight_matched is True
    assert bold.key != regular.key


def test_a_missing_weighted_face_falls_back_rather_than_dropping_text():
    """Never refuse to draw because a weight was unavailable."""
    from ct_engine.fonts.registry import REGISTRY

    resolved = REGISTRY.resolve("مرحبا", script="Arabic", bold=True)
    assert resolved.key  # usable either way
    if not resolved.weight_matched:
        assert resolved.builtin or resolved.path


def test_regular_resolution_is_unchanged():
    """The default path must behave exactly as before."""
    from ct_engine.fonts.registry import REGISTRY

    assert REGISTRY.resolve("Hello").key == REGISTRY.resolve("Hello", bold=False).key


def _files_only(registry, monkeypatch) -> None:
    """Hide MuPDF's built-in faces from `registry`.

    The built-in bold face is tried before any file and covers Latin, so
    tests of how font *files* are ranked would otherwise never reach them.
    """
    real = registry._load

    def load(*, builtin=None, path=None, cache=True):
        return None if builtin else real(path=path, cache=cache)

    monkeypatch.setattr(registry, "_load", load)


def _any_real_font() -> "Path":
    """A real, parseable font file guaranteed to exist wherever the engine's
    own dependencies are installed -- CI, the container image and every
    developer machine alike -- so weight-ranking tests can plant it under
    synthetic stems ("Test Bold.ttf", "Test Black.ttf", ...) instead of
    depending on which fonts happen to ship with the host OS. The ranking
    logic matches on filename only, so which real font backs the bytes is
    irrelevant to what it tests.
    """
    import reportlab
    from pathlib import Path

    return Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"


def test_a_true_bold_face_outranks_a_heavier_relative(tmp_path, monkeypatch):
    """"Bold" and "Black" are not the same weight.

    A flat marker set let the first alphabetical match win, so a family
    offering both a Bold and a Black face resolved to Black -- visibly
    heavier than the source's actual bold, and unreported because a real,
    covering face was found (`weight_matched` stayed True).
    """
    import shutil

    from ct_engine.fonts.registry import FontRegistry

    src = _any_real_font()

    # Named so the heavier face sorts first alphabetically -- proving the
    # win comes from rank, not from directory-listing or filename order.
    shutil.copy(src, tmp_path / "Aaa Black.ttf")
    shutil.copy(src, tmp_path / "Zzz Bold.ttf")

    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])
    _files_only(registry, monkeypatch)

    resolved = registry.resolve("Directions", bold=True)
    assert resolved.weight_matched is True
    assert "black" not in resolved.display_name.lower()
    assert "bold" in resolved.display_name.lower()


def test_a_family_with_only_a_heavier_relative_still_resolves_it(tmp_path, monkeypatch):
    """No Bold exists in this family -- Black is a legitimate fallback and
    must still come back `weight_matched=True`: a real face was found and
    will be drawn, so nothing should be reported as missing."""
    import shutil

    from ct_engine.fonts.registry import FontRegistry

    shutil.copy(_any_real_font(), tmp_path / "Only Black.ttf")

    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])
    _files_only(registry, monkeypatch)

    resolved = registry.resolve("Directions", bold=True)
    assert resolved.weight_matched is True
    assert "black" in resolved.display_name.lower()


def test_gathering_spans_every_search_directory_not_just_the_first(tmp_path, monkeypatch):
    """The better-ranked face may live in a directory searched *after* the
    first one that has any match at all -- Bold, found second, must still
    beat Black, found first. This is not a regression test for the most
    recent enumeration fix (that landed earlier, in the ranking commit);
    it is ordinary coverage for the "gather everything before choosing"
    rule that `_resolve_weighted` has followed since.
    """
    import shutil

    from ct_engine.fonts.registry import FontRegistry

    src = _any_real_font()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    shutil.copy(src, dir_a / "Test Black.ttf")
    shutil.copy(src, dir_b / "Test Bold.ttf")

    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [dir_a, dir_b])
    _files_only(registry, monkeypatch)

    resolved = registry.resolve("Directions", bold=True)
    assert resolved.weight_matched is True
    assert "bold" in resolved.display_name.lower()
    assert "black" not in resolved.display_name.lower()


def test_gathering_recurses_into_subdirectories(tmp_path, monkeypatch):
    """Debian's font packages install one level below the search directory
    itself (e.g. fonts-dejavu-core puts its faces under
    /usr/share/fonts/truetype/dejavu/, not directly in /usr/share/fonts/).
    A shallow, top-level-only scan of the search directory would silently
    find nothing there, even though a matching face is on disk.
    """
    import shutil

    from ct_engine.fonts.registry import FontRegistry

    nested = tmp_path / "truetype" / "somefont"
    nested.mkdir(parents=True)
    shutil.copy(_any_real_font(), nested / "Test Bold.ttf")

    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])
    _files_only(registry, monkeypatch)

    resolved = registry.resolve("Directions", bold=True)
    assert resolved.weight_matched is True
    assert "bold" in resolved.display_name.lower()


def test_unweighted_resolution_also_recurses_into_subdirectories(tmp_path, monkeypatch):
    """`_system_fonts` was already recursive before the enumeration fix --
    there was never a recursion bug on this path specifically. What this
    guards is that it goes through the *shared* enumeration: with
    `_search_dirs` mocked to a single directory, the old code's `[1:]`
    slice (skipping what it assumed was the bundled directory first) would
    have skipped this one too, scanning nothing at all.
    """
    import shutil

    from ct_engine.fonts.registry import FontRegistry

    nested = tmp_path / "truetype" / "somefont"
    nested.mkdir(parents=True)
    shutil.copy(_any_real_font(), nested / "Test Regular.ttf")

    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])

    assert any(p.name == "Test Regular.ttf" for p in registry._system_fonts())
