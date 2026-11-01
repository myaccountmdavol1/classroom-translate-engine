"""A worksheet's own Google Font is used for its translation.

No network: the index is a small stand-in and the "downloads" are the
engine's own copies of Lexend and Noto Sans.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ct_engine.fonts import google
from ct_engine.fonts.registry import FontRegistry

FACES = Path(google.__file__).parent / "faces"
INDEX = {
    "commit": "0123456789abcdef",
    "families": {
        "lexend": ["ofl/lexend", "Lexend", "s",
                   [[400, 0, "Lexend-Regular.ttf", 1000], [700, 0, "Lexend-Bold.ttf", 1000]], []],
        # Calibri's twin, played by Noto Sans.
        "carlito": ["ofl/carlito", "Carlito", "s",
                    [[400, 0, "NotoSans-Regular.ttf", 1000], [700, 0, "NotoSans-Bold.ttf", 1000],
                     [400, 1, "NotoSans-Italic.ttf", 1000]], []],
        "huge": ["ofl/huge", "Huge", "s", [[400, 0, "Huge.ttf", 50_000_000]], []],
    },
}


@pytest.fixture
def fonts(monkeypatch, tmp_path):
    """Google Fonts switched on, offline; the URLs asked for."""
    asked: list[str] = []

    def fetch(url):
        asked.append(url)
        found = list(FACES.glob(f"*/{url.rsplit('/', 1)[-1]}"))
        return found[0].read_bytes() if found else None

    monkeypatch.setenv("CT_GOOGLE_FONTS", "1")
    monkeypatch.setattr(google, "_index", lambda: INDEX)
    monkeypatch.setattr(google, "fetch", fetch)
    monkeypatch.setattr(google, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(google, "_failed", set())
    return asked


@pytest.mark.parametrize("name, bold, want", [
    ("ABCDEF+Lexend-Bold", False, ("lexend", 700, False)),
    ("Lexend-SemiBoldItalic", False, ("lexend", 600, True)),
    ("LexendBold", False, ("lexend", 700, False)),
    ("Lexend Regular", False, ("lexend", 400, False)),
    ("Lexend", True, ("lexend", 700, False)),
    ("Calibri", False, ("carlito", 400, False)),
    ("Calibri,BoldItalic", False, ("carlito", 700, True)),
    ("Calibri-Bold", False, ("carlito", 700, False)),
])
def test_a_pdf_font_name_is_its_family_weight_and_slant(fonts, name, bold, want):
    m = google.match(name, bold)
    assert (m.key, m.weight, m.italic) == want


@pytest.mark.parametrize("name", ["ArialMT", "Helvetica-Bold", "TimesNewRomanPSMT", "ComicSansMS", None, ""])
def test_a_font_that_is_not_a_google_font_is_not_matched(fonts, name):
    assert google.match(name) is None


def test_the_nearest_weight_is_fetched_once(fonts):
    path, exact = google.face_file(google.match("Lexend-SemiBold"))
    assert path.name == "Lexend-Bold.ttf" and not exact
    again, _ = google.face_file(google.match("Lexend-Bold"))
    assert again == path
    assert fonts == ["https://cdn.jsdelivr.net/gh/google/fonts@0123456789abcdef/ofl/lexend/Lexend-Bold.ttf"]


def test_a_font_too_big_to_fetch_is_not_fetched(fonts):
    assert google.face_file(google.Match("huge", "Huge", 400, False)) is None
    assert fonts == []


def test_the_translation_is_drawn_in_the_source_font(fonts):
    resolved = FontRegistry().resolve("Educación física", bold=True, source="ABCDEF+Lexend-Bold")
    assert resolved.path.name == "Lexend-Bold.ttf"
    assert resolved.display_name == "Lexend" and resolved.weight_matched


def test_her_chosen_font_wins_over_the_source_font(fonts):
    resolved = FontRegistry().resolve("Educación", family="noto-serif", source="Lexend-Bold")
    assert "Lexend" not in str(resolved.path)
    assert fonts == []


def test_letters_the_source_font_lacks_fall_back_as_before(fonts):
    resolved = FontRegistry().resolve("学校", script="Han", source="Lexend-Regular")
    assert "Lexend" not in str(resolved.path or resolved.builtin)


def test_a_failed_download_falls_back_and_is_not_tried_again(fonts, monkeypatch):
    calls = []

    def offline(url):
        calls.append(url)
        return None

    monkeypatch.setattr(google, "fetch", offline)
    registry = FontRegistry()
    first = registry.resolve("Hola", source="Lexend-Regular")
    registry.resolve("Adiós", source="Lexend-Regular")
    assert "Lexend" not in str(first.path or first.builtin)
    assert len(calls) == 1


def test_switched_off_nothing_is_fetched(fonts, monkeypatch):
    monkeypatch.setenv("CT_GOOGLE_FONTS", "0")
    FontRegistry().resolve("Hola", source="Lexend-Bold")
    assert fonts == []


def test_a_weight_cut_from_a_variable_font_is_named_as_a_static_file_would_be():
    assert google._cut_name("Roboto Serif", 700, True) == "RobotoSerif-BoldItalic.ttf"
    assert google._cut_name("Lexend", 400, False) == "Lexend-Regular.ttf"
    assert google._cut_name("Lexend", 400, True) == "Lexend-Italic.ttf"
