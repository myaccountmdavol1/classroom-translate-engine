"""Font search stays small: in memory, in time, and in the PDF it writes.

One emoji no font had made the registry open every font on the server and
keep them all (1.3 GB+ measured, on a 512 MB host). And every drawn font was
embedded whole, so a one-page worksheet downloaded at megabytes.
"""

from __future__ import annotations

import shutil

import fitz
import pytest

from ct_engine.fonts.registry import FontRegistry, FontResolutionError
from ct_engine.reconstruct import builder
from ct_engine.reconstruct.builder import rebuild_document
from test_fonts_and_fitting import _any_real_font, _files_only
from test_reconstruct import prepared_pages
from test_translate_pipeline import EchoProvider


def _counting_loads(registry, monkeypatch) -> list:
    """Record every font *file* the registry opens."""
    opened: list = []
    real = registry._load

    def load(*, builtin=None, path=None, cache=True):
        if path is not None:
            opened.append(path)
        return real(builtin=builtin, path=path, cache=cache)

    monkeypatch.setattr(registry, "_load", load)
    return opened


def _file_fonts_kept(registry) -> list[str]:
    return [k for k in registry._font_cache if "/" in k or "\\" in k]


def test_a_symbol_no_font_has_does_not_open_every_font(monkeypatch):
    registry = FontRegistry()
    opened = _counting_loads(registry, monkeypatch)

    with pytest.raises(FontResolutionError) as caught:
        registry.resolve("Instrucciones \U000f0001")

    assert caught.value.missing == "\U000f0001"
    # Only the handful of symbol faces, never the whole font directory.
    assert len(opened) <= 8, opened
    assert len(_file_fonts_kept(registry)) == 0


def test_an_emoji_beside_bold_words_does_not_search_every_bold_file(monkeypatch):
    registry = FontRegistry()
    opened = _counting_loads(registry, monkeypatch)

    with pytest.raises(FontResolutionError) as caught:
        registry.resolve("\U0001f4d8 Installation instructions", bold=True)

    assert caught.value.missing == "\U0001f4d8"
    assert len(opened) <= 12, opened


def test_a_failed_search_is_remembered(monkeypatch):
    registry = FontRegistry()
    with pytest.raises(FontResolutionError):
        registry.resolve("Hola \U000f0002")

    def must_not_search(*args, **kwargs):
        raise AssertionError("searched again for text already known to fail")

    monkeypatch.setattr(registry, "_resolve_uncached", must_not_search)
    with pytest.raises(FontResolutionError) as caught:
        registry.resolve("Hola \U000f0002")
    assert caught.value.missing == "\U000f0002"


def test_files_that_fall_short_are_not_kept(tmp_path, monkeypatch):
    """A broad search opens each file in turn and lets go of the ones that
    do not cover the text -- holding them all is what ran out of memory."""
    for name in ("One.ttf", "Two.ttf", "Three.ttf"):
        shutil.copy(_any_real_font(), tmp_path / name)
    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])
    opened = _counting_loads(registry, monkeypatch)

    # Adlam: letters, not symbols, and in none of these files.
    with pytest.raises(FontResolutionError):
        registry.resolve("\U0001e900\U0001e901")
    assert len(opened) == 3
    assert _file_fonts_kept(registry) == []

    # The character each one lacked is remembered: a second, different
    # string with it is ruled out without reopening the files.
    opened.clear()
    with pytest.raises(FontResolutionError):
        registry.resolve("\U0001e900\U0001e901\U0001e902")
    assert opened == []


def test_the_file_that_covers_the_text_is_kept(tmp_path, monkeypatch):
    shutil.copy(_any_real_font(), tmp_path / "Vera.ttf")
    registry = FontRegistry()
    monkeypatch.setattr(registry, "_search_dirs", lambda: [tmp_path])
    _files_only(registry, monkeypatch)

    resolved = registry.resolve("Lozenge")
    assert resolved.path is not None and resolved.path.name == "Vera.ttf"
    assert _file_fonts_kept(registry) == [str(tmp_path / "Vera.ttf")]


# --- Which face plain and styled text gets --------------------------------
def test_latin_text_uses_a_small_builtin_face_not_the_cjk_one():
    """Droid Sans Fallback covers Latin too, but embeds as 1.7 MB."""
    registry = FontRegistry()
    assert registry.resolve("Escribe tu respuesta, señor.").builtin == "helv"
    assert registry.resolve("你好").builtin == "china-s"


@pytest.mark.parametrize("look,bold,italic,expect", [
    ("sans", False, True, "heit"),
    ("sans", True, False, "hebo"),
    ("serif", False, False, "tiro"),
    ("serif", False, True, "tiit"),
    ("serif", True, True, "tibi"),
    ("mono", True, False, "cobo"),
])
def test_styled_latin_keeps_the_sources_look(look, bold, italic, expect):
    """An italic Helvetica line stays a sans italic, a Times one a serif
    italic -- not whatever italic file sorts first on the machine."""
    resolved = FontRegistry().resolve(
        "Instrucciones: lee cada pregunta.", bold=bold, italic=italic, look=look
    )
    assert resolved.builtin == expect
    assert resolved.weight_matched is True


def test_apple_private_faces_are_never_offered():
    from pathlib import Path

    assert FontRegistry._denied(Path("/System/Library/Fonts/.NewYork-RegularItalic.otf"))
    assert not FontRegistry._denied(Path("/usr/share/fonts/NotoSans-Italic.ttf"))


# --- The PDF written ----------------------------------------------------------
@pytest.mark.parametrize("target,transform,needle", [
    ("es", lambda t: "Instrucciones: " + t, "Instrucciones"),
    ("zh", lambda t: "说明 " + t, "说明"),
    ("es", lambda t: "\U0001f4d8 Indicaciones " + t, "Indicaciones"),
])
def test_a_translated_worksheet_stays_small_and_searchable(target, transform, needle):
    data, pages = prepared_pages(
        "01_clean_worksheet.pdf", EchoProvider(transform), target=target
    )
    out, _ = rebuild_document(data, pages, target_lang=target)
    # Was 2.4 MB (Spanish) and 1.7 MB (Chinese) with whole fonts embedded.
    assert len(out) < 200_000, len(out)
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert needle in doc[0].get_text()
    finally:
        doc.close()


def test_a_failed_subset_keeps_the_full_fonts(monkeypatch):
    data, pages = prepared_pages(
        "01_clean_worksheet.pdf", EchoProvider(lambda t: "说明 " + t), target="zh"
    )

    def broken(self, *args, **kwargs):
        raise RuntimeError("subsetting failed")

    monkeypatch.setattr(fitz.Document, "subset_fonts", broken)
    out, _ = rebuild_document(data, pages, target_lang="zh")
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert "说明" in doc[0].get_text()
    finally:
        doc.close()


def test_a_subset_that_changes_the_text_is_thrown_away(monkeypatch):
    written = fitz.open()
    page = written.new_page()
    page.insert_text((50, 80), "Hola", fontname="china-s")
    original = written.tobytes()
    written.close()

    def corrupting(self, *args, **kwargs):
        self[0].insert_text((50, 120), "extra", fontname="helv")

    monkeypatch.setattr(fitz.Document, "subset_fonts", corrupting)
    assert builder._with_subset_fonts(original) == original


def test_a_bold_heading_does_not_restyle_the_regular_words_after_it():
    """A bold file kept for a heading is not reused for plain text: the
    registry outlives a document, so it would also restyle the next one."""
    text = "สวัสดีครับ นักเรียน"
    try:
        fresh = FontRegistry().resolve(text, script="Thai")
        registry = FontRegistry()
        bolded = registry.resolve(text, script="Thai", bold=True)
    except FontResolutionError:
        pytest.skip("no Thai font on this machine")
    regular = registry.resolve(text, script="Thai")
    assert regular.key == fresh.key
    if bolded.path is not None and bolded.key != fresh.key:
        assert regular.key != bolded.key


def test_a_missing_vowel_sign_is_not_dropped_as_a_symbol():
    # A combining mark changes the word it sits on; only emoji and the like
    # may be left out to save the rest of the block.
    assert builder._without_symbols("नमस्ते᳐ कक्षा", "᳐") is None
    assert builder._without_symbols("Hi 🙂 class", "🙂") == "Hi  class"
