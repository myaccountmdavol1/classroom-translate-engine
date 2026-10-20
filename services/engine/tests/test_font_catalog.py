"""The fonts a teacher can choose by name (fonts/catalog.py).

Each claim the menu makes -- which languages a font has letters for -- is
checked against the real files, so the editor never promises letters a
font lacks. Choosing one never changes how anything else on the server is
drawn: the faces are not where automatic searches look.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import pytest

from ct_engine.fonts import catalog
from ct_engine.fonts.registry import FONT_STYLES, REGISTRY, FontRegistry
from ct_engine.pipeline.edit import EditError, _font_family, apply_layout
from test_text_styles import _styled

ENGINE = Path(__file__).resolve().parent.parent
WEB = ENGINE.parent.parent / "apps" / "web"

# Letters each language needs beyond plain ASCII.
SAMPLES = {
    "es": "¿Qué año? Niño ÁÉÍÓÚÜÑ áéíóúüñ ¡",
    "pt": "ÃÕÇÂÊÔÀ ãõçâêôà",
    "fr": "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ àâæçéèêëîïôœùûüÿ «»",
    "ht": "Èèòàńm",
    "vi": "Tiếng Việt ăâđêôơư ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ Đ",
    "pl": "ĄĆĘŁŃÓŚŹŻ ąćęłńóśźż",
    "ru": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "uk": "ҐЄІЇ ґєії",
    "he": "שלום עולם",
    "ar": "مرحبا بالعالم",
    "ur": "اردو زبان",
    "bn": "বাংলা",
    "zh": "你好世界",
    "ja": "日本語ひらがな",
    "ko": "안녕하세요",
}


def _has_all(font, text):
    return all(font.has_glyph(ord(c)) for c in text if not c.isspace())


def test_the_catalog_lists_real_openly_licensed_files():
    raw = json.loads(catalog.CATALOG_FILE.read_text())
    assert raw["version"] == 1
    assert [f["id"] for f in raw["faces"]] == [
        "noto-sans", "noto-serif", "open-sans", "andika", "lexend", "comic-neue",
    ]
    for face in raw["faces"]:
        assert face["look"] in ("sans", "serif") and face["name"] and face["hint"]
        for weight, name in face["files"].items():
            if name is None:
                assert weight in ("italic", "bold_italic")
                continue
            path = catalog.FACES_DIR / name
            assert path.is_file(), name
            licence = path.parent / "OFL.txt"
            assert "SIL Open Font License" in licence.read_text(), licence
    marks = catalog.FACES_DIR / raw["marks"]
    assert marks.is_file() and "SIL Open Font License" in (marks.parent / "OFL.txt").read_text()


@pytest.mark.parametrize("face_id", catalog.family_ids())
def test_every_coverage_claim_is_true_and_complete(face_id):
    entry = catalog.face(face_id)
    for weight, name in entry.files.items():
        if name is None:
            continue
        font = fitz.Font(fontfile=str(catalog.FACES_DIR / name))
        for lang, sample in SAMPLES.items():
            claimed = lang in entry.covers
            assert _has_all(font, sample) == claimed, (face_id, weight, lang)


def test_the_marks_face_has_the_bullets_text_faces_lack():
    font = fitz.Font(fontfile=str(catalog.marks_face()))
    assert all(font.has_glyph(ord(c)) for c in "•◦■✓")


def test_the_editor_has_the_same_files():
    copy = WEB / "public" / "fonts" / "v1"
    if not copy.is_dir():
        pytest.skip("the editor's copy of the fonts is not in this checkout")
    for path in sorted(catalog.FACES_DIR.rglob("*")):
        if path.is_file() and path.name != "catalog.json":
            other = copy / path.relative_to(catalog.FACES_DIR)
            assert other.is_file(), other
            assert hashlib.sha256(other.read_bytes()).digest() == hashlib.sha256(
                path.read_bytes()
            ).digest(), other
    web_catalog = WEB / "src" / "lib" / "font-catalog.json"
    if not web_catalog.is_file():
        pytest.skip("the editor's font-catalog.json is not in this checkout")
    assert web_catalog.read_bytes() == catalog.CATALOG_FILE.read_bytes()


def test_a_chosen_font_is_drawn_in_its_weight():
    plain = REGISTRY.resolve("Hola niño", family="andika")
    assert plain.path.name == "Andika-Regular.ttf" and plain.face == "andika"
    assert not plain.fell_back and plain.weight_matched
    bold = REGISTRY.resolve("Hola niño", family="andika", bold=True)
    assert bold.path.name == "Andika-Bold.ttf" and bold.weight_matched
    both = REGISTRY.resolve("Hola niño", family="noto-serif", bold=True, italic=True)
    assert both.path.name == "NotoSerif-BoldItalic.ttf"


def test_a_weight_the_font_lacks_is_drawn_in_its_regular_face():
    italic = REGISTRY.resolve("Hola niño", family="lexend", italic=True)
    assert italic.path.name == "Lexend-Regular.ttf"
    assert italic.face == "lexend" and not italic.weight_matched


@pytest.mark.parametrize("text,script", [("你好世界", "Han"), ("مرحبا بالعالم", "Arabic")])
def test_letters_the_font_lacks_fall_back_to_the_standard_face(text, script):
    from ct_engine.fonts.shaping import shape_logical

    shaped = shape_logical(text, script)
    got = REGISTRY.resolve(shaped, script=script, family="andika")
    assert got.fell_back and got.face is None
    assert got.path is None or not got.path.is_relative_to(catalog.FACES_DIR)
    font = REGISTRY.font_for(got)
    assert _has_all(font, shaped)


def test_automatic_searches_never_see_the_catalog():
    registry = FontRegistry()
    assert not any(p.is_relative_to(catalog.FACES_DIR) for p in registry._all_font_files())
    assert registry.resolve("Plain words").builtin == "helv"
    assert registry.resolve("Bold words", bold=True).builtin == "hebo"
    registry.resolve("Слова по-русски", script="Cyrillic")
    assert not any(p.is_relative_to(catalog.FACES_DIR) for p in registry._plain_chosen)
    assert not any(str(catalog.FACES_DIR) in key for key in registry._font_cache)
    # Choosing one keeps it; plain text afterwards still is not drawn in it.
    registry.resolve("Hola", family="andika")
    assert registry.resolve("Hola otra vez").builtin == "helv"
    assert not any(p.is_relative_to(catalog.FACES_DIR) for p in registry._plain_chosen)


@pytest.mark.parametrize("style", sorted(FONT_STYLES))
def test_older_styles_resolve_as_before(style):
    got = REGISTRY.resolve("Hola", family=style)
    assert got.face is None and not got.fell_back
    assert got.path is None or not got.path.is_relative_to(catalog.FACES_DIR)


def test_font_choices_are_checked():
    assert _font_family("andika") == "andika"
    assert _font_family("serif") == "serif"
    assert _font_family("auto") is None
    with pytest.raises(EditError) as exc:
        _font_family("wingdings")
    assert exc.value.message == "That font is not available."


def test_coverage_reads_language_variants():
    assert catalog.covers("open-sans", "pt-BR") and catalog.covers("open-sans", "he")
    assert not catalog.covers("andika", "zh-Hans") and not catalog.covers("nope", "es")


def test_the_page_and_the_editor_know_the_face(store):
    from ct_engine.api.routes import _block_out

    manifest, obj = _styled(store)
    apply_layout(manifest, 1, obj.id, font_family="andika", store=store, target_lang="es")
    assert obj.drawn_layout.face == "andika" and not obj.drawn_layout.face_fallback
    # The Directions are italic in the source, and Andika has an italic.
    assert obj.drawn_layout.font_path.endswith("andika/Andika-Italic.ttf")
    b = _block_out(obj)
    assert b["font_family"] == "andika" and b["font_face"] == "andika"
    assert b["font_fallback"] is False


def test_a_font_without_the_letters_says_so(store):
    from ct_engine.api.routes import _block_out

    manifest, obj = _styled(store, text="مرحبا بالعالم", lang="ar")
    result = apply_layout(manifest, 1, obj.id, font_family="andika", store=store, target_lang="ar")
    assert obj.drawn_layout.face is None and obj.drawn_layout.face_fallback
    assert not obj.drawn_layout.font_path or "faces" not in obj.drawn_layout.font_path
    b = _block_out(obj)
    assert b["font_family"] == "andika" and b["font_face"] is None and b["font_fallback"]
    # Recorded, not a warning: the words are drawn, in a face with their letters.
    assert not any("Andika" in w for w in result.warnings)


def test_a_list_in_a_chosen_font_draws_its_numbers_in_it(store):
    manifest, obj = _styled(store)
    apply_layout(manifest, 1, obj.id, font_family="comic-neue", list="number",
                 store=store, target_lang="es")
    marks = obj.drawn_layout.marks
    assert marks and all(m.font_path == obj.drawn_layout.font_path for m in marks)
    assert "comic-neue" in obj.drawn_layout.font_path
