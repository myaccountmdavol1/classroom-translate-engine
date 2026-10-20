"""A teacher can pick a type style; the PDF is drawn in a font that has it.

Styles, not font names: what is installed differs between a Mac and the
server, and every style has to work in every script. A style with no font
covering the text falls back to the automatic choice rather than failing.
"""

from __future__ import annotations

import pytest

from ct_engine.fonts.registry import REGISTRY
from ct_engine.pipeline.edit import EditError, apply_layout
from test_edit import first_editable, prepared


@pytest.mark.parametrize("family,expect", [
    ("serif", ("serif", "times", "georgia", "songti", "stsong", "baskerville", "palatino")),
    ("mono", ("mono", "courier", "menlo", "monaco")),
])
def test_a_style_resolves_to_a_font_of_that_style(family, expect):
    resolved = REGISTRY.resolve("Photosynthesis review", script="Latin", family=family)
    assert any(word in resolved.display_name.lower().replace(" ", "") for word in expect), resolved.display_name


def test_sans_is_not_a_serif_or_mono_face():
    name = REGISTRY.resolve("Photosynthesis review", script="Latin", family="sans").display_name.lower()
    assert "serif" not in name and "mono" not in name


def test_the_choice_is_stored_drawn_and_reset(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    apply_layout(manifest, 1, obj.id, font_family="serif", store=store, target_lang="es")
    assert obj.manual_font_family == "serif"
    assert obj.drawn_layout and obj.drawn_layout.font_path
    apply_layout(manifest, 1, obj.id, font_family="auto", store=store, target_lang="es")
    assert obj.manual_font_family is None


def test_an_unknown_style_is_refused(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    with pytest.raises(EditError):
        apply_layout(manifest, 1, obj.id, font_family="wingdings", store=store, target_lang="es")


def test_a_font_that_cannot_be_drawn_with_is_never_chosen():
    """macOS's Songti.ttc loads and claims every CJK glyph, but MuPDF cannot
    draw with it ("substitute font creation is not implemented"). Chosen as
    the serif style on a real packet, it made the whole page's black text
    vanish from the PDF. Every resolved font must actually draw."""
    import fitz

    for family in (None, "sans", "serif", "mono"):
        for text, script in (("找出能回答以下问题的数字：", "Han"), ("Find the numbers", "Latin")):
            resolved = REGISTRY.resolve(text, script=script, family=family)
            font = REGISTRY.font_for(resolved)
            doc = fitz.open()
            page = doc.new_page()
            writer = fitz.TextWriter(page.rect)
            writer.append((40, 80), text, font=font, fontsize=12)
            writer.write_text(page)  # must not raise
            assert text[:3] in page.get_text() or font.name, (family, resolved.display_name)
