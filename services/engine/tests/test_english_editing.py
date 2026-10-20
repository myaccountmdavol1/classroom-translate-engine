"""Editing in English, for a teacher who cannot read the translation.

An ENL teacher fixing a Simplified Chinese worksheet cannot edit Chinese.
So every block can be corrected by changing its ENGLISH and translating
again -- even a block whose translation was edited before, since she is
asking for it explicitly -- and new text can be typed in English.
"""

from __future__ import annotations

import pytest

from ct_engine.pipeline.edit import EditError, add_block, translate_from_english
from ct_engine.translate.one_off import OneOffTranslation
from test_edit import first_editable, prepared


@pytest.fixture
def fake_translate(monkeypatch):
    calls = []

    def fake(text, *, target_lang, source_lang=None, provider=None):
        calls.append(text)
        return OneOffTranslation(text=f"[{target_lang}] {text}", protected_intact=True, provider="fake")

    monkeypatch.setattr("ct_engine.translate.one_off.translate_one", fake)
    return calls


def test_new_english_replaces_the_source_and_the_translation(store, fake_translate):
    manifest = prepared(store)
    obj = first_editable(manifest)
    translate_from_english(manifest, 1, obj.id, "Write your name here.", store=store, target_lang="es")
    assert obj.flat_text == "Write your name here."
    assert obj.translated_text == "[es] Write your name here."
    assert fake_translate == ["Write your name here."]


def test_it_retranslates_even_a_hand_edited_block(store, fake_translate):
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.translated_text = "texto a mano"
    obj.manually_edited = True
    translate_from_english(manifest, 1, obj.id, "Name:", store=store, target_lang="es")
    assert obj.translated_text == "[es] Name:"
    assert not obj.manually_edited


def test_empty_english_is_refused(store, fake_translate):
    manifest = prepared(store)
    obj = first_editable(manifest)
    with pytest.raises(EditError):
        translate_from_english(manifest, 1, obj.id, "   ", store=store, target_lang="es")


def test_new_text_can_be_typed_in_english(store, fake_translate):
    manifest = prepared(store)
    _, created = add_block(
        manifest, 1,
        bbox={"x0": 100, "y0": 500, "x1": 300, "y1": 520},
        text="", source_text="What can you double?",
        store=store, target_lang="es",
    )
    assert created.flat_text == "What can you double?"
    assert created.translated_text == "[es] What can you double?"


def test_back_translation_reports_what_the_translation_says(client, monkeypatch):
    """A teacher who cannot read the translation checks its meaning by
    reading it translated back into English. Nothing is changed."""
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    seen = []

    def fake(text, *, target_lang, source_lang=None, provider=None):
        seen.append((text, target_lang))
        return OneOffTranslation(text=f"[{target_lang}] {text}", protected_intact=True, provider="fake")

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    assert _translate(client, doc_id).status_code in (200, 202)
    _wait_terminal(client, doc_id)
    page = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    block = next(b for b in page["blocks"] if b["editable"] and b["translation"])
    monkeypatch.setattr("ct_engine.translate.one_off.translate_one", fake)

    r = client.get(f"/api/documents/{doc_id}/pages/1/blocks/{block['id']}/back-translation")
    assert r.status_code == 200, r.text
    assert r.json()["text"] == f"[en] {block['translation']}"
    assert seen == [(block["translation"], "en")]
    after = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]
    assert after["revision"] == page["revision"]


def test_each_english_line_stays_a_line_in_the_translation(store, fake_translate):
    """She cannot place a line break in Chinese, only in the English."""
    manifest = prepared(store)
    obj = first_editable(manifest)
    translate_from_english(
        manifest, 1, obj.id, "Equation: a + a + 3 = 15\n\nSolution: a = 6",
        store=store, target_lang="es",
    )
    assert obj.translated_text == "[es] Equation: a + a + 3 = 15\n\n[es] Solution: a = 6"
    assert fake_translate == ["Equation: a + a + 3 = 15", "Solution: a = 6"]
    assert obj.flat_text == "Equation: a + a + 3 = 15\n\nSolution: a = 6"


def test_checking_the_english_releases_a_line_left_in_english(store, fake_translate):
    """A picture line read too uncertainly is left in English until she
    checks it; translating her English is that check."""
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.keep_original = True
    translate_from_english(manifest, 1, obj.id, "Add Chrome app or extension by ID",
                           store=store, target_lang="es")
    assert not obj.keep_original
    assert obj.translated_text == "[es] Add Chrome app or extension by ID"


def test_new_english_is_what_a_tight_box_draws(store, fake_translate):
    """A tight box draws its translation-only copy; it must not be the old one."""
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.translation_only_text = "texto viejo"
    translate_from_english(manifest, 1, obj.id, "Write your name.", store=store, target_lang="es")
    assert obj.translation_only_text in (None, "[es] Write your name.")
