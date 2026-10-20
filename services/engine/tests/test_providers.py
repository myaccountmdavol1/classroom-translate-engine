"""Provider registry, credentials, caching, and the real offline engine."""

from __future__ import annotations

import pytest

import ct_engine.providers.translation.local_argos  # noqa: F401  (registers)
import ct_engine.providers.translation.remote  # noqa: F401  (registers)
from ct_engine.providers import (
    TRANSLATION_PROVIDERS,
    ProviderNotAvailable,
    get_translation_provider,
)
from ct_engine.providers.translation.base import TranslationError, TranslationSegment
from ct_engine.providers.translation.cache import TranslationCache


def argos_available() -> bool:
    try:
        import argostranslate.translate  # noqa: F401

        return True
    except ImportError:
        return False


needs_argos = pytest.mark.skipif(
    not argos_available(), reason="argostranslate not installed"
)


def test_all_expected_providers_are_registered():
    for name in ("local", "google", "deepl", "openai", "anthropic"):
        assert name in TRANSLATION_PROVIDERS


def test_unknown_provider_raises_rather_than_falling_back():
    """Silently substituting an engine would mislead a district."""
    with pytest.raises(ProviderNotAvailable):
        get_translation_provider("does-not-exist")


@pytest.mark.parametrize("name", ["google", "deepl", "openai", "anthropic"])
def test_remote_providers_refuse_without_credentials(name):
    with pytest.raises((TranslationError, ProviderNotAvailable)) as exc:
        get_translation_provider(name)
    assert "configured" in str(exc.value).lower()


def test_remote_providers_declare_they_send_data_externally():
    """The privacy surface reads this flag; it must be accurate."""
    for name in ("google", "deepl", "openai", "anthropic"):
        assert TRANSLATION_PROVIDERS[name].sends_data_externally is True
    assert TRANSLATION_PROVIDERS["local"].sends_data_externally is False


def test_cache_hits_and_misses():
    cache = TranslationCache()
    assert cache.get("hola", "en", "es", "p") is None
    cache.put("hola", "en", "es", "p", "hello")
    assert cache.get("hola", "en", "es", "p") == "hello"
    # A different provider must not serve the previous engine's output.
    assert cache.get("hola", "en", "es", "other") is None
    # A different target language is a different key.
    assert cache.get("hola", "en", "fr", "p") is None
    assert cache.stats["hits"] == 1


def test_cache_evicts_oldest_entries():
    cache = TranslationCache(max_entries=3)
    for i in range(5):
        cache.put(f"t{i}", "en", "es", "p", f"r{i}")
    assert cache.stats["entries"] <= 3


# --- Real offline engine --------------------------------------------------
@needs_argos
def test_local_provider_reports_unsupported_languages_honestly():
    """Argos publishes no Traditional Chinese or Haitian Creole model.

    Serving Simplified to a family that reads Traditional would be worse than
    refusing.
    """
    provider = get_translation_provider("local")
    assert provider.supports("en", "es")
    assert not provider.supports("en", "zh-Hant")
    assert not provider.supports("en", "ht")


@needs_argos
@pytest.mark.slow
def test_local_provider_actually_translates_and_caches():
    provider = get_translation_provider("local")
    segments = [
        TranslationSegment(segment_id="a", text="Read each question carefully."),
        TranslationSegment(segment_id="b", text="Show your work."),
        TranslationSegment(segment_id="c", text="Read each question carefully."),
    ]
    results = provider.translate(segments, source_lang="en", target_lang="es")
    assert len(results) == 3
    assert [r.segment_id for r in results] == ["a", "b", "c"]
    for r in results[:2]:
        assert r.text and r.text != segments[0].text
        assert not r.error
    # The repeated string must come from cache, not a second model call.
    assert results[2].cached
    assert results[2].text == results[0].text


@needs_argos
@pytest.mark.slow
def test_placeholders_survive_the_real_engine_or_are_reported():
    """The measured guarantee: placeholders survive, or the caller is told."""
    from ct_engine.translate.protection import mask_text, restore_text

    provider = get_translation_provider("local")
    cases = [
        "Solve 3x + 7 = 22 and show your work.",
        "Water is H₂O. Explain why that matters.",
        "Visit www.example.org for the lab sheet.",
    ]
    for source in cases:
        masked = mask_text(source)
        result = provider.translate(
            [TranslationSegment(segment_id="s", text=masked.masked_text)],
            source_lang="en",
            target_lang="es",
        )[0]
        restored = restore_text(result.text, masked.spans)
        # Either every protected span is byte-identical, or it is reported
        # missing. Silent corruption is the one unacceptable outcome.
        if restored.ok:
            for span in masked.spans:
                assert span.text in restored.text
        else:
            assert restored.missing


# --- DeepL output hygiene ---------------------------------------------------
# Measured against the live API on 2026-10-11: a sentence ending in spaces,
# sent with preserve_formatting, came back as
# '长度比宽度的5倍还要长2英尺。Simplified Chinese (Mainland)    '.
# That text was drawn on page 30 of a real packet.
class _FakeDeepL:
    def __init__(self, reply):
        self.reply = reply
        self.sent = None

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.sent = json["text"]
        fake = self

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"translations": [{"text": fake.reply(t)} for t in fake.sent]}

        return R()


def _deepl(monkeypatch, reply):
    from ct_engine.providers.translation import remote

    monkeypatch.setattr(remote.get_settings(), "deepl_api_key", "k:fx", raising=False)
    fake = _FakeDeepL(reply)
    monkeypatch.setattr(remote.httpx, "Client", fake)
    return remote.DeepLTranslationProvider(), fake


def test_deepl_never_sees_edge_whitespace_and_gets_it_back(monkeypatch):
    from ct_engine.providers.translation.base import TranslationSegment

    provider, fake = _deepl(monkeypatch, lambda t: "长度。")
    seg = TranslationSegment(segment_id="a", text="The length.    ", object_ids=[], role="other")
    [out] = provider._translate_batch([seg], source_lang="en", target_lang="zh-Hans")
    assert fake.sent == ["The length."]
    assert out.text == "长度。    "


def test_a_language_name_deepl_adds_is_removed(monkeypatch):
    from ct_engine.providers.translation.base import TranslationSegment

    provider, _ = _deepl(monkeypatch, lambda t: "长度。Simplified Chinese (Mainland)")
    seg = TranslationSegment(segment_id="a", text="The length.", object_ids=[], role="other")
    [out] = provider._translate_batch([seg], source_lang="en", target_lang="zh-Hans")
    assert out.text == "长度。"


def test_a_language_name_in_the_source_is_kept(monkeypatch):
    from ct_engine.providers.translation.base import TranslationSegment

    provider, _ = _deepl(monkeypatch, lambda t: "学习 Simplified Chinese")
    seg = TranslationSegment(segment_id="a", text="Learn Simplified Chinese", object_ids=[], role="other")
    [out] = provider._translate_batch([seg], source_lang="en", target_lang="zh-Hans")
    assert out.text == "学习 Simplified Chinese"


# Measured against the live API on 2026-10-11, Simplified Chinese only: about
# 6 in 10 short sentences came back with a note appended -- in every
# model_type, with and without preserve_formatting, ZH and ZH-HANS alike.
# Spanish and Traditional Chinese came back clean (0 in 10).
@pytest.mark.parametrize("source,reply,expected", [
    ("If one solution, find the solution.",
     "如果只有一个解，就找出该解。1. 保留“解”的译法，符合中文数学术语习惯。",
     "如果只有一个解，就找出该解。"),
    ("Determine the number of solutions by inspection.",
     "通过观察确定解的个数。1. 简体中文（大陆）",
     "通过观察确定解的个数。"),
    ("Determine the number of solutions by inspection.",
     "通过观察确定解的个数。简体中文",
     "通过观察确定解的个数。"),
    # A real numbered list in the source is left alone.
    ("Steps: 1. Add the numbers.",
     "步骤：1. 把数字相加。",
     "步骤：1. 把数字相加。"),
    # A numbered item with no space after its number ("3.Undo") is still a
    # real item, and anything carrying a placeholder is never a note. Real
    # packet: the cleanup cut item 3 and its blanks off, and the teacher's
    # re-translation was refused as "changed a number".
    ("operations. 3.Undo the #0# or #1# using inverse",
     "运算。3. 使用逆运算撤销 #0# 或 #1#",
     "运算。3. 使用逆运算撤销 #0# 或 #1#"),
])
def test_notes_deepl_appends_to_chinese_are_removed(monkeypatch, source, reply, expected):
    from ct_engine.providers.translation.base import TranslationSegment

    provider, _ = _deepl(monkeypatch, lambda t: reply)
    seg = TranslationSegment(segment_id="a", text=source, object_ids=[], role="other")
    [out] = provider._translate_batch([seg], source_lang="en", target_lang="zh-Hans")
    assert out.text == expected
