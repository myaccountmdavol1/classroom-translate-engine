"""DeepL first, Google when DeepL's free allowance runs out.

Each free plan covers about 500,000 characters a month; chained, a school
gets both before anything costs money. No network: the engines are fakes
that answer the way the real ones do.
"""

from __future__ import annotations

import httpx
import pytest

from ct_engine.providers.registry import get_translation_provider
from ct_engine.providers.translation.base import TranslationSegment
from ct_engine.providers.translation.cache import GLOBAL_CACHE
from ct_engine.providers.translation.chain import (
    FallbackTranslationProvider, is_exhausted, reset_exhausted,
)
from ct_engine.providers.translation.remote import _CachingProvider


def _quota_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    return httpx.HTTPStatusError(
        "quota", request=request, response=httpx.Response(status, request=request),
    )


class _Fake(_CachingProvider):
    def __init__(self, name, *, fail_status=None, quota=frozenset()):
        self.name = name
        self.fail_status = fail_status
        self.QUOTA_STATUSES = quota
        self.calls = 0

    def supports(self, source_lang, target_lang):
        return True

    def _translate_batch(self, segments, *, source_lang, target_lang):
        from ct_engine.providers.translation.base import TranslationResult

        self.calls += 1
        if self.fail_status:
            raise _quota_error(self.fail_status)
        return [
            TranslationResult(segment_id=s.segment_id, text=f"[{self.name}] {s.text}", provider=self.name)
            for s in segments
        ]


@pytest.fixture(autouse=True)
def _clean():
    reset_exhausted()
    GLOBAL_CACHE.clear() if hasattr(GLOBAL_CACHE, "clear") else None
    yield
    reset_exhausted()


def _segs(*texts):
    return [TranslationSegment(segment_id=str(i), text=t) for i, t in enumerate(texts)]


def test_deepl_is_used_while_it_has_allowance():
    deepl, google = _Fake("deepl-a"), _Fake("google-a")
    out = FallbackTranslationProvider([deepl, google]).translate(
        _segs("Write your name."), source_lang="en", target_lang="es",
    )
    assert out[0].text == "[deepl-a] Write your name."
    assert google.calls == 0


def test_a_used_up_deepl_hands_the_rest_of_the_month_to_google():
    deepl = _Fake("deepl-b", fail_status=456, quota=frozenset({456}))
    google = _Fake("google-b")
    chain = FallbackTranslationProvider([deepl, google])
    out = chain.translate(_segs("Show your work."), source_lang="en", target_lang="es")
    assert out[0].text == "[google-b] Show your work." and not out[0].error
    assert is_exhausted("deepl-b")
    # The next request does not ask DeepL again this month.
    calls = deepl.calls
    chain.translate(_segs("Check your answer."), source_lang="en", target_lang="es")
    assert deepl.calls == calls


def test_when_every_allowance_is_used_up_the_text_is_kept_and_says_why():
    chain = FallbackTranslationProvider([
        _Fake("deepl-c", fail_status=456, quota=frozenset({456})),
        _Fake("google-c", fail_status=403, quota=frozenset({403})),
    ])
    out = chain.translate(_segs("Name:"), source_lang="en", target_lang="es")
    assert out[0].error and out[0].text == "Name:"


def test_an_engine_with_no_key_is_left_out_of_the_chain(monkeypatch):
    monkeypatch.setenv("CT_DEEPL_API_KEY", "abc:fx")
    monkeypatch.delenv("CT_GOOGLE_TRANSLATE_API_KEY", raising=False)
    from ct_engine.config import get_settings
    get_settings.cache_clear()
    try:
        provider = get_translation_provider("deepl,google")
        assert provider.name == "deepl"
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_deepl_tags_keep_the_spaces_around_protected_math():
    from ct_engine.providers.translation.remote import _from_tags, _to_tags

    source = "Plot (3, 4) & label it #0# then use #1#."
    assert _to_tags(source) == 'Plot (3, 4) &amp; label it <x i="0"/> then use <x i="1"/>.'
    # DeepL dropped one space and doubled another.
    back = _from_tags('Traza (3, 4) y ponle una etiqueta<x i="0"/> y usa  <x i="1"/>.', source)
    assert back == "Traza (3, 4) y ponle una etiqueta #0# y usa #1#."
