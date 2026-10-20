"""Checking a translation in English works behind the DeepL/Google chain.

English is not a student language, so it is missing from LANGUAGES. The
chain asked each engine whether it offered the language, both said no, and
"Check what it says in English" failed with "did not answer".
"""

import httpx

from ct_engine.languages import can_translate_into
from ct_engine.providers.translation.base import TranslationResult, TranslationSegment
from ct_engine.providers.translation.chain import (
    LANGUAGE_UNAVAILABLE,
    FallbackTranslationProvider,
    untranslated_reason,
)
from ct_engine.providers.translation.remote import (
    DeepLTranslationProvider,
    GoogleTranslationProvider,
)
from ct_engine.translate.one_off import translate_one


def test_english_is_a_language_engines_may_translate_into():
    assert can_translate_into("en")
    assert can_translate_into("en-US")
    assert can_translate_into("es")
    assert not can_translate_into("xx")
    assert not can_translate_into(None)


def test_deepl_and_google_offer_english():
    deepl = DeepLTranslationProvider.__new__(DeepLTranslationProvider)
    google = GoogleTranslationProvider.__new__(GoogleTranslationProvider)
    assert deepl.supports("es", "en")
    assert google.supports(None, "en")
    # DeepL's own gaps still hold.
    assert not deepl.supports("en", "ht")


class _Echo:
    """An engine that offers what `offers` says and answers in brackets."""

    sends_data_externally = False

    def __init__(self, name, offers):
        self.name = name
        self._offers = offers
        self.calls = 0

    def supports(self, source_lang, target_lang):
        return self._offers(target_lang)

    def translate(self, segments, *, source_lang, target_lang):
        self.calls += 1
        return [
            TranslationResult(segment_id=s.segment_id, text=f"[{target_lang}] {s.text}", provider=self.name)
            for s in segments
        ]

    def close(self):
        pass


def test_the_chain_translates_back_into_english():
    first = _Echo("deepl", can_translate_into)
    chain = FallbackTranslationProvider([first, _Echo("google", can_translate_into)])
    result = translate_one("Se coloca una planta en un armario.", target_lang="en", provider=chain)
    assert result.text == "[en] Se coloca una planta en un armario."
    assert first.calls == 1


def test_a_language_no_engine_offers_says_so_instead_of_try_again():
    chain = FallbackTranslationProvider([_Echo("deepl", lambda t: False), _Echo("google", lambda t: False)])
    [out] = chain.translate([TranslationSegment(segment_id="a", text="Hola")], source_lang=None, target_lang="xx")
    assert out.error == LANGUAGE_UNAVAILABLE
    assert untranslated_reason(chain, None, "xx") == LANGUAGE_UNAVAILABLE


def test_the_browser_transport_leaves_out_headers_a_page_cannot_set():
    from ct_browser import relay

    seen = {}

    def handle(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"ok": True})

    transport = relay._BrowserSafeTransport()
    transport._inner = httpx.MockTransport(handle)
    with httpx.Client(transport=transport) as client:
        client.post("https://example.test/api/mt/deepl", json={"text": ["Hola"]})
    for name in relay._FORBIDDEN_HEADERS:
        assert name not in seen
    assert seen.get("content-type") == "application/json"
