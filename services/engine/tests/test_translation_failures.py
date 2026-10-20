"""When the translation service fails, the teacher is told -- never handed
an English file marked "Complete".

She usually cannot read the target language, so a document that came back
untranslated looks finished unless the engine says otherwise. No network:
every engine here is a fake, or real code talking to a mocked transport.
"""

from __future__ import annotations

import datetime as _dt
import json
import time

import httpx
import pytest

from conftest import corpus_file
from ct_engine.model.document import Page
from ct_engine.model.enums import TranslationStatus
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan
from ct_engine.pipeline.translate import translate_page
from ct_engine.providers.translation import remote
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from ct_engine.providers.translation.cache import GLOBAL_CACHE, TranslationCache
from ct_engine.providers.translation.chain import (
    ALLOWANCE_USED_UP,
    ALLOWANCE_USED_UP_TODAY,
    SERVICE_UNREACHABLE,
    FallbackTranslationProvider,
    is_exhausted,
    is_resting,
    mark_exhausted,
    next_day_start,
    next_month_start,
    reset_exhausted,
    rest,
    untranslated_reason,
)
from ct_engine.providers.translation.remote import (
    DeepLTranslationProvider,
    GoogleTranslationProvider,
    _CachingProvider,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_exhausted()
    GLOBAL_CACHE.clear()
    # Retries wait in real life; here they must not.
    monkeypatch.setattr(remote, "_sleep", lambda _s: None)
    yield
    reset_exhausted()
    GLOBAL_CACHE.clear()


def _segs(n: int, prefix: str = "Line") -> list[TranslationSegment]:
    return [TranslationSegment(segment_id=str(i), text=f"{prefix} {i}") for i in range(n)]


def _mock_http(monkeypatch, handler):
    """Every httpx.Client built from here on talks to `handler`."""
    real = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(remote.httpx, "Client", client)


# --- DeepL ----------------------------------------------------------------
def test_deepl_never_sends_more_than_50_texts_in_one_request(monkeypatch):
    monkeypatch.setattr(DeepLTranslationProvider, "__init__", lambda self: None)
    provider = DeepLTranslationProvider()
    provider._key, provider._endpoint = "k:fx", "https://deepl.test/v2/translate"
    sizes = []

    def handler(request):
        texts = json.loads(request.content)["text"]
        sizes.append(len(texts))
        return httpx.Response(200, json={"translations": [{"text": f"es {t}"} for t in texts]})

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(60), source_lang="en", target_lang="es")
    assert sizes == [50, 10]
    assert all(not r.error for r in out) and out[59].text == "es Line 59"


# --- Google -----------------------------------------------------------------
def _google(monkeypatch):
    monkeypatch.setattr(GoogleTranslationProvider, "__init__", lambda self: None)
    provider = GoogleTranslationProvider()
    provider._key = "test"
    return provider


def _google_ok(request):
    q = json.loads(request.content)["q"]
    return httpx.Response(
        200, json={"data": {"translations": [{"translatedText": f"es {t}"} for t in q]}}
    )


def test_a_google_rate_limit_is_retried_and_does_not_set_google_aside(monkeypatch):
    provider = _google(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _google_ok(request)

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(2), source_lang="en", target_lang="es")
    assert len(calls) == 2
    assert not out[0].error and out[0].text == "es Line 0"
    assert not is_exhausted("google")


def test_a_dropped_connection_is_retried(monkeypatch):
    provider = _google(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("reset", request=request)
        return _google_ok(request)

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(1), source_lang="en", target_lang="es")
    assert len(calls) == 3 and not out[0].error


def test_a_service_that_hangs_is_not_asked_again_and_is_set_aside(monkeypatch):
    """Each hang costs the full read timeout; three of them per batch, page
    after page, held a 30-page packet for an hour and an Office download
    past the proxy's timeout."""
    provider = _google(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(70), source_lang="en", target_lang="es")
    # One request for the first batch; the second is not sent at all.
    assert len(calls) == 1
    assert all(r.error for r in out)
    assert is_resting("google") and not is_exhausted("google")


def test_retries_stop_at_the_batch_deadline(monkeypatch):
    provider = _google(monkeypatch)
    calls = []
    clock = iter([0.0, 69.0, 69.0])
    monkeypatch.setattr(remote, "_clock", lambda: next(clock))

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(1), source_lang="en", target_lang="es")
    assert len(calls) == 1 and out[0].error


def test_the_chain_moves_on_from_a_resting_engine():
    class _Ok(_Broken):
        def _translate_batch(self, segments, *, source_lang, target_lang):
            return [
                TranslationResult(segment_id=s.segment_id, text=f"es {s.text}", provider=self.name)
                for s in segments
            ]

    first, second = _Ok("deepl-r"), _Ok("google-r")
    rest("deepl-r", 60)
    out = FallbackTranslationProvider([first, second]).translate(
        _segs(1), source_lang="en", target_lang="es"
    )
    assert out[0].provider == "google-r" and not out[0].error


def test_a_service_that_keeps_failing_gives_up_after_three_tries(monkeypatch):
    provider = _google(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(1), source_lang="en", target_lang="es")
    assert len(calls) == 3 and out[0].error
    assert not is_exhausted("google")


def test_a_google_daily_cap_sets_it_aside_until_tomorrow_not_next_month(monkeypatch):
    provider = _google(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            403, json={"error": {"errors": [{"reason": "dailyLimitExceeded"}]}}
        )

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error and len(calls) == 1  # a quota answer is not retried
    assert is_exhausted("google")
    from ct_engine.providers.translation import chain

    assert chain._exhausted["google"] == next_day_start()


def test_a_refused_google_key_is_not_called_a_used_up_allowance(monkeypatch):
    provider = _google(monkeypatch)

    def handler(request):
        return httpx.Response(
            403, json={"error": {"message": "API key not valid", "status": "PERMISSION_DENIED"}}
        )

    _mock_http(monkeypatch, handler)
    out = provider.translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error
    assert not is_exhausted("google")


# --- Allowance memory ------------------------------------------------------
def test_an_allowance_comes_back_when_its_time_has_passed():
    past = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=1)
    mark_exhausted("engine-x", past)
    assert not is_exhausted("engine-x")
    mark_exhausted("engine-y")
    assert is_exhausted("engine-y")


def test_the_reset_times_are_in_the_future():
    now = _dt.datetime(2026, 12, 31, 23, 0, tzinfo=_dt.timezone.utc)
    assert next_month_start(now) == _dt.datetime(2027, 1, 1, tzinfo=_dt.timezone.utc)
    assert next_day_start(now) == _dt.datetime(2027, 1, 1, 8, tzinfo=_dt.timezone.utc)
    early = _dt.datetime(2026, 10, 13, 3, 0, tzinfo=_dt.timezone.utc)
    assert next_day_start(early) == _dt.datetime(2026, 10, 13, 8, tzinfo=_dt.timezone.utc)


class _Broken(_CachingProvider):
    def __init__(self, name, status=None):
        self.name = name
        self.status = status
        self.QUOTA_STATUSES = frozenset({456})

    def supports(self, source_lang, target_lang):
        return True

    def _translate_batch(self, segments, *, source_lang, target_lang):
        request = httpx.Request("POST", "https://example.test")
        if self.status is None:
            raise httpx.ConnectError("down", request=request)
        raise httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(self.status, request=request)
        )


def test_the_reason_says_used_up_only_when_it_is():
    down = FallbackTranslationProvider([_Broken("deepl-d"), _Broken("google-d")])
    out = down.translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error.startswith(SERVICE_UNREACHABLE)

    spent = FallbackTranslationProvider([_Broken("deepl-e", 456), _Broken("google-e", 456)])
    out = spent.translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error.startswith(ALLOWANCE_USED_UP)


def test_a_daily_cap_says_tomorrow_not_the_1st():
    google = _Broken("google-t")
    mark_exhausted("google-t", next_day_start())
    out = FallbackTranslationProvider([google]).translate(
        _segs(1), source_lang="en", target_lang="es"
    )
    assert out[0].error.startswith(ALLOWANCE_USED_UP_TODAY)

    # DeepL out for the month, Google for the day: back tomorrow.
    deepl = _Broken("deepl-t")
    mark_exhausted("deepl-t", next_month_start())
    chain = FallbackTranslationProvider([deepl, google])
    assert untranslated_reason(chain, "en", "es") == ALLOWANCE_USED_UP_TODAY


def test_an_engine_that_does_not_offer_the_language_does_not_count():
    """DeepL has no Haitian Creole; with Google used up, nothing can
    translate it, and "try again in a few minutes" would be false."""

    class _NoCreole(_Broken):
        def supports(self, source_lang, target_lang):
            return target_lang != "ht"

    chain = FallbackTranslationProvider([_NoCreole("deepl-h"), _Broken("google-h")])
    mark_exhausted("google-h", next_month_start())
    assert untranslated_reason(chain, "en", "ht") == ALLOWANCE_USED_UP
    # For a language DeepL offers, it still has its allowance.
    assert untranslated_reason(chain, "en", "es") == SERVICE_UNREACHABLE


# --- Cache -----------------------------------------------------------------
def test_the_cache_keeps_what_was_used_recently():
    cache = TranslationCache(max_entries=2)
    cache.put("a", "en", "es", "p", "A")
    cache.put("b", "en", "es", "p", "B")
    assert cache.get("a", "en", "es", "p") == "A"  # now the most recent
    cache.put("c", "en", "es", "p", "C")
    assert cache.get("b", "en", "es", "p") is None
    assert cache.get("a", "en", "es", "p") == "A"
    assert cache.stats["entries"] == 2


# --- The page ----------------------------------------------------------------
class _Silent(TranslationProvider):
    """A service that answers every segment with an error."""

    name = "silent"

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        return [
            TranslationResult(segment_id=s.segment_id, text=s.text, error="timed out")
            for s in segments
        ]


def _page(*texts: str) -> Page:
    objects = []
    for i, text in enumerate(texts):
        box = Rect(x0=50, y0=50 + 30 * i, x1=400, y1=70 + 30 * i)
        objects.append(
            TextObject(
                id=f"t{i}", bbox=box, norm_bbox=box.normalized(612, 792),
                lines=[TextLine(bbox=box, spans=[TextSpan(text=text, bbox=box)])],
            )
        )
    return Page(index=0, number=1, width=612, height=792, text_objects=objects)


def test_text_the_service_never_translated_is_not_blamed_on_a_formula():
    page = _page("Write your name on the line.", "Show all of your work.")
    stats = translate_page(page, _Silent(), source_lang="en", target_lang="es")
    assert stats.eligible == 2 and stats.provider_failed == 2
    codes = {f.code for f in stats.findings}
    assert codes == {"translation_unavailable"}
    assert all(o.translation_status is TranslationStatus.FAILED for o in page.text_objects)


# --- The whole document --------------------------------------------------------
def test_a_document_the_service_could_not_translate_fails_with_the_reason(
    client, monkeypatch
):
    chain = FallbackTranslationProvider([_Broken("deepl-z", 456)])
    monkeypatch.setattr("ct_engine.api.routes.get_translation_provider", lambda: chain)
    monkeypatch.setattr("ct_engine.jobs.runner.get_translation_provider", lambda: chain)

    up = client.post(
        "/api/documents",
        files={"file": ("w.pdf", corpus_file("01_clean_worksheet.pdf"), "application/pdf")},
    ).json()
    doc_id = up["document_id"]
    deadline = time.time() + 60
    while client.get(f"/api/documents/{doc_id}/job").json()["stage"] != "REVIEW_READY":
        assert time.time() < deadline
        time.sleep(0.05)

    r = client.post(
        f"/api/documents/{doc_id}/translate",
        json={"target_language": "es", "translate_image_text": False},
    )
    assert r.status_code == 200, r.text
    deadline = time.time() + 120
    while True:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("COMPLETE", "FAILED"):
            break
        assert time.time() < deadline
        time.sleep(0.05)
    assert job["stage"] == "FAILED"
    assert job["error"] == ALLOWANCE_USED_UP
