"""DeepL and Google through the web app's relay (ct_browser/relay.py).

The relay passes DeepL's and Google's answers through verbatim, so the
engine's own rules (456, Google's daily 403, Retry-After, the fallback
chain, the cache) must behave exactly as they do against the real services.
Only a refused class code (401), a relay timeout (504) and the character cap
are new. No network: the relay is an httpx.MockTransport here.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextvars
import json
import re
import time
from pathlib import Path

import httpx
import pytest

from conftest import corpus_file
from ct_browser import relay
from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.jobs import runner as runner_mod
from ct_engine.jobs.runner import JobCancelled, JobRunner
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import JobStage
from ct_engine.model.job import DocumentJob, JobSettings
from ct_engine.pipeline.manifest_store import ManifestStore
from ct_engine.providers.registry import (
    TRANSLATION_PROVIDERS,
    ProviderNotAvailable,
    get_translation_provider,
)
from ct_engine.providers.translation import remote
from ct_engine.providers.translation.base import TranslationSegment
from ct_engine.providers.translation.cache import GLOBAL_CACHE
from ct_engine.providers.translation.chain import (
    ALLOWANCE_USED_UP_TODAY,
    FallbackTranslationProvider,
    is_exhausted,
    is_resting,
    next_day_start,
    reset_exhausted,
    untranslated_reason,
)
from ct_engine.providers.translation.remote import (
    DeepLTranslationProvider,
    GoogleTranslationProvider,
)
from ct_engine.util.hashing import sha256_bytes

RELAY = "http://localhost:3000/api/mt"
CODE = "correct horse"
WEB_RELAY_TS = (
    Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "lib" / "server" / "mtRelay.ts"
)


@pytest.fixture(autouse=True)
def _relay(monkeypatch):
    reset_exhausted()
    GLOBAL_CACHE.clear()
    monkeypatch.setattr(remote, "_sleep", lambda _s: None)
    monkeypatch.setenv("CT_RELAY_URL", RELAY)
    monkeypatch.delenv("CT_RELAY_ORIGIN", raising=False)
    relay.install()
    token = relay.ACCESS_CODE.set(CODE)
    yield
    relay.ACCESS_CODE.reset(token)
    relay.uninstall()
    reset_exhausted()
    GLOBAL_CACHE.clear()


class Relay:
    """A stand-in for /api/mt: records every request, answers per service."""

    def __init__(self, monkeypatch, deepl=None, google=None) -> None:
        self.requests: list[httpx.Request] = []
        self._answer = {"deepl": deepl or deepl_ok, "google": google or google_ok}
        monkeypatch.setattr(relay, "_transport", httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        service = request.url.path.rsplit("/", 1)[-1]
        return self._answer[service](request)

    def calls(self, service: str) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/" + service))


def deepl_ok(request):
    texts = json.loads(request.content)["text"]
    return httpx.Response(200, json={"translations": [{"text": f"es {t}"} for t in texts]})


def google_ok(request):
    q = json.loads(request.content)["q"]
    return httpx.Response(
        200, json={"data": {"translations": [{"translatedText": f"g {t}"} for t in q]}}
    )


def refused(_request):
    # The relay's own 401 body (mtRelay.ts ACCESS_REQUIRED).
    return httpx.Response(
        401,
        json={
            "error": "This service needs the class access code.",
            "detail": "This service needs the class access code.",
            "diagnostic": None,
            "code": "access_required",
        },
    )


def _segs(n: int, prefix: str = "Line") -> list[TranslationSegment]:
    return [TranslationSegment(segment_id=str(i), text=f"{prefix} {i}") for i in range(n)]


def _chain():
    provider = get_translation_provider("deepl,google")
    assert isinstance(provider, FallbackTranslationProvider)
    return provider


# --- Installing -------------------------------------------------------------
def test_install_puts_the_relay_providers_in_the_registry_and_uninstall_restores():
    chain = _chain()
    assert [type(p) for p in chain.providers] == [relay.RelayDeepL, relay.RelayGoogle]
    # Same names, so the cache, the allowance memory and the chain's
    # messages are the engine's own.
    assert [p.name for p in chain.providers] == ["deepl", "google"]
    assert all(p.sends_data_externally for p in chain.providers)
    relay.uninstall()
    assert TRANSLATION_PROVIDERS["deepl"] is DeepLTranslationProvider
    assert TRANSLATION_PROVIDERS["google"] is GoogleTranslationProvider
    assert JobRunner._run_translation.__qualname__ == "JobRunner._run_translation"
    relay.install()  # for the fixture's uninstall


def test_without_a_relay_address_the_engine_says_translation_is_not_set_up(monkeypatch):
    monkeypatch.delenv("CT_RELAY_URL")
    with pytest.raises(ProviderNotAvailable):
        get_translation_provider("deepl,google")


# --- What is sent -------------------------------------------------------------
def test_requests_go_to_the_relay_with_the_class_code_and_no_key(monkeypatch):
    fake = Relay(monkeypatch)
    monkeypatch.setenv("CT_RELAY_ORIGIN", "http://localhost:3000")
    relay.RelayDeepL().translate(_segs(2), source_lang="en", target_lang="es")
    relay.RelayGoogle().translate(_segs(2, "Row"), source_lang="en", target_lang="es")

    deepl, google = fake.requests
    assert str(deepl.url) == f"{RELAY}/deepl"
    assert str(google.url) == f"{RELAY}/google"  # no ?key=, ever
    for r in (deepl, google):
        assert r.method == "POST"
        assert r.headers["X-CT-Access"] == CODE
        assert r.headers["Origin"] == "http://localhost:3000"
        assert "authorization" not in r.headers
        assert "key" not in r.url.params


def test_no_origin_header_unless_the_native_harness_asks_for_one(monkeypatch):
    fake = Relay(monkeypatch)
    relay.RelayDeepL().translate(_segs(1), source_lang="en", target_lang="es")
    # In a browser the browser sends Origin itself; Python cannot.
    assert "origin" not in fake.requests[0].headers


def test_no_class_code_sends_no_header(monkeypatch):
    fake = Relay(monkeypatch)
    token = relay.ACCESS_CODE.set(None)
    try:
        relay.RelayDeepL().translate(_segs(1), source_lang="en", target_lang="es")
    finally:
        relay.ACCESS_CODE.reset(token)
    assert "x-ct-access" not in fake.requests[0].headers


def test_the_provider_gives_up_before_the_relay_does():
    # mtRelay.ts aborts DeepL/Google at 55 s and answers 504. This side must
    # time out first, so a slow batch is abandoned (and never re-sent) here.
    ts = WEB_RELAY_TS.read_text()
    relay_ms = int(re.search(r"UPSTREAM_TIMEOUT_MS = ([\d_]+)", ts).group(1).replace("_", ""))
    assert relay.RELAY_TIMEOUT.read < relay_ms / 1000
    assert relay.RELAY_TIMEOUT.read < remote._TIMEOUT.read


def test_character_caps_match_the_web_relay():
    ts = WEB_RELAY_TS.read_text()

    def const(name):
        return int(re.search(rf"export const {name} = ([\d_]+);", ts).group(1).replace("_", ""))

    assert relay.MAX_TOTAL_CHARS == const("MAX_TOTAL_CHARS")
    assert relay.MAX_ITEM_CHARS == const("MAX_ITEM_CHARS")
    assert relay.RelayDeepL.batch_size <= const("DEEPL_MAX_ITEMS")
    assert relay.RelayGoogle.batch_size <= const("GOOGLE_MAX_ITEMS")


@pytest.mark.parametrize("cls", [relay.RelayDeepL, relay.RelayGoogle])
def test_a_big_page_is_split_under_the_character_cap_in_order(monkeypatch, cls):
    fake = Relay(monkeypatch)
    # 30 blocks of ~1,000 characters: well over the cap in one batch.
    segs = [
        TranslationSegment(segment_id=str(i), text=f"{i} " + "word " * 200) for i in range(30)
    ]
    out = cls().translate(segs, source_lang="en", target_lang="es")

    assert len(fake.requests) > 1
    for r in fake.requests:
        body = json.loads(r.content)
        texts = body.get("text") or body.get("q")
        assert sum(len(t) for t in texts) <= relay.MAX_TOTAL_CHARS
    assert [r.segment_id for r in out] == [s.segment_id for s in segs]
    assert all(not r.error for r in out)
    assert out[17].text.split()[1] == "17"


# --- Same body, same reading as the engine's own providers ---------------------
_TRICKY = [
    TranslationSegment(segment_id="a", text="  Solve #0# for x.  "),
    TranslationSegment(segment_id="b", text="Use < and > & keep #1#"),
    TranslationSegment(segment_id="c", text="Label the cell wall"),
    TranslationSegment(segment_id="d", text="What is photosynthesis?"),
]


def _deepl_reply(request):
    texts = json.loads(request.content)["text"]
    # DeepL's habits: tags carried through without their spaces, and a note
    # after a finished Chinese sentence.
    return httpx.Response(
        200,
        json={"translations": [{"text": f"{t}。1. 简体中文（大陆）"} for t in texts]},
    )


def _native_and_relay(monkeypatch, native_cls, relay_cls, reply, target):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return reply(request)

    real = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    # remote.httpx IS httpx, so this reaches both providers.
    monkeypatch.setattr(remote.httpx, "Client", client)
    monkeypatch.setattr(native_cls, "__init__", lambda self: None)
    native = native_cls()
    native._key, native._endpoint = "k:fx", "https://deepl.test/v2/translate"
    native_out = native.translate(list(_TRICKY), source_lang="en", target_lang=target)
    GLOBAL_CACHE.clear()
    relay_out = relay_cls().translate(list(_TRICKY), source_lang="en", target_lang=target)
    return sent, native_out, relay_out


def test_relay_deepl_sends_and_reads_exactly_what_the_engine_s_deepl_does(monkeypatch):
    sent, native_out, relay_out = _native_and_relay(
        monkeypatch, DeepLTranslationProvider, relay.RelayDeepL, _deepl_reply, "zh-Hans"
    )
    assert sent[0] == sent[1]
    assert [(r.segment_id, r.text, r.error) for r in native_out] == [
        (r.segment_id, r.text, r.error) for r in relay_out
    ]


def test_relay_google_sends_and_reads_exactly_what_the_engine_s_google_does(monkeypatch):
    sent, native_out, relay_out = _native_and_relay(
        monkeypatch, GoogleTranslationProvider, relay.RelayGoogle, google_ok, "zh-Hans"
    )
    assert sent[0] == sent[1]
    assert sent[1]["target"] == "zh-CN"
    assert [(r.segment_id, r.text, r.error) for r in native_out] == [
        (r.segment_id, r.text, r.error) for r in relay_out
    ]


def test_results_come_back_in_order_with_cache_hits_in_between(monkeypatch):
    fake = Relay(monkeypatch)
    provider = relay.RelayDeepL()
    provider.translate(_segs(3)[1:2], source_lang="en", target_lang="es")  # caches "Line 1"
    out = provider.translate(_segs(5), source_lang="en", target_lang="es")
    assert [r.segment_id for r in out] == ["0", "1", "2", "3", "4"]
    assert [r.text for r in out] == [f"es Line {i}" for i in range(5)]
    assert out[1].cached and not out[0].cached
    assert json.loads(fake.requests[-1].content)["text"] == [
        "Line 0", "Line 2", "Line 3", "Line 4",
    ]


# --- Inherited rules, through the relay ---------------------------------------
def test_456_sets_deepl_aside_and_google_takes_the_rest(monkeypatch):
    fake = Relay(monkeypatch, deepl=lambda r: httpx.Response(456, json={"message": "Quota Exceeded"}))
    chain = _chain()
    out = chain.translate(_segs(3), source_lang="en", target_lang="es")
    assert [r.text for r in out] == ["g Line 0", "g Line 1", "g Line 2"]
    assert is_exhausted("deepl")
    # Not asked again for the rest of the document.
    chain.translate(_segs(3, "More"), source_lang="en", target_lang="es")
    assert fake.calls("deepl") == 1
    assert fake.calls("google") == 2


def test_google_403_with_the_daily_marker_sets_google_aside_until_tomorrow(monkeypatch):
    daily = {"error": {"code": 403, "message": "Daily Limit Exceeded",
                       "errors": [{"reason": "dailyLimitExceeded"}]}}
    Relay(
        monkeypatch,
        deepl=lambda r: httpx.Response(456),
        google=lambda r: httpx.Response(403, json=daily),
    )
    chain = _chain()
    out = chain.translate(_segs(2), source_lang="en", target_lang="es")
    assert all(r.error for r in out)
    assert is_exhausted("google") and is_exhausted("deepl")
    from ct_engine.providers.translation import chain as chain_mod

    assert chain_mod._exhausted["google"] == next_day_start()
    assert untranslated_reason(chain, "en", "es") == ALLOWANCE_USED_UP_TODAY


def test_google_403_without_the_marker_is_not_a_used_up_allowance(monkeypatch):
    Relay(monkeypatch, google=lambda r: httpx.Response(403, json={"error": {"message": "API key not valid"}}))
    out = relay.RelayGoogle().translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error
    assert not is_exhausted("google")


def test_429_is_tried_again_after_its_retry_after(monkeypatch):
    waited = []
    monkeypatch.setattr(remote, "_sleep", waited.append)
    answers = iter([httpx.Response(429, headers={"Retry-After": "3"}), None])

    def deepl(request):
        return next(answers) or deepl_ok(request)

    fake = Relay(monkeypatch, deepl=deepl)
    out = relay.RelayDeepL().translate(_segs(2), source_lang="en", target_lang="es")
    assert [r.text for r in out] == ["es Line 0", "es Line 1"]
    assert waited == [3.0]
    assert fake.calls("deepl") == 2
    assert not is_exhausted("deepl") and not is_resting("deepl")


def test_a_relay_504_is_never_sent_again_and_rests_the_service(monkeypatch):
    fake = Relay(monkeypatch, deepl=lambda r: httpx.Response(504))
    chain = _chain()
    out = chain.translate(_segs(3), source_lang="en", target_lang="es")
    # DeepL may have done (and billed) the work: one request, no retry.
    assert fake.calls("deepl") == 1
    assert is_resting("deepl") and not is_exhausted("deepl")
    assert [r.text for r in out] == ["g Line 0", "g Line 1", "g Line 2"]
    chain.translate(_segs(2, "Next"), source_lang="en", target_lang="es")
    assert fake.calls("deepl") == 1


def test_a_502_from_the_relay_is_tried_again_like_any_outage(monkeypatch):
    answers = iter([httpx.Response(502), None])
    fake = Relay(monkeypatch, deepl=lambda r: next(answers) or deepl_ok(r))
    out = relay.RelayDeepL().translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].text == "es Line 0"
    assert fake.calls("deepl") == 2


# --- A refused class code -----------------------------------------------------
def test_class_code_refused_passes_the_engine_s_page_guards():
    # The engine guards every page and batch with `except Exception`; a wrong
    # code must not be absorbed there one page at a time.
    assert issubclass(relay.ClassCodeRefused, JobCancelled)
    assert not issubclass(relay.ClassCodeRefused, Exception)


def test_outside_a_job_a_401_asks_each_service_at_most_once(monkeypatch):
    fake = Relay(monkeypatch, deepl=refused, google=refused)
    chain = _chain()
    for _ in range(4):
        out = chain.translate(_segs(3), source_lang="en", target_lang="es")
        assert all(r.error and relay.CLASS_CODE_REFUSED in r.error for r in out)
    assert fake.calls("deepl") <= 1 and fake.calls("google") <= 1
    # A wrong code is not a used-up allowance.
    assert not is_exhausted("deepl") and not is_exhausted("google")


def test_a_401_that_is_not_about_the_code_is_an_ordinary_failure(monkeypatch):
    Relay(monkeypatch, deepl=lambda r: httpx.Response(401, json={"message": "nope"}))
    out = relay.RelayDeepL().translate(_segs(1), source_lang="en", target_lang="es")
    assert out[0].error and relay.CLASS_CODE_REFUSED not in out[0].error


# --- Inside a translation job -------------------------------------------------
SETTINGS = JobSettings(target_language="es", translate_image_text=False)


class _ContextPool(cf.ThreadPoolExecutor):
    """Runs each job in the context it was submitted from, as the browser's
    job pool does, so the job carries the request's class code."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


def _analyzed(store, document_id: str) -> ManifestStore:
    name = "13_rotated_pages.pdf"
    data = corpus_file(name)
    source_key = f"documents/{document_id}/source.pdf"
    store.put_bytes(source_key, data)
    manifest = DocumentManifest(
        document_id=document_id, filename=name, content_type="application/pdf",
        byte_size=len(data), sha256=sha256_bytes(data), source_key=source_key,
    )
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            manifest.pages.append(page)
    manifest.analysis.page_count = len(manifest.pages)
    manifests = ManifestStore(store)
    manifests.save_manifest(manifest)
    return manifests


def _run_job(store, document_id: str, monkeypatch) -> DocumentJob:
    monkeypatch.setattr(
        runner_mod, "get_translation_provider", lambda: get_translation_provider("deepl,google")
    )
    manifests = _analyzed(store, document_id)
    runner = JobRunner(store, manifests)
    runner._pool = _ContextPool(max_workers=1)
    try:
        job = runner.submit_translation(document_id, SETTINGS)
        deadline = time.time() + 60
        while True:
            done = manifests.load_job(document_id, job.job_id)
            if done is not None and done.is_terminal:
                return done
            assert time.time() < deadline, f"job still {done.stage if done else None}"
            time.sleep(0.02)
    finally:
        runner._pool.shutdown(wait=True)


def test_a_wrong_class_code_fails_the_job_after_one_request(store, monkeypatch):
    fake = Relay(monkeypatch, deepl=refused, google=refused)
    job = _run_job(store, "refused", monkeypatch)
    assert job.stage == JobStage.FAILED
    assert job.error == relay.CLASS_CODE_REFUSED
    assert "server" not in job.error.lower()
    # Not one request per page per service: the code is wrong for both.
    assert len(fake.requests) == 1
    assert not is_exhausted("deepl") and not is_exhausted("google")


def test_after_a_refused_job_the_next_one_with_a_right_code_runs(store, monkeypatch):
    answers = {"ok": False}
    fake = Relay(
        monkeypatch,
        deepl=lambda r: deepl_ok(r) if answers["ok"] else refused(r),
    )
    assert _run_job(store, "first", monkeypatch).stage == JobStage.FAILED
    answers["ok"] = True
    second = _run_job(store, "second", monkeypatch)
    assert second.stage == JobStage.COMPLETE, second.error
    assert fake.calls("google") == 0


def test_the_job_carries_the_class_code_of_the_request_that_started_it(store, monkeypatch):
    fake = Relay(monkeypatch)
    job = _run_job(store, "carried", monkeypatch)
    assert job.stage == JobStage.COMPLETE, job.error
    assert fake.requests and all(r.headers["X-CT-Access"] == CODE for r in fake.requests)


def test_a_stop_the_teacher_asked_for_still_ends_cancelled(store):
    # The wrapper only changes a stop caused by a refused code.
    manifests = ManifestStore(store)
    runner = JobRunner(store, manifests)
    job = DocumentJob(job_id="job_x", document_id="doc_x", settings=SETTINGS,
                      stage=JobStage.TRANSLATING)
    runner._mark_cancelled(job)
    assert job.stage == JobStage.CANCELLED
    runner._pool.shutdown()
