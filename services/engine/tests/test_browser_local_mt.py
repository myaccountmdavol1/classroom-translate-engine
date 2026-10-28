"""Mozilla's models in the browser (ct_browser.local_mt), checked natively.

In the browser, `js.ctMoz.translate(texts, code, html)` is the Bergamot
runtime called synchronously in the worker. Here a fake `js` module plays
that part, with answers shaped like the ones the real runtime gave on the
Unit 1 packet (2026-10-28 spike), including the ways it loses or pads a
placeholder.
"""

from __future__ import annotations

import sys
import types

import pytest

from ct_browser import local_mt
from ct_browser.local_mt import (
    LOST_PLACEHOLDER,
    NOT_READY,
    MozillaTranslationProvider,
    from_marked,
    mozilla_code,
    to_marked,
)
from ct_engine.providers.registry import TRANSLATION_PROVIDERS, get_translation_provider
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from ct_engine.providers.translation.cache import GLOBAL_CACHE
from ct_engine.providers.translation.chain import FallbackTranslationProvider, reset_exhausted


class FakeMoz:
    """The worker's js.ctMoz. `plain` and `marked` map source text to the
    runtime's answer in each mode; anything else comes back as "ES:" + text."""

    def __init__(self, langs=("es",)) -> None:
        self.ready = True
        self.langs = list(langs)
        self.plain: dict[str, str] = {}
        self.marked: dict[str, str] = {}
        self.calls: list[tuple[list[str], str, bool]] = []
        self.fail: Exception | None = None

    def translate(self, texts, code, html):
        texts = list(texts)
        self.calls.append((texts, code, html))
        if self.fail:
            raise self.fail
        table = self.marked if html else self.plain
        return [table.get(t, "ES:" + t) for t in texts]


@pytest.fixture
def moz(monkeypatch):
    fake = FakeMoz()
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(ctMoz=fake))
    GLOBAL_CACHE.clear()
    reset_exhausted()
    local_mt.install()
    yield fake
    local_mt.uninstall()
    GLOBAL_CACHE.clear()


def _seg(i: int, text: str) -> TranslationSegment:
    return TranslationSegment(segment_id=f"s{i}", text=text)


def test_codes_follow_the_models_mozilla_publishes():
    assert mozilla_code("es") == "es"
    assert mozilla_code("pt-BR") == "pt"
    assert mozilla_code("zh-Hans") == "zh"
    assert mozilla_code("zh-Hant") == "zh_hant"
    # No model, or the wrong variety: left to the next engine.
    assert mozilla_code("ht") is None
    assert mozilla_code("pt-PT") is None
    assert mozilla_code("pt") is None
    assert mozilla_code(None) is None


def test_every_code_has_a_model_in_the_lock():
    """apps/web/scripts/mt-models.lock.json is what the worker can load."""
    import json
    from pathlib import Path

    lock = Path(__file__).resolve().parents[3] / "apps" / "web" / "scripts" / "mt-models.lock.json"
    models = set(json.loads(lock.read_text())["models"])
    assert set(local_mt.MOZILLA_CODES.values()) == models


def test_supports_only_a_loaded_model_from_english(moz):
    p = MozillaTranslationProvider()
    assert p.supports("en", "es")
    assert p.supports(None, "es")
    assert not p.supports("fr", "es")  # models start from English
    assert not p.supports("en", "zh-Hans")  # not loaded in this worker
    assert not p.supports("en", "ht")


def test_supports_nothing_before_the_runtime_is_ready(monkeypatch):
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace())
    assert not MozillaTranslationProvider().supports("en", "es")


def test_translates_a_batch_in_one_call_and_keeps_placeholders(moz):
    moz.plain["Solve #0# and check your answer."] = "Resuelve #0# y comprueba tu respuesta."
    out = MozillaTranslationProvider().translate(
        [_seg(0, "Solve #0# and check your answer."), _seg(1, "Find the cost.")],
        source_lang="en", target_lang="es",
    )
    assert [r.text for r in out] == ["Resuelve #0# y comprueba tu respuesta.", "ES:Find the cost."]
    assert all(r.error is None and r.provider == "mozilla" for r in out)
    assert len(moz.calls) == 1 and moz.calls[0][1:] == ("es", False)


def test_a_lost_placeholder_is_tried_again_as_markup(moz):
    src = "Let #0# = Andrea’s running distance and #1# = Patrick’s."
    # Measured: plain mode dropped the closing '#' of the first one.
    moz.plain[src] = "Deja #0 = La distancia de carrera de Andrea y #1# = Patrick’s."
    moz.marked[to_marked(src)] = 'Deja  <x i="0"></x>= La distancia de carrera de Andrea y  <x i="1"></x>= de Patrick.'
    [r] = MozillaTranslationProvider().translate([_seg(0, src)], source_lang="en", target_lang="es")
    assert r.error is None
    assert r.text == "Deja #0# = La distancia de carrera de Andrea y #1# = de Patrick."
    assert [c[2] for c in moz.calls] == [False, True]


def test_a_placeholder_lost_twice_is_an_error_for_the_next_engine(moz):
    src = "Inverse Operations: Opposite operations that #0# each other out."
    moz.plain[src] = "逆向运行:相互对接的相反操作。"
    moz.marked[to_marked(src)] = "逆向运行:相互对接的相反操作。"
    [r] = MozillaTranslationProvider().translate([_seg(0, src)], source_lang="en", target_lang="es")
    assert r.error == LOST_PLACEHOLDER
    assert r.text == src


def test_a_runtime_failure_marks_every_segment(moz):
    moz.fail = RuntimeError("memory access out of bounds")
    out = MozillaTranslationProvider().translate(
        [_seg(0, "One."), _seg(1, "Two.")], source_lang="en", target_lang="es"
    )
    assert [r.error for r in out] == [NOT_READY, NOT_READY]


def test_cached_segments_are_not_sent_again(moz):
    p = MozillaTranslationProvider()
    p.translate([_seg(0, "Find the cost.")], source_lang="en", target_lang="es")
    p.translate([_seg(1, "Find the cost."), _seg(2, "Check it.")], source_lang="en", target_lang="es")
    assert moz.calls[1][0] == ["Check it."]


def test_markup_escaping_round_trips():
    src = "If a < b & #0# > 3, solve #1#."
    marked = to_marked(src)
    assert marked == 'If a &lt; b &amp; <x i="0"></x> &gt; 3, solve <x i="1"></x>.'
    assert from_marked(marked) == src


class _Backup(TranslationProvider):
    """Stands in for the DeepL relay at the end of the chain."""

    name = "backup"
    sends_data_externally = True

    def __init__(self) -> None:
        self.seen: list[str] = []

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        self.seen += [s.text for s in segments]
        return [TranslationResult(segment_id=s.segment_id, text="DL:" + s.text, provider=self.name) for s in segments]


def test_the_chain_sends_only_failed_segments_to_the_backup(moz):
    bad = "Inverse Operations: Opposite operations that #0# each other out."
    moz.plain[bad] = moz.marked[to_marked(bad)] = "lost it"
    backup = _Backup()
    chain = FallbackTranslationProvider([MozillaTranslationProvider(), backup])
    out = chain.translate([_seg(0, "Find the cost."), _seg(1, bad)], source_lang="en", target_lang="es")
    assert [r.text for r in out] == ["ES:Find the cost.", "DL:" + bad]
    assert backup.seen == [bad]


def test_registered_by_name_for_the_browser_chain(moz):
    assert TRANSLATION_PROVIDERS["mozilla"] is MozillaTranslationProvider
    assert isinstance(get_translation_provider("mozilla"), MozillaTranslationProvider)


# --- Chrome's built-in translator ------------------------------------------


class FakeChrome:
    """The worker's js.ctChrome: the host tab's Translator, synchronously."""

    def __init__(self, langs=("es",)) -> None:
        self.ready = bool(langs)
        self.langs = list(langs)
        self.answers: dict[str, str] = {}
        self.calls: list[tuple[list[str], str]] = []
        self.fail: Exception | None = None

    def translate(self, texts, code):
        texts = list(texts)
        self.calls.append((texts, code))
        if self.fail:
            raise self.fail
        return [self.answers.get(t, "CH:" + t) for t in texts]


@pytest.fixture
def both(monkeypatch):
    moz, chrome = FakeMoz(), FakeChrome()
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(ctMoz=moz, ctChrome=chrome))
    GLOBAL_CACHE.clear()
    reset_exhausted()
    local_mt.install()
    yield moz, chrome
    local_mt.uninstall()
    GLOBAL_CACHE.clear()


def test_chrome_gets_brackets_and_gives_back_placeholders(both):
    _, chrome = both
    chrome.answers["Solve ⟦0⟧ and check your answer."] = "Resuelve ⟦0⟧ y comprueba tu respuesta."
    [r] = local_mt.ChromeTranslationProvider().translate(
        [_seg(0, "Solve #0# and check your answer.")], source_lang="en", target_lang="es"
    )
    assert r.error is None and r.provider == "chrome"
    assert r.text == "Resuelve #0# y comprueba tu respuesta."
    assert chrome.calls == [(["Solve ⟦0⟧ and check your answer."], "es")]


def test_chrome_supports_only_what_the_host_said_is_ready(both):
    p = local_mt.ChromeTranslationProvider()
    assert p.supports("en", "es")
    assert not p.supports("en", "zh-Hans")
    assert not p.supports("fr", "es")
    assert not p.supports("en", "ht")


def test_chrome_then_mozilla_then_backup(both):
    moz, chrome = both
    lost = "Undo the #0# or #1# using inverse operations."
    chrome.answers[local_mt.to_brackets(lost)] = "Deshacer el ⟦0 o ⟦1⟧ usando operaciones inversas."
    backup = _Backup()
    chain = FallbackTranslationProvider(
        [local_mt.ChromeTranslationProvider(), MozillaTranslationProvider(), backup]
    )
    out = chain.translate([_seg(0, "Find the cost."), _seg(1, lost)], source_lang="en", target_lang="es")
    assert [r.provider for r in out] == ["chrome", "mozilla"]
    assert out[1].text == "ES:" + lost
    assert backup.seen == []


def test_chrome_failing_sends_everything_on(both):
    _, chrome = both
    chrome.fail = RuntimeError("Chrome's translator answered 503")
    out = local_mt.ChromeTranslationProvider().translate(
        [_seg(0, "One."), _seg(1, "Two.")], source_lang="en", target_lang="es"
    )
    assert [r.error for r in out] == [local_mt.CHROME_NOT_READY] * 2


def test_chrome_codes_match_the_page_copy():
    """src/engine/chrome-mt.ts primes the same languages on the click."""
    import re
    from pathlib import Path

    ts = (Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "engine" / "chrome-mt.ts").read_text()
    block = ts.split("CHROME_CODES: Record<string, string> = {", 1)[1].split("};", 1)[0]
    pairs = dict(re.findall(r'"?([\w-]+)"?:\s*"([\w-]+)"', block))
    assert pairs == local_mt.CHROME_CODES
