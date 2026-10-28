"""Translation on the teacher's own computer: Mozilla's models (Bergamot),
and Chrome's built-in translator (below).

The worker loads the Bergamot runtime (the WebAssembly build of Marian that
Firefox translates with) and the model for a job's language before the job
runs, and exposes them as `js.ctMoz`. Its translate() is a plain synchronous
call into the runtime, so the engine calls it mid-job exactly as it calls
DeepL, and nothing leaves the computer: no key, no allowance, no class code.

Mozilla's models translate from English only (one model per direction), so
a document detected as another language is left to the next engine in the
chain.

Placeholders. Segments carry ``#N#`` where the engine keeps something as it
is (a formula, a blank, a URL). In plain mode the models keep them in nearly
every segment (measured 13/13 es and pt, 12/13 zh on the Unit 1 packet). A
segment that loses one is sent again with the placeholders as empty
elements in HTML mode (``<x i="N"></x>``), which held 13/13 in all three.
If it still loses one, the segment comes back as an error, so the chain
hands it to the next engine rather than the page getting a broken formula.
"""

from __future__ import annotations

import html
import re
from typing import Any

from ct_engine.providers.registry import TRANSLATION_PROVIDERS
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from ct_engine.providers.translation.cache import GLOBAL_CACHE
from ct_engine.util.logging import get_logger

log = get_logger(__name__)

# The engine's target codes -> Mozilla's model names (en-<code>). Only
# languages Mozilla publishes a released English model for. Left out:
#   ht       Haitian Creole: no model.
#   pt-PT    Mozilla's "pt" writes Brazilian Portuguese ("você"); a family
#            in Portugal is better served by the next engine than by the
#            wrong variety, as Argos's Traditional Chinese rule says.
MOZILLA_CODES: dict[str, str] = {
    "es": "es", "fr": "fr", "ar": "ar", "bn": "bn", "ur": "ur", "ru": "ru",
    "ko": "ko", "ja": "ja", "vi": "vi", "pl": "pl", "he": "he", "uk": "uk",
    "pt-BR": "pt", "zh-Hans": "zh", "zh-Hant": "zh_hant",
}

# The engine's tolerant reading (protection._PLACEHOLDER_RE): a tokenizer
# may write "# 3 #", which still restores.
_PLACEHOLDER = re.compile(r"#\s*(\d{1,3})\s*#")
_MARK = re.compile(r'<x\s+i="(\d{1,3})"\s*>\s*</x>|<x\s+i="(\d{1,3})"\s*/>')

NOT_READY = "The translator on this computer is not ready for this language."
LOST_PLACEHOLDER = "The translator on this computer changed a formula or blank."


def mozilla_code(target_lang: str | None) -> str | None:
    """The model a job into `target_lang` needs, or None if there is none."""
    if not target_lang:
        return None
    if target_lang in MOZILLA_CODES:
        return MOZILLA_CODES[target_lang]
    # A bare base code ("pt", "zh") is not guessed: which variety it means
    # is exactly what the table decides.
    return None


def from_english(source_lang: str | None) -> bool:
    """Mozilla's models start from English. No detected language counts as
    English, as it does for the engine's other local provider."""
    return source_lang is None or source_lang.split("-")[0].lower() in ("en", "und", "")


def _bridge() -> Any | None:
    """The worker's js.ctMoz, once its runtime is loaded; else None.

    Looked up on every call: the worker loads models between Python
    entries, after this provider may already exist. Outside Pyodide (the
    native tests) a fake `js` module stands in.
    """
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return None
    bridge = getattr(js, "ctMoz", None)
    if bridge is None or not getattr(bridge, "ready", False):
        return None
    return bridge


def loaded_models() -> frozenset[str]:
    bridge = _bridge()
    if bridge is None:
        return frozenset()
    return frozenset(str(code) for code in bridge.langs)


def _to_js(texts: list[str]) -> Any:
    try:
        from pyodide.ffi import to_js  # type: ignore[import-not-found]
    except ImportError:
        return texts
    return to_js(texts)


def same_placeholders(source: str, out: str) -> bool:
    """Every placeholder in the source, each exactly once, and no others."""
    want = sorted(m.group(1) for m in _PLACEHOLDER.finditer(source))
    got = sorted(m.group(1) for m in _PLACEHOLDER.finditer(out))
    return want == got


def to_marked(text: str) -> str:
    """Plain text with placeholders -> the HTML the runtime's html mode reads."""
    escaped = html.escape(text, quote=False)
    return _PLACEHOLDER.sub(lambda m: f'<x i="{m.group(1)}"></x>', escaped)


def from_marked(text: str) -> str:
    """The runtime's HTML back to plain text with placeholders.

    The runtime puts the element against the next word and pads the space
    before it ('que  <x i="0"></x>se'), so each one is read back with one
    space either side and the spacing tidied.
    """
    restored = _MARK.sub(lambda m: f" #{m.group(1) or m.group(2)}# ", text)
    restored = html.unescape(restored)
    restored = re.sub(r"[ \t]{2,}", " ", restored)
    # No space before closing punctuation that followed a placeholder.
    restored = re.sub(r"(#\d{1,3}#) ([.,;:!?)\]。，；：！？）])", r"\1\2", restored)
    return restored.strip()


class MozillaTranslationProvider(TranslationProvider):
    """Mozilla's Bergamot models, run in the engine's own worker."""

    name = "mozilla"
    sends_data_externally = False

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        code = mozilla_code(target_lang)
        return code is not None and from_english(source_lang) and code in loaded_models()

    def _run(self, bridge: Any, texts: list[str], code: str, html_mode: bool) -> list[str]:
        out = bridge.translate(_to_js(texts), code, html_mode)
        to_py = getattr(out, "to_py", None)
        values = to_py() if to_py is not None else out
        result = [str(t) for t in values]
        if len(result) != len(texts):
            raise RuntimeError(f"expected {len(texts)} translations, got {len(result)}")
        return result

    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        if not segments:
            return []
        code = mozilla_code(target_lang)
        bridge = _bridge()
        if code is None or bridge is None or code not in loaded_models():
            # Every segment marked, never raised: the chain moves them on.
            return [self._failed(s, NOT_READY) for s in segments]

        done: dict[str, str] = {}
        errors: dict[str, str] = {}
        todo: list[TranslationSegment] = []
        for seg in segments:
            cached = GLOBAL_CACHE.get(seg.text, "en", code, self.name)
            if cached is not None:
                done[seg.segment_id] = cached
            else:
                todo.append(seg)

        fresh: list[TranslationSegment] = []
        if todo:
            again: list[TranslationSegment] = []
            try:
                plain = self._run(bridge, [s.text for s in todo], code, False)
            except Exception as exc:
                log.warning("Mozilla translator failed on %d segments: %r", len(todo), exc)
                plain = None
                for seg in todo:
                    errors[seg.segment_id] = NOT_READY
            for seg, out in zip(todo, plain or []):
                if same_placeholders(seg.text, out):
                    done[seg.segment_id] = out
                    fresh.append(seg)
                else:
                    again.append(seg)
            if again:
                try:
                    marked = self._run(bridge, [to_marked(s.text) for s in again], code, True)
                except Exception as exc:
                    log.warning("Mozilla translator (html) failed: %r", exc)
                    marked = [""] * len(again)
                for seg, out in zip(again, marked):
                    text = from_marked(out)
                    if text and same_placeholders(seg.text, text):
                        done[seg.segment_id] = text
                        fresh.append(seg)
                    else:
                        errors[seg.segment_id] = LOST_PLACEHOLDER
        for seg in fresh:
            GLOBAL_CACHE.put(seg.text, "en", code, self.name, done[seg.segment_id])

        return [
            TranslationResult(segment_id=s.segment_id, text=done[s.segment_id], provider=self.name)
            if s.segment_id in done
            else self._failed(s, errors.get(s.segment_id, NOT_READY))
            for s in segments
        ]

    def _failed(self, seg: TranslationSegment, why: str) -> TranslationResult:
        return TranslationResult(segment_id=seg.segment_id, text=seg.text, provider=self.name, error=why)


# --- Chrome's built-in translator ------------------------------------------
#
# Chrome and Edge (138+, on a computer) translate on the device with the
# Translator API, which exists only on a page. The worker's js.ctChrome
# sends each batch to the host tab (src/engine/chrome-mt.ts) as a
# synchronous request and returns the texts. Before a job the worker asks
# the host whether the language is ready; `langs` says which is.
#
# Placeholders. Chrome's Spanish model dropped the closing '#' of "#0#" in 5
# of 13 packet sentences; written as ⟦0⟧ they held 13/13 in es, pt and zh
# (2026-10-28 spike). Chrome has no markup mode, so a segment that still
# loses one goes to the next engine.

# Our target codes -> Chrome's. Mirrored in src/engine/chrome-mt.ts
# CHROME_CODES (used there to download the language on the Translate
# click). Left out: ht and ur (no Chrome model), pt-PT (Chrome's "pt" is
# Brazilian Portuguese).
CHROME_CODES: dict[str, str] = {
    "es": "es", "fr": "fr", "ar": "ar", "bn": "bn", "ru": "ru", "ko": "ko",
    "ja": "ja", "vi": "vi", "pl": "pl", "he": "he", "uk": "uk",
    "pt-BR": "pt", "zh-Hans": "zh", "zh-Hant": "zh-Hant",
}

_BRACKET = re.compile(r"⟦\s*(\d{1,3})\s*⟧")
CHROME_NOT_READY = "Chrome's translator is not ready for this language."


def chrome_code(target_lang: str | None) -> str | None:
    return CHROME_CODES.get(target_lang or "")


def _chrome_bridge() -> Any | None:
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return None
    bridge = getattr(js, "ctChrome", None)
    if bridge is None or not getattr(bridge, "ready", False):
        return None
    return bridge


def to_brackets(text: str) -> str:
    return _PLACEHOLDER.sub(lambda m: f"⟦{m.group(1)}⟧", text)


def from_brackets(text: str) -> str:
    return _BRACKET.sub(lambda m: f"#{m.group(1)}#", text)


class ChromeTranslationProvider(TranslationProvider):
    """Chrome's on-device translator, through the host tab."""

    name = "chrome"
    sends_data_externally = False

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        code = chrome_code(target_lang)
        bridge = _chrome_bridge()
        return (
            code is not None
            and from_english(source_lang)
            and bridge is not None
            and code in [str(c) for c in bridge.langs]
        )

    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        if not segments:
            return []
        code = chrome_code(target_lang)
        bridge = _chrome_bridge()
        if code is None or bridge is None or not self.supports(source_lang, target_lang):
            return [self._failed(s, CHROME_NOT_READY) for s in segments]

        done: dict[str, str] = {}
        errors: dict[str, str] = {}
        todo = []
        for seg in segments:
            cached = GLOBAL_CACHE.get(seg.text, "en", code, self.name)
            if cached is not None:
                done[seg.segment_id] = cached
            else:
                todo.append(seg)
        if todo:
            try:
                out = bridge.translate(_to_js([to_brackets(s.text) for s in todo]), code)
                to_py = getattr(out, "to_py", None)
                texts = [str(t) for t in (to_py() if to_py is not None else out)]
                if len(texts) != len(todo):
                    raise RuntimeError(f"expected {len(todo)} translations, got {len(texts)}")
            except Exception as exc:
                log.warning("Chrome translator failed on %d segments: %r", len(todo), exc)
                texts = None
            for i, seg in enumerate(todo):
                if texts is None:
                    errors[seg.segment_id] = CHROME_NOT_READY
                    continue
                text = from_brackets(texts[i])
                if same_placeholders(seg.text, text):
                    done[seg.segment_id] = text
                    GLOBAL_CACHE.put(seg.text, "en", code, self.name, text)
                else:
                    errors[seg.segment_id] = LOST_PLACEHOLDER
        return [
            TranslationResult(segment_id=s.segment_id, text=done[s.segment_id], provider=self.name)
            if s.segment_id in done
            else self._failed(s, errors.get(s.segment_id, CHROME_NOT_READY))
            for s in segments
        ]

    def _failed(self, seg: TranslationSegment, why: str) -> TranslationResult:
        return TranslationResult(segment_id=seg.segment_id, text=seg.text, provider=self.name, error=why)


def install() -> None:
    """Register 'mozilla' and 'chrome' (CT_TRANSLATION_PROVIDER=chrome,mozilla,...)."""
    TRANSLATION_PROVIDERS["mozilla"] = MozillaTranslationProvider
    TRANSLATION_PROVIDERS["chrome"] = ChromeTranslationProvider


def uninstall() -> None:
    """Take them out again (for tests)."""
    if TRANSLATION_PROVIDERS.get("mozilla") is MozillaTranslationProvider:
        del TRANSLATION_PROVIDERS["mozilla"]
    if TRANSLATION_PROVIDERS.get("chrome") is ChromeTranslationProvider:
        del TRANSLATION_PROVIDERS["chrome"]
