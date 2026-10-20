"""Remote translation providers: Google, DeepL, and LLM-based (OpenAI /
Anthropic).

All are opt-in. Each declares `sends_data_externally = True`, which the
privacy surface reads, and each refuses to construct without its credential
rather than failing later mid-document.

The LLM providers carry the exam-integrity contract in their system prompt
(spec sec. 23 / 24): translate, never explain, never answer, never simplify.
Their output is still verified by the protection layer afterward -- a prompt
is a request, not a guarantee.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import time

import httpx

from ...config import get_settings
from ...languages import get_language
from ...util.logging import get_logger
from ..registry import register_translation
from .base import (
    TranslationError,
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from .cache import GLOBAL_CACHE
from .chain import (
    is_exhausted,
    is_resting,
    mark_exhausted,
    next_day_start,
    next_month_start,
    rest,
)

log = get_logger(__name__)

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Batch sizes chosen to stay well inside each API's request limits while
# keeping round-trips down on a 100-page document.
_HTTP_BATCH = 64
_LLM_BATCH = 20

# A dropped connection, a rate limit or a 5xx usually clears in seconds.
# Without a retry one of them leaves a whole page in English.
_RETRY_DELAYS = (1.0, 3.0)  # seconds before the 2nd and 3rd attempt
_RETRY_AFTER_CAP = 10.0
# No batch is tried again once this long has passed since its first attempt.
# Office downloads translate inside the request, behind a proxy timeout.
_BATCH_DEADLINE = 70.0
# A service that stopped answering mid-request is set aside this long, so the
# rest of the document goes to the next engine (or fails) without each page
# waiting out the same timeout.
_REST_AFTER_TIMEOUT = 120.0
_sleep = time.sleep  # replaced in tests
_clock = time.monotonic  # replaced in tests


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
        # The service took the request and then hung for the full read
        # timeout. Asking again would most likely hang just as long.
        return False
    if isinstance(exc, httpx.TransportError):  # connect errors, dropped links
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _retry_delay(exc: Exception, default: float) -> float:
    """The server's Retry-After when it gives one, within reason."""
    if isinstance(exc, httpx.HTTPStatusError):
        raw = exc.response.headers.get("Retry-After", "")
        try:
            return min(max(float(raw), 0.0), _RETRY_AFTER_CAP)
        except ValueError:
            pass
    return default


def _require(value: str | None, env_name: str, provider: str) -> str:
    if not value:
        raise TranslationError(
            f"{provider} translation is selected but not configured on this "
            f"server.",
            diagnostic=f"{env_name} is not set",
        )
    return value


class _CachingProvider(TranslationProvider):
    """Shared cache handling so each provider only implements its transport."""

    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        if not segments:
            return []

        results: dict[str, TranslationResult] = {}
        pending: list[TranslationSegment] = []
        for seg in segments:
            hit = GLOBAL_CACHE.get(seg.text, source_lang, target_lang, self.name)
            if hit is not None:
                results[seg.segment_id] = TranslationResult(
                    segment_id=seg.segment_id,
                    text=hit,
                    provider=self.name,
                    cached=True,
                )
            else:
                pending.append(seg)

        batch_size = self.batch_size
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            try:
                if is_exhausted(self.name):
                    raise TranslationError(
                        f"{self.name} has no translation allowance left.",
                    )
                if is_resting(self.name):
                    raise TranslationError(
                        f"{self.name} stopped answering a moment ago.",
                    )
                out = self._translate_with_retry(
                    chunk, source_lang=source_lang, target_lang=target_lang
                )
            except Exception as exc:
                until = (
                    self._quota_until(exc)
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                if until is not None:
                    # Out of allowance: every further batch would fail the
                    # same way, so the engine is set aside until the
                    # allowance comes back and the rest of the document goes
                    # to the next one.
                    mark_exhausted(self.name, until)
                elif isinstance(exc, httpx.TimeoutException):
                    rest(self.name, _REST_AFTER_TIMEOUT)
                log.warning("%s batch failed: %r", self.name, exc)
                # Degrade per-segment rather than losing the document.
                out = [
                    TranslationResult(
                        segment_id=s.segment_id,
                        text=s.text,
                        provider=self.name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    for s in chunk
                ]
            for r in out:
                results[r.segment_id] = r
                if not r.error and r.text:
                    src = next(
                        (s.text for s in chunk if s.segment_id == r.segment_id), None
                    )
                    if src:
                        GLOBAL_CACHE.put(
                            src, source_lang, target_lang, self.name, r.text
                        )

        # Preserve input order and guarantee one result per input segment.
        return [
            results.get(
                s.segment_id,
                TranslationResult(
                    segment_id=s.segment_id,
                    text=s.text,
                    provider=self.name,
                    error="no result returned",
                ),
            )
            for s in segments
        ]

    batch_size: int = _HTTP_BATCH
    # HTTP statuses that mean "no allowance left", not "this request failed".
    QUOTA_STATUSES: frozenset[int] = frozenset()

    def _translate_with_retry(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        """One batch, tried again after brief failures.

        Quota and auth answers are never retried: asking again cannot change
        them, and each try would spend a request. Nor is a batch tried again
        past _BATCH_DEADLINE, so a slow service cannot hold a page (or an
        Office download) for minutes.
        """
        started = _clock()
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                return self._translate_batch(
                    segments, source_lang=source_lang, target_lang=target_lang
                )
            except Exception as exc:
                quota = (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in self.QUOTA_STATUSES
                )
                if quota or not _is_transient(exc) or attempt == len(_RETRY_DELAYS):
                    raise
                delay = _retry_delay(exc, _RETRY_DELAYS[attempt])
                if _clock() - started + delay >= _BATCH_DEADLINE:
                    raise
                log.info(
                    "%s batch failed (%r); trying again in %.0fs",
                    self.name, exc, delay,
                )
                _sleep(delay)
        raise AssertionError("unreachable")  # the loop returns or raises

    def _quota_until(self, exc: httpx.HTTPStatusError) -> _dt.datetime | None:
        """When the allowance comes back, if this error means it ran out.

        None means the request failed for another reason and the engine
        stays in use. Monthly plans are the default.
        """
        if exc.response.status_code in self.QUOTA_STATUSES:
            return next_month_start()
        return None

    def _translate_batch(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        raise NotImplementedError


@register_translation("google")
class GoogleTranslationProvider(_CachingProvider):
    name = "google"
    sends_data_externally = True
    # 403 is a daily quota -- or a key that may not be used, which is not a
    # quota at all. 429 is a per-minute rate limit: brief, so it is retried
    # and never sets Google aside. See _quota_until.
    QUOTA_STATUSES = frozenset({403})
    _DAILY_MARKERS = (
        "dailylimitexceeded", "quotaexceeded", "daily limit", "per day",
    )
    ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self) -> None:
        self._key = _require(
            get_settings().google_translate_api_key,
            "CT_GOOGLE_TRANSLATE_API_KEY",
            "Google",
        )

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        return get_language(target_lang) is not None

    def _quota_until(self, exc: httpx.HTTPStatusError) -> _dt.datetime | None:
        if exc.response.status_code != 403:
            return None
        try:
            body = exc.response.text.lower()
        except Exception:
            body = ""
        if any(marker in body for marker in self._DAILY_MARKERS):
            # Google's caps are per day (the admin sets one in the Cloud
            # console), so the engine comes back tomorrow, not next month.
            return next_day_start()
        # A key that is wrong or restricted: setting Google aside would
        # hide the real problem behind "allowance used up".
        log.error("Google refused the request (403): %s", body[:300])
        return None

    @staticmethod
    def _code(target: str) -> str:
        # Google uses zh-CN / zh-TW for the Chinese variants.
        return {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}.get(target, target.split("-")[0])

    def _translate_batch(self, segments, *, source_lang, target_lang):
        payload = {
            "q": [s.text for s in segments],
            "target": self._code(target_lang),
            "format": "text",
        }
        if source_lang:
            payload["source"] = source_lang.split("-")[0]
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                self.ENDPOINT, params={"key": self._key}, json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        translations = data.get("data", {}).get("translations", [])
        if len(translations) != len(segments):
            raise TranslationError(
                "The translation service returned an unexpected response.",
                diagnostic=f"expected {len(segments)} results, got {len(translations)}",
            )
        return [
            TranslationResult(
                segment_id=seg.segment_id,
                text=t.get("translatedText", ""),
                provider=self.name,
            )
            for seg, t in zip(segments, translations)
        ]


_PLACEHOLDER = re.compile(r"#(\d+)#")
_TAG = re.compile(r'<x\s+i="(\d+)"\s*/>|<x\s+i="(\d+)"\s*>\s*</x>')


def _to_tags(text: str) -> str:
    """Plain text with placeholders, as the XML DeepL's tag mode expects."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _PLACEHOLDER.sub(lambda m: f'<x i="{m.group(1)}"/>', escaped)


def _from_tags(text: str, source: str) -> str:
    """DeepL's XML back to plain text with placeholders.

    In tag mode DeepL drops the space between a word and a tag ("ponle una
    etiqueta#0#"); a space the source had there is put back.
    """
    restored = _TAG.sub(lambda m: f"#{m.group(1) or m.group(2)}#", text)
    restored = restored.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    for m in _PLACEHOLDER.finditer(source):
        token = m.group(0)
        spaced_before = m.start() > 0 and source[m.start() - 1].isspace()
        spaced_after = m.end() < len(source) and source[m.end()].isspace()
        if spaced_before:
            restored = re.sub(rf"(?<=\w){re.escape(token)}", f" {token}", restored)
        if spaced_after:
            restored = re.sub(rf"{re.escape(token)}(?=\w)", f"{token} ", restored)
    if "  " not in source:
        # And sometimes adds one beside a tag ("Utiliza  3x+2").
        restored = re.sub(r"(?<=\S) {2,}(?=\S)", " ", restored)
    return restored


@register_translation("deepl")
class DeepLTranslationProvider(_CachingProvider):
    name = "deepl"
    sends_data_externally = True
    # 456: "Quota exceeded", the free plan's monthly characters are used up.
    QUOTA_STATUSES = frozenset({456})
    # DeepL accepts at most 50 `text` values per /v2/translate request; a
    # bigger batch is refused whole.
    batch_size = 50

    def __init__(self) -> None:
        self._key = _require(get_settings().deepl_api_key, "CT_DEEPL_API_KEY", "DeepL")
        # Free-tier keys carry a ":fx" suffix and use a different host.
        self._endpoint = (
            "https://api-free.deepl.com/v2/translate"
            if self._key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

    # DeepL's supported set is narrower than our registry; naming the
    # unsupported ones explicitly avoids a mid-document failure.
    _UNSUPPORTED = {"ht", "bn", "ur", "he"}

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        lang = get_language(target_lang)
        return lang is not None and lang.base not in self._UNSUPPORTED

    @staticmethod
    def _code(target: str) -> str:
        return {
            "zh-Hans": "ZH-HANS", "zh-Hant": "ZH-HANT",
            "pt-BR": "PT-BR", "pt-PT": "PT-PT", "en": "EN-US",
        }.get(target, target.split("-")[0].upper())

    @staticmethod
    def _language_names(target_lang: str) -> list[str]:
        """How DeepL might name the target language in its own output."""
        lang = get_language(target_lang)
        if lang is None:
            return []
        names = [lang.english_name, lang.native_name]
        words = lang.english_name.split()
        if len(words) == 2:  # "Simplified Chinese" -> "Chinese (Simplified)"
            names.append(f"{words[1]} ({words[0]})")
        return [n for n in names if n]

    def _clean(self, text: str, source: str, target_lang: str) -> str:
        """Remove what DeepL appended that is not a translation.

        Measured on the live API (2026-10-11), Simplified Chinese only: about
        6 in 10 short sentences came back with a note after them -- the
        language's name ("Simplified Chinese (Mainland)", "1. 简体中文（大陆）")
        or a translator's remark ("1. 保留“解”的译法，符合中文数学术语习惯。").
        These were drawn on a teacher's worksheet.
        """
        # A numbered note after a finished sentence, when the source has no
        # numbered list of its own.
        # "3.Undo" counts as a numbered item too: no space is needed. And a
        # tail that carries a placeholder is the teacher's content, never a
        # note -- cutting it once lost item 3 and its answer blanks.
        if not re.search(r"(?:^|\s)\d+[.)]", source):
            note = re.search(r"(?<=[。！？.!?])\s*\d+[.)]\s.*$", text, flags=re.DOTALL)
            if note and "#" not in note.group(0):
                text = text[: note.start()]
        for name in self._language_names(target_lang):
            if name.lower() in source.lower():
                continue
            text = re.sub(
                rf"\s*(?:\d+[.)]\s*)?{re.escape(name)}(?:\s*[(（][^)）]*[)）])?\s*$",
                "",
                text,
                flags=re.IGNORECASE,
            )
        return text

    def _translate_batch(self, segments, *, source_lang, target_lang):
        # Edge whitespace is sent to nobody: with preserve_formatting, DeepL
        # treats trailing spaces as room to say something, and fills it.
        edges = []
        for seg in segments:
            core = seg.text.strip()
            start = seg.text.find(core) if core else len(seg.text)
            edges.append((seg.text[:start], seg.text[start + len(core):], core))
        payload = {
            "text": [_to_tags(core) for _, _, core in edges],
            "target_lang": self._code(target_lang),
            # Placeholders go as tags DeepL is built to carry through. Sent
            # as "#0#", "Solve #0#" came back "Resuelve el n.º #0#": the #
            # read as a number sign.
            "tag_handling": "xml",
            # Keep formatting literal: DeepL otherwise "fixes" spacing around
            # our placeholders.
            "preserve_formatting": True,
        }
        if source_lang:
            payload["source_lang"] = source_lang.split("-")[0].upper()
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                self._endpoint,
                headers={"Authorization": f"DeepL-Auth-Key {self._key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        out = data.get("translations", [])
        if len(out) != len(segments):
            raise TranslationError(
                "The translation service returned an unexpected response.",
                diagnostic=f"expected {len(segments)}, got {len(out)}",
            )
        return [
            TranslationResult(
                segment_id=seg.segment_id,
                text=lead
                + self._clean(_from_tags(t.get("text", ""), core), core, target_lang).strip()
                + trail,
                provider=self.name,
            )
            for seg, t, (lead, trail, core) in zip(segments, out, edges)
        ]


# The instruction that protects assessment integrity. Read spec sec. 23/24
# before editing: every clause here exists to stop the model turning a
# translation into a lesson.
_LLM_SYSTEM = """You translate K-12 classroom materials from {source} into {target}.

You are translating an assessment or instructional document. Follow these rules exactly.

TRANSLATE:
- Render the meaning naturally in {target}, at the same reading level as the source.
- Keep the same tone and register a teacher would use.
- Keep the instructional intent identical.

NEVER:
- Never answer, solve, or hint at the answer to any question.
- Never add explanations, definitions, examples, or teaching content.
- Never simplify or clarify something the source left difficult.
- Never change numbers, values, units, or the order of answer choices.
- Never add or remove sentences.

PLACEHOLDERS:
- Tokens of the form #0#, #1#, #2# are protected content (equations, formulas,
  URLs). Reproduce each one exactly as written, including both # characters.
- Keep them in the position the meaning requires. Do not translate them.

OUTPUT:
- Return a JSON array of objects: [{{"id": "<id>", "text": "<translation>"}}]
- One object per input item, in the same order. No commentary, no markdown."""


class _LLMProvider(_CachingProvider):
    """Shared logic for chat-completion style translation back ends."""

    batch_size = _LLM_BATCH
    sends_data_externally = True

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        return get_language(target_lang) is not None

    def _build_prompt(self, segments, source_lang, target_lang) -> tuple[str, str]:
        src = get_language(source_lang)
        source_name = src.english_name if src else (source_lang or "the source language")
        tgt = get_language(target_lang)
        target_name = tgt.english_name if tgt else target_lang
        system = _LLM_SYSTEM.format(source=source_name, target=target_name)
        items = [
            {"id": s.segment_id, "role": s.role, "text": s.text} for s in segments
        ]
        return system, json.dumps(items, ensure_ascii=False)

    @staticmethod
    def _parse(raw: str, segments) -> list[TranslationResult]:
        text = raw.strip()
        # Models sometimes wrap JSON in a fenced block despite instructions.
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TranslationError(
                "The translation service returned a response we could not read.",
                diagnostic=f"{exc}: {text[:200]}",
            ) from exc
        by_id = {
            str(item.get("id")): str(item.get("text", ""))
            for item in parsed
            if isinstance(item, dict)
        }
        return [
            TranslationResult(
                segment_id=s.segment_id,
                text=by_id.get(s.segment_id, s.text),
                provider="llm",
                error=None if s.segment_id in by_id else "segment missing from response",
            )
            for s in segments
        ]


@register_translation("openai")
class OpenAITranslationProvider(_LLMProvider):
    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-4o-mini"

    def __init__(self) -> None:
        self._key = _require(
            get_settings().openai_api_key, "CT_OPENAI_API_KEY", "OpenAI"
        )

    def _translate_batch(self, segments, *, source_lang, target_lang):
        system, user = self._build_prompt(segments, source_lang, target_lang)
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self.MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            body = resp.json()
        content = body["choices"][0]["message"]["content"]
        results = self._parse(content, segments)
        for r in results:
            r.provider = self.name
        return results


@register_translation("anthropic")
class AnthropicTranslationProvider(_LLMProvider):
    name = "anthropic"
    ENDPOINT = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-5"

    def __init__(self) -> None:
        self._key = _require(
            get_settings().anthropic_api_key, "CT_ANTHROPIC_API_KEY", "Anthropic"
        )

    def _translate_batch(self, segments, *, source_lang, target_lang):
        system, user = self._build_prompt(segments, source_lang, target_lang)
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                self.ENDPOINT,
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "max_tokens": 8192,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
        content = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        results = self._parse(content, segments)
        for r in results:
            r.provider = self.name
        return results
