"""Several translation engines, one after another, to stay inside free plans.

`CT_TRANSLATION_PROVIDER=deepl,google` translates with DeepL and, when DeepL
says its monthly allowance is used up (or a request to it fails), sends the
rest to Google. Each free plan covers roughly 500,000 characters a month, so
two of them double what a school can translate for nothing.

An engine that ran out is set aside until its allowance comes back: the
start of next month for a monthly plan, the next day for a daily cap. That
memory is in this process only: after a restart the first request tries it
again, is told the same, and sets it aside again -- one wasted request.
"""

from __future__ import annotations

import datetime as _dt
import threading

from ...util.logging import get_logger
from .base import TranslationProvider, TranslationResult, TranslationSegment

log = get_logger(__name__)

_lock = threading.Lock()
# Engine name -> the moment (UTC) its allowance is expected back.
_exhausted: dict[str, _dt.datetime] = {}
# Engine name -> the moment (UTC) it is worth asking again after it stopped
# answering. Kept apart from `_exhausted` so the teacher is never told an
# allowance ran out when a service was only slow.
_resting: dict[str, _dt.datetime] = {}

# What a teacher is told when nothing could translate the text. Which one
# depends on why: a used-up allowance waits for the reset, a service that did
# not answer is worth trying again shortly.
ALLOWANCE_USED_UP = (
    "The free translation allowance for this month is used up. "
    "It resets on the 1st."
)
# A daily cap (Google's) comes back the next morning; telling her to wait for
# the 1st would keep her from trying again tomorrow.
ALLOWANCE_USED_UP_TODAY = (
    "The free translation allowance for today is used up. "
    "It comes back tomorrow."
)
SERVICE_UNREACHABLE = (
    "The translation service did not answer. Please try again in a few minutes."
)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def next_month_start(now: _dt.datetime | None = None) -> _dt.datetime:
    now = now or _now()
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return _dt.datetime(year, month, 1, tzinfo=_dt.timezone.utc)


def next_day_start(now: _dt.datetime | None = None) -> _dt.datetime:
    """When a daily quota comes back.

    Google resets daily quotas at midnight Pacific time; 08:00 UTC is that
    moment in winter and an hour after it in summer, so it is never early.
    """
    now = now or _now()
    reset = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return reset if reset > now else reset + _dt.timedelta(days=1)


def mark_exhausted(name: str, until: _dt.datetime | None = None) -> None:
    until = until or next_month_start()
    with _lock:
        if _exhausted.get(name) != until:
            log.warning(
                "%s translation allowance is used up until %s",
                name, until.isoformat(timespec="minutes"),
            )
        _exhausted[name] = until


def is_exhausted(name: str) -> bool:
    with _lock:
        until = _exhausted.get(name)
        if until is None:
            return False
        if _now() >= until:
            del _exhausted[name]
            return False
        return True


def exhausted_until(
    provider: TranslationProvider, source_lang: str | None, target_lang: str
) -> _dt.datetime | None:
    """When translation into `target_lang` comes back, if every engine
    behind `provider` that offers it has run out of allowance; else None.

    Engines that do not offer the language are left out: the chain never
    asks them, so they cannot keep it going. The earliest reset counts,
    because that is when the first engine can take the text again.
    """
    members = getattr(provider, "providers", None) or [provider]
    usable = [p for p in members if p.supports(source_lang, target_lang)]
    if not usable:
        return None
    with _lock:
        now = _now()
        untils = [_exhausted.get(p.name) for p in usable]
    if any(u is None or now >= u for u in untils):
        return None
    return min(untils)  # type: ignore[type-var]


def untranslated_reason(
    provider: TranslationProvider, source_lang: str | None, target_lang: str
) -> str:
    """What to tell a teacher when nothing translated her text.

    Only say "used up" when it is true: an engine that timed out or answered
    5xx still has its allowance, and she should try again rather than wait.
    """
    until = exhausted_until(provider, source_lang, target_lang)
    if until is None:
        return SERVICE_UNREACHABLE
    if until - _now() <= _dt.timedelta(days=1, hours=1):
        return ALLOWANCE_USED_UP_TODAY
    return ALLOWANCE_USED_UP


def rest(name: str, seconds: float) -> None:
    """Set an engine aside briefly after it stopped answering.

    Not a used-up allowance: it is still worth trying again in a few minutes,
    and the teacher is told so. Without this every later page waits out the
    same timeout before moving on to the next engine.
    """
    until = _now() + _dt.timedelta(seconds=seconds)
    with _lock:
        log.warning("%s is not answering; set aside for %.0fs", name, seconds)
        _resting[name] = until


def is_resting(name: str) -> bool:
    with _lock:
        until = _resting.get(name)
        if until is None:
            return False
        if _now() >= until:
            del _resting[name]
            return False
        return True


def reset_exhausted() -> None:
    """For tests."""
    with _lock:
        _exhausted.clear()
        _resting.clear()


class FallbackTranslationProvider(TranslationProvider):
    """Each segment is translated by the first engine that manages it."""

    def __init__(self, providers: list[TranslationProvider]) -> None:
        if not providers:
            raise ValueError("At least one translation engine is needed.")
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self.sends_data_externally = any(p.sends_data_externally for p in providers)

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        return any(p.supports(source_lang, target_lang) for p in self.providers)

    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        results: dict[str, TranslationResult] = {}
        pending = list(segments)
        for provider in self.providers:
            if not pending:
                break
            if (
                is_exhausted(provider.name)
                or is_resting(provider.name)
                or not provider.supports(source_lang, target_lang)
            ):
                continue
            failed = []
            for r, seg in zip(
                provider.translate(pending, source_lang=source_lang, target_lang=target_lang),
                pending,
            ):
                results[seg.segment_id] = r
                if r.error:
                    failed.append(seg)
            pending = failed
        if pending:
            reason = untranslated_reason(self, source_lang, target_lang)
            for seg in pending:
                prior = results.get(seg.segment_id)
                results[seg.segment_id] = TranslationResult(
                    segment_id=seg.segment_id, text=seg.text, provider=self.name,
                    error=reason if prior is None else f"{reason} ({prior.error})",
                )
        return [results[s.segment_id] for s in segments]

    def close(self) -> None:
        for p in self.providers:
            p.close()
