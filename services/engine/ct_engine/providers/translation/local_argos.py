"""Offline translation via Argos Translate (CTranslate2 + OpenNMT models).

This is the default provider: free, and nothing leaves the machine, which is
the right posture for student work (spec sec. 10 / 43).

Argos is an OPTIONAL dependency -- it pulls in torch and spaCy, which roughly
triples the install size. Install it with:

    pip install -r requirements-local-translation.txt

If it is absent, this provider reports that clearly instead of pretending to
translate.

Language packs download once, on first use, from the Argos package index.
A pack that does not exist is reported as unsupported -- notably
Traditional Chinese, which Argos does not publish. Silently substituting
Simplified would hand a Hong Kong family the wrong script.
"""

from __future__ import annotations

import threading

from ...util.logging import get_logger
from ..registry import register_translation
from .base import (
    TranslationError,
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from .cache import GLOBAL_CACHE

log = get_logger(__name__)

# Our BCP-47 target codes mapped to the ISO-639-1 codes Argos publishes.
# Entries mapping to None are explicitly NOT available offline.
_ARGOS_CODES: dict[str, str | None] = {
    "es": "es", "fr": "fr", "ru": "ru", "ko": "ko", "ja": "ja",
    "zh-Hans": "zh", "zh-Hant": None,  # Argos publishes Simplified only
    "ar": "ar", "pt-BR": "pt", "pt-PT": "pt", "he": "he", "uk": "uk",
    "pl": "pl", "vi": "vi", "bn": "bn", "ur": "ur",
    "ht": None,  # No Haitian Creole model published
}


class ArgosUnavailable(TranslationError):
    pass


@register_translation("local")
class LocalArgosTranslationProvider(TranslationProvider):
    name = "local"
    sends_data_externally = False

    # Model loading is expensive and not thread-safe to do concurrently.
    _install_lock = threading.Lock()

    def __init__(self) -> None:
        self._checked_pairs: set[tuple[str, str]] = set()

    # -- availability ------------------------------------------------------
    @staticmethod
    def _import():
        try:
            import argostranslate.package as package
            import argostranslate.translate as translate
        except ImportError as exc:
            raise ArgosUnavailable(
                "Offline translation is not installed on this server.",
                diagnostic=(
                    "argostranslate is missing; install "
                    "requirements-local-translation.txt or set "
                    "CT_TRANSLATION_PROVIDER to a configured remote provider"
                ),
            ) from exc
        return package, translate

    @staticmethod
    def argos_code(target: str) -> str | None:
        if target in _ARGOS_CODES:
            return _ARGOS_CODES[target]
        base = target.split("-")[0]
        return _ARGOS_CODES.get(base, base if len(base) == 2 else None)

    def supports(self, source_lang: str | None, target_lang: str) -> bool:
        code = self.argos_code(target_lang)
        if code is None:
            return False
        try:
            self._import()
        except ArgosUnavailable:
            return False
        return True

    def _ensure_pair(self, source: str, target: str) -> None:
        """Install the language pack for this pair if it is not present."""
        if (source, target) in self._checked_pairs:
            return
        package, translate = self._import()

        with self._install_lock:
            installed = {
                (p.from_code, p.to_code) for p in package.get_installed_packages()
            }
            if (source, target) not in installed:
                log.info("Installing offline language pack %s->%s", source, target)
                try:
                    package.update_package_index()
                    available = package.get_available_packages()
                except Exception as exc:
                    raise ArgosUnavailable(
                        "The offline language pack could not be downloaded. "
                        "Check this server's network access, or choose a "
                        "different language.",
                        diagnostic=repr(exc),
                    ) from exc

                match = [
                    p
                    for p in available
                    if p.from_code == source and p.to_code == target
                ]
                if not match:
                    raise ArgosUnavailable(
                        "That language is not available for offline translation "
                        "on this server.",
                        diagnostic=f"no argos package {source}->{target}",
                    )
                try:
                    package.install_from_path(match[0].download())
                except Exception as exc:
                    raise ArgosUnavailable(
                        "The offline language pack could not be installed.",
                        diagnostic=repr(exc),
                    ) from exc
            self._checked_pairs.add((source, target))

    # -- translation -------------------------------------------------------
    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        if not segments:
            return []

        target = self.argos_code(target_lang)
        if target is None:
            raise ArgosUnavailable(
                f"Offline translation does not support that language.",
                diagnostic=f"no argos mapping for {target_lang}",
            )
        # Argos models are all pivoted through English; with no detected
        # source we assume English rather than guessing per segment.
        source = (source_lang or "en").split("-")[0]
        if source == target:
            return [
                TranslationResult(
                    segment_id=s.segment_id, text=s.text, provider=self.name
                )
                for s in segments
            ]

        self._ensure_pair(source, target)
        _, translate_mod = self._import()

        results: list[TranslationResult] = []
        for seg in segments:
            cached = GLOBAL_CACHE.get(seg.text, source, target, self.name)
            if cached is not None:
                results.append(
                    TranslationResult(
                        segment_id=seg.segment_id,
                        text=cached,
                        provider=self.name,
                        cached=True,
                    )
                )
                continue
            try:
                out = translate_mod.translate(seg.text, source, target)
            except Exception as exc:
                # One bad segment must never fail the document.
                log.warning("Segment %s failed: %r", seg.segment_id, exc)
                results.append(
                    TranslationResult(
                        segment_id=seg.segment_id,
                        text=seg.text,
                        provider=self.name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            out = (out or "").strip()
            if out:
                GLOBAL_CACHE.put(seg.text, source, target, self.name, out)
            results.append(
                TranslationResult(
                    segment_id=seg.segment_id,
                    text=out or seg.text,
                    provider=self.name,
                    # Argos exposes no calibrated score, and inventing one
                    # would make QA trust a number that means nothing.
                    confidence=None,
                    error=None if out else "empty translation",
                )
            )
        return results
