"""Provider selection from environment configuration.

Deliberately fails loudly for a provider that is named but not implemented or
not credentialed. Silently falling back to a different engine would mean a
district thinks it is using DeepL while actually getting something else.
"""

from __future__ import annotations

from ..config import get_settings
from ..util.logging import get_logger
from .ocr.base import OCRProvider
from .translation.base import TranslationProvider

log = get_logger(__name__)


class ProviderNotAvailable(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


# Registries are populated by implementation modules at import time. Phase 2
# adds the translation engines; phase 3 adds OCR.
TRANSLATION_PROVIDERS: dict[str, type[TranslationProvider]] = {}
OCR_PROVIDERS: dict[str, type[OCRProvider]] = {}


def register_translation(name: str):
    def deco(cls: type[TranslationProvider]) -> type[TranslationProvider]:
        TRANSLATION_PROVIDERS[name] = cls
        return cls

    return deco


def register_ocr(name: str):
    def deco(cls: type[OCRProvider]) -> type[OCRProvider]:
        OCR_PROVIDERS[name] = cls
        return cls

    return deco


def get_translation_provider(name: str | None = None) -> TranslationProvider:
    key = (name or get_settings().translation_provider).lower()
    if "," in key:
        # "deepl,google": each in turn, so a used-up free plan is not the end
        # of the month's translating. An engine with no key is left out (and
        # said so in the log); with none at all, that is the loud failure.
        from .translation.chain import FallbackTranslationProvider

        members = []
        for part in (p.strip() for p in key.split(",") if p.strip()):
            try:
                members.append(get_translation_provider(part))
            except Exception as exc:
                log.warning("Translation engine %r left out: %s", part, exc)
        if not members:
            raise ProviderNotAvailable(
                f"None of the translation engines in {key!r} could be started."
            )
        return members[0] if len(members) == 1 else FallbackTranslationProvider(members)
    cls = TRANSLATION_PROVIDERS.get(key)
    if cls is None:
        raise ProviderNotAvailable(
            f"Translation provider {key!r} is not available. "
            f"Configured providers: {sorted(TRANSLATION_PROVIDERS) or 'none yet'}. "
            "Set CT_TRANSLATION_PROVIDER to one of these."
        )
    return cls()


def get_ocr_provider(name: str | None = None) -> OCRProvider:
    key = (name or get_settings().ocr_provider).lower()
    cls = OCR_PROVIDERS.get(key)
    if cls is None:
        raise ProviderNotAvailable(
            f"OCR provider {key!r} is not available. "
            f"Configured providers: {sorted(OCR_PROVIDERS) or 'none yet'}. "
            "Set CT_OCR_PROVIDER to one of these."
        )
    return cls()
