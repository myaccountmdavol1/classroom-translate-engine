"""Translating one string on demand.

The pipeline translates documents. This translates a sentence, for the cover
box: a teacher types the replacement wording in English and needs it in the
target language immediately, with the same protection the pipeline applies so
an equation cannot be paraphrased into nonsense.

Deliberately stateless. It reads no manifest and writes nothing, so it is safe
to call while an edit is in flight.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.registry import ProviderNotAvailable, get_translation_provider
from ..providers.translation.base import (
    TranslationError,
    TranslationProvider,
    TranslationSegment,
)
from .protection import mask_text, restore_text

# The same ceiling the block editor enforces, so the two cannot disagree
# about what "too long" means.
MAX_TEXT_LENGTH = 8000


class OneOffError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic


@dataclass
class OneOffTranslation:
    text: str
    provider: str
    # False when a protected span (maths, a URL, a number) did not survive.
    # The text is still returned; the interface warns instead of hiding it.
    protected_intact: bool
    cached: bool = False


def translate_one(
    text: str,
    *,
    target_lang: str,
    source_lang: str | None = None,
    provider: TranslationProvider | None = None,
) -> OneOffTranslation:
    """Translate one string, protecting maths and leaving no state behind."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise OneOffError("There is nothing to translate.")
    if len(cleaned) > MAX_TEXT_LENGTH:
        raise OneOffError("That text is too long to translate in one go.")

    if provider is None:
        try:
            provider = get_translation_provider()
        except ProviderNotAvailable as exc:
            raise OneOffError(
                "No translation service is available right now.", diagnostic=str(exc)
            ) from exc

    masked = mask_text(cleaned)
    segment = TranslationSegment(segment_id="one-off", text=masked.masked_text)
    try:
        results = provider.translate(
            [segment], source_lang=source_lang, target_lang=target_lang
        )
    except TranslationError as exc:
        raise OneOffError(
            "That could not be translated just now. Please try again.",
            diagnostic=exc.diagnostic or str(exc),
        ) from exc

    if not results:
        raise OneOffError("The translation service returned nothing.")
    result = results[0]
    if result.error:
        raise OneOffError(
            "That could not be translated just now. Please try again.",
            diagnostic=result.error,
        )

    restored = restore_text(result.text, masked.spans)
    return OneOffTranslation(
        text=restored.text,
        provider=result.provider or getattr(provider, "name", "unknown"),
        protected_intact=restored.ok,
        cached=result.cached,
    )


def translate_many(
    texts: list[str],
    *,
    target_lang: str,
    source_lang: str | None = None,
) -> list[OneOffTranslation | None]:
    """Translate a list of short texts in one request, in order.

    For text that is not a document -- a Google Form's questions and
    choices. Each item is protected like any block; an item whose numbers or
    formulas did not survive, or that failed, comes back as None so the
    caller keeps the original rather than a damaged translation. Blank
    items come back blank.
    """
    try:
        provider = get_translation_provider()
    except ProviderNotAvailable as exc:
        raise OneOffError(
            "No translation service is available right now.", diagnostic=str(exc)
        ) from exc

    out: list[OneOffTranslation | None] = [None] * len(texts)
    masked, segments = {}, []
    for i, text in enumerate(texts):
        cleaned = (text or "").strip()
        if not cleaned:
            out[i] = OneOffTranslation(text=text or "", provider="none", protected_intact=True)
            continue
        if len(cleaned) > MAX_TEXT_LENGTH:
            continue
        masked[i] = mask_text(cleaned)
        segments.append(TranslationSegment(segment_id=str(i), text=masked[i].masked_text))
    if not segments:
        return out
    try:
        results = provider.translate(segments, source_lang=source_lang, target_lang=target_lang)
    except TranslationError as exc:
        raise OneOffError(
            "That could not be translated just now. Please try again.",
            diagnostic=exc.diagnostic or str(exc),
        ) from exc
    for result in results:
        i = int(result.segment_id)
        if result.error:
            continue
        restored = restore_text(result.text, masked[i].spans)
        if not restored.ok:
            continue
        out[i] = OneOffTranslation(
            text=restored.text,
            provider=result.provider or getattr(provider, "name", "unknown"),
            protected_intact=True,
            cached=result.cached,
        )
    return out
