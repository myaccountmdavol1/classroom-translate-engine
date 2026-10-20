"""Translation provider interface (spec sec. 10).

Nothing in the pipeline may import a concrete translation service. Everything
goes through `TranslationProvider`, selected by `CT_TRANSLATION_PROVIDER`, so
a district can move from a local model to DeepL or a hosted LLM by changing
environment variables.

Two contracts matter and are non-negotiable for any implementation:

  1. **Placeholders survive.** Segments arrive containing tokens such as
     ``{{MATH_001}}``. A provider MUST return them intact and unreordered
     within the sentence's meaning. The protection layer verifies this after
     the fact and rejects results that dropped or mangled a token.
  2. **No added content.** Providers must not explain, define, answer, or
     expand. A translation that solves the question destroys the assessment
     (spec sec. 23/24).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranslationSegment:
    """One semantic unit of text to translate.

    A segment is a whole question, direction or paragraph -- never a
    fragment -- because fragments cannot be translated correctly
    (spec sec. 15/22).
    """

    segment_id: str
    text: str
    # Ids of the page objects this segment came from, so results can be
    # mapped back to exact coordinates.
    object_ids: list[str] = field(default_factory=list)
    # Semantic role, which lets a provider adjust register (a QUESTION and a
    # READING_PASSAGE want different tone).
    role: str = "OTHER"
    # Neighbouring text, supplied for disambiguation only. Never translated.
    context: str | None = None


@dataclass
class TranslationResult:
    segment_id: str
    text: str
    # Provider-reported quality in [0, 1] where available; None otherwise.
    # Never fabricated -- absence of a score is itself information.
    confidence: float | None = None
    provider: str = ""
    # True when the result was served from cache (spec sec. 50).
    cached: bool = False
    error: str | None = None


class TranslationError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic


class TranslationProvider(ABC):
    name: str = "abstract"
    # Whether the provider sends document text off this machine. The UI and
    # the privacy notice both read this, so it must be accurate.
    sends_data_externally: bool = True

    @abstractmethod
    def supports(self, source_lang: str | None, target_lang: str) -> bool: ...

    @abstractmethod
    def translate(
        self,
        segments: list[TranslationSegment],
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> list[TranslationResult]:
        """Translate a batch, returning one result per input segment.

        Implementations MUST return results for every segment, in the same
        order, using `TranslationResult.error` for individual failures rather
        than raising -- one bad segment must not fail a 100-page document.
        """

    def close(self) -> None:
        """Release any held resources. Safe to call more than once."""
