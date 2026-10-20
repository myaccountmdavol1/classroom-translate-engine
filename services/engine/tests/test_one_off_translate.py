"""Translating one string on demand, for the cover box's English field.

The batch pipeline translates a whole document. A teacher retyping a line
into a cover box needs one string, right now, with the same maths protection
the pipeline uses -- "Solve 3x + 5 = 20" must not come back rearranged.
"""

from __future__ import annotations

import pytest

from ct_engine.providers.translation.base import (
    TranslationError,
    TranslationProvider,
    TranslationResult,
)
from ct_engine.translate.one_off import OneOffError, translate_one


class StubProvider(TranslationProvider):
    """Returns a fixed string, recording what it was asked to translate."""

    name = "stub"

    def __init__(self, reply: str = "RESULTADO"):
        self.reply = reply
        self.seen: list[str] = []

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        self.seen = [s.text for s in segments]
        return [
            TranslationResult(
                segment_id=s.segment_id, text=self.reply, provider=self.name
            )
            for s in segments
        ]


class EchoingProvider(StubProvider):
    """Returns the masked text unchanged, so placeholders survive."""

    def translate(self, segments, *, source_lang, target_lang):
        self.seen = [s.text for s in segments]
        return [
            TranslationResult(
                segment_id=s.segment_id, text=s.text, provider=self.name
            )
            for s in segments
        ]


class FailingProvider(StubProvider):
    def translate(self, segments, *, source_lang, target_lang):
        raise TranslationError("The translation service is unavailable.")


def test_a_plain_string_comes_back_translated():
    result = translate_one(
        "Show your work below.", target_lang="es", provider=StubProvider()
    )
    assert result.text == "RESULTADO"
    assert result.provider == "stub"
    assert result.protected_intact is True


def test_maths_is_masked_before_the_provider_sees_it():
    """The provider must never be handed a raw equation to paraphrase."""
    provider = EchoingProvider()
    translate_one("Solve 3x + 5 = 20 for x.", target_lang="es", provider=provider)
    assert "3x + 5 = 20" not in provider.seen[0]


def test_maths_is_restored_after_translation():
    result = translate_one(
        "Solve 3x + 5 = 20 for x.", target_lang="es", provider=EchoingProvider()
    )
    assert "3x + 5 = 20" in result.text
    assert result.protected_intact is True


def test_a_lost_placeholder_is_reported_not_raised():
    """Withholding the result would be worse: she can see it and fix it."""
    result = translate_one(
        "Solve 3x + 5 = 20 for x.",
        target_lang="es",
        provider=StubProvider("RESULTADO SIN NADA"),
    )
    assert result.protected_intact is False
    assert result.text  # still returned


def test_empty_text_is_rejected():
    with pytest.raises(OneOffError):
        translate_one("   ", target_lang="es", provider=StubProvider())


def test_overlong_text_is_rejected():
    with pytest.raises(OneOffError):
        translate_one("x" * 9000, target_lang="es", provider=StubProvider())


def test_a_provider_failure_becomes_a_teacher_facing_error():
    with pytest.raises(OneOffError) as exc:
        translate_one("Hello", target_lang="es", provider=FailingProvider())
    assert "_" not in exc.value.message


def test_a_per_segment_error_becomes_a_teacher_facing_error():
    class ErroringProvider(StubProvider):
        def translate(self, segments, *, source_lang, target_lang):
            return [
                TranslationResult(
                    segment_id=s.segment_id, text="", error="quota exceeded"
                )
                for s in segments
            ]

    with pytest.raises(OneOffError):
        translate_one("Hello", target_lang="es", provider=ErroringProvider())
