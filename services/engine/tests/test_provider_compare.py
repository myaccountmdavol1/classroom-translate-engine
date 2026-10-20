"""The provider comparison harness itself."""

from __future__ import annotations

import pytest

import ct_engine.providers.translation.local_argos  # noqa: F401  (registers)
import ct_engine.providers.translation.remote  # noqa: F401  (registers)
from ct_engine.evaluation.provider_compare import (
    COST_PER_MILLION_CHARS,
    EVAL_SET,
    evaluate_provider,
    run,
)
from ct_engine.model.enums import SemanticRole


def test_eval_set_covers_the_failure_modes_that_matter():
    roles = {role for _key, role, _text in EVAL_SET}
    for required in (
        SemanticRole.DIRECTION,
        SemanticRole.QUESTION,
        SemanticRole.ANSWER_CHOICE,
        SemanticRole.IMAGE_LABEL,
        SemanticRole.RESPONSE_LABEL,
        SemanticRole.READING_PASSAGE,
    ):
        assert required in roles, f"eval set has no {required.value}"

    joined = " ".join(text for _k, _r, text in EVAL_SET)
    # Must include the content classes that disqualify a provider.
    assert "3x + 7 = 22" in joined
    assert "H₂O" in joined
    assert "www." in joined
    assert "250 mL" in joined
    assert "______" in joined


def test_unconfigured_providers_are_skipped_with_a_reason():
    """The harness must be useful before any key exists."""
    for name in ("google", "deepl", "openai", "anthropic"):
        report = evaluate_provider(name, target_lang="es")
        assert not report.available
        assert report.skip_reason
        assert "configured" in report.skip_reason.lower()
        assert report.outcomes == []


def test_unknown_provider_is_skipped_not_crashed():
    report = evaluate_provider("nonexistent", target_lang="es")
    assert not report.available and report.skip_reason


def test_unsupported_language_is_skipped_with_that_reason():
    """Argos publishes no Traditional Chinese model."""
    report = evaluate_provider("local", target_lang="zh-Hant")
    assert not report.available
    assert "does not support" in report.skip_reason


def test_every_provider_has_a_cost_estimate():
    from ct_engine.providers import TRANSLATION_PROVIDERS

    for name in TRANSLATION_PROVIDERS:
        assert name in COST_PER_MILLION_CHARS, f"no cost figure for {name}"
    assert COST_PER_MILLION_CHARS["local"] == 0.0


def test_local_provider_declares_it_keeps_data_on_the_machine():
    report = evaluate_provider("local", target_lang="es")
    if not report.available:
        pytest.skip(report.skip_reason or "offline engine unavailable")
    assert report.sends_data_externally is False


@pytest.mark.slow
def test_harness_produces_bounded_metrics_for_the_offline_engine():
    report = evaluate_provider("local", target_lang="es")
    if not report.available:
        pytest.skip(report.skip_reason or "offline engine unavailable")

    assert len(report.outcomes) == len(EVAL_SET)
    assert 0.0 <= report.protection_rate <= 1.0
    assert 0.0 <= report.untranslated_rate <= 1.0
    assert 0.0 <= report.label_success_rate <= 1.0
    assert report.expansion_ratio > 0
    assert report.seconds > 0
    assert report.cost_per_100_pages() == 0.0
    # Spanish is the best-supported pair; protection must be perfect there or
    # something has regressed in the protection layer.
    assert report.protection_rate == 1.0


@pytest.mark.slow
def test_run_returns_one_report_per_requested_provider():
    reports = run(["local", "deepl"], "es")
    assert [r.provider for r in reports] == ["local", "deepl"]
