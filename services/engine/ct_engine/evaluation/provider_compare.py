"""Compare translation providers on what this product actually needs.

Not BLEU. The questions that decide whether a provider is usable for K-12
assessment material are:

  1. Does protected content survive? A provider that mangles an equation is
     disqualified regardless of how fluent its prose is (spec sec. 20/24).
  2. Does it translate at all? Bare labels like "Cell Wall" come back
     unchanged from weaker engines, which silently leaves a diagram in
     English.
  3. Does it add content? A translation that explains or answers a question
     has destroyed the assessment (spec sec. 23).
  4. What does it cost per document, and how long does it take?

Run:
    python -m ct_engine.evaluation.provider_compare
    python -m ct_engine.evaluation.provider_compare --providers local,deepl --target es

Providers without a credential are skipped with the reason printed, so the
harness is useful before any key exists and unchanged after one is added.

Two caveats on reading the numbers:

  * `untranslated` cannot tell "the engine failed" from "the word is the same
    in both languages". "mitochondrion" is identical in French, and shows as
    unchanged. Read the per-segment detail before concluding anything.
  * Latency is measured with the translation cache cleared, so it reflects a
    cold run. Caching is what makes the real pipeline fast on repeated
    strings, and including it here would flatter whichever provider ran
    second.
  * A low protection rate does NOT mean the pipeline would ship corrupted
    formulas. Those segments fail verification, keep their source text, and
    flag the page. It means reduced translation COVERAGE for that language,
    which is still a reason to prefer a better provider.

Measured with the offline engine at the time of writing: en->es and en->zh
score 100% protection, while en->ar scores 67% -- equations and fill-in
blanks are the segments it loses.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from ..model.enums import SemanticRole
from ..providers import ProviderNotAvailable, get_translation_provider
from ..providers.translation.base import TranslationSegment
from ..providers.translation.cache import GLOBAL_CACHE
from ..translate.protection import mask_text, restore_text
from ..translate.short_label import (
    frame,
    looks_untranslated,
    should_frame,
    unframe,
)

# Cost per million characters, in USD. Published list prices at the time of
# writing, for ORDER-OF-MAGNITUDE comparison only -- verify before quoting
# these to anyone. LLM providers are billed per token, so the figure is a
# conversion at roughly 4 characters per token, input plus output.
COST_PER_MILLION_CHARS: dict[str, float] = {
    "local": 0.0,
    "google": 20.0,
    "deepl": 25.0,
    "openai": 0.6,
    "anthropic": 4.5,
}

# A representative slice of real classroom material: each entry is something
# the corpus actually contains, chosen to exercise a distinct failure mode.
EVAL_SET: list[tuple[str, SemanticRole, str]] = [
    ("direction", SemanticRole.DIRECTION,
     "Directions: Read each question carefully. Write your answers in "
     "complete sentences on the lines provided."),
    ("question", SemanticRole.QUESTION,
     "Explain why chlorophyll is necessary for photosynthesis."),
    ("question_predict", SemanticRole.QUESTION,
     "A plant is placed in a dark closet for one week. Predict what will "
     "happen to its leaves and explain your reasoning."),
    ("math_inline", SemanticRole.QUESTION,
     "Solve 3x + 7 = 22 and show your work on the lines below."),
    ("math_parens", SemanticRole.QUESTION,
     "Use 5(y - 2) = 35 to find the value of y."),
    ("chemistry", SemanticRole.READING_PASSAGE,
     "Water is H₂O. Explain why 2H₂ + O₂ = 2H₂O is a balanced equation."),
    ("measurement", SemanticRole.QUESTION,
     "The beaker holds 250 mL of solution at 37 °C. Record your observation."),
    ("url", SemanticRole.DIRECTION,
     "Visit www.example.org/lab for the data sheet before you begin."),
    ("answer_choice", SemanticRole.ANSWER_CHOICE, "mitochondrion"),
    ("answer_choice_phrase", SemanticRole.ANSWER_CHOICE,
     "regulate what enters and leaves"),
    ("label_two_word", SemanticRole.IMAGE_LABEL, "Cell Wall"),
    ("label_one_word", SemanticRole.IMAGE_LABEL, "Nucleus"),
    ("label_technical", SemanticRole.IMAGE_LABEL, "Chloroplast"),
    ("label_axis", SemanticRole.IMAGE_LABEL, "Time (hours)"),
    ("response_label", SemanticRole.RESPONSE_LABEL, "Name: ______________"),
    ("passage", SemanticRole.READING_PASSAGE,
     "The monarch butterfly begins its life as a tiny egg attached to the "
     "underside of a milkweed leaf. Within a few days, a caterpillar emerges "
     "and begins to eat."),
]


@dataclass
class SegmentOutcome:
    key: str
    role: str
    source: str
    output: str
    protected_expected: int = 0
    protected_survived: int = 0
    untranslated: bool = False
    recovered_by_framing: bool = False
    error: str | None = None

    @property
    def protection_ok(self) -> bool:
        return self.protected_survived == self.protected_expected


@dataclass
class ProviderReport:
    provider: str
    available: bool
    skip_reason: str | None = None
    sends_data_externally: bool | None = None
    seconds: float = 0.0
    outcomes: list[SegmentOutcome] = field(default_factory=list)

    # --- headline metrics ---
    @property
    def protection_rate(self) -> float:
        graded = [o for o in self.outcomes if o.protected_expected]
        if not graded:
            return 1.0
        return sum(o.protection_ok for o in graded) / len(graded)

    @property
    def untranslated_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.untranslated for o in self.outcomes) / len(self.outcomes)

    @property
    def label_success_rate(self) -> float:
        labels = [o for o in self.outcomes if o.role == SemanticRole.IMAGE_LABEL.value]
        if not labels:
            return 1.0
        return sum(not o.untranslated for o in labels) / len(labels)

    @property
    def expansion_ratio(self) -> float:
        """Median output/source length. A large value can mean added content."""
        ratios = [
            len(o.output) / len(o.source)
            for o in self.outcomes
            if o.source and o.output and not o.error
        ]
        return round(statistics.median(ratios), 2) if ratios else 0.0

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.error)

    def cost_per_100_pages(self, chars_per_page: int = 1800) -> float:
        rate = COST_PER_MILLION_CHARS.get(self.provider)
        if rate is None:
            return float("nan")
        return rate * (chars_per_page * 100) / 1_000_000


def evaluate_provider(
    name: str, *, source_lang: str = "en", target_lang: str = "es"
) -> ProviderReport:
    try:
        provider = get_translation_provider(name)
    except (ProviderNotAvailable, Exception) as exc:  # noqa: BLE001
        return ProviderReport(
            provider=name,
            available=False,
            skip_reason=getattr(exc, "message", None) or str(exc),
        )

    if not provider.supports(source_lang, target_lang):
        return ProviderReport(
            provider=name,
            available=False,
            skip_reason=f"does not support {source_lang} -> {target_lang}",
            sends_data_externally=provider.sends_data_externally,
        )

    report = ProviderReport(
        provider=name,
        available=True,
        sends_data_externally=provider.sends_data_externally,
    )

    # Mask exactly as the pipeline does, so protection is measured on the
    # real input the provider would receive.
    masked = {key: mask_text(text) for key, _role, text in EVAL_SET}
    segments = [
        TranslationSegment(
            segment_id=key, text=masked[key].masked_text, role=role.value
        )
        for key, role, _text in EVAL_SET
    ]

    # Measure cold. The translation cache is process-global, so a second run
    # of the same provider would otherwise report 0.0s and look infinitely
    # faster than a provider being measured for the first time.
    GLOBAL_CACHE.clear()

    start = time.perf_counter()
    try:
        results = provider.translate(
            segments, source_lang=source_lang, target_lang=target_lang
        )
    except Exception as exc:  # noqa: BLE001
        report.available = False
        report.skip_reason = f"call failed: {exc}"
        return report
    by_id = {r.segment_id: r for r in results}

    # Second pass for bare labels, matching the pipeline's recovery step.
    retry_ids = [
        key
        for key, _role, _text in EVAL_SET
        if (r := by_id.get(key))
        and not r.error
        and should_frame(masked[key].masked_text)
        and looks_untranslated(masked[key].masked_text, r.text)
    ]
    recovered: dict[str, str] = {}
    if retry_ids:
        try:
            framed = provider.translate(
                [
                    TranslationSegment(
                        segment_id=key, text=frame(masked[key].masked_text)
                    )
                    for key in retry_ids
                ],
                source_lang=source_lang,
                target_lang=target_lang,
            )
            for r in framed:
                if r.error:
                    continue
                candidate = unframe(
                    r.text, target_lang, original=masked[r.segment_id].masked_text
                )
                if candidate and not looks_untranslated(
                    masked[r.segment_id].masked_text, candidate
                ):
                    recovered[r.segment_id] = candidate
        except Exception:  # noqa: BLE001
            pass
    report.seconds = round(time.perf_counter() - start, 3)

    for key, role, text in EVAL_SET:
        result = by_id.get(key)
        mask = masked[key]
        if result is None:
            report.outcomes.append(
                SegmentOutcome(key=key, role=role.value, source=text, output="",
                               error="no result")
            )
            continue

        output = recovered.get(key, result.text)
        restored = restore_text(output, mask.spans)
        survived = sum(1 for s in mask.spans if s.text in restored.text)
        report.outcomes.append(
            SegmentOutcome(
                key=key,
                role=role.value,
                source=text,
                output=restored.text,
                protected_expected=len(mask.spans),
                protected_survived=survived,
                untranslated=looks_untranslated(text, restored.text),
                recovered_by_framing=key in recovered,
                error=result.error,
            )
        )
    return report


def run(providers: list[str], target_lang: str, source_lang: str = "en") -> list[ProviderReport]:
    return [
        evaluate_provider(name, source_lang=source_lang, target_lang=target_lang)
        for name in providers
    ]


def print_report(reports: list[ProviderReport], target_lang: str) -> None:
    print(f"\nTranslation provider comparison  (en -> {target_lang})")
    print(f"{len(EVAL_SET)} segments from the test corpus\n")

    header = (
        f"{'provider':11s} {'external':>8s} {'protect':>8s} {'labels':>7s} "
        f"{'untrans':>8s} {'expand':>7s} {'errors':>7s} {'sec':>6s} {'$/100pg':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        if not r.available:
            print(f"{r.provider:11s} {'-':>8s}   skipped: {r.skip_reason[:60]}")
            continue
        cost = r.cost_per_100_pages()
        print(
            f"{r.provider:11s} "
            f"{('yes' if r.sends_data_externally else 'no'):>8s} "
            f"{r.protection_rate * 100:7.0f}% "
            f"{r.label_success_rate * 100:6.0f}% "
            f"{r.untranslated_rate * 100:7.0f}% "
            f"{r.expansion_ratio:7.2f} "
            f"{r.errors:7d} "
            f"{r.seconds:6.1f} "
            f"{cost:8.2f}"
        )

    print("\n  protect  protected values (equations, formulas, URLs) returned intact")
    print("  labels   bare diagram labels that actually translated")
    print("  untrans  segments returned unchanged")
    print("  expand   median output/source length; well above ~1.5 may mean added text")
    print("  $/100pg  list-price estimate at 1,800 chars/page -- verify before quoting")
    print(
        "\n  Note: 'untrans' counts words that are identical in both languages\n"
        "  as well as genuine failures, and a low 'protect' score means reduced\n"
        "  coverage, not shipped corruption -- those segments keep their source\n"
        "  text and flag the page."
    )

    for r in reports:
        if not r.available:
            continue
        problems = [
            o
            for o in r.outcomes
            if not o.protection_ok or o.untranslated or o.error
        ]
        if not problems:
            continue
        print(f"\n  {r.provider}: {len(problems)} segment(s) to look at")
        for o in problems[:8]:
            why = (
                "protected value lost"
                if not o.protection_ok
                else ("unchanged" if o.untranslated else f"error: {o.error}")
            )
            print(f"    [{o.key}] {why}")
            print(f"       {o.source[:64]!r}")
            print(f"    -> {o.output[:64]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="local,google,deepl,openai,anthropic",
        help="comma-separated provider names",
    )
    parser.add_argument("--target", default="es", help="target language code")
    parser.add_argument("--source", default="en", help="source language code")
    parser.add_argument("--json", dest="json_path", help="also write JSON here")
    args = parser.parse_args()

    # Importing the provider modules is what registers them.
    from ..providers.translation import local_argos as _l  # noqa: F401
    from ..providers.translation import remote as _r  # noqa: F401

    names = [n.strip() for n in args.providers.split(",") if n.strip()]
    reports = run(names, args.target, args.source)
    print_report(reports, args.target)

    if args.json_path:
        payload = [
            {
                **{k: v for k, v in asdict(r).items() if k != "outcomes"},
                "protection_rate": r.protection_rate,
                "label_success_rate": r.label_success_rate,
                "untranslated_rate": r.untranslated_rate,
                "expansion_ratio": r.expansion_ratio,
                "cost_per_100_pages": r.cost_per_100_pages(),
                "outcomes": [asdict(o) for o in r.outcomes],
            }
            for r in reports
        ]
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
