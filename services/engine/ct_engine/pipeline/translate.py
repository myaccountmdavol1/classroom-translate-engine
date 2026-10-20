"""The TRANSLATE stage.

Turns a page's text objects into translation segments, runs them through the
configured provider, and maps results back onto the exact objects they came
from so a single edit can later re-render exactly one object (spec sec. 22).

PROTECTED CONTENT -- three tiers, in order:

  1. Mask protected spans as ``#N#`` placeholders, translate, restore.
     Measured ~92% clean on Argos across es/fr/zh/ar/ru.
  2. If a placeholder did not survive, retry the segment UNMASKED and check
     that every protected string appears verbatim in the output. NMT often
     reproduces equations untouched, so this recovers most of the remainder.
  3. If protected content still cannot be guaranteed, the segment FAILS: the
     source text is kept and the page is flagged. A visibly untranslated
     sentence is recoverable in review; a silently corrupted equation is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.math_runs import MATH_FONT_REASON
from ..model.document import Page, QAFinding
from ..model.enums import (
    OutputLanguageMode,
    SemanticRole,
    TextSource,
    TranslationStatus,
)
from ..providers.translation.base import (
    TranslationProvider,
    TranslationResult,
    TranslationSegment,
)
from ..translate.short_label import (
    frame as frame_label,
    looks_untranslated,
    should_frame,
    unframe,
)
from ..translate.protection import (
    MaskResult,
    has_translatable_content,
    is_math_dominant,
    mask_text,
    normalize_for_translation,
    peel_edges,
    repair_entities,
    restore_text,
    split_leading_label,
)
from ..util.logging import get_logger

log = get_logger(__name__)

# Roles whose text is never sent to a translator.
SKIP_ROLES = frozenset({SemanticRole.PAGE_NUMBER, SemanticRole.FORMULA})


# Answer-choice markers. Splitting on these is what keeps "(1) ribosome
# (2) mitochondrion ..." from being reordered or merged by the translator --
# an exam-integrity requirement, not a cosmetic one (spec sec. 24).
_CHOICE_SPLIT = re.compile(r"(\(\s*[A-Da-d1-9]\s*\)|(?<![\w)])[A-Da-d][.)](?=\s))")


@dataclass
class SegmentPart:
    """One independently translated piece of a text object.

    Most blocks have exactly one part. Answer-choice rows have one per
    choice, so each choice keeps its own marker and its position in the list.
    """

    part_id: str
    prefix: str           # marker kept verbatim, e.g. "(2) "
    separator: str        # whitespace that preceded this part in the source
    source_body: str
    mask: MaskResult
    # Protected content peeled off the start/end of the body, re-attached
    # verbatim after translation. See protection.peel_edges.
    edge_prefix: str = ""
    edge_suffix: str = ""


@dataclass
class SegmentPlan:
    """What we intend to do with one text object."""

    object_id: str
    label: str            # leading question number, kept verbatim
    parts: list[SegmentPart]
    role: SemanticRole


def split_answer_choices(text: str) -> list[tuple[str, str, str]]:
    """Split a choice row into (separator, marker, body) triples.

    Returns a single ("", "", text) triple when the text is not a multiple
    choice row, so callers can treat both shapes identically.
    """
    pieces = _CHOICE_SPLIT.split(text)
    # re.split with one capture group yields [pre, marker, body, marker, body...]
    if len(pieces) < 3:
        return [("", "", text)]

    out: list[tuple[str, str, str]] = []
    lead = pieces[0]
    if lead.strip():
        out.append(("", "", lead.strip()))
    for i in range(1, len(pieces) - 1, 2):
        marker = pieces[i]
        body = pieces[i + 1]
        stripped = body.lstrip()
        gap = body[: len(body) - len(stripped)]
        separator = " " if not out else " "
        out.append((separator, marker + (gap or " "), stripped.rstrip()))
    # Fewer than two real choices means this was not a choice row after all.
    if sum(1 for _, marker, _ in out if marker) < 2:
        return [("", "", text)]
    return out


@dataclass
class TranslateStats:
    translated: int = 0
    cached: int = 0
    skipped: int = 0
    protected_blocks: int = 0
    failed: int = 0
    recovered_unmasked: int = 0
    # Short labels that only translated once wrapped in a noun phrase.
    short_labels_recovered: int = 0
    repaired_placeholders: int = 0
    # Blocks the translation service never translated (it did not answer,
    # or its allowance is used up), as distinct from blocks deliberately
    # kept in English to protect a formula. The runner adds these up to
    # decide whether the document was translated at all.
    provider_failed: int = 0
    # Blocks that had text to translate on this page: the denominator for
    # provider_failed.
    eligible: int = 0
    findings: list[QAFinding] = field(default_factory=list)


def _record_regions(page: Page, object_id: str, masked: MaskResult) -> None:
    """Record protected spans so QA and the debug overlay can see them."""
    from ..model.objects import ProtectedRegion

    for span in masked.spans:
        page.protected_regions.append(
            ProtectedRegion(
                object_id=object_id,
                start=span.start,
                end=span.end,
                text=span.text,
                category=span.category,
                placeholder=span.token,
            )
        )


def plan_page(page: Page) -> list[SegmentPlan]:
    """Decide, per text object, what gets translated and what is protected."""
    plans: list[SegmentPlan] = []
    page.protected_regions = []

    for obj in page.text_objects:
        # A person-created block has no source text to translate, and a
        # suppressed one was deliberately removed. Neither is touched.
        if obj.is_manual_block or obj.suppressed:
            continue
        text = obj.flat_text.strip()
        if not text:
            obj.translation_status = TranslationStatus.NOT_REQUIRED
            continue

        # Glyphs in a math font were cut out at block splitting because their
        # text layer is not the equation. Never translate or redraw them.
        if obj.protected_reason == MATH_FONT_REASON:
            obj.translation_status = TranslationStatus.PROTECTED
            obj.role = SemanticRole.FORMULA
            continue

        if obj.role in SKIP_ROLES:
            obj.translation_status = TranslationStatus.PROTECTED
            obj.protected = True
            obj.protected_reason = f"role={obj.role.value}"
            # A formula block is entirely protected content; recording its
            # spans is what makes it visible to QA and the debug overlay.
            _record_regions(page, obj.id, mask_text(text))
            continue

        if not has_translatable_content(text):
            obj.translation_status = TranslationStatus.NOT_REQUIRED
            continue

        # Recognised text only: a misread can turn an equation into
        # something that looks translatable. On a real scanned homework
        # page, `-51 + 15x - 11x = 41` was read as `"91+ ox - 11x = 41`,
        # translated to `"91+ buey - 11x = 41`, and painted over the
        # original -- changing the numbers in a maths question. Native text
        # is exact by construction and is never second-guessed this way.
        if obj.source in (TextSource.OCR, TextSource.IMAGE_OCR) and is_math_dominant(
            text
        ):
            obj.translation_status = TranslationStatus.PROTECTED
            obj.protected = True
            obj.protected_reason = "recognised text is an expression"
            _record_regions(page, obj.id, mask_text(text))
            continue

        label, body = split_leading_label(text)

        # Answer-choice rows are split so each choice translates on its own.
        chunks = (
            split_answer_choices(body)
            if obj.role is SemanticRole.ANSWER_CHOICE
            else [("", "", body)]
        )

        parts: list[SegmentPart] = []
        for i, (separator, prefix, chunk) in enumerate(chunks):
            edge_prefix, core, edge_suffix = peel_edges(chunk)
            # Symbols the engine mangles are rewritten before masking, but
            # `source_body` keeps the original: it is what bilingual output
            # shows as the English half, and what the seam-whitespace logic
            # in `rejoin_part` measures.
            masked = mask_text(normalize_for_translation(core))
            _record_regions(page, obj.id, masked)
            if edge_prefix or edge_suffix:
                # Peeled edges are protected too, just handled positionally
                # rather than by placeholder.
                _record_regions(page, obj.id, mask_text(edge_prefix + edge_suffix))
            parts.append(
                SegmentPart(
                    part_id=f"seg_{obj.id}_{i}",
                    prefix=prefix,
                    separator=separator,
                    source_body=core,
                    mask=masked,
                    edge_prefix=edge_prefix,
                    edge_suffix=edge_suffix,
                )
            )

        translatable = [
            p
            for p in parts
            if not p.mask.is_fully_protected and has_translatable_content(p.source_body)
        ]
        if not translatable:
            obj.translation_status = TranslationStatus.PROTECTED
            obj.protected = True
            obj.protected_reason = "no translatable content outside protected spans"
            continue

        plans.append(
            SegmentPlan(
                object_id=obj.id, label=label, parts=parts, role=obj.role
            )
        )
    return plans


def rejoin_part(part: SegmentPart, body: str) -> str:
    """Reassemble one part, restoring the whitespace at its seams.

    `peel_edges` splits protected content off the ends of a body, and both
    the provider and `compose_output` strip whatever they return. Simply
    concatenating the pieces back therefore closes a gap the source had:
    "Solve for " + "x: 3x + 7 = 22" came back as "Resuelve parax: 3x + 7 = 22",
    which reads as a different variable name. The source's own spacing at the
    seam is authoritative, so it is put back.
    """
    lead = trail = ""
    if part.edge_prefix and part.source_body[:1].isspace():
        stripped = part.source_body.lstrip()
        lead = part.source_body[: len(part.source_body) - len(stripped)]
    if part.edge_suffix and part.source_body[-1:].isspace():
        trail = part.source_body[len(part.source_body.rstrip()) :]
    return (
        part.separator
        + part.prefix
        + part.edge_prefix
        + lead
        + body
        + trail
        + part.edge_suffix
    )


def compose_output(
    label: str, translated: str, source: str, mode: OutputLanguageMode
) -> str:
    """Assemble the final string for a block, honouring the output mode."""
    translated = translated.strip()
    source = source.strip()
    if mode is OutputLanguageMode.BILINGUAL_TRANSLATION_FIRST:
        body = f"{translated} ({source})" if translated != source else translated
    elif mode is OutputLanguageMode.BILINGUAL_ENGLISH_FIRST:
        body = f"{source} ({translated})" if translated != source else source
    else:
        body = translated
    # The label keeps its original spacing so numbering lines up exactly.
    return f"{label}{body}" if label else body


def translate_page(
    page: Page,
    provider: TranslationProvider,
    *,
    source_lang: str | None,
    target_lang: str,
    output_mode: OutputLanguageMode = OutputLanguageMode.TRANSLATED_ONLY,
) -> TranslateStats:
    """Translate one page in place. Never raises for segment-level problems."""
    stats = TranslateStats()
    plans = plan_page(page)
    by_id = {o.id: o for o in page.text_objects}
    stats.protected_blocks = sum(1 for o in page.text_objects if o.protected)
    stats.skipped = sum(
        1
        for o in page.text_objects
        if o.translation_status is TranslationStatus.NOT_REQUIRED
    )

    stats.eligible = len(plans)
    if not plans:
        page.translated = True
        return stats

    # Flatten every part of every plan into one provider call, so a page is
    # a single round trip rather than one per block.
    segments: list[TranslationSegment] = []
    part_index: dict[str, tuple[SegmentPlan, SegmentPart]] = {}
    for plan in plans:
        for part in plan.parts:
            if part.mask.is_fully_protected or not has_translatable_content(
                part.source_body
            ):
                continue
            part_index[part.part_id] = (plan, part)
            segments.append(
                TranslationSegment(
                    segment_id=part.part_id,
                    text=part.mask.masked_text,
                    object_ids=[plan.object_id],
                    role=plan.role.value,
                )
            )

    try:
        results = provider.translate(
            segments, source_lang=source_lang, target_lang=target_lang
        )
    except Exception as exc:
        log.warning("Page %s translation failed: %r", page.number, exc)
        for plan in plans:
            obj = by_id.get(plan.object_id)
            if obj:
                obj.translation_status = TranslationStatus.FAILED
        stats.failed = len(plans)
        stats.provider_failed = len(plans)
        stats.findings.append(
            QAFinding(
                severity="error",
                code="translation_failed",
                message=f"Page {page.number} could not be translated.",
                details={"reason": str(exc)[:200]},
            )
        )
        return stats

    results_by_id = {r.segment_id: r for r in results}

    # --- Short-label recovery pass ---------------------------------------
    # Bare labels ("Cell Wall") often come back untranslated because there is
    # no sentence around them. Retry those inside a minimal noun phrase, in
    # ONE extra batch for the whole page rather than a call per label.
    retry_segments: list[TranslationSegment] = []
    for part_id, (plan, part) in part_index.items():
        result = results_by_id.get(part_id)
        if result is None or result.error:
            continue
        source = part.mask.masked_text
        if should_frame(source) and looks_untranslated(source, result.text):
            retry_segments.append(
                TranslationSegment(
                    segment_id=part_id + "_framed",
                    text=frame_label(source),
                    object_ids=[plan.object_id],
                    role=plan.role.value,
                )
            )

    if retry_segments:
        try:
            framed = provider.translate(
                retry_segments, source_lang=source_lang, target_lang=target_lang
            )
        except Exception as exc:
            log.debug("Framed retry failed on page %s: %r", page.number, exc)
            framed = []
        for r in framed:
            if r.error:
                continue
            part_id = r.segment_id.removesuffix("_framed")
            entry = part_index.get(part_id)
            original_result = results_by_id.get(part_id)
            if entry is None or original_result is None:
                continue
            source = entry[1].mask.masked_text
            recovered = unframe(r.text, target_lang, original=source)
            # Keep it only if it is actually different from the source; a
            # frame that still round-trips unchanged has told us nothing.
            # The frame itself echoed back ("the cell wall") is also nothing:
            # unframe only strips TARGET-language articles, so the English
            # "the" would otherwise pass as a translation.
            if (recovered and not looks_untranslated(source, recovered)
                    and not looks_untranslated(frame_label(source), r.text)):
                original_result.text = recovered
                stats.short_labels_recovered += 1

    for plan in plans:
        obj = by_id.get(plan.object_id)
        if obj is None:
            continue

        rendered: list[str] = []
        # Built in parallel with `rendered`: the translation on its own, kept
        # so reconstruction can drop the bilingual pairing on a page that
        # cannot be made to fit.
        rendered_translation_only: list[str] = []
        failure: str | None = None
        # True when the translation service gave no translation at all, as
        # opposed to one that lost a protected formula.
        provider_error = False

        for part in plan.parts:
            if part.part_id not in part_index:
                # Fully protected part (a bare formula among choices): keep it.
                passthrough = (
                    part.separator
                    + part.prefix
                    + part.edge_prefix
                    + part.source_body
                    + part.edge_suffix
                )
                rendered.append(passthrough)
                rendered_translation_only.append(passthrough)
                continue

            result = results_by_id.get(part.part_id)
            if result is None or result.error:
                failure = (result.error if result else "no result") or "no result"
                provider_error = True
                break

            restored = restore_text(result.text, part.mask.spans)
            if restored.repaired:
                stats.repaired_placeholders += len(restored.repaired)

            body = repair_entities(restored.text, part.source_body)
            if not restored.ok:
                recovered = _retry_unmasked(
                    provider,
                    plan,
                    part,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                if recovered is None:
                    failure = f"protected content missing: {restored.missing}"
                    break
                body = recovered
                stats.recovered_unmasked += 1

            composed = compose_output("", body, part.source_body, output_mode)
            rendered.append(rejoin_part(part, composed))
            translated_only = compose_output(
                "", body, part.source_body, OutputLanguageMode.TRANSLATED_ONLY
            )
            rendered_translation_only.append(rejoin_part(part, translated_only))
            if result.cached:
                stats.cached += 1

        if failure is not None:
            if provider_error:
                _fail_provider(obj, stats, failure)
            else:
                _fail(obj, stats, failure)
            continue

        # A human edit outranks the machine: never overwrite it on a re-run.
        if obj.manually_edited and obj.translated_text:
            stats.translated += 1
            continue

        obj.segment_id = plan.parts[0].part_id
        composed_text = (
            (plan.label + "".join(rendered).lstrip()).strip()
            if plan.label
            else "".join(rendered).strip()
        )
        obj.translated_text = composed_text
        obj.machine_translated_text = composed_text
        translation_only = (
            (plan.label + "".join(rendered_translation_only).lstrip()).strip()
            if plan.label
            else "".join(rendered_translation_only).strip()
        )
        obj.translation_only_text = translation_only or composed_text
        obj.translation_status = TranslationStatus.TRANSLATED
        stats.translated += 1

    page.translated = True
    return stats


def _retry_unmasked(
    provider: TranslationProvider,
    plan: SegmentPlan,
    part: SegmentPart,
    *,
    source_lang: str | None,
    target_lang: str,
) -> str | None:
    """Translate the raw text and accept it only if protected spans survived.

    Machine translation frequently reproduces equations and URLs untouched,
    so this recovers segments the placeholder round-trip lost -- but only
    when every protected string is verifiably present byte-for-byte.
    """
    if not part.mask.spans:
        return None
    try:
        out = provider.translate(
            [
                TranslationSegment(
                    segment_id=part.part_id + "_raw",
                    text=part.source_body,
                    object_ids=[plan.object_id],
                    role=plan.role.value,
                )
            ],
            source_lang=source_lang,
            target_lang=target_lang,
        )
    except Exception:
        return None
    if not out or out[0].error:
        return None
    candidate = out[0].text
    for span in part.mask.spans:
        if span.text not in candidate:
            return None
    return candidate


def _fail(obj, stats: TranslateStats, reason: str | None) -> None:
    """Keep the source text and flag the page rather than risk corruption."""
    obj.translation_status = TranslationStatus.FAILED
    obj.translated_text = None
    stats.failed += 1
    stats.findings.append(
        QAFinding(
            severity="warning",
            code="segment_untranslated",
            message=(
                "Some text on this page was left in the original language "
                "because its formula or number could not be preserved safely."
            ),
            object_ids=[obj.id],
            bbox=obj.bbox,
            details={"reason": (reason or "unknown")[:200]},
        )
    )


def _fail_provider(obj, stats: TranslateStats, reason: str | None) -> None:
    """Keep the source text when the translation service gave nothing back.

    Kept apart from `_fail`: telling a teacher a formula was protected when
    the service simply never answered sends her looking for a formula, and
    hides that the text was never translated.
    """
    obj.translation_status = TranslationStatus.FAILED
    obj.translated_text = None
    stats.failed += 1
    stats.provider_failed += 1
    stats.findings.append(
        QAFinding(
            severity="warning",
            code="translation_unavailable",
            message=(
                "This text was not translated because the translation service "
                "did not answer, or its free allowance is used up."
            ),
            object_ids=[obj.id],
            bbox=obj.bbox,
            details={"reason": (reason or "unknown")[:200]},
        )
    )
