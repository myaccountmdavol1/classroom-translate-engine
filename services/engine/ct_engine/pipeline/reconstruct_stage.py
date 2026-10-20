"""RECONSTRUCT / VERIFY / EXPORT stages.

Rebuilds the translated PDF, renders translated page previews for the review
viewer, runs the per-page checks, and enforces the completion gate: a
document is never marked complete unless its output has exactly as many pages
as its input (spec sec. 46).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image

from ..config import get_settings
from ..model.document import DocumentManifest, Page, PageQA, QAFinding
from ..model.enums import PageStatus, TranslationStatus
from ..model.formats import SourceFormat, format_of
from ..model.geometry import Rect
from ..reconstruct.builder import rebuild_document, rebuild_page, subset_fonts_file
from ..reconstruct.docx_builder import rebuild_docx
from ..reconstruct.pptx_builder import rebuild_pptx
from ..reconstruct.xlsx_builder import rebuild_xlsx
from ..render.previews import make_thumbnail
from ..storage import keys as K
from ..storage.base import ObjectNotFoundError, ObjectStore
from ..util.logging import get_logger

log = get_logger(__name__)


@dataclass
class ExportResult:
    export_key: str
    byte_size: int
    source_page_count: int
    output_page_count: int
    findings_by_page: dict[int, list[QAFinding]] = field(default_factory=dict)
    # For reflowable sources, where "page count" is a section count that is
    # preserved trivially. Paragraph counts are the completeness evidence
    # that actually means something there. Both None for PDFs.
    source_unit_count: int | None = None
    output_unit_count: int | None = None
    unit_noun: str = "page"

    @property
    def page_counts_match(self) -> bool:
        return self.source_page_count == self.output_page_count


def reconstruct_and_export(
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    target_lang: str,
    on_progress=None,
    check_cancel=None,
) -> ExportResult:
    """Rebuild the document in its own format and persist the output.

    `check_cancel`, when given, is called between pages of a PDF rebuild and
    may raise to stop it (a cancelled job). Office write-backs are a single
    package step; `on_progress` is their only checkpoint.
    """
    fmt = format_of(manifest)
    if fmt is SourceFormat.DOCX:
        return _reconstruct_and_export_docx(
            manifest, store=store, on_progress=on_progress
        )
    if fmt is SourceFormat.PPTX:
        return _reconstruct_and_export_pptx(
            manifest, store=store, on_progress=on_progress
        )
    if fmt is SourceFormat.XLSX:
        return _reconstruct_and_export_xlsx(
            manifest, store=store, on_progress=on_progress
        )
    return _reconstruct_and_export_pdf(
        manifest,
        store=store,
        target_lang=target_lang,
        on_progress=on_progress,
        check_cancel=check_cancel,
    )


def _reconstruct_and_export_docx(
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    on_progress=None,
) -> ExportResult:
    """Write translations back into the uploaded Word file.

    No preview is rendered and no page image exists, because Word lays the
    document out when it opens it. The completeness evidence is therefore not
    "the output has as many pages as the input" -- that is trivially true of
    a section count -- but "the output has as many paragraphs as the input",
    which is exact and is checked here by reopening what was written.
    """
    from ..ingest.docx_parser import DocxParser

    source = store.get_bytes(manifest.source_key)
    result = rebuild_docx(source, manifest.pages)

    export_key = K.export_key(manifest.document_id, "translated.docx")
    store.put_bytes(export_key, result.data, content_type=manifest.content_type)

    source_paragraphs = output_paragraphs = 0
    opened = False
    try:
        with DocxParser().open_bytes(source) as doc:
            source_paragraphs = doc.paragraph_count
        # Reopen the bytes actually written, not the object in memory: this
        # is the only check that proves the file a teacher downloads opens
        # (spec sec. 53).
        with DocxParser().open_bytes(result.data) as doc:
            output_paragraphs = doc.paragraph_count
        opened = True
    except Exception as exc:
        log.exception("Rebuilt Word document could not be reopened")
        for page in manifest.pages:
            page.error = "The translated Word document could not be reopened."
            page.error_stage = "RECONSTRUCTING"
            page.diagnostics.extra["docx_reopen_error"] = f"{type(exc).__name__}: {exc}"

    total = len(manifest.pages)
    for i, page in enumerate(manifest.pages):
        page.reconstructed = opened
        page.revision += 1
        if on_progress:
            on_progress(i + 1, total)

    findings = {idx: s.findings for idx, s in result.stats.items() if s.findings}
    if result.unresolved:
        # Already recorded per page by the builder; repeated at document level
        # so the gate below has something unambiguous to fail on.
        log.error("%d DOCX anchors did not resolve", len(result.unresolved))

    return ExportResult(
        export_key=export_key,
        byte_size=len(result.data),
        source_page_count=len(manifest.pages),
        output_page_count=len(manifest.pages) if opened else 0,
        findings_by_page=findings,
        source_unit_count=source_paragraphs,
        output_unit_count=output_paragraphs,
        unit_noun="paragraph",
    )


def _reconstruct_and_export_pptx(
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    on_progress=None,
) -> ExportResult:
    """Write translations back into the uploaded PowerPoint file.

    Unlike a Word section, a slide IS a page: one slide in, one slide out,
    so the ordinary page-count gate is exact here and is what the completion
    check uses. The paragraph count is carried as well, because a slide that
    kept its slot while losing a bullet would otherwise pass.
    """
    from ..ingest.pptx_parser import PptxParser

    source = store.get_bytes(manifest.source_key)
    result = rebuild_pptx(source, manifest.pages)

    export_key = K.export_key(manifest.document_id, "translated.pptx")
    store.put_bytes(export_key, result.data, content_type=manifest.content_type)

    source_paragraphs = output_paragraphs = 0
    output_slides = 0
    opened = False
    try:
        with PptxParser().open_bytes(source) as doc:
            source_paragraphs = doc.paragraph_count
        with PptxParser().open_bytes(result.data) as doc:
            output_paragraphs = doc.paragraph_count
            output_slides = doc.page_count
        opened = True
    except Exception as exc:
        log.exception("Rebuilt presentation could not be reopened")
        for page in manifest.pages:
            page.error = "The translated presentation could not be reopened."
            page.error_stage = "RECONSTRUCTING"
            page.diagnostics.extra["pptx_reopen_error"] = f"{type(exc).__name__}: {exc}"

    total = len(manifest.pages)
    for i, page in enumerate(manifest.pages):
        page.reconstructed = opened
        page.revision += 1
        if on_progress:
            on_progress(i + 1, total)

    return ExportResult(
        export_key=export_key,
        byte_size=len(result.data),
        source_page_count=len(manifest.pages),
        output_page_count=output_slides,
        findings_by_page={
            idx: s.findings for idx, s in result.stats.items() if s.findings
        },
        source_unit_count=source_paragraphs,
        output_unit_count=output_paragraphs,
        unit_noun="paragraph",
    )


def _reconstruct_and_export_xlsx(
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    on_progress=None,
) -> ExportResult:
    """Write translations back into the uploaded Excel workbook.

    One sheet in, one sheet out, so the page-count gate is exact. The cell
    count is carried as well -- every cell holding a string, whatever it
    says -- because a sheet that kept its tab while losing a cell would
    otherwise pass.
    """
    from ..ingest.xlsx_parser import XlsxParser

    source = store.get_bytes(manifest.source_key)
    result = rebuild_xlsx(source, manifest.pages)

    export_key = K.export_key(manifest.document_id, "translated.xlsx")
    store.put_bytes(export_key, result.data, content_type=manifest.content_type)

    source_cells = output_cells = 0
    output_sheets = 0
    opened = False
    try:
        with XlsxParser().open_bytes(source) as doc:
            source_cells = doc.cell_count
        # Reopen the bytes actually written (spec sec. 53).
        with XlsxParser().open_bytes(result.data) as doc:
            output_cells = doc.cell_count
            output_sheets = doc.page_count
        opened = True
    except Exception as exc:
        log.exception("Rebuilt workbook could not be reopened")
        for page in manifest.pages:
            page.error = "The translated workbook could not be reopened."
            page.error_stage = "RECONSTRUCTING"
            page.diagnostics.extra["xlsx_reopen_error"] = f"{type(exc).__name__}: {exc}"

    total = len(manifest.pages)
    for i, page in enumerate(manifest.pages):
        page.reconstructed = opened
        page.revision += 1
        if on_progress:
            on_progress(i + 1, total)

    return ExportResult(
        export_key=export_key,
        byte_size=len(result.data),
        source_page_count=len(manifest.pages),
        output_page_count=output_sheets,
        findings_by_page={
            idx: s.findings for idx, s in result.stats.items() if s.findings
        },
        source_unit_count=source_cells,
        output_unit_count=output_cells,
        unit_noun="cell",
    )


def _render_thumbnail(pdf_page, width_px: int) -> bytes:
    """A thumbnail drawn straight from the page, at most `width_px` wide.

    Rendering small costs far less than decoding the full-size preview PNG
    and shrinking it. The scale stays a hair under the exact ratio because
    MuPDF rounds the pixel box outward; if it still comes out wider, the
    old downscale is the fallback.
    """
    width = pdf_page.rect.width
    if width <= 0:
        raise ValueError("page has no width")
    scale = min(1.0, (width_px - 0.01) / width) if width > width_px else 1.0
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    png = pix.tobytes("png")
    if pix.width > width_px:
        png, _, _ = make_thumbnail(png, width_px)
    return png


def _scratch_pdf(store: ObjectStore) -> Path:
    fd, name = tempfile.mkstemp(dir=str(store.scratch_dir()), suffix=".pdf")
    os.close(fd)
    return Path(name)


def _rebuild_to_store(
    manifest: DocumentManifest,
    export_key: str,
    *,
    store: ObjectStore,
    target_lang: str,
    on_plate=None,
    before_page=None,
):
    """Rebuild the PDF into the store without holding it in memory whole.

    The source is read from its file and the output written to a scratch
    file that the store takes over. Returns (byte_size, build_stats).
    """
    source: bytes | Path | None = store.local_path(manifest.source_key)
    if source is None:
        source = store.get_bytes(manifest.source_key)
    out = _scratch_pdf(store)
    try:
        written, build_stats = rebuild_document(
            source,
            manifest.pages,
            target_lang=target_lang,
            on_plate=on_plate,
            before_page=before_page,
            out_path=out,
        )
        del source
        if isinstance(written, (bytes, bytearray)):
            # A stand-in rebuild (tests) may still hand back bytes.
            out.write_bytes(written)
            del written
        byte_size = out.stat().st_size
        store.put_file(export_key, out, content_type="application/pdf")
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return byte_size, build_stats


def _store_plate(manifest, page, pdf_page, *, store: ObjectStore) -> None:
    """Save the live editor's background for one page.

    Called by the rebuild at the moment the page holds everything except the
    translations, so it costs one render and no second rebuild. A failure
    only costs the live editor its background; the PDF is unaffected.
    """
    try:
        pix = pdf_page.get_pixmap(dpi=get_settings().preview_dpi, alpha=False)
        key = K.page_plate_key(manifest.document_id, page.index)
        store.put_bytes(key, pix.tobytes("png"), content_type="image/png")
        page.renders.plate_key = key
    except Exception as exc:
        log.warning("Plate render failed for page %s: %r", page.number, exc)


def _reconstruct_and_export_pdf(
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    target_lang: str,
    on_progress=None,
    check_cancel=None,
) -> ExportResult:
    """Rebuild every page, persist the output PDF, and render previews."""
    # Whatever the old export was built from no longer describes it.
    store.delete(K.export_state_key(manifest.document_id))
    # Memory: neither the source nor the output is held whole. Both used to
    # stay in memory, beside two open documents, through the slowest part
    # of the job -- the per-page preview loop below -- and a large scanned
    # packet ran the free server out of memory at "Rebuilding pages".
    export_key = K.export_key(manifest.document_id)
    byte_size, build_stats = _rebuild_to_store(
        manifest,
        export_key,
        store=store,
        target_lang=target_lang,
        on_plate=lambda index, pdf_page: _store_plate(
            manifest, manifest.pages[index], pdf_page, store=store
        ),
        before_page=check_cancel,
    )

    # Reopen what we actually wrote. Validating the file on disk -- rather
    # than trusting the in-memory object -- is the only check that proves the
    # file a teacher downloads will open (spec sec. 53).
    settings = get_settings()
    doc = _open_pdf(store, export_key)
    source_doc = _open_pdf(store, manifest.source_key)
    try:
        output_pages = doc.page_count
        total = min(output_pages, len(manifest.pages))
        for i, page in enumerate(manifest.pages):
            if i >= output_pages:
                break
            # Count the artwork actually present in the file that was
            # written, against the same count taken the same way from the
            # source. This replaces asking whether the engine managed to
            # save its own extra COPY of each image, which is a question
            # about the review UI's cache and not about the teacher's
            # document -- two 3-pixel divider rules that failed to extract
            # were being reported as artwork lost, on a page where the
            # output is pixel-identical.
            try:
                page.diagnostics.extra["source_image_count"] = count_artwork(
                    source_doc.load_page(i)
                )
                page.diagnostics.extra["output_image_count"] = count_artwork(
                    doc.load_page(i)
                )
            except Exception as exc:
                log.warning("Could not count images on page %s: %r", i + 1, exc)
            try:
                out_page = doc.load_page(i)
                pix = out_page.get_pixmap(dpi=settings.preview_dpi, alpha=False)
                png = pix.tobytes("png")
                del pix
                key = K.page_translated_render_key(manifest.document_id, i)
                store.put_bytes(key, png, content_type="image/png")
                del png
                thumb = _render_thumbnail(out_page, settings.thumbnail_width_px)
                thumb_key = K.page_translated_thumb_key(manifest.document_id, i)
                store.put_bytes(thumb_key, thumb, content_type="image/png")
                page.renders.translated_preview_key = key
                page.renders.translated_thumb_key = thumb_key
                page.reconstructed = True
            except Exception as exc:
                log.warning("Translated preview failed for page %s: %r", i + 1, exc)
                page.reconstructed = True  # the PDF page exists; only the preview failed
            if on_progress:
                on_progress(i + 1, total)
    finally:
        doc.close()
        source_doc.close()

    _save_export_state(manifest, store, target_lang)
    findings = {idx: s.findings for idx, s in build_stats.items() if s.findings}
    return ExportResult(
        export_key=export_key,
        byte_size=byte_size,
        source_page_count=len(manifest.pages),
        output_page_count=output_pages,
        findings_by_page=findings,
    )


# --- Per-page verification -------------------------------------------------
def count_artwork(pdf_page) -> int:
    """Pictures on a page, not counting images that only serve as a mask.

    A layered scan holds its sharp foreground behind a full-resolution
    transparency mask, and scanners often list that mask as a resource of
    its own. It is part of its picture, not a second one: the rebuilt file
    keeps it attached without listing it, so counting it made every page of
    such a scan report a missing picture.
    """
    images = pdf_page.get_images(full=True)
    masks = {smask for _, smask, *_ in images if smask}
    # Pictures DRAWN on the page, not every picture its resources list: a
    # Word file laid out by LibreOffice lists all seven of its pictures on
    # each of its five pages, the rebuilt file lists only the drawn ones,
    # and every page reported six pictures missing.
    drawn = {info["xref"] for info in pdf_page.get_image_info(xrefs=True) if info.get("xref")}
    return sum(1 for xref, *_ in images if xref not in masks and xref in drawn)


def verify_page(page: Page, extra_findings: list[QAFinding] | None = None) -> PageQA:
    """Run the automated checks for one page and derive its status.

    The score is an internal triage heuristic, not a claim about translation
    quality (spec sec. 30).
    """
    qa = PageQA()
    qa.findings = list(extra_findings or [])
    from ..translate.value_checks import value_findings
    qa.findings.extend(value_findings(page))
    for obj in page.text_objects:
        if obj.keep_original:
            qa.findings.append(QAFinding(severity="warning", code="original_kept",
                message="This section is shown in the original language: it was hard to read, or the original was put back. Check its English to translate it.",
                object_ids=[obj.id], bbox=obj.bbox))
        # A raised or lowered glyph with no Unicode script form is redrawn on
        # the baseline, so "xq" may now read as a product. Say so.
        if obj.translated_text and not obj.keep_original and any(
            span.script and not span.script_converted
            for line in obj.lines for span in line.spans
        ):
            qa.findings.append(QAFinding(severity="warning", code="math_script_flattened",
                message="An exponent or subscript in this section is now on the line. Compare it with the original.",
                object_ids=[obj.id], bbox=obj.bbox))
    qa.checks_run = [
        "page_present",
        "images_preserved",
        "protected_content_intact",
        "translation_coverage",
    ]
    # Checks 4-6 read coordinates. On a reflowable page those coordinates are
    # a synthesised reading-order flow, not a layout -- the layout does not
    # exist until Word opens the file. Running them would report overlaps and
    # out-of-bounds text about boxes nobody will ever see, which is worse
    # than not checking: it would send reviewers to inspect pages that are
    # fine. Word reflowing the text is also what makes them unnecessary.
    if page.has_fixed_layout:
        qa.checks_run += ["objects_within_bounds", "response_areas_preserved"]

    # 1. Images must still be present and un-distorted.
    #
    # Presence is checked against the rebuilt file, counted the same way on
    # both sides -- not against whether the engine saved its own copy of
    # each image for the review UI. Reconstruction is configured never to
    # remove artwork (PDF_REDACT_IMAGE_NONE), so a drop here means something
    # genuinely went wrong and is worth a reviewer's time; the previous
    # check fired on hairline rules that were in fact pixel-identical.
    before = page.diagnostics.extra.get("source_image_count")
    after = page.diagnostics.extra.get("output_image_count")
    if isinstance(before, int) and isinstance(after, int) and after < before:
        qa.findings.append(
            QAFinding(
                severity="warning",
                code="image_not_preserved",
                message=(
                    f"{before - after} picture(s) on this page are missing "
                    "from the translated document."
                ),
                details={"source": before, "output": after},
            )
        )

    for img in page.image_objects:
        if img.pixel_aspect_ratio and img.aspect_ratio:
            drift = abs(img.aspect_ratio - img.pixel_aspect_ratio) / img.pixel_aspect_ratio
            if drift > 0.05:
                qa.findings.append(
                    QAFinding(
                        severity="warning",
                        code="image_aspect_changed",
                        message="An image on this page may be stretched.",
                        object_ids=[img.id],
                        bbox=img.bbox,
                        details={"drift": round(drift, 3)},
                    )
                )

    # 2. Protected content must still appear verbatim in the output text.
    for region in page.protected_regions:
        owner = page.find_text_object(region.object_id)
        if owner is None or owner.translated_text is None:
            continue
        if region.text not in owner.translated_text:
            qa.findings.append(
                QAFinding(
                    severity="error",
                    code="protected_content_lost",
                    message=(
                        "A formula or value on this page did not survive "
                        "translation. Compare it with the original before sharing."
                    ),
                    object_ids=[owner.id],
                    bbox=owner.bbox,
                    details={"category": region.category},
                )
            )

    # 3. Translation coverage.
    eligible = [
        t
        for t in page.text_objects
        if t.translation_status
        not in (TranslationStatus.NOT_REQUIRED, TranslationStatus.PROTECTED)
    ]
    untranslated = [
        t for t in eligible if t.translation_status is not TranslationStatus.TRANSLATED
        and t.translation_status is not TranslationStatus.MANUAL_EDIT
    ]
    if eligible and untranslated:
        qa.findings.append(
            QAFinding(
                severity="warning",
                code="untranslated_text",
                message=(
                    f"{len(untranslated)} block(s) on this page were left in the "
                    "original language."
                ),
                object_ids=[t.id for t in untranslated],
            )
        )

    # 4. Translated text must stay inside the page.
    for t in page.text_objects if page.has_fixed_layout else []:
        if t.translation_status is not TranslationStatus.TRANSLATED:
            continue
        if (
            t.bbox.x1 > page.width + 2
            or t.bbox.y1 > page.height + 2
            or t.bbox.x0 < -2
            or t.bbox.y0 < -2
        ):
            qa.findings.append(
                QAFinding(
                    severity="error",
                    code="out_of_bounds",
                    message="Translated text on this page falls outside the page.",
                    object_ids=[t.id],
                    bbox=t.bbox,
                )
            )

    # 5. Response areas must not be covered by translated text.
    for area in page.response_areas if page.has_fixed_layout else []:
        for t in page.text_objects:
            if t.translation_status is not TranslationStatus.TRANSLATED:
                continue
            if area.bbox.overlap_fraction_of_self(t.bbox) > 0.25:
                qa.findings.append(
                    QAFinding(
                        severity="error",
                        code="collision_response_area",
                        message=(
                            "Translated text may cover space a student needs "
                            "for writing."
                        ),
                        object_ids=[t.id],
                        bbox=area.bbox,
                    )
                )
                break

    # 6. Text-on-text collisions among translated blocks, judged on where
    # the words were really drawn (widening and reflow move them), and in
    # both directions: a short line printed inside a paragraph covers little
    # of the paragraph but is wholly on top of it. Two translations printed
    # over each other is never acceptable, so a real overlap is an error.
    drawn = [
        t for t in page.text_objects
        if t.translation_status in (TranslationStatus.TRANSLATED, TranslationStatus.MANUAL_EDIT)
        and t.translated_text and not t.keep_original and not t.suppressed
    ] if page.has_fixed_layout else []
    for i, a in enumerate(drawn):
        for b in drawn[i + 1 :]:
            ra, rb = _collision_box(a), _collision_box(b)
            share = max(ra.overlap_fraction_of_self(rb), rb.overlap_fraction_of_self(ra))
            if share <= 0.30:
                continue
            qa.findings.append(
                QAFinding(
                    severity="error" if share > 0.5 else "warning",
                    code="collision_text",
                    message=(
                        "Two translations are printed on top of each other. "
                        "Remove one, or move it."
                    ),
                    object_ids=[a.id, b.id],
                    bbox=ra.union(rb),
                )
            )

    # --- Score and status ---
    errors = len(qa.errors)
    warnings = len(qa.warnings)
    score = 1.0 - (0.25 * errors) - (0.07 * warnings)
    qa.layout_confidence = round(max(0.0, min(1.0, score)), 3)

    if page.error:
        qa.status = PageStatus.FAILED
    elif errors:
        qa.status = PageStatus.REVIEW_REQUIRED
    elif warnings:
        qa.status = PageStatus.REVIEW_SUGGESTED
    else:
        qa.status = PageStatus.READY

    page.qa = qa
    page.verified = True
    page.recompute_diagnostics()
    return qa


def _collision_box(t) -> Rect:
    """Where a block's words were drawn, for the text-on-text check. A
    shape is judged by its words' rows, not its frame: a large oval round
    a short caption merely surrounds other text, it is not printed over it."""
    layout = t.drawn_layout
    if t.manual_shape and layout is not None and layout.rows:
        from ..reconstruct.builder import ASCENT_RATIO, DESCENT_RATIO, _turn_point

        inked = [r for r in layout.rows if r.x1 > r.x0]
        if inked:
            size = layout.size
            corners = [
                (x, y)
                for r in inked
                for x in (r.x0, r.x1)
                for y in (r.baseline - ASCENT_RATIO * size, r.baseline + DESCENT_RATIO * size)
            ]
            if layout.angle and layout.pivot is not None:
                corners = [_turn_point(x, y, layout.pivot, layout.angle) for x, y in corners]
            xs, ys = zip(*corners)
            return Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    return t.drawn_bbox or t.bbox


@dataclass
class CompletionGate:
    """The check that decides whether a download may be offered."""

    source_pages: int
    rendered_pages: int
    verified_pages: int
    output_pages: int
    opens_correctly: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons


def check_completion(
    manifest: DocumentManifest, export: ExportResult, output_opens: bool
) -> CompletionGate:
    gate = CompletionGate(
        source_pages=manifest.source_page_count,
        rendered_pages=manifest.rendered_page_count,
        verified_pages=manifest.verified_page_count,
        output_pages=export.output_page_count,
        opens_correctly=output_opens,
    )
    if export.output_page_count != manifest.source_page_count:
        gate.reasons.append(
            f"The translated document has {export.output_page_count} pages but the "
            f"original has {manifest.source_page_count}."
        )
    # For a Word document the page count above is a section count, which
    # survives by construction and therefore proves nothing. The paragraph
    # count is the real complete-document check (spec sec. 3): if the output
    # lost a paragraph, a teacher would hand out a worksheet with a missing
    # question and no warning.
    if export.source_unit_count is not None:
        if export.output_unit_count != export.source_unit_count:
            gate.reasons.append(
                f"The translated document has {export.output_unit_count} "
                f"{export.unit_noun}s but the original has "
                f"{export.source_unit_count}."
            )
    if not output_opens:
        gate.reasons.append("The translated document could not be reopened.")
    if manifest.rendered_page_count != manifest.source_page_count:
        gate.reasons.append(
            f"Only {manifest.rendered_page_count} of {manifest.source_page_count} "
            "pages were rebuilt."
        )
    # Backstop: a page with words waiting to be translated that was never
    # translated, and never reported as failing, would go out in English
    # with nothing said. That is a bug upstream, so it is logged loudly and
    # the download is held back rather than offered as complete.
    skipped = never_translated_pages(manifest)
    if skipped:
        log.error(
            "Pages never translated and never flagged: %s (document %s)",
            skipped, manifest.document_id,
        )
        numbers = ", ".join(str(n) for n in skipped[:10])
        more = "" if len(skipped) <= 10 else f" and {len(skipped) - 10} more"
        noun = "Page" if len(skipped) == 1 else "Pages"
        verb = "was" if len(skipped) == 1 else "were"
        gate.reasons.append(
            f"{noun} {numbers}{more} {verb} not translated. "
            "Please translate the document again."
        )
    return gate


def never_translated_pages(manifest: DocumentManifest) -> list[int]:
    """Page numbers with text still waiting to be translated, where the
    page was never translated and no error was recorded for it."""
    return [
        page.number
        for page in manifest.pages
        if not page.translated
        and page.error is None
        and any(
            t.translation_status is TranslationStatus.PENDING
            and not t.keep_original
            and not t.suppressed
            and t.flat_text.strip()
            for t in page.text_objects
        )
    ]


def rebuild_single_page(
    manifest: DocumentManifest,
    page: Page,
    *,
    store: ObjectStore,
    target_lang: str,
) -> list[QAFinding]:
    """Rebuild one page and refresh its translated preview.

    Used after a manual edit. Opens the source, touches only this page, and
    leaves every other page's preview alone -- which is what keeps a one-word
    correction from costing a 100-page re-render (spec sec. 36).
    """
    if format_of(manifest).rebuilds_in_place:
        # An Office page has no preview to refresh, and its write-back is a
        # whole-package operation rather than a per-page one. The export is
        # rebuilt lazily on download (job.export_stale), so the edit just has
        # to be recorded. Rebuilding the file here instead would cost a full
        # package rewrite on every keystroke, for nothing anyone sees.
        page.reconstructed = True
        return []

    settings = get_settings()
    doc = _open_pdf(store, manifest.source_key)
    try:
        if page.index >= doc.page_count:
            raise IndexError(f"page {page.index} not in source document")
        pdf_page = doc.load_page(page.index)
        stats = rebuild_page(
            pdf_page,
            page,
            target_lang=target_lang,
            on_plate=lambda p: _store_plate(manifest, page, p, store=store),
        )

        pix = pdf_page.get_pixmap(dpi=settings.preview_dpi, alpha=False)
        png = pix.tobytes("png")
        del pix
        preview_key = K.page_translated_render_key(manifest.document_id, page.index)
        store.put_bytes(preview_key, png, content_type="image/png")
        del png
        thumb = _render_thumbnail(pdf_page, settings.thumbnail_width_px)
        thumb_key = K.page_translated_thumb_key(manifest.document_id, page.index)
        store.put_bytes(thumb_key, thumb, content_type="image/png")
        page.renders.translated_preview_key = preview_key
        page.renders.translated_thumb_key = thumb_key
        page.reconstructed = True
        return list(stats.findings)
    finally:
        doc.close()


def rebuild_export_only(
    manifest: DocumentManifest, *, store: ObjectStore, target_lang: str
) -> ExportResult:
    """Rebuild the downloadable file without re-rendering any previews.

    The download is a single file, so it has to be regenerated after an edit.
    This does that from the stored page models: no translation, no OCR, and
    no per-page preview work.
    """
    fmt = format_of(manifest)
    if fmt is SourceFormat.DOCX:
        # The Word write-back does not render anything, so there is no
        # preview work to skip -- the full path is already the cheap one.
        return _reconstruct_and_export_docx(manifest, store=store)
    if fmt is SourceFormat.PPTX:
        return _reconstruct_and_export_pptx(manifest, store=store)
    if fmt is SourceFormat.XLSX:
        return _reconstruct_and_export_xlsx(manifest, store=store)

    patched = _patch_export(manifest, store=store, target_lang=target_lang)
    if patched is not None:
        return patched

    store.delete(K.export_state_key(manifest.document_id))
    export_key = K.export_key(manifest.document_id)
    byte_size, build_stats = _rebuild_to_store(
        manifest, export_key, store=store, target_lang=target_lang
    )

    doc = _open_pdf(store, export_key)
    try:
        output_pages = doc.page_count
    finally:
        doc.close()

    _save_export_state(manifest, store, target_lang)
    return ExportResult(
        export_key=export_key,
        byte_size=byte_size,
        source_page_count=len(manifest.pages),
        output_page_count=output_pages,
        findings_by_page={
            idx: s.findings for idx, s in build_stats.items() if s.findings
        },
    )


# --- Reading stored PDFs, and rebuilding only what changed ------------------
def _open_pdf(store: ObjectStore, key: str) -> "fitz.Document":
    """Open a stored PDF from its file, so MuPDF reads it as needed.

    Opening from bytes keeps the whole file in memory for as long as the
    document is open. Backends with no local file fall back to bytes.
    """
    path = store.local_path(key)
    if path is not None:
        return fitz.open(str(path), filetype="pdf")
    return fitz.open(stream=store.get_bytes(key), filetype="pdf")


# Fields the rebuild never reads, or that change without changing what is
# drawn (previews, checks, undo history, counters). Everything else on a page
# -- every text object, picture and region -- is in the fingerprint, so a
# change anywhere on the page, by any route, marks it for rebuilding. Leaving
# a field in costs at most an unneeded rebuild; leaving one out wrongly would
# serve a stale page, so this list stays short.
_NOT_DRAWN = {
    "renders", "qa", "diagnostics", "edit_history", "redo_history",
    "revision", "reconstructed", "verified", "error", "error_stage", "attempts",
}


def _page_fingerprint(page: Page) -> str:
    raw = page.model_dump_json(exclude=_NOT_DRAWN)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _save_export_state(
    manifest: DocumentManifest, store: ObjectStore, target_lang: str
) -> None:
    """Record what each page looked like when the stored export was built."""
    state = {
        "target_lang": target_lang,
        "pages": {str(p.index): _page_fingerprint(p) for p in manifest.pages},
    }
    try:
        store.put_text(
            K.export_state_key(manifest.document_id),
            json.dumps(state),
            content_type="application/json",
        )
    except Exception as exc:
        # Without it the next download after an edit rebuilds every page:
        # slower, never wrong.
        log.warning("Could not record export state: %r", exc)


def _stale_pages(
    manifest: DocumentManifest, store: ObjectStore, target_lang: str
) -> list[int] | None:
    """Pages that changed since the stored export was built.

    None when that cannot be told -- no record, another language, a page
    count that moved -- and the whole file must be rebuilt.
    """
    try:
        state = json.loads(store.get_text(K.export_state_key(manifest.document_id)))
    except ObjectNotFoundError:
        return None
    except Exception as exc:
        log.warning("Export state unreadable, rebuilding everything: %r", exc)
        return None
    built = state.get("pages") or {}
    if state.get("target_lang") != target_lang or len(built) != len(manifest.pages):
        return None
    return [
        p.index for p in manifest.pages
        if built.get(str(p.index)) != _page_fingerprint(p)
    ]


def _internal_links(pdf_page, pages: set[int] | None = None) -> bool:
    """Whether this page links within the document (to `pages`, if given)."""
    for link in pdf_page.get_links():
        kind = link.get("kind")
        if kind == fitz.LINK_NAMED:
            return True  # a named destination: cannot tell where it goes
        if kind == fitz.LINK_GOTO and (pages is None or link.get("page") in pages):
            return True
    return False


def _subset_fonts_file(path: Path, scratch: Path) -> Path:
    """The PDF at `path` with its fonts cut down to the glyphs it uses.

    Each rebuilt page embeds the whole font file again (a 46 KB export
    became 15 MB after one page was swapped in), so a patched export is
    subset before it is stored, as a full rebuild is, with the builder's
    same text check. Returns the smaller file and removes the other;
    returns `path` unchanged whenever the subset cannot be made, would
    change the text, or is not smaller.
    """
    fd, name = tempfile.mkstemp(dir=str(scratch), suffix=".pdf")
    os.close(fd)
    subset = Path(name)
    if not subset_fonts_file(path, subset):
        subset.unlink(missing_ok=True)
        return path
    path.unlink(missing_ok=True)
    return subset


def _patch_export(
    manifest: DocumentManifest, *, store: ObjectStore, target_lang: str
) -> ExportResult | None:
    """Rebuild only the edited pages into the stored export.

    A one-word fix used to redraw every page of the document inside the
    download request, with the document locked against further edits the
    whole time. The edited pages are rebuilt from the source and swapped
    into the stored file instead.

    Returns None, and the caller rebuilds everything, whenever this cannot
    be done without losing something: no record of what the export was built
    from, a fillable form, bookmarks or links into a swapped page (a page
    swap drops them), or more than half the pages changed (no saving).
    """
    export_key = K.export_key(manifest.document_id)
    stale = _stale_pages(manifest, store, target_lang)
    if not stale or not store.exists(export_key):
        # An empty list with the export marked stale means something changed
        # that the fingerprint does not see; the full rebuild is the safe
        # answer to that.
        return None
    if len(stale) > len(manifest.pages) // 2:
        return None

    stale_set = set(stale)
    tmp: Path | None = None
    findings: dict[int, list[QAFinding]] = {}
    out = _open_pdf(store, export_key)
    src = _open_pdf(store, manifest.source_key)
    try:
        if (
            out.page_count != src.page_count
            or out.page_count != len(manifest.pages)
            or out.is_form_pdf
            or out.get_toc(simple=True)
            or any(_internal_links(out.load_page(i), stale_set)
                   for i in range(out.page_count))
            or any(_internal_links(src.load_page(i)) for i in stale)
        ):
            return None
        for i in sorted(stale):
            page = manifest.page(i)
            if page is None:
                return None
            try:
                stats = rebuild_page(src.load_page(i), page, target_lang=target_lang)
            except Exception:
                # The full rebuild records a page that cannot be rebuilt as a
                # finding and carries it over unchanged; let it.
                log.exception("Page %s could not be patched; rebuilding all", i + 1)
                return None
            if stats.findings:
                findings[i] = list(stats.findings)
            out.delete_page(i)
            out.insert_pdf(src, from_page=i, to_page=i, start_at=i)
        fd, name = tempfile.mkstemp(dir=str(store.scratch_dir()), suffix=".pdf")
        os.close(fd)
        tmp = Path(name)
        out.save(str(tmp), garbage=3, deflate=True)
    except Exception:
        log.exception("Could not patch the export; rebuilding all pages")
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        return None
    finally:
        out.close()
        src.close()
    tmp = _subset_fonts_file(tmp, store.scratch_dir())

    byte_size = tmp.stat().st_size
    store.put_file(export_key, tmp, content_type="application/pdf")
    # The file written must open, and with every page (spec sec. 46/53).
    check = _open_pdf(store, export_key)
    try:
        output_pages = check.page_count
    finally:
        check.close()
    _save_export_state(manifest, store, target_lang)
    log.info("Export patched: %d of %d pages rebuilt", len(stale), len(manifest.pages))
    return ExportResult(
        export_key=export_key,
        byte_size=byte_size,
        source_page_count=len(manifest.pages),
        output_page_count=output_pages,
        findings_by_page=findings,
    )
