"""The ANALYZE stage.

Decomposes every page of the source document into the page object model,
preserves every original image asset, renders previews, and aggregates a
document-level analysis.

Completeness is the contract: this stage processes every page or records
exactly why a page could not be processed. It never stops early, and it never
reports success for a page it skipped (spec sec. 3).
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from dataclasses import dataclass
from typing import Callable

from ..analysis.language import detect_language, sample_document_text
from ..analysis.response_areas import detect_response_areas
from ..config import get_settings
from ..ingest.base import OpenDocument, PageParseError
from ..model.document import DocumentAnalysis, DocumentManifest, Page
from ..model.enums import Orientation, PageTextMode
from ..render.previews import render_source_page
from ..storage.base import ObjectStore
from ..util.logging import get_logger
from .assets import extract_page_assets

log = get_logger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass
class PageOutcome:
    index: int
    page: Page | None
    error: str | None = None
    diagnostic: str | None = None


def analyze_page(
    doc: OpenDocument,
    index: int,
    *,
    document_id: str,
    store: ObjectStore,
    seen_hashes: set[str],
    hashes_lock: threading.Lock,
    render_previews: bool = True,
) -> PageOutcome:
    """Fully analyze one page. Never raises for page-local problems."""
    try:
        page = doc.parse_page(index)
    except PageParseError as exc:
        return PageOutcome(index, None, exc.message, exc.diagnostic)
    except Exception as exc:  # defensive: parser contract violation
        return PageOutcome(
            index,
            None,
            f"Page {index + 1} could not be read.",
            f"{type(exc).__name__}: {exc}",
        )

    # Assets and previews are best-effort: losing a preview must not lose the
    # page's object model, which is the expensive part.
    try:
        with hashes_lock:
            local_seen = seen_hashes
        stored, failed = extract_page_assets(
            doc, page, document_id=document_id, store=store, seen_hashes=local_seen
        )
        if failed:
            page.diagnostics.extra["assets_not_preserved"] = failed
    except Exception as exc:
        log.warning("Asset extraction failed on page %s: %r", index + 1, exc)
        page.diagnostics.extra["asset_error"] = str(exc)

    try:
        page.response_areas = detect_response_areas(page)
    except Exception as exc:
        log.warning("Response-area detection failed on page %s: %r", index + 1, exc)

    if render_previews and not doc.supports_preview:
        page.diagnostics.extra["preview_unavailable"] = (
            "This format has no page images; review is by text."
        )
    elif render_previews:
        try:
            render_source_page(doc, page, document_id=document_id, store=store)
        except Exception as exc:
            log.warning("Preview render failed on page %s: %r", index + 1, exc)
            page.diagnostics.extra["preview_error"] = str(exc)

    page.recompute_diagnostics()
    page.analyzed = True
    return PageOutcome(index, page)


def analyze_document(
    doc: OpenDocument,
    manifest: DocumentManifest,
    *,
    store: ObjectStore,
    on_progress: ProgressFn | None = None,
    concurrency: int | None = None,
    render_previews: bool = True,
    only_indices: list[int] | None = None,
) -> DocumentManifest:
    """Analyze the whole document, filling in `manifest.pages`.

    `only_indices` restricts work to specific pages, which is how a retry of
    one failed page avoids reprocessing the other 99.
    """
    settings = get_settings()
    workers = max(1, concurrency or settings.page_concurrency)
    total_pages = doc.page_count

    manifest.metadata = doc.metadata()

    targets = (
        sorted(set(only_indices))
        if only_indices is not None
        else list(range(total_pages))
    )
    targets = [i for i in targets if 0 <= i < total_pages]

    # Seed the page list so that every source page has a slot from the start.
    # A page that fails still occupies its index, which is what makes
    # "source_page_count == len(pages)" a meaningful check.
    existing = {p.index: p for p in manifest.pages}
    seen_hashes: set[str] = set()
    hashes_lock = threading.Lock()
    completed = 0
    progress_lock = threading.Lock()

    def report() -> None:
        nonlocal completed
        with progress_lock:
            completed += 1
            done = completed
        if on_progress:
            try:
                on_progress(done, len(targets))
            except Exception:
                log.debug("progress callback raised", exc_info=True)

    def work(index: int) -> PageOutcome:
        try:
            return analyze_page(
                doc,
                index,
                document_id=manifest.document_id,
                store=store,
                seen_hashes=seen_hashes,
                hashes_lock=hashes_lock,
                render_previews=render_previews,
            )
        finally:
            report()

    outcomes: list[PageOutcome] = []
    if workers == 1 or len(targets) == 1:
        outcomes = [work(i) for i in targets]
    else:
        # PyMuPDF document objects are not safe to share across threads for
        # writes, but page reads and pixmap rendering are fine and release the
        # GIL, which is where the wall-clock win is. Keep the pool modest.
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, i): i for i in targets}
            for fut in cf.as_completed(futures):
                idx = futures[fut]
                try:
                    outcomes.append(fut.result())
                except Exception as exc:
                    outcomes.append(
                        PageOutcome(
                            idx,
                            None,
                            f"Page {idx + 1} could not be read.",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )

    for outcome in outcomes:
        if outcome.page is not None:
            existing[outcome.index] = outcome.page
        else:
            # Keep a placeholder page carrying the failure, so the page still
            # exists in the manifest and can be retried individually.
            placeholder = existing.get(outcome.index) or Page(
                index=outcome.index,
                number=outcome.index + 1,
                width=0.0,
                height=0.0,
            )
            placeholder.analyzed = False
            placeholder.error = outcome.error
            placeholder.error_stage = "ANALYZING"
            placeholder.attempts += 1
            existing[outcome.index] = placeholder

    # Any page never attempted (e.g. targeted retry) still needs a slot.
    for i in range(total_pages):
        if i not in existing:
            existing[i] = Page(index=i, number=i + 1, width=0.0, height=0.0)

    manifest.pages = [existing[i] for i in range(total_pages)]
    manifest.analysis = aggregate_analysis(manifest, total_pages)
    # Set after the roll-up, which builds a fresh DocumentAnalysis: formats
    # that keep their pictures untouched report them from the package, not
    # from parsed page objects, of which they have none.
    embedded = getattr(doc, "embedded_media_count", None)
    if embedded is not None:
        manifest.analysis.embedded_media_count = embedded
    return manifest


def aggregate_analysis(manifest: DocumentManifest, page_count: int) -> DocumentAnalysis:
    """Roll page facts up into the document-level summary shown at configure."""
    pages = manifest.pages
    analysis = DocumentAnalysis(page_count=page_count)

    size_counts: dict[tuple[float, float], int] = {}
    orientation_counts: dict[str, int] = {}

    for p in pages:
        if not p.analyzed:
            continue
        analysis.total_text_objects += len(p.text_objects)
        analysis.total_images += len(p.image_objects)
        analysis.total_vectors += len(p.vector_objects)
        analysis.total_characters += sum(t.char_count for t in p.text_objects)

        key = (round(p.width, 1), round(p.height, 1))
        size_counts[key] = size_counts.get(key, 0) + 1
        orientation_counts[p.orientation.value] = (
            orientation_counts.get(p.orientation.value, 0) + 1
        )

        if p.text_mode == PageTextMode.SCANNED:
            analysis.scanned_page_count += 1
        elif p.text_mode == PageTextMode.NATIVE:
            analysis.native_page_count += 1
        elif p.text_mode == PageTextMode.EMPTY:
            analysis.empty_page_count += 1
        else:  # MIXED counts toward both signals
            analysis.scanned_page_count += 1
            analysis.native_page_count += 1

    analysis.has_selectable_text = analysis.total_characters > 0
    analysis.needs_ocr = any(p.needs_ocr for p in pages if p.analyzed)

    analysis.page_sizes = [
        {"width": w, "height": h, "count": c}
        for (w, h), c in sorted(size_counts.items(), key=lambda kv: -kv[1])
    ]
    analysis.has_mixed_page_sizes = len(size_counts) > 1
    analysis.orientation_counts = orientation_counts
    analysis.has_mixed_orientation = (
        len([k for k, v in orientation_counts.items() if v > 0]) > 1
    )

    sample = sample_document_text([p_text(p) for p in pages if p.analyzed])
    lang, conf, nchars = detect_language(sample)
    analysis.detected_language = lang
    analysis.detected_language_confidence = conf
    analysis.language_sample_chars = nchars

    failed = [p.number for p in pages if p.error]
    if failed:
        preview = ", ".join(str(n) for n in failed[:8])
        more = f" and {len(failed) - 8} more" if len(failed) > 8 else ""
        analysis.warnings.append(
            f"{len(failed)} page(s) could not be read: {preview}{more}."
        )
    if analysis.scanned_page_count:
        analysis.warnings.append(
            f"{analysis.scanned_page_count} page(s) appear to be scanned and will "
            "need to be read with text recognition."
        )
    if analysis.has_mixed_page_sizes:
        analysis.warnings.append("This document mixes more than one page size.")
    if not analysis.has_selectable_text and not analysis.scanned_page_count:
        analysis.warnings.append("No text was found in this document.")

    return analysis


def p_text(page: Page) -> str:
    return " ".join(t.flat_text for t in page.text_objects)
