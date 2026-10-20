"""Background job runner.

Documents are processed by a worker pool, not inside the HTTP request that
uploaded them, so a 100-page document survives the browser closing
(spec sec. 39). Job state is written to storage after every page, so a
restart resumes instead of restarting (spec sec. 40).

This implementation is a single-process pool, which is the right size for the
initial internal deployment. The interface -- submit, poll, retry -- is what a
distributed queue would expose too, so swapping in Redis/RQ or Celery later is
a change behind this class rather than a change to the pipeline.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import threading
import uuid
from typing import Callable

from ..config import get_settings
from ..ingest.base import ParseError
from ..ingest.registry import parser_for_manifest
from ..model.formats import SourceFormat, format_of
from ..model.document import DocumentManifest, QAFinding
from ..model.enums import JobStage
from ..model.job import DocumentJob, JobSettings, STAGE_LABELS
from ..analysis.block_split import split_page_blocks
from ..analysis.classify import classify_page
from ..analysis.dedupe import drop_duplicate_text
from ..analysis.ocr_grouping import group_ocr_lines
from ..analysis.reading_order import apply_reading_order
from ..pipeline.analyze import analyze_document
from ..pipeline.image_text import extract_image_text
from ..util.memory import give_back_memory, release_page_memory
from ..pipeline.ocr_stage import ocr_page, pages_needing_ocr
from ..pipeline.manifest_store import ManifestStore, document_lock
from ..pipeline.reconstruct_stage import (
    check_completion,
    reconstruct_and_export,
    verify_page,
)
from ..pipeline.translate import translate_page
from ..providers import (
    ProviderNotAvailable,
    get_ocr_provider,
    get_translation_provider,
)
from ..providers.translation.base import TranslationError
from ..providers.translation.chain import untranslated_reason
from ..storage import keys as K
from ..storage.base import ObjectStore
from ..util.logging import get_logger

log = get_logger(__name__)

# A job is picked up again at most this many times. Beyond that it is failed
# with a message, so a document that crashes the worker on every boot cannot
# restart-loop indefinitely.
MAX_RESUME_ATTEMPTS = 3

# When the translation service gave nothing back for at least this share of
# the blocks that needed translating, the document was not translated: it
# fails with the reason instead of finishing "Complete" in English.
UNTRANSLATED_FAIL_SHARE = 0.5

# The manifest holds every page's translations and is much bigger than the
# job, so during translation it is written less often than the job is.
_MANIFEST_SAVE_INTERVAL = 5.0


def _pictures_not_read(page_number: int, reason: str) -> QAFinding:
    """A page whose pictures were never read, said so rather than implied.

    Without this the report says "No readable text found inside pictures"
    when the truth is that nobody looked.
    """
    return QAFinding(
        severity="warning",
        code="image_text_not_read",
        message=(
            f"Words in pictures on page {page_number} could not be read, so "
            "they were left unchanged."
        ),
        details={"reason": reason[:200]},
    )


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


# What a teacher sees in the job's history once she stops it.
CANCELLED_MESSAGE = "Stopped. Nothing more will be done with this file."


class JobCancelled(BaseException):
    """Raised at a checkpoint once the job has been cancelled.

    A BaseException on purpose, like KeyboardInterrupt: every stage guards
    each page with `except Exception` so one bad page cannot stop the
    document, and those guards must let a cancellation through instead of
    recording it as a page that failed and carrying on.
    """


class _StopWhenCancelled:
    """An open document that raises JobCancelled once its job is cancelled.

    Analysis reads pages through the pipeline's own page loop, which keeps
    going past errors and runs pages on a small pool. Checking on every use
    of the document is what makes the pages still waiting there stop at
    once instead of each being read first.
    """

    def __init__(self, doc, check: Callable[[], None]) -> None:
        self._doc = doc
        self._check = check

    def __getattr__(self, name: str):
        self._check()
        return getattr(self._doc, name)


class JobRunner:
    def __init__(self, store: ObjectStore, manifests: ManifestStore) -> None:
        settings = get_settings()
        self.store = store
        self.manifests = manifests
        self._pool = cf.ThreadPoolExecutor(
            max_workers=max(1, settings.max_job_concurrency),
            thread_name_prefix="ct-job",
        )
        # Tracks in-flight jobs so a duplicate submit is rejected rather than
        # running the same document twice concurrently.
        self._active: dict[str, str] = {}  # document_id -> job_id
        # The pool's handle for each job, so one still waiting for a worker
        # can be dropped outright when it is cancelled.
        self._futures: dict[str, cf.Future] = {}  # job_id -> future
        # Jobs asked to stop. Checked between pages and between stages.
        self._cancel_requested: set[str] = set()
        self._guard = threading.Lock()
        self._shutdown = False

    # -- submission --------------------------------------------------------
    def submit_analysis(
        self, document_id: str, settings: JobSettings | None = None
    ) -> DocumentJob:
        job_settings = settings or JobSettings(target_language="und")
        job = DocumentJob(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            settings=job_settings,
            stage=JobStage.QUEUED,
        )
        sp = job.stage_progress(JobStage.ANALYZING)
        sp.label = STAGE_LABELS[JobStage.ANALYZING]
        job.log("info", "Document queued for analysis.")
        return self._enqueue(job, self._run_analysis)

    def _enqueue(self, job: DocumentJob, work: Callable[[str, str], None]) -> DocumentJob:
        """Save and start `job`, or hand back the job already running.

        The new job is written only once it is known to be the one that
        runs. Saved first, a repeated request left a QUEUED job on disk
        that nothing would ever pick up -- and, being the newest, it was
        the one the document's progress was read from, so the screen said
        "Waiting to start" for ever while the real job finished.
        """
        with self._guard:
            if self._shutdown:
                raise RuntimeError("Job runner is shutting down")
            active = self._active.get(job.document_id)
            # A job being stopped is not handed back: the new request runs
            # after it, once it lets go of the document.
            if active and active not in self._cancel_requested:
                existing = self.manifests.load_job(job.document_id, active)
                if existing and not existing.is_terminal:
                    return existing
            self.manifests.save_job(job)
            self._active[job.document_id] = job.job_id
            self._futures[job.job_id] = self._pool.submit(
                work, job.document_id, job.job_id
            )
        return job

    # -- cancelling -----------------------------------------------------------
    def cancel(self, document_id: str) -> bool:
        """Stop the document's running or waiting job.

        A job still waiting for a worker is dropped at once. A running one
        is told to stop and does so at its next page or stage, ending
        CANCELLED; its worker slot and memory are given back as it ends.
        Returns False when nothing was running for the document; True means
        the job is dropped or will end CANCELLED (the final COMPLETE save
        checks for a stop under the same guard). The request is also
        written down, so a restart before the job's next checkpoint ends it
        rather than resuming it.
        """
        with self._guard:
            job_id = self._active.get(document_id)
            if not job_id:
                return False
            job = self.manifests.load_job(document_id, job_id)
            if job is None or job.is_terminal:
                return False
            self._cancel_requested.add(job_id)
            future = self._futures.get(job_id)
            dropped = future is not None and future.cancel()
            if dropped:
                # It never started, so no worker will tidy up after it.
                self._futures.pop(job_id, None)
                self._cancel_requested.discard(job_id)
                self._active.pop(document_id, None)
                self._mark_cancelled(job)
            else:
                # The request lives in memory until the job's next
                # checkpoint. Written down too, so a restart before then
                # (most likely on a big file stuck in one heavy step) ends
                # the job instead of picking it up again.
                self._write_cancel_marker(document_id, job_id)
        log.info("Job %s cancelled%s", job_id, "" if dropped else " (stopping)")
        return True

    @staticmethod
    def _cancel_marker_key(document_id: str, job_id: str) -> str:
        # Beside the job's own file, not inside it: the worker saves its
        # in-memory copy of the job often and would write over a field.
        # list_jobs reads only .json files, so this is never taken for a job.
        return f"{K.jobs_prefix(document_id)}/{job_id}.cancel"

    def _write_cancel_marker(self, document_id: str, job_id: str) -> None:
        try:
            self.store.put_text(self._cancel_marker_key(document_id, job_id), "1")
        except Exception as exc:  # the in-memory request still stops the job
            log.warning("Could not record the stop for job %s: %r", job_id, exc)

    def _clear_cancel_marker(self, document_id: str, job_id: str) -> None:
        try:
            self.store.delete(self._cancel_marker_key(document_id, job_id))
        except Exception as exc:
            log.warning("Could not clear the stop for job %s: %r", job_id, exc)

    def _cancel_marked(self, document_id: str, job_id: str) -> bool:
        try:
            return self.store.exists(self._cancel_marker_key(document_id, job_id))
        except Exception:
            return False

    def _check_cancel(self, job: DocumentJob) -> None:
        if job.job_id in self._cancel_requested:
            raise JobCancelled()

    def _mark_cancelled(self, job: DocumentJob) -> None:
        job.stage = JobStage.CANCELLED
        job.error = None
        job.error_diagnostic = None
        job.finished_at = _utcnow()
        job.log("info", CANCELLED_MESSAGE)
        self.manifests.save_job(job)

    def _job_ended(self, document_id: str, job_id: str) -> None:
        with self._guard:
            if self._active.get(document_id) == job_id:
                self._active.pop(document_id, None)
            self._futures.pop(job_id, None)
            self._cancel_requested.discard(job_id)
        # The job has ended one way or another; its stop no longer matters.
        self._clear_cancel_marker(document_id, job_id)
        # Between documents the engine should be back near its fresh
        # size, so the next upload has room (see util/memory.py).
        give_back_memory()

    # -- execution ---------------------------------------------------------
    def _run_analysis(self, document_id: str, job_id: str) -> None:
        job = self.manifests.load_job(document_id, job_id)
        if job is None:
            log.error("Job %s vanished before it started", job_id)
            return
        try:
            self._check_cancel(job)
            self._analysis_stage(job)
        except JobCancelled:
            self._mark_cancelled(job)
        except Exception as exc:  # last-resort guard: a worker must not die
            log.exception("Job %s failed unexpectedly", job_id)
            job.stage = JobStage.FAILED
            job.error = "This document could not be processed."
            job.error_diagnostic = f"{type(exc).__name__}: {exc}"
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
        finally:
            self._job_ended(document_id, job_id)

    def _analysis_stage(self, job: DocumentJob) -> None:
        manifest = self.manifests.load_manifest(job.document_id)
        if manifest is None:
            job.stage = JobStage.FAILED
            job.error = "That document could not be found."
            job.error_diagnostic = f"no manifest for {job.document_id}"
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
            return

        job.stage = JobStage.ANALYZING
        job.started_at = job.started_at or _utcnow()
        progress = job.stage_progress(JobStage.ANALYZING)
        progress.started_at = _utcnow()
        progress.total = manifest.analysis.page_count or 0
        progress.completed = 0
        job.log("info", "Analyzing document.")
        self.manifests.save_job(job)

        source = self.store.local_path(manifest.source_key)
        parser = parser_for_manifest(manifest)

        # Persist progress at most every half second: a 100-page document
        # would otherwise spend more time writing JSON than reading pages.
        last_write = 0.0
        write_lock = threading.Lock()

        def on_progress(done: int, total: int) -> None:
            nonlocal last_write
            import time

            self._check_cancel(job)
            now = time.monotonic()
            release_page_memory()
            progress.completed = done
            progress.total = total
            with write_lock:
                if now - last_write >= 0.5 or done >= total:
                    last_write = now
                    self.manifests.save_job(job)

        try:
            with document_lock(job.document_id):
                if source is not None:
                    doc = parser.open_path(source)
                else:
                    doc = parser.open_bytes(self.store.get_bytes(manifest.source_key))
                with doc:
                    manifest.analysis.page_count = doc.page_count
                    progress.total = doc.page_count
                    analyze_document(
                        _StopWhenCancelled(doc, lambda: self._check_cancel(job)),
                        manifest,
                        store=self.store,
                        on_progress=on_progress,
                    )
                self._check_cancel(job)
                self.manifests.save_manifest(manifest)
        except ParseError as exc:
            job.stage = JobStage.FAILED
            job.error = exc.message
            job.error_diagnostic = exc.diagnostic
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
            return

        progress.finished_at = _utcnow()
        progress.completed = progress.total

        for page in manifest.pages:
            if page.error:
                job.record_page_error(
                    page.number,
                    page.error,
                    diagnostic=f"stage={page.error_stage}",
                )
            else:
                job.mark_page_done(JobStage.ANALYZING, page.index)

        failed = len(manifest.failed_pages)
        total = manifest.analysis.page_count
        if failed and failed == total:
            job.stage = JobStage.FAILED
            job.error = "None of the pages in this document could be read."
        else:
            # Phase 1 ends at review-ready: the document is decomposed and can
            # be inspected. Translation stages attach here in phase 2.
            job.stage = JobStage.REVIEW_READY
            if failed:
                job.log(
                    "warning",
                    f"{failed} of {total} pages could not be read and are flagged.",
                )
            else:
                job.log("info", f"Analyzed {total} of {total} pages.")
        job.finished_at = _utcnow()
        self.manifests.save_job(job)

    # -- translation --------------------------------------------------------
    def submit_translation(
        self, document_id: str, settings: JobSettings
    ) -> DocumentJob:
        """Queue the full translate -> reconstruct -> verify -> export run."""
        job = DocumentJob(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            settings=settings,
            stage=JobStage.QUEUED,
        )
        job.log("info", "Translation queued.")
        return self._enqueue(job, self._run_translation)

    def _run_translation(self, document_id: str, job_id: str) -> None:
        job = self.manifests.load_job(document_id, job_id)
        if job is None:
            return
        try:
            self._check_cancel(job)
            self._translation_stages(job)
        except JobCancelled:
            self._mark_cancelled(job)
        except Exception as exc:
            log.exception("Translation job %s failed", job_id)
            job.stage = JobStage.FAILED
            job.error = "This document could not be translated."
            job.error_diagnostic = f"{type(exc).__name__}: {exc}"
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
        finally:
            self._job_ended(document_id, job_id)

    def _translation_stages(self, job: DocumentJob) -> None:
        manifest = self.manifests.load_manifest(job.document_id)
        if manifest is None:
            job.stage = JobStage.FAILED
            job.error = "That document could not be found."
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
            return

        settings = job.settings
        try:
            provider = get_translation_provider()
        except (ProviderNotAvailable, TranslationError) as exc:
            job.stage = JobStage.FAILED
            job.error = getattr(exc, "message", str(exc))
            job.error_diagnostic = getattr(exc, "diagnostic", None) or repr(exc)
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
            return

        if not provider.supports(settings.source_language, settings.target_language):
            job.stage = JobStage.FAILED
            job.error = (
                "The translation engine on this server does not support that "
                "language yet. Please choose a different language."
            )
            job.error_diagnostic = (
                f"provider={provider.name} target={settings.target_language}"
            )
            job.finished_at = _utcnow()
            self.manifests.save_job(job)
            return

        import time as _time

        def stop_if_cancelled() -> None:
            self._check_cancel(job)

        last_write = 0.0
        last_manifest_write = _time.monotonic()
        # Pages translated in this run whose translations are not on disk
        # yet. The job only calls a page done once the manifest holding it
        # is saved: a manifest left over from an earlier translation (into
        # another language) also says `translated`, so after a restart the
        # job's word is the only thing that tells the two apart.
        unsaved_translated: list[int] = []

        def save_manifest_and_mark() -> None:
            self.manifests.save_manifest(manifest)
            for index in unsaved_translated:
                job.mark_page_done(JobStage.TRANSLATING, index)
            unsaved_translated.clear()

        def save_throttled(force: bool = False, manifest_too: bool = False) -> None:
            nonlocal last_write, last_manifest_write
            now = _time.monotonic()
            if manifest_too and (
                force or now - last_manifest_write >= _MANIFEST_SAVE_INTERVAL
            ):
                # Manifest first, so the job never claims a page whose
                # translations are not on disk yet.
                last_manifest_write = now
                save_manifest_and_mark()
            if force or now - last_write >= 0.5:
                last_write = now
                self.manifests.save_job(job)

        with document_lock(job.document_id):
            # Read the manifest again now the document is ours. A job queued
            # behind one being stopped waits here while that job finishes its
            # page and saves; the copy read above would be older than what is
            # on disk, and saving it later would undo that write (or an edit
            # made in the meantime).
            manifest = self.manifests.load_manifest(job.document_id)
            if manifest is None:
                job.stage = JobStage.FAILED
                job.error = "That document could not be found."
                job.finished_at = _utcnow()
                self.manifests.save_job(job)
                return
            source_lang = (
                settings.source_language or manifest.analysis.detected_language
            )
            total = len(manifest.pages)

            # --- OCR (only for pages with no usable text layer) ---
            ocr_findings: dict[int, list] = {}
            scanned = pages_needing_ocr(manifest.pages)
            # Scanned pages and pictures are one "Reading pages" step to a
            # teacher, so they share one counter that runs to the end once.
            # Counted separately, the picture pass raised the total and never
            # moved the count: the screen sat on "7 / 22 pages" until
            # translation began, and looked frozen.
            image_pages = (
                [p for p in manifest.pages if p.image_objects]
                if settings.translate_image_text
                else []
            )
            if scanned or image_pages:
                reading = job.stage_progress(JobStage.OCR)
                reading.total = len(scanned) + len(image_pages)
                reading.completed = 0
                reading.started_at = _utcnow()
            if scanned:
                job.stage = JobStage.OCR
                ocr_progress = job.stage_progress(JobStage.OCR)
                ocr_progress.label = "Reading scanned pages"
                job.log("info", f"Reading {len(scanned)} scanned page(s).")
                self.manifests.save_job(job)

                try:
                    ocr_provider = get_ocr_provider()
                except Exception as exc:
                    # Missing OCR is not fatal: pages without a text layer are
                    # carried through unchanged and flagged, and the rest of
                    # the document still translates.
                    ocr_provider = None
                    job.log(
                        "warning",
                        "Scanned pages could not be read; they will be left "
                        "as they are.",
                        detail=str(exc)[:200],
                    )

                if ocr_provider is not None:
                    source = self.store.local_path(manifest.source_key)
                    parser = parser_for_manifest(manifest)
                    reader = (
                        parser.open_path(source)
                        if source is not None
                        else parser.open_bytes(
                            self.store.get_bytes(manifest.source_key)
                        )
                    )
                    with reader:
                        for page in scanned:
                            stop_if_cancelled()
                            try:
                                st = ocr_page(
                                    reader,
                                    page,
                                    ocr_provider,
                                    language=source_lang,
                                )
                                if st.findings:
                                    ocr_findings.setdefault(page.index, []).extend(
                                        st.findings
                                    )
                                # Order matters. Tesseract's page-mode line
                                # detection reads ACROSS a column gutter on a
                                # multi-column design: it emitted the single
                                # line "SERVICE TO HANDS-ON" from two separate
                                # captions. Splitting wide-gap lines first
                                # breaks those apart, and only then are lines
                                # joined into paragraphs. Grouping first fused
                                # the crossed lines into blocks that no later
                                # split could undo.
                                split_page_blocks(page)
                                group_ocr_lines(page)
                            except Exception as exc:
                                log.exception("OCR failed on page %s", page.number)
                                job.record_page_error(
                                    page.number,
                                    f"Page {page.number} could not be read.",
                                    diagnostic=repr(exc),
                                )
                            release_page_memory()
                            ocr_progress.completed += 1
                            save_throttled()
                # Counted as read even when no reader was available, so the
                # bar never stalls short of a step that has moved on.
                ocr_progress.completed = max(ocr_progress.completed, len(scanned))
                self.manifests.save_manifest(manifest)

            # --- Text inside images ---
            # Shares the OCR stage label: to a teacher this is all "reading
            # the page", and the distinction is an implementation detail.
            if image_pages:
                job.stage = JobStage.OCR
                img_progress = job.stage_progress(JobStage.OCR)
                img_progress.label = "Reading words in pictures"
                pages_with_images = image_pages
                job.log(
                    "info", f"Reading text inside {len(pages_with_images)} page(s) of pictures."
                )
                self.manifests.save_job(job)
                try:
                    ocr_provider2 = get_ocr_provider()
                except Exception as exc:
                    # Not fatal, but never quiet: each page with pictures
                    # says its pictures were not read.
                    ocr_provider2 = None
                    job.log(
                        "warning",
                        "Words in pictures could not be read on this server; "
                        "pictures are left as they are.",
                        detail=str(exc)[:200],
                    )
                    for page in pages_with_images:
                        ocr_findings.setdefault(page.index, []).append(
                            _pictures_not_read(page.number, repr(exc))
                        )
                if ocr_provider2 is not None:
                    source = self.store.local_path(manifest.source_key)
                    parser = parser_for_manifest(manifest)
                    reader = (
                        parser.open_path(source)
                        if source is not None
                        else parser.open_bytes(
                            self.store.get_bytes(manifest.source_key)
                        )
                    )
                    label_cache: dict = {}
                    with reader:
                        for page in pages_with_images:
                            stop_if_cancelled()
                            try:
                                st = extract_image_text(
                                    reader,
                                    page,
                                    ocr_provider2,
                                    language=source_lang,
                                    seen_hashes=label_cache,
                                )
                                if st.findings:
                                    ocr_findings.setdefault(page.index, []).extend(
                                        st.findings
                                    )
                            except Exception as exc:
                                log.exception(
                                    "Image text failed on page %s", page.number
                                )
                                ocr_findings.setdefault(page.index, []).append(
                                    _pictures_not_read(page.number, repr(exc))
                                )
                                job.log(
                                    "warning",
                                    f"Words in pictures on page {page.number} "
                                    "could not be read.",
                                    detail=repr(exc)[:200],
                                )
                            release_page_memory()
                            img_progress.completed += 1
                            save_throttled()
                    self.manifests.save_manifest(manifest)
                img_progress.completed = img_progress.total
            if scanned or image_pages:
                job.stage_progress(JobStage.OCR).finished_at = _utcnow()

            # --- TRANSLATING ---
            stop_if_cancelled()
            job.stage = JobStage.TRANSLATING
            job.started_at = job.started_at or _utcnow()
            progress = job.stage_progress(JobStage.TRANSLATING)
            progress.total = total
            progress.completed = 0
            progress.started_at = _utcnow()
            job.log("info", f"Translating {total} page(s).")
            self.manifests.save_job(job)

            page_findings: dict[int, list] = {}
            # Blocks that needed translating, and those the translation
            # service gave nothing back for, across the pages run now.
            eligible = provider_failed = 0
            for page in manifest.pages:
                stop_if_cancelled()
                # Done only if the manifest agrees. The job is only told
                # once the manifest is saved (save_manifest_and_mark), and
                # this check is a second guard against a manifest that was
                # lost or replaced: skipping such a page would export it in
                # English.
                if job.is_page_done(JobStage.TRANSLATING, page.index) and page.translated:
                    progress.completed += 1
                    continue
                try:
                    # Both OCR stages have run by now, so this is the first
                    # point where the same glyphs read twice can be seen and
                    # collapsed -- before anything is translated or drawn.
                    drop_duplicate_text(page)
                    split_page_blocks(page)
                    apply_reading_order(page)
                    classify_page(page, is_first_page=(page.index == 0))
                    stats = translate_page(
                        page,
                        provider,
                        source_lang=source_lang,
                        target_lang=settings.target_language,
                        output_mode=settings.output_mode,
                    )
                    page_findings.setdefault(page.index, []).extend(stats.findings)
                    eligible += stats.eligible
                    provider_failed += stats.provider_failed
                    unsaved_translated.append(page.index)
                except Exception as exc:
                    # One page failing must not stop the document.
                    log.exception("Page %s translation failed", page.number)
                    page.error = f"Page {page.number} could not be translated."
                    page.error_stage = "TRANSLATING"
                    job.record_page_error(
                        page.number, page.error, diagnostic=repr(exc)
                    )
                release_page_memory()
                progress.completed += 1
                save_throttled(manifest_too=True)
            progress.finished_at = _utcnow()
            save_manifest_and_mark()
            self.manifests.save_job(job)

            if eligible and provider_failed >= eligible * UNTRANSLATED_FAIL_SHARE:
                # The service did not translate the document. Offering it as
                # "Complete" would hand a teacher who cannot read the target
                # language a file that is still in English.
                job.stage = JobStage.FAILED
                job.error = untranslated_reason(
                    provider, source_lang, settings.target_language
                )
                job.error_diagnostic = (
                    f"provider={provider.name} untranslated={provider_failed} "
                    f"of {eligible} blocks"
                )
                job.log("error", job.error)
                job.finished_at = _utcnow()
                self.manifests.save_job(job)
                return

            # --- RECONSTRUCTING ---
            stop_if_cancelled()
            job.stage = JobStage.RECONSTRUCTING
            rebuild_progress = job.stage_progress(JobStage.RECONSTRUCTING)
            rebuild_progress.total = total
            rebuild_progress.completed = 0
            rebuild_progress.started_at = _utcnow()
            job.log("info", "Rebuilding pages.")
            self.manifests.save_job(job)

            def on_render(done: int, of: int) -> None:
                stop_if_cancelled()
                release_page_memory()
                rebuild_progress.completed = done
                rebuild_progress.total = of or total
                save_throttled()

            export = reconstruct_and_export(
                manifest,
                store=self.store,
                target_lang=settings.target_language,
                on_progress=on_render,
                check_cancel=stop_if_cancelled,
            )
            rebuild_progress.completed = rebuild_progress.total
            rebuild_progress.finished_at = _utcnow()

            # --- VERIFYING ---
            stop_if_cancelled()
            job.stage = JobStage.VERIFYING
            verify_progress = job.stage_progress(JobStage.VERIFYING)
            verify_progress.total = total
            verify_progress.completed = 0
            verify_progress.started_at = _utcnow()
            self.manifests.save_job(job)

            for page in manifest.pages:
                stop_if_cancelled()
                extra = list(ocr_findings.get(page.index, []))
                extra.extend(page_findings.get(page.index, []))
                extra.extend(export.findings_by_page.get(page.index, []))
                try:
                    verify_page(page, extra)
                except Exception as exc:
                    log.warning("QA failed on page %s: %r", page.number, exc)
                release_page_memory()
                verify_progress.completed += 1
                save_throttled()
            verify_progress.finished_at = _utcnow()

            # --- EXPORTING + completion gate ---
            stop_if_cancelled()
            job.stage = JobStage.EXPORTING
            self.manifests.save_job(job)

            # Reopen the export with a reader for that format. This is the
            # last line of defence before a teacher downloads the file: an
            # export that cannot be reopened must fail the gate rather than
            # be offered (spec sec. 53).
            opens = False
            try:
                fmt = format_of(manifest)
                # An Office file is small and its readers take bytes. A PDF
                # can be large, so it is never read into memory here: it is
                # opened from its file, as the rebuild already did.
                export_bytes = (
                    self.store.get_bytes(export.export_key)
                    if fmt in (SourceFormat.DOCX, SourceFormat.PPTX, SourceFormat.XLSX)
                    else b""
                )
                if fmt is SourceFormat.DOCX:
                    from ..ingest.docx_parser import DocxParser

                    with DocxParser().open_bytes(export_bytes) as check:
                        opens = check.paragraph_count == export.output_unit_count
                elif fmt is SourceFormat.PPTX:
                    from ..ingest.pptx_parser import PptxParser

                    with PptxParser().open_bytes(export_bytes) as check:
                        opens = (
                            check.page_count == export.output_page_count
                            and check.paragraph_count == export.output_unit_count
                        )
                elif fmt is SourceFormat.XLSX:
                    from ..ingest.xlsx_parser import XlsxParser

                    with XlsxParser().open_bytes(export_bytes) as check:
                        opens = (
                            check.page_count == export.output_page_count
                            and check.cell_count == export.output_unit_count
                        )
                else:
                    import fitz

                    path = self.store.local_path(export.export_key)
                    check = (
                        fitz.open(str(path), filetype="pdf")
                        if path is not None
                        else fitz.open(
                            stream=self.store.get_bytes(export.export_key),
                            filetype="pdf",
                        )
                    )
                    try:
                        opens = check.page_count == export.output_page_count
                    finally:
                        check.close()
                del export_bytes
            except Exception as exc:
                log.error("Exported document failed to reopen: %r", exc)

            gate = check_completion(manifest, export, opens)
            job.export_key = export.export_key
            self.manifests.save_manifest(manifest)

            if not gate.passed:
                job.stage = JobStage.FAILED
                job.error = gate.reasons[0]
                job.error_diagnostic = "; ".join(gate.reasons)
                job.log("error", "Completion checks did not pass.")
                job.finished_at = _utcnow()
                self.manifests.save_job(job)
                return
            # The last word on whether the job finished. Checked and saved
            # under the guard cancel() holds, so a stop asked for up to now
            # ends the job CANCELLED, and one asked for after finds it
            # COMPLETE and reports that nothing was stopped.
            with self._guard:
                stop_if_cancelled()
                job.stage = JobStage.COMPLETE
                job.log(
                    "info",
                    f"Complete: {export.output_page_count} of "
                    f"{manifest.source_page_count} pages.",
                )
                job.finished_at = _utcnow()
                self.manifests.save_job(job)

    # -- resume after a restart ---------------------------------------------
    def resume_interrupted(self) -> list[str]:
        """Pick up jobs that were mid-flight when the process stopped.

        Page-level state is already persisted after every page, which is what
        makes this cheap: a resumed job skips the pages it finished and
        continues. Without this, a deploy, a crash, or a host suspending an
        idle machine silently abandons a job forever -- and the module
        docstring's promise of resumability would not be true end to end.

        Returns the job ids that were requeued.
        """
        resumed: list[str] = []
        try:
            document_ids = {
                key.split("/")[1]
                for key in self.store.list_prefix("documents")
                if len(key.split("/")) > 1
            }
        except Exception as exc:
            log.warning("Could not scan for interrupted jobs: %r", exc)
            return resumed

        for document_id in sorted(document_ids):
            try:
                jobs = self.manifests.list_jobs(document_id)
            except Exception:
                continue
            if not jobs:
                continue
            job = jobs[-1]
            if job.is_terminal:
                continue
            # REVIEW_READY is not terminal, but nothing is in flight: analysis
            # finished and the job is waiting for a person to choose a
            # language. Resuming it would pointlessly re-analyse the document.
            if job.stage is JobStage.REVIEW_READY:
                continue
            # The teacher stopped this job before the restart, and it never
            # reached a checkpoint to end itself. Running it again would
            # spend memory and translation quota on a file they abandoned.
            if self._cancel_marked(document_id, job.job_id):
                self._mark_cancelled(job)
                self._clear_cancel_marker(document_id, job.job_id)
                log.info("Job %s was stopped before the restart", job.job_id)
                continue

            if job.resume_count >= MAX_RESUME_ATTEMPTS:
                # Give up loudly rather than restart-looping forever.
                job.stage = JobStage.FAILED
                if (job.settings.target_language or "und") == "und":
                    # Still being opened: there is nothing to translate yet.
                    job.error = (
                        "This file could not be opened after the service "
                        "restarted several times. Please upload it again."
                    )
                else:
                    job.error = (
                        "This document could not be finished after the service "
                        "restarted several times. Please try translating it again."
                    )
                job.error_diagnostic = f"resume_count={job.resume_count}"
                job.finished_at = _utcnow()
                self.manifests.save_job(job)
                log.error(
                    "Job %s exceeded resume attempts; marked failed", job.job_id
                )
                continue

            job.resume_count += 1
            job.log(
                "info",
                "Picking this document up again after the service restarted.",
            )
            self.manifests.save_job(job)

            # A translation job goes back to the translation stages whatever
            # stage it had reached -- including QUEUED, when it was waiting
            # for a free worker. Sending that one to analysis would drop the
            # teacher's request without a word. Analysis jobs never carry a
            # target language.
            is_translation = (job.settings.target_language or "und") != "und"
            work = (
                self._run_translation
                if is_translation
                or job.stage
                in (
                    JobStage.TRANSLATING,
                    JobStage.RECONSTRUCTING,
                    JobStage.VERIFYING,
                    JobStage.EXPORTING,
                    JobStage.OCR,
                )
                else self._run_analysis
            )
            with self._guard:
                if self._shutdown:
                    break
                self._active[document_id] = job.job_id
                self._futures[job.job_id] = self._pool.submit(
                    work, document_id, job.job_id
                )
            resumed.append(job.job_id)

        if resumed:
            log.info("Resumed %d interrupted job(s): %s", len(resumed), resumed)
        return resumed

    # -- retry -------------------------------------------------------------
    def retry_pages(
        self, document_id: str, job_id: str, page_numbers: list[int]
    ) -> DocumentJob | None:
        """Re-run analysis for specific pages only (spec sec. 48)."""
        job = self.manifests.load_job(document_id, job_id)
        manifest = self.manifests.load_manifest(document_id)
        if job is None or manifest is None:
            return None

        indices = [n - 1 for n in page_numbers if 1 <= n <= len(manifest.pages)]
        if not indices:
            return job

        self._pool.submit(self._retry_worker, document_id, job_id, indices)
        return job

    def _retry_worker(
        self, document_id: str, job_id: str, indices: list[int]
    ) -> None:
        job = self.manifests.load_job(document_id, job_id)
        manifest = self.manifests.load_manifest(document_id)
        if job is None or manifest is None:
            return
        job.log("info", f"Retrying {len(indices)} page(s).")
        self.manifests.save_job(job)
        parser = parser_for_manifest(manifest)
        try:
            with document_lock(document_id):
                source = self.store.local_path(manifest.source_key)
                doc = (
                    parser.open_path(source)
                    if source is not None
                    else parser.open_bytes(self.store.get_bytes(manifest.source_key))
                )
                with doc:
                    analyze_document(
                        doc,
                        manifest,
                        store=self.store,
                        only_indices=indices,
                    )
                for i in indices:
                    page = manifest.page(i)
                    if page and not page.error:
                        page.error = None
                        page.error_stage = None
                        job.clear_page_error(i + 1)
                        job.mark_page_done(JobStage.ANALYZING, i)
                self.manifests.save_manifest(manifest)
        except Exception as exc:
            log.exception("Retry failed for %s pages %s", document_id, indices)
            job.log("error", "Retry did not succeed.", detail=str(exc))
        if not manifest.failed_pages and job.stage == JobStage.FAILED:
            job.stage = JobStage.REVIEW_READY
            job.error = None
        self.manifests.save_job(job)

    # -- lifecycle ---------------------------------------------------------
    def shutdown(self, wait: bool = True) -> None:
        with self._guard:
            self._shutdown = True
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
