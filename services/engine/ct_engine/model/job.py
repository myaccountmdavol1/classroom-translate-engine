"""Durable job model.

A 100-page document must not depend on one HTTP request, one model call or a
browser staying open (spec sec. 39). A job is therefore a persisted record
that a worker advances stage by stage, writing progress after each page.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from pydantic import BaseModel, Field

from .enums import JobStage, OutputLanguageMode, ProcessingMode


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


# Stages shown to teachers, in order, with plain-language labels. The UI never
# invents its own copy for these -- the engine owns the vocabulary so that
# "OCR" is never shown to a teacher (spec sec. 57).
STAGE_LABELS: dict[JobStage, str] = {
    JobStage.QUEUED: "Waiting to start",
    JobStage.ANALYZING: "Reading your document",
    JobStage.OCR: "Reading pages",
    JobStage.TRANSLATING: "Translating",
    JobStage.RECONSTRUCTING: "Preparing pages",
    JobStage.VERIFYING: "Checking pages",
    JobStage.REVIEW_READY: "Ready for review",
    JobStage.EXPORTING: "Preparing document",
    JobStage.COMPLETE: "Complete",
    JobStage.FAILED: "Could not be completed",
    JobStage.CANCELLED: "Cancelled",
}


class JobSettings(BaseModel):
    """Teacher-chosen options for one translation job."""

    target_language: str
    source_language: str | None = None  # None = auto-detect
    output_mode: OutputLanguageMode = OutputLanguageMode.TRANSLATED_ONLY
    processing_mode: ProcessingMode = ProcessingMode.HIGH_FIDELITY
    translate_image_text: bool = True
    translate_handwriting: bool = False


class StageProgress(BaseModel):
    """Real counters for one stage. Never synthesised or time-based."""

    stage: JobStage
    label: str = ""
    total: int = 0
    completed: int = 0
    started_at: _dt.datetime | None = None
    finished_at: _dt.datetime | None = None

    @property
    def fraction(self) -> float:
        return (self.completed / self.total) if self.total else 0.0


class JobEvent(BaseModel):
    """Append-only audit trail entry."""

    at: _dt.datetime = Field(default_factory=_utcnow)
    stage: JobStage
    level: str = "info"  # info | warning | error
    message: str
    page_number: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PageError(BaseModel):
    """A per-page failure that did not stop the document."""

    page_number: int
    stage: JobStage
    # Plain-language text a teacher can act on, e.g. "Page 27 could not be
    # read because the scan was too faint." Never a stack trace.
    message: str
    # Full technical detail, kept out of the teacher-facing surface.
    diagnostic: str | None = None
    at: _dt.datetime = Field(default_factory=_utcnow)
    retryable: bool = True
    attempts: int = 0


class DocumentJob(BaseModel):
    job_id: str
    document_id: str
    settings: JobSettings

    stage: JobStage = JobStage.QUEUED
    created_at: _dt.datetime = Field(default_factory=_utcnow)
    updated_at: _dt.datetime = Field(default_factory=_utcnow)
    started_at: _dt.datetime | None = None
    finished_at: _dt.datetime | None = None

    progress: list[StageProgress] = Field(default_factory=list)
    events: list[JobEvent] = Field(default_factory=list)
    page_errors: list[PageError] = Field(default_factory=list)

    # Set only when the whole job could not proceed.
    error: str | None = None
    error_diagnostic: str | None = None

    export_key: str | None = None
    report_key: str | None = None
    # Set when a manual edit changes a page after the PDF was built. The
    # download rebuilds on demand rather than on every keystroke: rebuilding
    # a 100-page export costs ~16s, so doing it per edit would make the
    # editor unusable and would repeat the work for every correction.
    export_stale: bool = False
    # How many times this job has been picked up again after a restart. Capped
    # so a job that crashes on every boot cannot loop forever.
    resume_count: int = 0

    # Indices of pages already completed per stage, so a resumed job skips
    # finished work instead of redoing it (spec sec. 40).
    completed_pages: dict[str, list[int]] = Field(default_factory=dict)

    def stage_progress(self, stage: JobStage) -> StageProgress:
        for p in self.progress:
            if p.stage == stage:
                return p
        sp = StageProgress(stage=stage, label=STAGE_LABELS.get(stage, stage.value))
        self.progress.append(sp)
        return sp

    def mark_page_done(self, stage: JobStage, page_index: int) -> None:
        bucket = self.completed_pages.setdefault(stage.value, [])
        if page_index not in bucket:
            bucket.append(page_index)

    def is_page_done(self, stage: JobStage, page_index: int) -> bool:
        return page_index in self.completed_pages.get(stage.value, [])

    def log(
        self,
        level: str,
        message: str,
        *,
        page_number: int | None = None,
        **details: Any,
    ) -> JobEvent:
        ev = JobEvent(
            stage=self.stage,
            level=level,
            message=message,
            page_number=page_number,
            details=details,
        )
        self.events.append(ev)
        # Keep the trail bounded; a 100-page job with per-page events would
        # otherwise grow without limit across retries.
        if len(self.events) > 2000:
            del self.events[: len(self.events) - 2000]
        self.updated_at = _utcnow()
        return ev

    def record_page_error(
        self,
        page_number: int,
        message: str,
        *,
        diagnostic: str | None = None,
        retryable: bool = True,
    ) -> PageError:
        for existing in self.page_errors:
            if existing.page_number == page_number and existing.stage == self.stage:
                existing.message = message
                existing.diagnostic = diagnostic
                existing.attempts += 1
                existing.at = _utcnow()
                return existing
        err = PageError(
            page_number=page_number,
            stage=self.stage,
            message=message,
            diagnostic=diagnostic,
            retryable=retryable,
            attempts=1,
        )
        self.page_errors.append(err)
        return err

    def clear_page_error(self, page_number: int) -> None:
        self.page_errors = [e for e in self.page_errors if e.page_number != page_number]

    @property
    def is_terminal(self) -> bool:
        return self.stage in (JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED)

    def touch(self) -> None:
        self.updated_at = _utcnow()
