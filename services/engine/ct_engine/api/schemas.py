"""API response shapes.

Deliberately not the internal model. A 100-page manifest with every span of
every block is megabytes; the viewer needs one page at a time. So the wire
format is: a light document summary with per-page rows, plus a detail
endpoint for the page currently on screen.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..languages import Language, source_language_name
from ..model.formats import format_of
from ..model.document import DocumentManifest, Page
from ..model.enums import JobStage, PageStatus
from ..model.job import STAGE_LABELS, DocumentJob


class LanguageOut(Language):
    pass


class PageSummary(BaseModel):
    index: int
    number: int
    width: float
    height: float
    rotation: int
    orientation: str
    text_mode: str
    needs_ocr: bool
    status: str
    layout_confidence: float
    text_objects: int
    images: int
    vectors: int
    response_areas: int
    characters: int
    has_preview: bool
    # False for a Word section: there is no page image and no layout to
    # review against, so the viewer shows the text itself instead.
    has_fixed_layout: bool = True
    # What one of these is called to a teacher. Carried per page rather than
    # derived in the client, so "page", "section" and "slide" are decided in
    # one place and cannot disagree between the viewer and the report.
    unit_noun: str = "page"
    revision: int = 0
    error: str | None = None

    @classmethod
    def of(cls, page: Page, unit_noun: str = "page") -> "PageSummary":
        return cls(
            index=page.index,
            number=page.number,
            width=page.width,
            height=page.height,
            rotation=page.rotation,
            orientation=page.orientation.value,
            text_mode=page.text_mode.value,
            needs_ocr=page.needs_ocr,
            status=page.qa.status.value,
            layout_confidence=page.qa.layout_confidence,
            text_objects=len(page.text_objects),
            images=len(page.image_objects),
            vectors=len(page.vector_objects),
            response_areas=len(page.response_areas),
            characters=sum(t.char_count for t in page.text_objects),
            has_preview=bool(page.renders.source_preview_key),
            has_fixed_layout=page.has_fixed_layout,
            unit_noun=unit_noun,
            revision=page.revision,
            error=page.error,
        )


class AnalysisOut(BaseModel):
    page_count: int
    detected_language: str | None
    detected_language_name: str | None
    detected_language_confidence: float
    has_selectable_text: bool
    needs_ocr: bool
    scanned_page_count: int
    native_page_count: int
    empty_page_count: int
    total_text_objects: int
    total_images: int
    total_vectors: int
    total_characters: int
    page_sizes: list[dict[str, Any]]
    has_mixed_page_sizes: bool
    has_mixed_orientation: bool
    orientation_counts: dict[str, int]
    warnings: list[str]


class StageProgressOut(BaseModel):
    stage: str
    label: str
    total: int
    completed: int
    fraction: float


class JobOut(BaseModel):
    job_id: str
    document_id: str
    stage: str
    stage_label: str
    is_terminal: bool
    progress: list[StageProgressOut]
    page_errors: list[dict[str, Any]]
    error: str | None
    export_stale: bool = False
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, job: DocumentJob) -> "JobOut":
        return cls(
            job_id=job.job_id,
            document_id=job.document_id,
            stage=job.stage.value,
            stage_label=STAGE_LABELS.get(job.stage, job.stage.value),
            is_terminal=job.is_terminal,
            progress=[
                StageProgressOut(
                    stage=p.stage.value,
                    label=p.label or STAGE_LABELS.get(p.stage, p.stage.value),
                    total=p.total,
                    completed=p.completed,
                    fraction=round(p.fraction, 4),
                )
                for p in job.progress
            ],
            page_errors=[
                {
                    "page_number": e.page_number,
                    "stage": e.stage.value,
                    "message": e.message,
                    "retryable": e.retryable,
                    "attempts": e.attempts,
                }
                for e in job.page_errors
            ],
            error=job.error,
            export_stale=job.export_stale,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
        )


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    byte_size: int
    content_type: str
    created_at: str
    metadata: dict[str, Any]
    analysis: AnalysisOut
    pages: list[PageSummary]
    # Completion counters -- the numbers the completion gate checks.
    source_page_count: int
    processed_page_count: int
    translated_page_count: int
    rendered_page_count: int
    verified_page_count: int
    failed_pages: list[int]
    pages_needing_review: list[int]
    job: JobOut | None = None

    @classmethod
    def of(
        cls,
        m: DocumentManifest,
        job: DocumentJob | None,
        upload_warnings: list[str] | None = None,
    ) -> "DocumentOut":
        a = m.analysis
        # Warnings found at upload come first, once: the analysis stage
        # rebuilds `a.warnings` from the pages and does not carry them over.
        warnings = list(dict.fromkeys([*(upload_warnings or []), *a.warnings]))
        return cls(
            document_id=m.document_id,
            filename=m.filename,
            byte_size=m.byte_size,
            content_type=m.content_type,
            created_at=m.created_at.isoformat(),
            metadata=m.metadata.model_dump(),
            analysis=AnalysisOut(
                page_count=a.page_count,
                detected_language=a.detected_language,
                detected_language_name=source_language_name(a.detected_language),
                detected_language_confidence=a.detected_language_confidence,
                has_selectable_text=a.has_selectable_text,
                needs_ocr=a.needs_ocr,
                scanned_page_count=a.scanned_page_count,
                native_page_count=a.native_page_count,
                empty_page_count=a.empty_page_count,
                total_text_objects=a.total_text_objects,
                total_images=a.total_images,
                total_vectors=a.total_vectors,
                total_characters=a.total_characters,
                page_sizes=a.page_sizes,
                has_mixed_page_sizes=a.has_mixed_page_sizes,
                has_mixed_orientation=a.has_mixed_orientation,
                orientation_counts=a.orientation_counts,
                warnings=warnings,
            ),
            pages=[PageSummary.of(p, format_of(m).page_noun) for p in m.pages],
            source_page_count=m.source_page_count,
            processed_page_count=m.processed_page_count,
            translated_page_count=m.translated_page_count,
            rendered_page_count=m.rendered_page_count,
            verified_page_count=m.verified_page_count,
            failed_pages=m.failed_pages,
            pages_needing_review=m.pages_needing_review,
            job=JobOut.of(job) if job else None,
        )


class UploadResponse(BaseModel):
    document_id: str
    job_id: str
    filename: str
    byte_size: int
    page_count: int
    warnings: list[str] = []


class ErrorOut(BaseModel):
    """Teacher-facing error. `diagnostic` is only populated in development."""

    error: str
    diagnostic: str | None = None
