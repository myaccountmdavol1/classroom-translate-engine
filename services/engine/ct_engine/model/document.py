"""Document and page containers, plus the page manifest."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from .enums import (
    Orientation,
    PageStatus,
    PageTextMode,
    SemanticRole,
    TranslationStatus,
)
from .geometry import Rect
from .objects import (
    FormFieldObject,
    ImageObject,
    PageDiagnostics,
    ProtectedRegion,
    ResponseArea,
    TableObject,
    TextObject,
    VectorObject,
)


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class QAFinding(BaseModel):
    """One automated quality observation about a page.

    Findings are additive evidence, never a verdict on their own: page status
    is derived from the whole set so that a single noisy heuristic cannot
    block a document.
    """

    # "error" downgrades a page to REVIEW_REQUIRED, "warning" to
    # REVIEW_SUGGESTED, "info" is informational only.
    severity: str = "info"
    code: str
    message: str
    object_ids: list[str] = Field(default_factory=list)
    bbox: Rect | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PageQA(BaseModel):
    """Result of the automated checks for one page."""

    checks_run: list[str] = Field(default_factory=list)
    findings: list[QAFinding] = Field(default_factory=list)
    # Internal heuristic in [0, 1]. Explicitly NOT a scientific measure of
    # translation quality; it is a triage signal for reviewer attention
    # (spec sec. 30).
    layout_confidence: float = 1.0
    status: PageStatus = PageStatus.PENDING

    @property
    def errors(self) -> list[QAFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[QAFinding]:
        return [f for f in self.findings if f.severity == "warning"]


class PageRenderRefs(BaseModel):
    """Storage keys for rasterized previews of a page."""

    source_preview_key: str | None = None
    source_thumb_key: str | None = None
    translated_preview_key: str | None = None
    translated_thumb_key: str | None = None
    # The live editor's background: everything but the translated text.
    plate_key: str | None = None
    preview_dpi: int = 0
    preview_width_px: int = 0
    preview_height_px: int = 0


class Page(BaseModel):
    """One page of the document, fully decomposed.

    `index` is 0-based and is the canonical identity of a page throughout the
    pipeline. `number` is the 1-based label shown to teachers. Every stage
    keys off `index`, which is what makes page correspondence checkable
    (spec sec. 46).
    """

    index: int
    number: int

    width: float
    height: float
    # /Rotate value from the page dictionary, in degrees. Retained for export;
    # object coordinates are already visually correct without applying it.
    rotation: int = 0
    orientation: Orientation = Orientation.PORTRAIT

    # False for reflowable sources (DOCX sections). Their object boxes are
    # a synthesised reading-order flow, NOT measured layout: nothing has been
    # laid out yet, because Word does that when the file is opened. Fitting,
    # reflow and overlap QA are all meaningless on such a page and are
    # skipped -- running them would produce confident nonsense.
    has_fixed_layout: bool = True

    text_mode: PageTextMode = PageTextMode.NATIVE
    # Fraction of the page area covered by native text, used to distinguish a
    # genuinely sparse page from a scanned one.
    text_coverage: float = 0.0
    is_scanned: bool = False
    needs_ocr: bool = False

    text_objects: list[TextObject] = Field(default_factory=list)
    image_objects: list[ImageObject] = Field(default_factory=list)
    vector_objects: list[VectorObject] = Field(default_factory=list)
    table_objects: list[TableObject] = Field(default_factory=list)
    form_fields: list[FormFieldObject] = Field(default_factory=list)

    response_areas: list[ResponseArea] = Field(default_factory=list)
    protected_regions: list[ProtectedRegion] = Field(default_factory=list)

    renders: PageRenderRefs = Field(default_factory=PageRenderRefs)
    qa: PageQA = Field(default_factory=PageQA)
    diagnostics: PageDiagnostics = Field(default_factory=PageDiagnostics)

    # Per-stage completion flags. These are what the completion gate reads;
    # a page that merely "didn't crash" is not a page that was processed.
    analyzed: bool = False
    ocr_done: bool = False
    translated: bool = False
    reconstructed: bool = False
    verified: bool = False

    # Bounded stack of previous editable states, for undo. Only the fields a
    # person can change are stored, so a snapshot is small enough to keep in
    # the manifest; ids of person-created blocks are recorded too, so undoing
    # an "add" removes the block again.
    edit_history: list[dict] = Field(default_factory=list)
    redo_history: list[dict] = Field(default_factory=list)

    # Bumped whenever this page is re-rendered. The viewer appends it to
    # preview URLs so an edited page is not served from browser cache.
    revision: int = 0

    # Teacher-facing error text when a stage failed on this page. A failed
    # page never aborts the document; it is recorded and retried (spec sec. 3).
    error: str | None = None
    error_stage: str | None = None
    attempts: int = 0

    @property
    def bounds(self) -> Rect:
        return Rect(x0=0.0, y0=0.0, x1=self.width, y1=self.height)

    @property
    def all_objects(self) -> list[Any]:
        return [
            *self.text_objects,
            *self.image_objects,
            *self.vector_objects,
            *self.table_objects,
            *self.form_fields,
        ]

    @property
    def translatable_objects(self) -> list[TextObject]:
        """Text objects eligible for translation.

        Excludes fully protected blocks and blocks with no translatable
        content, so that progress counts reflect real work rather than
        inflated totals.
        """
        return [
            t
            for t in self.text_objects
            if not t.protected
            and t.translation_status
            not in (TranslationStatus.NOT_REQUIRED, TranslationStatus.PROTECTED)
        ]

    def find_text_object(self, object_id: str) -> TextObject | None:
        for t in self.text_objects:
            if t.id == object_id:
                return t
        # Image-embedded OCR regions are text objects too, and a manual edit
        # may target one of them.
        for img in self.image_objects:
            for region in img.ocr_regions:
                if region.id == object_id:
                    return region
        return None

    def recompute_diagnostics(self) -> PageDiagnostics:
        """Recount the page's contents.

        `extra` is carried over, not recomputed. Everything else here is a
        count derived from the page, but `extra` holds observations a stage
        made and cannot make again -- an asset that failed to copy, a
        truncated vector list, a preview that would not render. Rebuilding
        the object without them silently discarded exactly the evidence the
        reviewer needs.
        """
        carried = dict(self.diagnostics.extra)
        d = PageDiagnostics(
            text_objects=len(self.text_objects),
            image_objects=len(self.image_objects),
            vector_objects=len(self.vector_objects),
            table_objects=len(self.table_objects),
            form_fields=len(self.form_fields),
            ocr_regions=sum(len(i.ocr_regions) for i in self.image_objects)
            + sum(1 for t in self.text_objects if t.source.value == "OCR"),
            protected_regions=len(self.protected_regions),
            response_areas=len(self.response_areas),
            translated_regions=sum(
                1
                for t in self.text_objects
                if t.translation_status
                in (TranslationStatus.TRANSLATED, TranslationStatus.MANUAL_EDIT)
            ),
            potential_collisions=sum(
                1 for f in self.qa.findings if f.code.startswith("collision")
            ),
            native_chars=sum(
                t.char_count for t in self.text_objects if t.source.value == "NATIVE"
            ),
            ocr_chars=sum(
                t.char_count for t in self.text_objects if t.source.value == "OCR"
            ),
        )
        d.extra = carried
        self.diagnostics = d
        return d


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    is_encrypted: bool = False
    is_form: bool = False
    pdf_version: str | None = None


class DocumentAnalysis(BaseModel):
    """Everything learned about the source document before translation.

    This is what the configure screen shows the teacher, and what the planner
    uses to decide whether OCR is needed.
    """

    page_count: int = 0
    # ISO-639-1 where detectable, else None. Never guessed from filename.
    detected_language: str | None = None
    detected_language_confidence: float = 0.0
    language_sample_chars: int = 0

    has_selectable_text: bool = False
    scanned_page_count: int = 0
    native_page_count: int = 0
    empty_page_count: int = 0
    needs_ocr: bool = False

    total_text_objects: int = 0
    total_images: int = 0
    # Pictures held in the source package, for formats that carry their
    # artwork through untouched instead of parsing it (DOCX). `total_images`
    # counts parsed page objects and is therefore 0 for those.
    embedded_media_count: int = 0
    total_vectors: int = 0
    total_characters: int = 0

    # Distinct (width, height) pairs in points, most common first.
    page_sizes: list[dict[str, Any]] = Field(default_factory=list)
    has_mixed_page_sizes: bool = False
    has_mixed_orientation: bool = False
    orientation_counts: dict[str, int] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)


class DocumentManifest(BaseModel):
    """The durable, complete description of a document in the pipeline.

    Persisted after every stage so processing is resumable: a crash at page 72
    of 100 loses nothing but page 72 (spec sec. 40).
    """

    document_id: str
    filename: str
    content_type: str
    byte_size: int
    sha256: str

    source_key: str
    original_upload_key: str | None = None
    original_sha256: str | None = None
    original_content_type: str | None = None
    # A Word or PowerPoint file shown as pages: `source_key` holds the PDF
    # LibreOffice made from it, which every page tool works on, and this
    # holds the original file, which the download is written back into.
    # Read from `word_source_key` too, its name before slides were added.
    office_source_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("office_source_key", "word_source_key"),
    )
    # "docx" or "pptx": what `office_source_key` is.
    office_format: str = "docx"
    created_at: _dt.datetime = Field(default_factory=_utcnow)
    updated_at: _dt.datetime = Field(default_factory=_utcnow)

    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    analysis: DocumentAnalysis = Field(default_factory=DocumentAnalysis)
    pages: list[Page] = Field(default_factory=list)

    reviewed_findings: list[str] = Field(default_factory=list)
    schema_version: int = 1

    # --- Counters the completion gate checks (spec sec. 3 / 46) ---
    @property
    def source_page_count(self) -> int:
        return self.analysis.page_count

    @property
    def processed_page_count(self) -> int:
        return sum(1 for p in self.pages if p.analyzed)

    @property
    def translated_page_count(self) -> int:
        return sum(1 for p in self.pages if p.translated)

    @property
    def rendered_page_count(self) -> int:
        return sum(1 for p in self.pages if p.reconstructed)

    @property
    def verified_page_count(self) -> int:
        return sum(1 for p in self.pages if p.verified)

    @property
    def failed_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.error]

    @property
    def pages_needing_review(self) -> list[int]:
        return [
            p.number
            for p in self.pages
            if p.qa.status in (PageStatus.REVIEW_REQUIRED, PageStatus.REVIEW_SUGGESTED)
        ]

    def page(self, index: int) -> Page | None:
        for p in self.pages:
            if p.index == index:
                return p
        return None

    def touch(self) -> None:
        self.updated_at = _utcnow()


# Roles whose text we never translate even when it looks like language.
NON_TRANSLATABLE_ROLES: frozenset[SemanticRole] = frozenset(
    {SemanticRole.PAGE_NUMBER, SemanticRole.FORMULA}
)
