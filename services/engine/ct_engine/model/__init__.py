"""Typed page object model for Classroom Translate."""

from .document import (
    DocumentAnalysis,
    DocumentManifest,
    DocumentMetadata,
    Page,
    PageQA,
    PageRenderRefs,
    QAFinding,
)
from .enums import (
    JobStage,
    ObjectKind,
    Orientation,
    OutputLanguageMode,
    PageStatus,
    PageTextMode,
    ProcessingMode,
    SemanticRole,
    TextAlignment,
    TextSource,
    TranslationStatus,
)
from .geometry import NormRect, Point, Rect
from .job import DocumentJob, JobEvent, JobSettings, PageError, StageProgress
from .objects import (
    FormFieldObject,
    ImageObject,
    PageDiagnostics,
    ProtectedRegion,
    ResponseArea,
    TableObject,
    TextLine,
    TextObject,
    TextSpan,
    TextStyle,
    VectorObject,
)

__all__ = [
    "DocumentAnalysis", "DocumentManifest", "DocumentMetadata", "Page", "PageQA",
    "PageRenderRefs", "QAFinding", "JobStage", "ObjectKind", "Orientation",
    "OutputLanguageMode", "PageStatus", "PageTextMode", "ProcessingMode",
    "SemanticRole", "TextAlignment", "TextSource", "TranslationStatus",
    "NormRect", "Point", "Rect", "DocumentJob", "JobEvent", "JobSettings",
    "PageError", "StageProgress", "FormFieldObject", "ImageObject",
    "PageDiagnostics", "ProtectedRegion", "ResponseArea", "TableObject",
    "TextLine", "TextObject", "TextSpan", "TextStyle", "VectorObject",
]
