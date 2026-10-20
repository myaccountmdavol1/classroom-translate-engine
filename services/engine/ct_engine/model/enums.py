"""Enumerations shared across the page object model and the pipeline."""

from __future__ import annotations

from enum import Enum


class ObjectKind(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    VECTOR = "VECTOR"
    TABLE = "TABLE"
    LINE = "LINE"
    SHAPE = "SHAPE"
    FORM_FIELD = "FORM_FIELD"
    UNKNOWN = "UNKNOWN"


class SemanticRole(str, Enum):
    """Instructional role of a text block.

    Drives translation behavior, typography, and QA expectations. `OTHER` is
    the honest default -- we never guess a role we cannot justify from
    evidence, because a wrong role silently changes translation handling.
    """

    TITLE = "TITLE"
    HEADING = "HEADING"
    DIRECTION = "DIRECTION"
    QUESTION = "QUESTION"
    ANSWER_CHOICE = "ANSWER_CHOICE"
    READING_PASSAGE = "READING_PASSAGE"
    CAPTION = "CAPTION"
    TABLE_CELL = "TABLE_CELL"
    IMAGE_LABEL = "IMAGE_LABEL"
    FOOTER = "FOOTER"
    HEADER = "HEADER"
    PAGE_NUMBER = "PAGE_NUMBER"
    FORMULA = "FORMULA"
    RESPONSE_LABEL = "RESPONSE_LABEL"
    OTHER = "OTHER"


class TranslationStatus(str, Enum):
    PENDING = "PENDING"            # eligible, not yet processed
    NOT_REQUIRED = "NOT_REQUIRED"  # no translatable content (e.g. pure digits)
    PROTECTED = "PROTECTED"        # must survive byte-identical (formula, URL)
    TRANSLATED = "TRANSLATED"
    MANUAL_EDIT = "MANUAL_EDIT"    # human overrode the machine translation
    FAILED = "FAILED"


class TextAlignment(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    JUSTIFY = "JUSTIFY"
    UNKNOWN = "UNKNOWN"


class TextSource(str, Enum):
    """Where a text object's characters came from.

    QA treats these very differently: NATIVE text is trustworthy, OCR text
    carries confidence and may be wrong, and IMAGE_OCR text lives inside a
    raster asset and needs the inpainting pipeline rather than a text overlay.
    """

    NATIVE = "NATIVE"
    OCR = "OCR"
    IMAGE_OCR = "IMAGE_OCR"
    # Created by a person in the review editor. There is no source text
    # beneath it, so reconstruction must draw it WITHOUT redacting or
    # covering anything -- needed where OCR missed stylised text entirely.
    MANUAL = "MANUAL"


class PageTextMode(str, Enum):
    """How a page's text must be obtained."""

    NATIVE = "NATIVE"      # reliable embedded text layer
    SCANNED = "SCANNED"    # no usable text layer; OCR required
    MIXED = "MIXED"        # some native text, but suspiciously sparse
    EMPTY = "EMPTY"        # genuinely blank page (no text, no images)


class PageStatus(str, Enum):
    READY = "READY"
    REVIEW_SUGGESTED = "REVIEW_SUGGESTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class JobStage(str, Enum):
    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    OCR = "OCR"
    TRANSLATING = "TRANSLATING"
    RECONSTRUCTING = "RECONSTRUCTING"
    VERIFYING = "VERIFYING"
    REVIEW_READY = "REVIEW_READY"
    EXPORTING = "EXPORTING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OutputLanguageMode(str, Enum):
    TRANSLATED_ONLY = "TRANSLATED_ONLY"
    BILINGUAL_TRANSLATION_FIRST = "BILINGUAL_TRANSLATION_FIRST"
    BILINGUAL_ENGLISH_FIRST = "BILINGUAL_ENGLISH_FIRST"


class ProcessingMode(str, Enum):
    STANDARD = "STANDARD"
    HIGH_FIDELITY = "HIGH_FIDELITY"


class Orientation(str, Enum):
    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"
    SQUARE = "SQUARE"
