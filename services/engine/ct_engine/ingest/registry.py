"""Choosing a parser for an upload.

Separated from the parsers themselves so every stage asks the same question
the same way. The job runner reopens the source several times -- to analyze,
to OCR, to retry one page, to rebuild -- and each of those used to name
`PdfParser` directly. With more than one format that would be four places to
forget.

Selection is by extension first and content type second, matching
`DocumentParser.can_parse`. Browsers and learning-management systems send
Office files under a variety of generic content types, so the filename is the
more reliable signal; the content type decides only when there is no usable
extension.
"""

from __future__ import annotations

from pathlib import Path

from ..util.logging import get_logger
from .base import DocumentParser, ParseError
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .pptx_parser import PptxParser
from .xlsx_parser import XlsxParser

log = get_logger(__name__)

# Order matters only for the content-type fallback; extensions are unique.
PARSERS: tuple[DocumentParser, ...] = (
    PdfParser(),
    DocxParser(),
    PptxParser(),
    XlsxParser(),
)


def parser_for(content_type: str | None, filename: str | None) -> DocumentParser:
    """Return the parser for this upload, or raise with a teacher-facing message."""
    for parser in PARSERS:
        if parser.can_parse(content_type, filename):
            return parser
    ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
    raise ParseError(
        f"Classroom Translate cannot open .{ext} files yet."
        if ext
        else "Classroom Translate could not tell what kind of file that is.",
        diagnostic=f"content_type={content_type!r} filename={filename!r}",
    )


def parser_for_manifest(manifest) -> DocumentParser:
    """Convenience for the stages, which always have a manifest to hand."""
    return parser_for(manifest.content_type, manifest.filename)


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    ext for p in PARSERS for ext in p.supported_extensions
)
