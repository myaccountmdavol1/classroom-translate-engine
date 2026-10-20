"""Parser interfaces.

Parsing and rasterizing sit behind these abstractions so a library can be
replaced without touching the pipeline (spec sec. 9). DOCX and PPTX parsers
(phase 8) implement the same two types.

`OpenDocument` owns rasterization as well as parsing because both need the
same open file handle; the higher-level preview/thumbnail policy lives in
`render/`, which only talks to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path

from ..model.document import DocumentMetadata, Page


class ParseError(RuntimeError):
    """Raised when a document cannot be parsed at all.

    Carries a teacher-facing message separate from the technical detail, so
    the API never leaks a stack trace into the UI (spec sec. 48).
    """

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic


class PageParseError(ParseError):
    """Raised when one page fails but the document is otherwise fine."""

    def __init__(
        self, page_index: int, message: str, *, diagnostic: str | None = None
    ) -> None:
        super().__init__(message, diagnostic=diagnostic)
        self.page_index = page_index


# AbstractContextManager is itself an ABC, so inheriting both would give
# an inconsistent MRO.
class OpenDocument(AbstractContextManager):
    """An opened source document ready for page-by-page work."""

    # False for formats that have no page images to show, because the
    # application that owns the format decides pagination when it opens the
    # file (DOCX). Declared rather than discovered by catching an exception,
    # so the analyze stage can skip previews as a normal path instead of
    # logging a failure per page for something that was never possible.
    supports_preview: bool = True

    @property
    @abstractmethod
    def page_count(self) -> int: ...

    @abstractmethod
    def metadata(self) -> DocumentMetadata: ...

    @abstractmethod
    def parse_page(self, index: int) -> Page:
        """Decompose one page into the page object model.

        Must raise PageParseError (not a bare exception) on failure so the
        job runner can flag exactly one page and continue (spec sec. 3).
        """

    @abstractmethod
    def render_page_png(
        self, index: int, *, dpi: int = 144, gray: bool = False
    ) -> bytes:
        """Rasterize one page to PNG bytes at the given DPI.

        `gray` renders luminance only. It exists for OCR, which discards
        colour anyway: a 300 DPI letter page is ~25 MB as RGB and ~8 MB as
        grey, and that difference is most of what a small container has.
        """

    @abstractmethod
    def render_region_png(
        self, index: int, bbox: tuple[float, float, float, float], *, dpi: int = 300
    ) -> bytes:
        """Rasterize a sub-region of a page, in page points."""

    @abstractmethod
    def extract_image_asset(self, xref: int) -> tuple[bytes, str] | None:
        """Return (raw_bytes, extension) for an embedded image, or None.

        Returns the *original* embedded bytes wherever possible -- never a
        re-encode -- so the exported document reuses the exact source asset
        (spec sec. 14).
        """

    @abstractmethod
    def close(self) -> None: ...

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DocumentParser(ABC):
    """Factory for OpenDocument handles for one family of formats."""

    name: str = "abstract"
    supported_content_types: frozenset[str] = frozenset()
    supported_extensions: frozenset[str] = frozenset()

    def can_parse(self, content_type: str | None, filename: str | None) -> bool:
        if content_type and content_type.lower() in self.supported_content_types:
            return True
        if filename:
            ext = Path(filename).suffix.lower().lstrip(".")
            return ext in self.supported_extensions
        return False

    @abstractmethod
    def open_path(self, path: Path) -> OpenDocument: ...

    @abstractmethod
    def open_bytes(self, data: bytes) -> OpenDocument: ...
