"""Rendering interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.document import Page


class PageRasterizer(ABC):
    """Produces raster previews of pages."""

    @abstractmethod
    def render_page(self, index: int, *, dpi: int) -> bytes: ...


class RenderError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic
