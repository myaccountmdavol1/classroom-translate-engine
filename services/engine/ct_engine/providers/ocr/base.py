"""OCR provider interface (spec sec. 10 / 16).

Selected by `CT_OCR_PROVIDER`. Every implementation must return bounding
boxes and per-region confidence: text without coordinates cannot be placed,
and text without confidence cannot be triaged for review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OCRWord:
    """A single recognized word, in pixel coordinates of the input image."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class OCRRegion:
    """One recognized region of text, in pixel coordinates of the input image."""

    text: str
    # (x0, y0, x1, y1) in pixels of the supplied image. The caller converts to
    # page points; providers must not guess the page geometry.
    bbox: tuple[float, float, float, float]
    confidence: float
    # Rotation of the text baseline in degrees, where the engine reports it.
    rotation: float = 0.0
    # True when the region was classified as handwriting rather than print
    # (spec sec. 17). None when the provider cannot tell.
    handwritten: bool | None = None
    line_index: int = 0
    # The words making up this line. Needed for two things the line box
    # cannot do: splitting a line that spans a column gap, and covering only
    # where glyphs actually are -- a line box drawn over "Name: ___ Date: ___"
    # would erase the answer blanks along with the labels.
    words: list[OCRWord] = field(default_factory=list)
    # Boxes where the engine found glyphs but the text was too unreliable to
    # use. Their pixels must still be covered during reconstruction: leaving
    # them uncovered produces a line that is half translated and half
    # original, which is worse than either.
    discarded_boxes: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )


@dataclass
class OCRResult:
    regions: list[OCRRegion] = field(default_factory=list)
    # Mean confidence weighted by character count.
    mean_confidence: float = 0.0
    detected_language: str | None = None
    provider: str = ""
    cached: bool = False
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(r.text for r in self.regions)


class OCRError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic


class OCRProvider(ABC):
    name: str = "abstract"
    sends_data_externally: bool = True

    @abstractmethod
    def supports_language(self, lang: str | None) -> bool: ...

    @abstractmethod
    def recognize(
        self,
        image_png: bytes,
        *,
        language: str | None = None,
        detect_handwriting: bool = False,
        sparse: bool = False,
    ) -> OCRResult:
        """Recognize text in an image.

        `sparse` selects scattered-text segmentation, for diagram and chart
        labels that are not laid out as a page of prose. Measured on the
        corpus, page-mode segmentation missed "Chloroplast" and "Nucleolus"
        entirely where sparse mode found both.

        Must return an `OCRResult` with `error` set rather than raising for
        recoverable problems, so a single unreadable page is flagged instead
        of aborting the document.
        """

    def recognize_line(
        self, image_png: bytes, *, language: str | None = None
    ) -> OCRResult | None:
        """Read an image known to hold exactly one level line of text.

        Used to re-read tilted words in a picture once they have been cut
        out and turned level (analysis/text_angle.py). None means this
        provider has no such mode, and the caller trusts only the geometry.
        """
        return None

    def close(self) -> None: ...
