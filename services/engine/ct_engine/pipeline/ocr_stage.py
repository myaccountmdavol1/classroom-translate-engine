"""The OCR stage (spec sec. 16).

Pages with no usable text layer are rendered at high resolution, read, and
turned into ordinary TextObjects so that everything downstream --
protection, translation, fitting, reconstruction -- works on a scanned page
exactly as it does on a digital one.

Two things distinguish an OCR'd object from a native one and both matter:

  * `source = OCR`, which tells the reconstruction engine that the original
    glyphs live in a raster and must be covered rather than redacted.
  * `confidence`, which is real and is used to flag pages for review. OCR
    that is probably wrong must not be presented as if it were certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import get_settings
from ..ingest.base import OpenDocument
from ..ingest.pdf_parser import garbled_ratio
from ..model.document import Page, QAFinding
from ..model.enums import ObjectKind, TextSource
from ..model.geometry import Rect
from ..model.objects import TextLine, TextObject, TextSpan, TextStyle, _new_id
from ..providers.ocr.base import OCRProvider, OCRRegion
from ..util.logging import get_logger
from .ocr_prep import remove_grid_lines

log = get_logger(__name__)

# Lines below this mean confidence are kept but flagged for a human.
REVIEW_CONFIDENCE = 0.72
# Below this the text is more likely noise than content; it is dropped and
# the page is flagged, because inventing words is worse than admitting a gap.
DISCARD_CONFIDENCE = 0.35
# A run of text this full of U+FFFD or private-use codes comes from a font
# whose ToUnicode is broken: its letters exist only in the picture, which is
# what OCR reads. The parser uses the same share to call a page scanned.
UNREADABLE_RUN = 0.35
# An OCR line mostly lying on readable native text is that text read again.
ALREADY_READ = 0.5


def _drop_unreadable_text(page: Page) -> list[Rect]:
    """Keep only the native text that reads, and return where it is.

    A rubric re-saved by macOS Preview had one font intact and one broken,
    so a single block held "1d" plus a label of U+FFFD. Kept whole, the
    block was translated as "1d U R"; dropped whole, "1d" was left to OCR,
    which read it as "ld". Runs are judged one by one instead: a block of
    nothing readable is left to OCR, and one with some is cut down to the
    lines that still read.
    """
    readable: list[Rect] = []
    objects: list[TextObject] = []
    for obj in page.text_objects:
        if obj.source != TextSource.NATIVE:
            objects.append(obj)
            continue
        lines: list[TextLine] = []
        changed = False
        for line in obj.lines:
            spans = [s for s in line.spans if garbled_ratio(s.text) <= UNREADABLE_RUN]
            if len(spans) != len(line.spans):
                changed = True
                if not "".join(s.text for s in spans).strip():
                    continue
                line = line.model_copy(update={
                    "spans": spans,
                    "bbox": Rect.bounding(s.bbox for s in spans),
                })
            lines.append(line)
        if changed:
            if not lines:
                continue
            bbox = Rect.bounding(line.bbox for line in lines)
            obj = obj.model_copy(update={
                "lines": lines,
                "bbox": bbox,
                "norm_bbox": bbox.normalized(page.width, page.height),
            })
        objects.append(obj)
        readable.extend(line.bbox for line in obj.lines if line.text.strip())
    page.text_objects = objects
    return readable


@dataclass
class OCRStats:
    pages_read: int = 0
    regions: int = 0
    low_confidence_regions: int = 0
    discarded_regions: int = 0
    mean_confidence: float = 0.0
    findings: list[QAFinding] = field(default_factory=list)


def _region_to_object(
    region: OCRRegion, *, scale: float, page: Page, index: int
) -> TextObject:
    """Convert one OCR line from image pixels to a page-space TextObject."""
    x0, y0, x1, y1 = (v * scale for v in region.bbox)
    bbox = Rect(x0=x0, y0=y0, x1=x1, y1=y1).clamped_to(page.bounds)

    # Estimate the type size from the words' heights, not the line's. A
    # Tesseract box in points is within a few percent of the source size,
    # but the LINE box takes the height of its tallest member, and one junk
    # "word" (a starburst read as "2" on a real packet) made a 12pt label
    # 65pt. The lower median of the word heights ignores one tall outlier.
    heights = sorted((w.bbox[3] - w.bbox[1]) * scale for w in region.words)
    typical = heights[(len(heights) - 1) // 2] if heights else bbox.height
    font_size = round(max(5.0, min(72.0, typical, bbox.height)), 1)

    style = TextStyle(font_family=None, font_size=font_size)

    # One span per recognized word. This is what lets block splitting break a
    # line at a column gap, and what lets the reconstruction engine paint over
    # only the glyphs -- a single line-wide span would erase the fill-in
    # blanks sitting between "Name:", "Date:" and "Period:".
    spans: list[TextSpan] = []
    for word in region.words:
        wx0, wy0, wx1, wy1 = (v * scale for v in word.bbox)
        spans.append(
            TextSpan(
                text=word.text,
                bbox=Rect(x0=wx0, y0=wy0, x1=wx1, y1=wy1).clamped_to(page.bounds),
                style=style,
                confidence=word.confidence,
            )
        )
    if not spans:
        spans = [
            TextSpan(
                text=region.text, bbox=bbox, style=style,
                confidence=region.confidence,
            )
        ]
    else:
        # Spans carry no trailing spaces, so rebuild the line text with them.
        for i in range(len(spans) - 1):
            spans[i] = spans[i].model_copy(update={"text": spans[i].text + " "})

    line = TextLine(bbox=bbox, spans=spans, rotation=region.rotation)

    cover_boxes = [
        Rect(x0=b[0] * scale, y0=b[1] * scale, x1=b[2] * scale, y1=b[3] * scale)
        .clamped_to(page.bounds)
        for b in region.discarded_boxes
    ]

    return TextObject(
        id=_new_id("ocr"),
        kind=ObjectKind.TEXT,
        bbox=bbox,
        norm_bbox=bbox.normalized(page.width, page.height),
        z_order=1000 + index,  # drawn above the scanned image
        rotation=region.rotation,
        lines=[line],
        source=TextSource.OCR,
        style=style,
        confidence=region.confidence,
        cover_boxes=cover_boxes,
    )


def ocr_page(
    doc: OpenDocument,
    page: Page,
    provider: OCRProvider,
    *,
    language: str | None,
    dpi: int | None = None,
) -> OCRStats:
    """Read a scanned page and attach the recognized text to it."""
    stats = OCRStats()
    settings = get_settings()
    dpi = dpi or settings.ocr_dpi

    try:
        # Grey, not colour: Tesseract converts to luminance itself, so the
        # two extra channels are pure cost on a memory-bound container.
        png = doc.render_page_png(page.index, dpi=dpi, gray=True)
    except Exception as exc:
        stats.findings.append(
            QAFinding(
                severity="error",
                code="ocr_render_failed",
                message=f"Page {page.number} could not be prepared for reading.",
                details={"reason": repr(exc)[:200]},
            )
        )
        return stats

    # Ruled tables defeat Tesseract, so it reads a copy with the grid taken
    # out. Same size, so its boxes still map onto the page unchanged.
    result = provider.recognize(
        remove_grid_lines(png, dpi=dpi) or png, language=language
    )
    if result.error:
        stats.findings.append(
            QAFinding(
                severity="error",
                code="ocr_failed",
                message=(
                    f"Page {page.number} could not be read. {result.error}"
                ),
            )
        )
        page.ocr_done = True
        return stats

    scale = 72.0 / dpi
    readable = _drop_unreadable_text(page)
    kept: list[TextObject] = []
    for i, region in enumerate(result.regions):
        if not region.text.strip():
            continue
        if region.confidence < DISCARD_CONFIDENCE:
            stats.discarded_regions += 1
            continue
        obj = _region_to_object(region, scale=scale, page=page, index=i)
        if obj.bbox.is_degenerate(epsilon=1.0):
            continue
        if any(obj.bbox.overlap_fraction_of_self(r) > ALREADY_READ for r in readable):
            continue
        if region.confidence < REVIEW_CONFIDENCE:
            stats.low_confidence_regions += 1
        kept.append(obj)

    # OCR objects are appended, not substituted: a MIXED page keeps whatever
    # real text layer it had, and we only add what was read from the image.
    page.text_objects.extend(kept)
    page.ocr_done = True
    stats.pages_read = 1
    stats.regions = len(kept)
    stats.mean_confidence = result.mean_confidence

    if not kept:
        stats.findings.append(
            QAFinding(
                severity="warning",
                code="ocr_empty",
                message=(
                    f"No readable text was found on page {page.number}. It will "
                    "be kept exactly as it is."
                ),
            )
        )
    elif stats.low_confidence_regions:
        stats.findings.append(
            QAFinding(
                severity="warning",
                code="ocr_low_confidence",
                message=(
                    f"{stats.low_confidence_regions} line(s) on page "
                    f"{page.number} were hard to read and may be wrong."
                ),
                object_ids=[
                    o.id
                    for o in kept
                    if o.confidence is not None and o.confidence < REVIEW_CONFIDENCE
                ],
            )
        )
    if stats.discarded_regions:
        stats.findings.append(
            QAFinding(
                severity="warning",
                code="ocr_discarded",
                message=(
                    f"{stats.discarded_regions} marking(s) on page {page.number} "
                    "were too unclear to read and were left unchanged."
                ),
            )
        )

    page.recompute_diagnostics()
    return stats


def pages_needing_ocr(pages: list[Page]) -> list[Page]:
    return [p for p in pages if p.needs_ocr and not p.ocr_done]
