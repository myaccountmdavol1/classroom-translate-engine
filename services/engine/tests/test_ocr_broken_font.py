"""OCR on a page whose text layer is half readable.

A rubric re-saved by macOS Preview kept its heading font intact while its
table font's ToUnicode broke: PyMuPDF read "1d" and the column headings
correctly but every table letter as U+FFFD, so the page went to OCR. OCR
objects were appended beside the whole native layer. Text in both fonts
was then drawn twice ("THE FRAMEWORK FOR TEACHING" over itself), a block
holding "1d" plus the broken label came out as "1d U R", and OCR's own
misread of a label ("le" for "1e") was translated too.
"""

from __future__ import annotations

from conftest import corpus_file
from ct_engine.config import get_settings
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
from ct_engine.pipeline.ocr_stage import ocr_page
from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult

BROKEN = "�"


class _StubOCR(OCRProvider):
    name = "stub"
    sends_data_externally = False

    def __init__(self, regions):
        self._regions = regions

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, *, language=None, detect_handwriting=False):
        return OCRResult(regions=self._regions, mean_confidence=0.9, provider=self.name)


def _px(x0, y0, x1, y1):
    """Points to the pixels OCR reports at the page's OCR resolution."""
    k = get_settings().ocr_dpi / 72.0
    return (x0 * k, y0 * k, x1 * k, y1 * k)


def _line(*spans: tuple[str, Rect]) -> TextLine:
    style = TextStyle(font_size=9.0)
    return TextLine(
        bbox=Rect.bounding(b for _, b in spans),
        spans=[TextSpan(text=t, bbox=b, style=style) for t, b in spans],
    )


def _native(obj_id: str, lines: list[TextLine], page) -> TextObject:
    bbox = Rect.bounding(l.bbox for l in lines)
    return TextObject(id=obj_id, kind=ObjectKind.TEXT, bbox=bbox,
                      norm_bbox=bbox.normalized(page.width, page.height),
                      lines=lines, style=TextStyle(font_size=9.0))


def _read(page, regions):
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        ocr_page(doc, page, _StubOCR(regions), language=None)


def _page():
    with PdfParser().open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
    page.text_objects = []
    return page


def test_a_label_keeps_its_readable_line_and_loses_the_broken_ones():
    page = _page()
    label = _native("label", [
        _line(("1d ", Rect(x0=63, y0=392, x1=75, y1=401))),
        _line((f"U{BROKEN * 4} ", Rect(x0=63, y0=408, x1=85, y1=417))),
        _line((f"R{BROKEN * 8} ", Rect(x0=63, y0=420, x1=105, y1=427))),
    ], page)
    page.text_objects = [label]
    _read(page, [
        OCRRegion(text="ld", bbox=_px(63, 392, 74, 401), confidence=0.9),
        OCRRegion(text="Using", bbox=_px(63, 408, 85, 417), confidence=0.9),
        OCRRegion(text="Resources", bbox=_px(63, 420, 105, 427), confidence=0.9),
    ])
    natives = [o for o in page.text_objects if o.source == TextSource.NATIVE]
    ocr = [o.text.strip() for o in page.text_objects if o.source == TextSource.OCR]
    assert [o.text.strip() for o in natives] == ["1d"]
    # Shrunk to the line it still holds, so the lines OCR replaces are
    # covered by OCR's objects, not by this one.
    assert natives[0].bbox.y1 <= 402
    # "ld" is OCR re-reading the readable "1d": not drawn a second time.
    assert ocr == ["Using", "Resources"]


def test_a_block_with_no_readable_text_is_left_to_ocr():
    page = _page()
    page.text_objects = [_native("cell", [
        _line((f"T{BROKEN * 2} {BROKEN * 7} ", Rect(x0=142, y0=119, x1=230, y1=128))),
        _line((f"{BROKEN * 9} {BROKEN * 9}", Rect(x0=142, y0=131, x1=230, y1=140))),
    ], page)]
    _read(page, [
        OCRRegion(text="The teacher lacks", bbox=_px(142, 119, 230, 128), confidence=0.9),
    ])
    assert [o.source for o in page.text_objects] == [TextSource.OCR]


def test_a_readable_heading_is_not_read_again():
    page = _page()
    page.text_objects = [_native("title", [
        _line(("THE FRAMEWORK FOR TEACHING ", Rect(x0=228, y0=46, x1=384, y1=56))),
    ], page)]
    _read(page, [
        OCRRegion(text="THE FRAMEWORK FOR TEACHING", bbox=_px(228, 48, 384, 57),
                  confidence=0.9),
        OCRRegion(text="Unrelated line", bbox=_px(100, 300, 300, 312), confidence=0.9),
    ])
    assert [o.text.strip() for o in page.text_objects] == [
        "THE FRAMEWORK FOR TEACHING", "Unrelated line"]
