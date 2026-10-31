"""A page that is one picture of a Word file is read as a picture: no table-line
removal, and sparse reading, so a lone hand-lettered word is found."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from ct_engine.ingest.image_import import image_to_pdf
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.pipeline import ocr_stage
from ct_engine.providers.ocr.base import OCRResult


class _Recorder:
    def __init__(self):
        self.calls = []

    def recognize(self, png, *, language=None, detect_handwriting=False, sparse=False):
        self.calls.append({"sparse": sparse})
        return OCRResult()


def _one_page():
    img = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img).rectangle((50, 50, 300, 120), fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    pdf, _ = image_to_pdf(buf.getvalue())
    doc = PdfParser().open_bytes(pdf)
    return doc, doc.parse_page(0)


def _run(monkeypatch, **kw):
    grid_calls = []

    def fake_grid(png, dpi=None):
        grid_calls.append(1)
        return None

    monkeypatch.setattr(ocr_stage, "remove_grid_lines", fake_grid)
    provider = _Recorder()
    doc, page = _one_page()
    with doc:
        ocr_stage.ocr_page(doc, page, provider, language="en", **kw)
    return grid_calls, provider.calls


def test_picture_pages_skip_grid_removal_and_read_sparse(monkeypatch):
    grid, calls = _run(monkeypatch, pictures=True)
    assert grid == []
    assert calls == [{"sparse": True}]


def test_ordinary_pages_are_read_as_before(monkeypatch):
    grid, calls = _run(monkeypatch)
    assert grid == [1]
    assert calls == [{"sparse": False}]


# --- Big lettering: the label fills the picture, centred --------------------
class _BigWord:
    """One line whose word is drawn 200pt tall on the page."""

    def __init__(self, dpi):
        self.px = 200 * dpi / 72.0

    def recognize(self, png, *, language=None, detect_handwriting=False, sparse=False):
        from ct_engine.providers.ocr.base import OCRRegion, OCRWord

        box = (10.0, 10.0, 10.0 + 3 * self.px, 10.0 + self.px)
        word = OCRWord(text="MAY", bbox=box, confidence=0.95)
        region = OCRRegion(text="MAY", bbox=box, confidence=0.95, words=[word])
        return OCRResult(regions=[region], mean_confidence=0.95)


def _big_page():
    img = Image.new("RGB", (1100, 400), "white")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    pdf, _ = image_to_pdf(buf.getvalue())
    doc = PdfParser().open_bytes(pdf)
    return doc, doc.parse_page(0)


def _read_big(monkeypatch, **kw):
    from ct_engine.model.enums import TextSource

    monkeypatch.setattr(ocr_stage, "remove_grid_lines", lambda png, dpi=None: None)
    doc, page = _big_page()
    dpi = 100
    with doc:
        ocr_stage.ocr_page(doc, page, _BigWord(dpi), language="en", dpi=dpi, **kw)
    return [t for t in page.text_objects if t.source is TextSource.OCR]


def test_picture_lettering_keeps_its_size_and_is_centred(monkeypatch):
    from ct_engine.model.enums import TextAlignment

    [label] = _read_big(monkeypatch, pictures=True)
    assert label.style.font_size > 72
    assert label.alignment is TextAlignment.CENTER
    assert label.fill_box is True


def test_scanned_page_lines_keep_the_old_cap(monkeypatch):
    from ct_engine.model.enums import TextAlignment

    [label] = _read_big(monkeypatch)
    assert label.style.font_size <= 72
    assert label.alignment is TextAlignment.UNKNOWN
    assert label.fill_box is False


# --- Drawn: as large as the box allows, wrapping when it must ---------------
def _drawn(text, width, height, size, fill_box, below=0):
    import fitz

    from ct_engine.model.document import Page
    from ct_engine.model.enums import ObjectKind, TextAlignment, TextSource, TranslationStatus
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
    from ct_engine.reconstruct.builder import rebuild_document

    pdf = fitz.open()
    pdf.new_page(width=width + 20, height=height + 20 + below)
    r = Rect(x0=10, y0=10, x1=10 + width, y1=10 + height)
    style = TextStyle(font_size=size)
    obj = TextObject(
        id="label", kind=ObjectKind.TEXT, bbox=r,
        norm_bbox=r.normalized(width + 20, height + 20 + below),
        source=TextSource.OCR, style=style, fill_box=fill_box,
        alignment=TextAlignment.CENTER if fill_box else TextAlignment.UNKNOWN,
        translated_text=text, translation_status=TranslationStatus.TRANSLATED,
        lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r, style=style)])],
    )
    page = Page(index=0, number=1, width=width + 20, height=height + 20 + below,
                text_objects=[obj], is_scanned=True)
    _, stats = rebuild_document(pdf.tobytes(), [page], target_lang="es")
    codes = {f.code for s in stats.values() for f in s.findings}
    return page.text_objects[0], codes


def test_a_longer_picture_translation_is_sized_to_fit_its_box():
    # Starting from the lettering's 115pt, the fitter stops shrinking at 71pt,
    # where it needs two lines the box cannot hold. Sized to the box instead,
    # it takes the largest size that fits (one line, here) and stays large.
    obj, codes = _drawn("EL SEPTIEMBRE PASADO", 700, 120, 115, fill_box=True)
    assert "text_overflow" not in codes
    layout = obj.drawn_layout
    assert layout.size > 50
    assert layout.size * 1.02 * len(layout.lines) <= 120 + 0.5


def test_a_short_picture_translation_fills_the_picture():
    obj, codes = _drawn("MAYO", 762, 251, 250, fill_box=True)
    assert "text_overflow" not in codes
    assert obj.drawn_layout.size > 200


def test_without_the_flag_a_long_translation_is_fitted_as_before():
    _, codes = _drawn("EL SEPTIEMBRE PASADO", 700, 120, 115, fill_box=False)
    assert "text_overflow" in codes


def test_a_picture_label_never_grows_down_into_the_picture():
    # Room below the box on the page: a picture label still stays in its
    # box (its words are painted into the Word picture), and stays large.
    obj, codes = _drawn("EL SEPTIEMBRE PASADO", 700, 120, 115, fill_box=True, below=150)
    assert "text_overflow" not in codes
    assert obj.drawn_bbox.y1 <= obj.bbox.y1 + 0.5
    assert obj.drawn_layout.size > 50


# --- Erased whole: hand lettering overshoots the reader's word box -----------
def _overshoot_page(fill_box, *, right_room=0, line_at=None, stale=False):
    """A 100pt-tall word box over a black bar that runs 5pt past its bottom
    edge; returns the dark pixels left in that 5pt strip after building."""
    import fitz

    from ct_engine.model.document import Page
    from ct_engine.model.enums import ObjectKind, TextAlignment, TextSource, TranslationStatus
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
    from ct_engine.reconstruct.builder import rebuild_document

    w, h, below = 300, 100, 40
    pw, ph = w + 20 + right_room, h + 20 + below
    pdf = fitz.open()
    pg = pdf.new_page(width=pw, height=ph)
    r = Rect(x0=10, y0=10, x1=10 + w, y1=10 + h)
    pg.draw_rect(fitz.Rect(40, 40, 43, 10 + h + 5), color=None, fill=(0, 0, 0))
    if line_at is not None:
        pg.draw_rect(fitz.Rect(200, line_at, 260, line_at + 1), color=None, fill=(0, 0, 0))
    style = TextStyle(font_size=60)
    obj = TextObject(
        id="label", kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(pw, ph),
        source=TextSource.OCR, style=style, fill_box=fill_box, erase_grown=stale,
        alignment=TextAlignment.CENTER if fill_box else TextAlignment.UNKNOWN,
        translated_text="a", translation_status=TranslationStatus.TRANSLATED,
        lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r, style=style)])],
    )
    page = Page(index=0, number=1, width=pw, height=ph, text_objects=[obj], is_scanned=True)
    out, _ = rebuild_document(pdf.tobytes(), [page], target_lang="es")
    return fitz.open(stream=out, filetype="pdf")[0], page.text_objects[0], r


def fitz_rect(*a):
    import fitz

    return fitz.Rect(*a)


def _dark_in(pg, x0, y0, x1, y1):
    pix = pg.get_pixmap(clip=fitz_rect(x0, y0, x1, y1), dpi=72)
    return sum(1 for i in range(0, len(pix.samples), pix.n) if pix.samples[i] < 128)


def test_a_picture_label_is_erased_past_its_word_box():
    pg, _, r = _overshoot_page(True)
    # 6% of the 100pt box is 6pt: the 5pt the lettering overshot is gone.
    assert _dark_in(pg, 40, r.y1 + 0.5, 43, r.y1 + 4.5) == 0


def test_without_the_flag_the_erase_margin_is_unchanged():
    pg, _, r = _overshoot_page(False)
    assert _dark_in(pg, 40, r.y1 + 1.0, 43, r.y1 + 4.5) > 0


def test_erase_margin_rule():
    from ct_engine.model.geometry import Rect
    from ct_engine.reconstruct.shapes import erase_margin

    class O:
        fill_box = True
        erase_grown = True

    tall, short = Rect(x0=0, y0=0, x1=50, y1=100), Rect(x0=0, y0=0, x1=50, y1=5)
    assert erase_margin(O(), tall) == 6.0
    assert erase_margin(O(), short) == 0.6
    O.fill_box = False
    assert erase_margin(O(), tall) == 0.6


def test_a_picture_label_does_not_widen_sideways_into_empty_space():
    obj, _ = _widen(True)
    assert obj.drawn_bbox.x1 <= obj.bbox.x1 + 0.5
    assert obj.widen_x1 is None


def test_other_labels_still_widen_into_empty_space():
    obj, _ = _widen(False)
    assert obj.drawn_bbox.x1 > obj.bbox.x1 + 4


def _widen(fill_box):
    # Narrow box, long text, empty page to its right.
    import fitz

    from ct_engine.model.document import Page
    from ct_engine.model.enums import ObjectKind, TextAlignment, TextSource, TranslationStatus
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
    from ct_engine.reconstruct.builder import rebuild_document

    pw, ph = 600, 100
    pdf = fitz.open()
    pdf.new_page(width=pw, height=ph)
    r = Rect(x0=10, y0=10, x1=110, y1=40)
    style = TextStyle(font_size=20)
    obj = TextObject(
        id="label", kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(pw, ph),
        source=TextSource.OCR, style=style, fill_box=fill_box,
        alignment=TextAlignment.CENTER if fill_box else TextAlignment.UNKNOWN,
        translated_text="EL SEPTIEMBRE PASADO", translation_status=TranslationStatus.TRANSLATED,
        lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r, style=style)])],
    )
    # Text further right, so the page's text margin leaves room to widen into.
    far = Rect(x0=500, y0=60, x1=590, y1=90)
    other = TextObject(
        id="far", kind=ObjectKind.TEXT, bbox=far, norm_bbox=far.normalized(pw, ph),
        source=TextSource.OCR, style=style, keep_original=True,
        lines=[TextLine(bbox=far, spans=[TextSpan(text="y", bbox=far, style=style)])],
    )
    page = Page(index=0, number=1, width=pw, height=ph, text_objects=[obj, other],
                is_scanned=True)
    rebuild_document(pdf.tobytes(), [page], target_lang="es")
    return page.text_objects[0], page


def test_the_picture_copy_covers_the_grown_erase_area():
    from types import SimpleNamespace

    from ct_engine.model.document import Page
    from ct_engine.model.enums import ObjectKind, TextSource
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle
    from ct_engine.pipeline.word_pictures import _labels_in

    r = Rect(x0=20, y0=20, x1=220, y1=120)  # 100pt tall: erased 6pt past
    style = TextStyle(font_size=60)

    def area(fill_box):
        t = TextObject(
            id="l", kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(300, 300),
            source=TextSource.OCR, style=style, fill_box=fill_box, erase_grown=fill_box,
            suppressed=True,
            lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r, style=style)])],
        )
        page = Page(index=0, number=1, width=300, height=300, text_objects=[t])
        image = SimpleNamespace(bbox=Rect(x0=0, y0=0, x1=300, y1=300))
        return _labels_in(page, image)

    assert area(True)[0].y1 >= r.y1 + 6.0
    assert area(False)[0].y1 == r.y1 + 2.5


def test_artwork_just_outside_the_box_stops_the_grown_erase():
    # A thin line 5pt below the box: past the 2.5pt ring, inside the 6pt
    # margin. The label is erased as before (0.6pt) and the line is kept.
    pg, obj, r = _overshoot_page(True, line_at=110 + 5)
    assert obj.erase_grown is False
    assert _dark_in(pg, 200, r.y1 + 4.5, 260, r.y1 + 6.5) > 0
    # The overshoot bar is not erased past 0.6pt either.
    assert _dark_in(pg, 40, r.y1 + 1.0, 43, r.y1 + 4.5) > 0


def test_a_plain_background_grows_the_erase():
    _, obj, _ = _overshoot_page(True)
    assert obj.erase_grown is True


def _line_survives(line_at):
    pg, obj, r = _overshoot_page(True, line_at=line_at)
    return obj.erase_grown, _dark_in(pg, 200, line_at - 0.5, 260, line_at + 1.5) > 0


def test_a_hairline_just_outside_the_box_survives():
    # 1.5pt out: inside the old 2.5pt ring's reach, off the grown ring.
    grown, kept = _line_survives(110 + 1.5)
    assert kept and grown is False


def test_a_hairline_inside_the_band_survives():
    # 3pt out of a 100pt box: past the 2.5pt ring, inside the 6pt band, on
    # neither ring.
    grown, kept = _line_survives(110 + 3)
    assert kept and grown is False


def test_a_stale_erase_flag_is_reset_for_every_object():
    _, obj, _ = _overshoot_page(False, stale=True)
    assert obj.erase_grown is False
