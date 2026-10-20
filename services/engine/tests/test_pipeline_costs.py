"""Work the free host should not do twice, and failures it must not hide.

A picture repeated on every page is read once; a thumbnail is rendered
small rather than decoded and shrunk; a Word or PowerPoint download asks the
translation service once, not once per line; and a picture or paragraph
that could not be put into the saved file is reported, not only logged.
"""

from __future__ import annotations

import io

from PIL import Image

from conftest import corpus_file
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.enums import TextSource
from ct_engine.pipeline.image_text import extract_image_text
from ct_engine.providers.ocr.base import OCRProvider, OCRResult
from test_image_text_doubt import _region


class _Counting(OCRProvider):
    name = "counting"
    sends_data_externally = False

    def __init__(self, regions, *, error=None):
        self.regions = regions
        self.error = error
        self.calls = 0

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, *, language=None, detect_handwriting=False, sparse=False):
        self.calls += 1
        return OCRResult(regions=list(self.regions), provider=self.name, error=self.error)


# --- Words in pictures --------------------------------------------------------
def test_the_same_picture_on_another_page_is_not_read_again():
    provider = _Counting([_region("Cell Wall", 20, 20, [0.95, 0.93])])
    cache: dict = {}
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        first = doc.parse_page(0)
        extract_image_text(doc, first, provider, language="en", seen_hashes=cache)
        second = doc.parse_page(0)  # the same pixels, as on a repeated slide
        stats = extract_image_text(doc, second, provider, language="en", seen_hashes=cache)

    assert provider.calls == 1
    assert stats.labels_accepted == 1
    a = [t for t in first.text_objects if t.source is TextSource.IMAGE_OCR]
    b = [t for t in second.text_objects if t.source is TextSource.IMAGE_OCR]
    assert [t.flat_text for t in a] == [t.flat_text for t in b]
    # Fresh blocks, not shared ones: an edit to one page must not move the other.
    assert {t.id for t in a}.isdisjoint({t.id for t in b})
    assert a[0].bbox == b[0].bbox


def test_a_failed_read_is_tried_again_next_time():
    provider = _Counting([], error="reader crashed")
    cache: dict = {}
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        for _ in range(2):
            extract_image_text(doc, doc.parse_page(0), provider, language="en", seen_hashes=cache)
    assert provider.calls == 2


def test_a_picture_that_cannot_be_prepared_is_reported():
    class _NoRender:
        def __init__(self, doc):
            self.doc = doc

        def render_region_png(self, *a, **k):
            raise MemoryError("no room")

    provider = _Counting([])
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(_NoRender(doc), page, provider, language="en")
    assert provider.calls == 0
    assert [f.code for f in stats.findings] == ["image_text_not_read"]


# --- Thumbnails ------------------------------------------------------------
def test_a_thumbnail_is_rendered_small_at_the_page_shape(store):
    from ct_engine.config import get_settings
    from ct_engine.render.previews import render_source_page

    with PdfParser().open_bytes(corpus_file("01_clean_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
        refs = render_source_page(doc, page, document_id="thumbs", store=store)
    thumb = Image.open(io.BytesIO(store.get_bytes(refs.source_thumb_key)))
    want = get_settings().thumbnail_width_px
    # Never wider than its slot, and not much narrower.
    assert want * 0.9 <= thumb.width <= want
    page_ratio = refs.preview_height_px / refs.preview_width_px
    assert abs(thumb.height / thumb.width - page_ratio) < 0.03


# --- Pictures in the saved Office file ------------------------------------------
def test_pictures_left_unchanged_are_counted_with_their_page():
    import zipfile

    from ct_engine.pipeline.word_pictures import replace_pictures

    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
    picture = Image.new("RGB", (200, 100), "white")
    _, stats = replace_pictures(
        docx.getvalue(),
        [(4, picture, None), (7, picture, picture.copy()), (9, None, None)],
    )
    # Cropped (page 4), no matching file picture (page 7), unopenable (page 9).
    assert stats.translated == 0 and stats.unmatched == 3
    assert sorted(stats.kept_pages) == [4, 7, 9]


# --- Paragraphs translated on their own -------------------------------------------
def test_unmatched_paragraphs_go_to_the_service_in_one_request(store, monkeypatch):
    """Paragraphs the pages cannot stand in for are sent together."""
    from ct_engine.pipeline import word_export
    from ct_engine.translate import one_off
    from ct_engine.translate.one_off import OneOffTranslation

    calls: list[list[str]] = []

    def fake_many(texts, *, target_lang, source_lang=None):
        calls.append(list(texts))
        return [
            None if t == "Broken line" else
            OneOffTranslation(text=f"es:{t}", provider="fake", protected_intact=True)
            for t in texts
        ]

    monkeypatch.setattr(one_off, "translate_many", fake_many)

    class _Unit:
        def __init__(self, text):
            self.text = text
            self.anchor = "a"
            self.translated_text = None
            self.translation_status = None
            self.manual_color = None
            self.manual_font_size = None
            self.suppressed = False

    units = [_Unit("First paragraph"), _Unit("Note one\nNote two"), _Unit("Broken line")]

    class _WordPage:
        text_objects = units

    class _Parser:
        def open_bytes(self, data):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        page_count = 1

        def parse_page(self, i):
            return _WordPage()

    class _Result:
        data = b"docx"

    monkeypatch.setattr("ct_engine.ingest.docx_parser.DocxParser", _Parser)
    monkeypatch.setattr(word_export, "rebuild_docx", lambda source, pages: _Result())

    class _Analysis:
        detected_language = "en"

    class _Manifest:
        office_format = "docx"
        office_source_key = "documents/w/source.docx"
        document_id = "w"
        analysis = _Analysis()
        pages: list = []

    store.put_bytes("documents/w/source.docx", b"docx")
    result = word_export.build_office_export(_Manifest(), store=store, target_lang="es")

    assert calls == [["First paragraph", "Note one", "Note two", "Broken line"]]
    assert units[0].translated_text == "es:First paragraph"
    assert units[1].translated_text == "es:Note one\nes:Note two"
    assert units[2].translated_text is None
    assert result.paragraphs_kept_original == 1
    assert result.pictures_kept_pages == []

    # Kept for the next download: only the one that failed is asked again.
    calls.clear()
    for u in units:
        u.translated_text = None
    word_export.build_office_export(_Manifest(), store=store, target_lang="es")
    assert calls == [["Broken line"]]
