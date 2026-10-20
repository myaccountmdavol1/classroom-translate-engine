"""Analysis pipeline: completeness, aggregation, resumability, failure isolation."""

from __future__ import annotations

from conftest import corpus_file
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest
from ct_engine.pipeline.analyze import analyze_document
from ct_engine.pipeline.manifest_store import ManifestStore
from ct_engine.util.hashing import sha256_bytes


def _manifest(name: str, data: bytes) -> DocumentManifest:
    return DocumentManifest(
        document_id=f"test_{name[:2]}",
        filename=name,
        content_type="application/pdf",
        byte_size=len(data),
        sha256=sha256_bytes(data),
        source_key="documents/test/source.pdf",
    )


def _run(name: str, store, **kwargs) -> DocumentManifest:
    data = corpus_file(name)
    m = _manifest(name, data)
    with PdfParser().open_bytes(data) as doc:
        analyze_document(doc, m, store=store, **kwargs)
    return m


def test_every_page_is_present_and_analyzed(store):
    m = _run("08_exam_multiple_choice.pdf", store)
    assert m.analysis.page_count == 4
    assert len(m.pages) == 4
    assert m.processed_page_count == 4
    assert m.failed_pages == []
    assert [p.number for p in m.pages] == [1, 2, 3, 4]


def test_hundred_page_document_completes_fully(store):
    m = _run("12_long_100p.pdf", store)
    assert m.source_page_count == 100
    assert m.processed_page_count == 100
    assert len(m.pages) == 100
    assert m.failed_pages == []


def test_language_is_detected_from_whole_document(store):
    m = _run("07_reading_passage.pdf", store)
    assert m.analysis.detected_language == "en"
    assert m.analysis.detected_language_confidence > 0.8


def test_scanned_document_reports_ocr_need_and_warns(store):
    m = _run("14_scanned_exam.pdf", store)
    assert m.analysis.needs_ocr
    assert m.analysis.scanned_page_count == 4
    assert not m.analysis.has_selectable_text
    assert any("scanned" in w for w in m.analysis.warnings)


def test_mixed_page_sizes_are_reported(store):
    m = _run("10_mixed_orientation.pdf", store)
    assert m.analysis.has_mixed_page_sizes
    assert m.analysis.has_mixed_orientation
    assert any("page size" in w for w in m.analysis.warnings)
    assert len(m.analysis.page_sizes) > 1


def test_previews_and_thumbnails_are_written(store):
    m = _run("01_clean_worksheet.pdf", store)
    page = m.pages[0]
    assert page.renders.source_preview_key and page.renders.source_thumb_key
    assert store.exists(page.renders.source_preview_key)
    assert store.exists(page.renders.source_thumb_key)
    assert page.renders.preview_width_px > 0
    # A thumbnail must be materially smaller than the preview.
    assert store.size(page.renders.source_thumb_key) < store.size(
        page.renders.source_preview_key
    )


def test_original_image_assets_are_preserved(store):
    """Original bytes are reused, never regenerated (spec sec. 14)."""
    m = _run("04_science_diagram.pdf", store)
    img = m.pages[0].image_objects[0]
    assert img.asset_key and store.exists(img.asset_key)
    assert img.image_hash
    stored = store.get_bytes(img.asset_key)
    assert len(stored) > 500
    assert stored[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xdb")


def test_repeated_images_are_stored_once(store):
    """The same diagram on many pages must not be stored many times."""
    m = _run("11_long_50p.pdf", store)
    keys = {
        img.asset_key
        for p in m.pages
        for img in p.image_objects
        if img.asset_key
    }
    placements = sum(len(p.image_objects) for p in m.pages)
    assert placements > len(keys), "content addressing did not dedupe"
    for k in keys:
        assert store.exists(k)


def test_response_areas_are_detected_on_worksheets(store):
    m = _run("01_clean_worksheet.pdf", store)
    areas = m.pages[0].response_areas
    assert areas, "answer lines were not detected"
    assert any(a.kind == "writing_lines" for a in areas)
    for a in areas:
        assert 0.0 < a.confidence <= 1.0
        assert a.bbox.width > 0 and a.bbox.height > 0


def test_response_areas_do_not_swallow_the_whole_page(store):
    m = _run("01_clean_worksheet.pdf", store)
    page = m.pages[0]
    page_area = page.width * page.height
    for a in page.response_areas:
        assert a.bbox.area < 0.6 * page_area


def test_diagnostics_counts_match_objects(store):
    m = _run("04_science_diagram.pdf", store)
    p = m.pages[0]
    d = p.diagnostics
    assert d.text_objects == len(p.text_objects)
    assert d.image_objects == len(p.image_objects)
    assert d.response_areas == len(p.response_areas)


def test_targeted_reanalysis_touches_only_requested_pages(store):
    """Resumability: retrying one page must not redo the rest (spec sec. 40)."""
    data = corpus_file("08_exam_multiple_choice.pdf")
    m = _manifest("08", data)
    with PdfParser().open_bytes(data) as doc:
        analyze_document(doc, m, store=store)
        original_ids = [p.text_objects[0].id if p.text_objects else None for p in m.pages]
        analyze_document(doc, m, store=store, only_indices=[2])
        after = [p.text_objects[0].id if p.text_objects else None for p in m.pages]
    assert len(m.pages) == 4
    assert after[2] != original_ids[2], "target page was not reprocessed"
    assert after[0] == original_ids[0], "untargeted page was reprocessed"
    assert after[3] == original_ids[3], "untargeted page was reprocessed"


def test_manifest_survives_a_save_load_round_trip(store):
    m = _run("04_science_diagram.pdf", store)
    ms = ManifestStore(store)
    ms.save_manifest(m)
    loaded = ms.load_manifest(m.document_id)
    assert loaded is not None
    assert len(loaded.pages) == len(m.pages)
    a, b = m.pages[0].text_objects[0], loaded.pages[0].text_objects[0]
    assert a.id == b.id
    assert a.bbox.as_tuple() == b.bbox.as_tuple()
    assert a.text == b.text
    assert loaded.pages[0].image_objects[0].asset_key == (
        m.pages[0].image_objects[0].asset_key
    )


def test_failed_page_does_not_abort_the_document(store, monkeypatch):
    """A page-level failure is recorded and the rest still complete."""
    from ct_engine.ingest import pdf_parser as pp

    data = corpus_file("08_exam_multiple_choice.pdf")
    m = _manifest("08fail", data)
    real = pp.PyMuPDFDocument.parse_page

    def flaky(self, index):
        if index == 1:
            raise pp.PageParseError(1, "Page 2 could not be read.", diagnostic="injected")
        return real(self, index)

    monkeypatch.setattr(pp.PyMuPDFDocument, "parse_page", flaky)
    with PdfParser().open_bytes(data) as doc:
        analyze_document(doc, m, store=store)

    assert len(m.pages) == 4, "the failed page must still occupy its slot"
    assert m.failed_pages == [2]
    assert m.processed_page_count == 3
    assert m.pages[1].error and m.pages[1].error_stage == "ANALYZING"
    assert m.pages[0].analyzed and m.pages[2].analyzed and m.pages[3].analyzed
    assert any("could not be read" in w for w in m.analysis.warnings)
