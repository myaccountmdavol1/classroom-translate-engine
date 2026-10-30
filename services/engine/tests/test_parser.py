"""Parser tests: page correspondence, coordinate fidelity, asset preservation.

The ink test is the important one. Checking that a box is "in bounds" only
proves it is on the page; checking that it contains dark pixels proves it is
on the *text*, which is what coordinate fidelity actually means.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from conftest import corpus_file
from ct_engine.ingest.pdf_parser import (
    PdfParser,
    color_to_hex,
    garbled_ratio,
    normalize_font_family,
)
from ct_engine.model.enums import Orientation, PageTextMode

ALL_DOCS = [
    "01_clean_worksheet.pdf", "02_scanned_worksheet.pdf", "03_math_worksheet.pdf",
    "04_science_diagram.pdf", "05_tables.pdf", "06_photographs.pdf",
    "07_reading_passage.pdf", "08_exam_multiple_choice.pdf", "09_handwriting.pdf",
    "10_mixed_orientation.pdf", "13_rotated_pages.pdf", "14_scanned_exam.pdf",
]

EXPECTED_PAGES = {
    "01_clean_worksheet.pdf": 1, "02_scanned_worksheet.pdf": 1,
    "07_reading_passage.pdf": 3, "08_exam_multiple_choice.pdf": 4,
    "10_mixed_orientation.pdf": 3, "11_long_50p.pdf": 52,
    "12_long_100p.pdf": 100, "13_rotated_pages.pdf": 4,
}


# -- font / colour helpers -------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ABCDEE+Calibri-Bold", "Calibri"),
        ("Helvetica-Bold", "Helvetica"),
        ("Arial", "Arial"),
        ("XYZABC+MinionPro-Regular", "MinionPro"),
        (None, None),
    ],
)
def test_normalize_font_family(raw, expected):
    assert normalize_font_family(raw) == expected


def test_color_conversion_handles_signed_ints():
    """PyMuPDF returns black as a negative int once alpha is packed in."""
    assert color_to_hex(-16777216) == "#000000"
    assert color_to_hex(0) == "#000000"
    assert color_to_hex(255) == "#0000ff"
    assert color_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
    assert color_to_hex(None) is None


def test_garbled_ratio_flags_private_use_area():
    assert garbled_ratio("normal text") == 0.0
    assert garbled_ratio("ab") == pytest.approx(0.5)


# -- page correspondence ---------------------------------------------------
@pytest.mark.parametrize("name,expected", sorted(EXPECTED_PAGES.items()))
def test_page_count_is_exact(parser, name, expected):
    """Never silently drop pages (spec sec. 3)."""
    with parser.open_bytes(corpus_file(name)) as doc:
        assert doc.page_count == expected
        parsed = [doc.parse_page(i) for i in range(doc.page_count)]
    assert len(parsed) == expected
    assert [p.number for p in parsed] == list(range(1, expected + 1))
    assert [p.index for p in parsed] == list(range(expected))


@pytest.mark.parametrize("name", ALL_DOCS)
def test_every_page_parses_without_error(parser, name):
    with parser.open_bytes(corpus_file(name)) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            assert page.analyzed
            assert page.width > 0 and page.height > 0


@pytest.mark.parametrize("name", ALL_DOCS)
def test_all_objects_are_within_page_bounds(parser, name):
    """An object outside the page means the coordinate transform is wrong."""
    with parser.open_bytes(corpus_file(name)) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            for obj in page.all_objects:
                assert obj.bbox.x0 >= -2.0, f"{name} p{i+1} {obj.kind}"
                assert obj.bbox.y0 >= -2.0, f"{name} p{i+1} {obj.kind}"
                assert obj.bbox.x1 <= page.width + 2.0, f"{name} p{i+1} {obj.kind}"
                assert obj.bbox.y1 <= page.height + 2.0, f"{name} p{i+1} {obj.kind}"


# -- coordinate fidelity ---------------------------------------------------
def _ink_hit_rate(doc, index, dpi=110) -> tuple[float, int]:
    page = doc.parse_page(index)
    if not page.text_objects:
        return 1.0, 0
    arr = np.asarray(
        Image.open(io.BytesIO(doc.render_page_png(index, dpi=dpi))).convert("L")
    )
    ink = arr < 160
    ph, pw = arr.shape
    scale = dpi / 72.0
    hits = 0
    for t in page.text_objects:
        x0, y0 = max(0, int(t.bbox.x0 * scale)), max(0, int(t.bbox.y0 * scale))
        x1 = min(pw, int(np.ceil(t.bbox.x1 * scale)))
        y1 = min(ph, int(np.ceil(t.bbox.y1 * scale)))
        if x1 > x0 and y1 > y0 and ink[y0:y1, x0:x1].mean() > 0.01:
            hits += 1
    return hits / len(page.text_objects), len(page.text_objects)


@pytest.mark.parametrize(
    "name",
    ["01_clean_worksheet.pdf", "04_science_diagram.pdf", "05_tables.pdf",
     "08_exam_multiple_choice.pdf", "07_reading_passage.pdf"],
)
def test_text_boxes_land_on_actual_ink(parser, name):
    with parser.open_bytes(corpus_file(name)) as doc:
        for i in range(doc.page_count):
            rate, count = _ink_hit_rate(doc, i)
            assert rate >= 0.95, f"{name} page {i+1}: only {rate:.0%} of {count} boxes had ink"


def test_rotated_pages_keep_correct_coordinates(parser):
    """/Rotate pages are where naive coordinate handling breaks silently."""
    with parser.open_bytes(corpus_file("13_rotated_pages.pdf")) as doc:
        rotations = []
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            rotations.append(page.rotation)
            rate, count = _ink_hit_rate(doc, i)
            assert rate >= 0.95, f"rotated page {i+1}: {rate:.0%} of {count} boxes had ink"
        # The corpus builds one /Rotate 90 and one /Rotate 270 page.
        assert 90 in rotations and 270 in rotations


def test_rotated_page_dimensions_are_visual_not_raw(parser):
    """A 90-rotated landscape page must report portrait visual dimensions."""
    with parser.open_bytes(corpus_file("13_rotated_pages.pdf")) as doc:
        page = doc.parse_page(1)
        assert page.rotation == 90
        # Underlying mediabox is landscape 792x612; rotated it presents as portrait.
        assert page.height > page.width
        assert page.orientation == Orientation.PORTRAIT


# -- structure -------------------------------------------------------------
def test_mixed_orientation_is_detected(parser):
    with parser.open_bytes(corpus_file("10_mixed_orientation.pdf")) as doc:
        orientations = {doc.parse_page(i).orientation for i in range(doc.page_count)}
    assert Orientation.PORTRAIT in orientations
    assert Orientation.LANDSCAPE in orientations


def test_scanned_pages_are_flagged_for_ocr(parser):
    with parser.open_bytes(corpus_file("02_scanned_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
    assert page.text_mode == PageTextMode.SCANNED
    assert page.needs_ocr and page.is_scanned
    assert not page.text_objects
    assert page.image_objects, "a scan must still expose its page image"


def test_native_pages_are_not_flagged_for_ocr(parser):
    with parser.open_bytes(corpus_file("01_clean_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
    assert page.text_mode == PageTextMode.NATIVE
    assert not page.needs_ocr
    assert page.text_objects


def test_text_blocks_carry_typography(parser):
    with parser.open_bytes(corpus_file("01_clean_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
    styled = [t for t in page.text_objects if t.style.font_size > 0]
    assert styled, "no block recovered a font size"
    assert any(t.style.bold for t in page.text_objects), "bold was not detected"
    assert any(t.style.font_family for t in page.text_objects)
    for t in page.text_objects:
        assert t.lines and all(ln.spans for ln in t.lines)


def test_flat_text_joins_soft_wraps_and_repairs_hyphenation(parser):
    """Translation needs sentences, not shredded PDF line fragments."""
    with parser.open_bytes(corpus_file("07_reading_passage.pdf")) as doc:
        page = doc.parse_page(0)
    multiline = [t for t in page.text_objects if len(t.lines) > 2]
    assert multiline
    block = multiline[0]
    assert "\n" in block.text
    assert "\n" not in block.flat_text
    assert len(block.flat_text) > 40


def test_images_are_extracted_with_geometry_and_hash(parser):
    with parser.open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        page = doc.parse_page(0)
        assert len(page.image_objects) == 1
        img = page.image_objects[0]
        assert img.pixel_width > 0 and img.pixel_height > 0
        assert img.bbox.width > 100 and img.bbox.height > 100
        assert img.xref
        # Aspect ratio must survive: a distorted diagram is a fidelity failure.
        assert img.aspect_ratio == pytest.approx(img.pixel_aspect_ratio, rel=0.02)
        raw = doc.extract_image_asset(img.xref)
        assert raw is not None
        data, ext = raw
        assert len(data) > 100 and ext in {"png", "jpeg", "jpg"}
        # contains_text must be tri-state: unexamined is not "no text".
        assert img.contains_text is None


def test_vector_rules_are_recognized(parser):
    """Writing lines are the evidence for student response areas."""
    with parser.open_bytes(corpus_file("01_clean_worksheet.pdf")) as doc:
        page = doc.parse_page(0)
    rules = [v for v in page.vector_objects if v.looks_like_rule]
    assert len(rules) >= 6, f"expected answer lines, found {len(rules)}"


def test_z_order_is_recorded(parser):
    with parser.open_bytes(corpus_file("05_tables.pdf")) as doc:
        page = doc.parse_page(0)
    zs = [t.z_order for t in page.text_objects]
    assert len(set(zs)) > 1, "all text blocks share one z-order"


def test_render_page_png_matches_page_aspect(parser):
    with parser.open_bytes(corpus_file("10_mixed_orientation.pdf")) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            im = Image.open(io.BytesIO(doc.render_page_png(i, dpi=72)))
            assert im.width / im.height == pytest.approx(
                page.width / page.height, rel=0.02
            )


def test_render_region_clips_correctly_on_rotated_page(parser):
    """Region rendering must undo the visual transform, not ignore it."""
    with parser.open_bytes(corpus_file("13_rotated_pages.pdf")) as doc:
        page = doc.parse_page(3)
        assert page.rotation == 270
        target = page.text_objects[0]
        png = doc.render_region_png(3, target.bbox.as_tuple(), dpi=150)
        arr = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
        # The clip is a text box, so it must contain ink.
        assert (arr < 160).mean() > 0.01


def test_out_of_range_page_raises_page_error(parser):
    from ct_engine.ingest.base import PageParseError

    with parser.open_bytes(corpus_file("01_clean_worksheet.pdf")) as doc:
        with pytest.raises(PageParseError):
            doc.parse_page(99)


def test_corrupt_pdf_raises_parse_error():
    from ct_engine.ingest.base import ParseError

    with pytest.raises(ParseError):
        PdfParser().open_bytes(b"%PDF-1.4\nthis is not a pdf at all")


def test_hundred_page_document_is_complete(parser):
    """The headline completeness requirement (spec sec. 3)."""
    with parser.open_bytes(corpus_file("12_long_100p.pdf")) as doc:
        assert doc.page_count == 100
        pages = [doc.parse_page(i) for i in range(100)]
    assert len(pages) == 100
    assert all(p.analyzed for p in pages)
    assert sum(len(p.text_objects) for p in pages) > 300
    assert sum(len(p.image_objects) for p in pages) > 0


def test_blank_lines_do_not_stretch_a_block(parser):
    """Word saves empty paragraphs as lines of spaces. Behind a picture of a
    table they joined the paragraph above, and its box grew over the whole
    table, so every word read from the picture "overlapped" it."""
    import fitz

    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_text(
        (200, 100), "with your parents and school counselor.\n" + " \n" * 40,
        fontsize=11,
    )
    with parser.open_bytes(pdf.tobytes()) as doc:
        parsed = doc.parse_page(0)

    [block] = parsed.text_objects
    assert block.bbox.y1 < 110
    assert [ln.text.strip() for ln in block.lines] == [
        "with your parents and school counselor."
    ]
    # Measured with the blank lines, which sit at the paragraph's own pitch.
    assert block.line_spacing == pytest.approx(15.1, abs=0.5)
