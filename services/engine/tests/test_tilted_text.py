"""Tilted words in pictures: read, covered and redrawn along their slant.

The real case: "HARBOR" printed on a drawing of a tablet held at ~13
degrees, on a Google Slides deck. Its upright box held the tablet's frame
and a big red letter, so the automatic cover refused, the teacher forced a
white box over the artwork, and the translation was drawn level at twice
the letters' size. Here the same shape of problem is built synthetically.
"""

from __future__ import annotations

import io
import math
from types import SimpleNamespace

import fitz
import numpy as np
import pytest
from PIL import Image

from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest, Page, PageQA, QAFinding
from ct_engine.model.enums import TextSource, TranslationStatus
from ct_engine.model.geometry import Rect
from ct_engine.pipeline import image_text
from ct_engine.pipeline.image_text import extract_image_text
from ct_engine.reconstruct.builder import rebuild_page, turned_frame
from tilted_scenes import PICTURE_AT, ScriptedTilt, picture_pdf, region_for, scene, words_box

TEXT = "HARBOR SCHOOL"


def _read(angle=13.0, *, text=TEXT, line=(TEXT, 0.96), conf=0.65, as_is=None, **kw):
    doc = scene(text, angle, **kw)
    pdf, rect = picture_pdf(doc)
    provider = ScriptedTilt([region_for(doc, text, conf)], line=line, as_is=as_is)
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        stats = extract_image_text(parsed, page, provider, language="en")
    labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    return pdf, page, stats, provider, labels


def _translate(label, text="ESCUELA HARBOR"):
    label.translated_text = text
    label.machine_translated_text = text
    label.translation_status = TranslationStatus.TRANSLATED


def _rebuild(pdf, page):
    """(final page render, plate render, the rebuilt pdf) at 144 dpi."""
    doc = fitz.open(stream=pdf, filetype="pdf")
    plates = {}
    stats = rebuild_page(
        doc[0], page, target_lang="es",
        on_plate=lambda p: plates.setdefault("plate", _pixels(p)),
    )
    final = _pixels(doc[0])
    out = doc.tobytes()
    doc.close()
    return final, plates.get("plate"), out, stats


def _pixels(pdf_page, dpi=144):
    pix = pdf_page.get_pixmap(dpi=dpi, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def _at(arr, x, y, dpi=144):
    return arr[int(y * dpi / 72), int(x * dpi / 72)].astype(int)


def _source(pdf):
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return _pixels(doc[0])


# --- Reading ---------------------------------------------------------------
def test_a_tilted_label_gets_its_turned_box_and_its_letters_size():
    _, _, stats, provider, labels = _read()
    (label,) = labels
    assert label.oriented is not None
    assert abs(label.oriented.angle - 13.0) <= 1.0
    assert label.orientation_checked
    # The size comes from the letters' height across the line, not from
    # the upright box, which is far taller for a long tilted line.
    assert label.style.font_size < 0.5 * label.bbox.height
    assert 16.0 < label.style.font_size < 26.0
    assert stats.labels_tilted == 1
    # The straightened re-read was surer than the tilted read: it is used.
    assert provider.line_calls == 1
    assert label.confidence == pytest.approx(0.96)
    assert not label.keep_original
    assert label.cover_boxes == []


def test_a_less_sure_reread_keeps_the_original_reading():
    _, _, _, _, (label,) = _read(line=("MILT0N SCH00L", 0.63))
    assert label.flat_text == TEXT
    assert label.confidence == pytest.approx(0.65)
    # It still read as words, so the slant is confirmed.
    assert label.oriented is not None


def test_without_a_reread_the_slant_needs_letters_that_agree_strongly():
    _, _, _, provider, (label,) = _read(line=None)
    assert provider.line_calls == 1
    assert label.oriented is not None  # 13 letters on one line
    assert label.flat_text == TEXT


def test_a_garbled_reread_does_not_replace_the_words():
    _, _, _, _, (label,) = _read(line=("@@ ~~", 0.99))
    assert label.flat_text == TEXT


def test_a_level_label_is_drawn_exactly_as_before(monkeypatch):
    """Regression guard: under 2.5 degrees nothing about the block changes,
    and no extra reading is spent."""
    _, _, _, provider, (label,) = _read(1.0)
    assert label.oriented is None and label.orientation_checked
    assert provider.line_calls == 0

    monkeypatch.setattr(image_text, "measure_region", lambda *a, **k: None)
    _, _, _, _, (before,) = _read(1.0)
    skip = {"id", "lines", "orientation_checked"}
    assert label.model_dump(exclude=skip) == before.model_dump(exclude=skip)
    assert [s.model_dump(exclude={"id"}) for s in label.lines[0].spans] == [
        s.model_dump(exclude={"id"}) for s in before.lines[0].spans
    ]


def test_curved_words_are_left_in_english_and_reported():
    _, _, stats, provider, (label,) = _read(
        0.0, text="WELCOME STUDENTS", line=("WELCOME STUDENTS", 0.96),
        size=20.0, arc=(230.0, 40.0),
    )
    assert label.keep_original
    assert label.oriented is None
    assert provider.line_calls == 0
    (finding,) = [f for f in stats.findings if f.code == "image_text_curved"]
    assert finding.object_ids == [label.id]
    assert "curve" in finding.message
    assert not [f for f in stats.findings if f.code == "ocr_low_confidence"]


def test_steeply_turned_words_are_left_in_english_and_reported():
    _, _, stats, _, (label,) = _read(60.0, text="Photosynthesis Lab",
                                     line=("Photosynthesis Lab", 0.96), size=20.0)
    assert label.keep_original
    assert [f.object_ids for f in stats.findings if f.code == "image_text_steep"] == [[label.id]]


def test_a_tilted_line_lost_to_scraps_is_rescued_by_a_reread():
    """Sparse reading returned only junk for a tilted line; measuring the
    scraps, straightening and reading once more recovers it."""
    doc = scene("Photosynthesis Lab", 25.0, size=20.0)
    pdf, _ = picture_pdf(doc)
    box = words_box(doc)
    half = fitz.Rect(box.x0, box.y0, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    scraps = [region_for(doc, "rh 7j", 0.2, box=half)]
    provider = ScriptedTilt(scraps, line=("Photosynthesis Lab", 0.95))
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        stats = extract_image_text(parsed, page, provider, language="en")
    labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert [t.flat_text for t in labels] == ["Photosynthesis Lab"]
    assert abs(labels[0].oriented.angle - 25.0) <= 1.0
    assert stats.labels_rescued == 1 and stats.labels_rejected == 0
    assert not [f for f in stats.findings if f.code == "image_label_unreadable"]


def test_scraps_that_do_not_reread_as_words_stay_rejected():
    doc = scene("Photosynthesis Lab", 25.0, size=20.0)
    pdf, _ = picture_pdf(doc)
    provider = ScriptedTilt([region_for(doc, "rh 7j", 0.2)], line=("rh", 0.4))
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        stats = extract_image_text(parsed, page, provider, language="en")
    assert not [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert stats.labels_rejected == 1


# --- Rebuilding ------------------------------------------------------------
def test_the_translation_is_drawn_along_the_slant_without_touching_the_artwork():
    pdf, page, _, _, (label,) = _read(frame=True, big_letter=True)
    _translate(label)
    final, plate, out, stats = _rebuild(pdf, page)
    assert stats.covered == 1 and stats.uncovered == 0
    assert label.background_color == "#ffffff"

    # The words are written turned, in the parser's clockwise convention.
    with fitz.open(stream=out, filetype="pdf") as doc:
        dirs = [
            line["dir"]
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if "ESCUELA" in "".join(s["text"] for s in line["spans"])
        ]
    assert dirs
    drawn = math.degrees(math.atan2(dirs[0][1], dirs[0][0]))
    assert abs(drawn - label.oriented.angle) <= 0.5

    # The frame above and the big letter below are exactly as they were.
    source = _source(pdf)
    ob = label.oriented
    t = math.radians(ob.angle)
    ox, oy = PICTURE_AT
    for along, across in ((-150, -0.95 * 24), (150, -0.95 * 24), (0, -0.95 * 24)):
        x = ob.cx + along * math.cos(t) - across * math.sin(t)
        y = ob.cy + along * math.sin(t) + across * math.cos(t)
        if ox < x < ox + 400 and oy < y < oy + 200:
            assert np.abs(_at(final, x, y) - _at(source, x, y)).max() < 10
    assert (np.abs(final.astype(int) - source.astype(int)).max(axis=2) > 40).sum() > 0

    # The source letters are gone from the plate: inside the turned box it
    # is all paper.
    xs, ys = np.meshgrid(np.arange(plate.shape[1]), np.arange(plate.shape[0]))
    px, py = (xs + 0.5) * 72 / 144, (ys + 0.5) * 72 / 144
    du = (px - ob.cx) * math.cos(t) + (py - ob.cy) * math.sin(t)
    dv = -(px - ob.cx) * math.sin(t) + (py - ob.cy) * math.cos(t)
    inside = (np.abs(du) < ob.width / 2 - 0.5) & (np.abs(dv) < ob.height / 2 - 0.5)
    assert inside.sum() > 500
    assert (plate[inside].min(axis=1) > 215).mean() > 0.99

    # The upright box the editor frames is exactly the turned frame's.
    xs_, ys_ = zip(*label.drawn_layout.quad)
    assert label.drawn_bbox.x0 == pytest.approx(min(xs_), abs=0.01)
    assert label.drawn_bbox.y1 == pytest.approx(max(ys_), abs=0.01)


def test_the_same_label_drawn_upright_would_be_refused():
    """Why the turned box matters: the upright box's surroundings are the
    frame and the letter, which the old path rightly refused to paint."""
    from ct_engine.reconstruct.builder import estimate_background

    pdf, page, _, _, (label,) = _read(frame=True, big_letter=True)
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        color, _ = estimate_background(doc[0], label.bbox)
    assert color is None


def _side_lengths(quad):
    (ax, ay), (bx, by), _, (dx, dy) = quad
    return math.hypot(bx - ax, by - ay), math.hypot(dx - ax, dy - ay)


def test_a_dragged_turned_block_keeps_its_slant_and_size():
    pdf, page, _, _, (label,) = _read()
    _translate(label)
    _rebuild(pdf, page)
    quad, box, size = label.drawn_layout.quad, label.drawn_bbox, label.drawn_layout.size
    label.manual_bbox = Rect(x0=box.x0 + 10, y0=box.y0 + 6, x1=box.x1 + 10, y1=box.y1 + 6)
    _rebuild(pdf, page)
    assert label.drawn_layout.angle == pytest.approx(label.oriented.angle)
    assert label.drawn_layout.size == pytest.approx(size)
    for (x, y), (x2, y2) in zip(quad, label.drawn_layout.quad):
        assert x2 == pytest.approx(x + 10, abs=0.05)
        assert y2 == pytest.approx(y + 6, abs=0.05)


def test_a_box_resized_only_wider_keeps_the_letters_height():
    w, h = turned_frame(Rect(x0=0, y0=0, x1=200, y1=40), 13.0, 22.0)
    assert h == pytest.approx(22.0)
    assert w > 150


def test_her_background_on_a_turned_block_is_turned_too():
    pdf, page, _, _, (label,) = _read(frame=False, big_letter=False)
    _translate(label)
    label.manual_fill = "#ff0000"
    final, _, _, _ = _rebuild(pdf, page)
    red = (final[:, :, 0] > 200) & (final[:, :, 1] < 60) & (final[:, :, 2] < 60)
    assert red.sum() > 200
    # The upright box's corner lies outside the turned frame: not painted.
    box = label.drawn_bbox
    assert not red[int((box.y0 + 1) * 2), int((box.x0 + 1) * 2)]
    assert not red[int((box.y1 - 1) * 2), int((box.x1 - 1) * 2)]


def test_straight_draws_the_translation_level_and_still_covers_the_slant():
    pdf, page, _, _, (label,) = _read(frame=True, big_letter=True)
    _translate(label)
    label.manual_rotation = 0.0
    final, plate, _, stats = _rebuild(pdf, page)
    assert label.render_angle == 0.0
    assert label.drawn_layout.angle == 0.0 and label.drawn_layout.quad is None
    assert stats.covered == 1


def test_a_label_read_before_slants_were_measured_is_measured_on_rebuild():
    pdf, page, _, _, (label,) = _read()
    label.oriented = None
    label.orientation_checked = False
    label.rotation = 0.0
    label.style = label.style.model_copy(update={"font_size": 50.0})
    _translate(label)
    _rebuild(pdf, page)
    assert label.orientation_checked
    assert label.oriented is not None and abs(label.oriented.angle - 13.0) <= 1.0
    assert label.style.font_size < 26.0
    assert label.drawn_layout.angle == pytest.approx(label.oriented.angle)


def sprite_on_page(obj, scale=2.0):
    """The block's sprite as the editor shows it: drawn unturned, turned
    by the block's angle about its frame's centre (CSS rotate), and placed
    on the page. Returns (alpha mask in page pixels at `scale`, its top-left
    in those pixels)."""
    from ct_engine.reconstruct.builder import turned_frame_of
    from ct_engine.reconstruct.sprite import render_sprite, sprite_area

    sprite = Image.open(io.BytesIO(render_sprite(obj, scale=scale))).convert("RGBA")
    area = sprite_area(obj)
    layout = obj.drawn_layout
    if layout.quad:
        frame = turned_frame_of(layout)[0]
        # PIL turns counter-clockwise for a positive angle.
        turned = sprite.rotate(-layout.angle, resample=Image.BICUBIC, expand=True)
        cx, cy = frame.center
        x0 = int(round(cx * scale - turned.width / 2))
        y0 = int(round(cy * scale - turned.height / 2))
        return np.asarray(turned)[:, :, 3] > 128, (x0, y0)
    return np.asarray(sprite)[:, :, 3] > 128, (
        int(round(area.x0 * scale)), int(round(area.y0 * scale)),
    )


def _sprite_overlap(obj, final, plate):
    ink, (x0, y0) = sprite_on_page(obj)
    changed = np.abs(final.astype(int) - plate.astype(int)).max(axis=2) > 60
    canvas = np.zeros_like(changed)
    h = min(ink.shape[0], canvas.shape[0] - y0)
    w = min(ink.shape[1], canvas.shape[1] - x0)
    canvas[y0:y0 + h, x0:x0 + w] = ink[:h, :w]
    return (canvas & changed).sum() / max(1, (canvas | changed).sum())


def test_the_sprite_is_the_same_turned_ink_as_the_page():
    pdf, page, _, _, (label,) = _read(frame=False, big_letter=False)
    _translate(label)
    final, plate, _, _ = _rebuild(pdf, page)
    # Drawn unturned: the picture is the frame's size, not the upright box's.
    from ct_engine.reconstruct.builder import turned_frame_of
    from ct_engine.reconstruct.sprite import sprite_area

    frame = turned_frame_of(label.drawn_layout)[0]
    assert sprite_area(label).width == pytest.approx(frame.width + 4, abs=0.01)
    assert _sprite_overlap(label, final, plate) > 0.6


# --- Exports ---------------------------------------------------------------
def test_office_pictures_copy_only_the_turned_strip(store):
    from ct_engine.pipeline.word_pictures import translated_pictures

    doc = scene(TEXT, 13.0, frame=True, big_letter=True)
    original = doc[0].get_pixmap(dpi=144, alpha=False).tobytes("png")
    pdf, page, _, _, (label,) = _read(frame=True, big_letter=True)
    _translate(label)
    _, _, out, _ = _rebuild(pdf, page)
    image = page.image_objects[0]
    image.asset_key = "pictures/original.png"
    store.put_bytes(image.asset_key, original)
    manifest = SimpleNamespace(pages=[page])
    ((_, before, after),) = translated_pictures(manifest, out, store)
    base = np.asarray(before.convert("RGB").resize(after.size, Image.LANCZOS)).astype(int)
    new = np.asarray(after).astype(int)
    zoom = after.width / image.bbox.width
    changed = np.abs(new - base).max(axis=2) > 40
    assert changed.sum() > 100
    # The upright box's corners hold the frame and the letter: unchanged.
    box = label.bbox
    for x, y in ((box.x0 + 1, box.y0 + 1), (box.x1 - 1, box.y1 - 1)):
        px, py = int((x - image.bbox.x0) * zoom), int((y - image.bbox.y0) * zoom)
        assert not changed[py, px]


def test_editable_powerpoint_keeps_turned_words_in_the_picture():
    import io as _io

    from pptx import Presentation

    from ct_engine.reconstruct.layered_pptx import export_layered_pptx

    with fitz.open() as pdf:
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 90), "Texto editable", fontsize=16)
        writer = fitz.TextWriter(page.rect)
        writer.append((300, 300), "Inclinado", fontsize=16)
        writer.write_text(page, morph=(fitz.Point(300, 300), fitz.Matrix(-13)))
        data = pdf.tobytes()
    deck = Presentation(_io.BytesIO(export_layered_pptx(data)))
    shapes = deck.slides[0].shapes
    assert [s.text for s in shapes if s.has_text_frame] == ["Texto editable"]
    background = Image.open(_io.BytesIO(shapes[0].image.blob)).convert("L")
    # The turned word is still visible, in the background picture.
    assert background.crop((580, 560, 700, 640)).getextrema()[0] < 100
    # ...and the slide says so where she will see it.
    from ct_engine.reconstruct.layered_pptx import TURNED_TEXT_NOTE

    assert deck.slides[0].notes_slide.notes_text_frame.text == TURNED_TEXT_NOTE


def test_a_level_page_gets_no_note_about_turned_words():
    import io as _io

    from pptx import Presentation

    from ct_engine.reconstruct.layered_pptx import export_layered_pptx

    with fitz.open() as pdf:
        pdf.new_page(width=612, height=792).insert_text((72, 90), "Texto", fontsize=16)
        data = pdf.tobytes()
    slide = Presentation(_io.BytesIO(export_layered_pptx(data))).slides[0]
    assert not slide.has_notes_slide


def test_new_slant_findings_have_titles():
    from ct_engine.api.findings import TITLES

    for code in ("image_text_curved", "image_text_steep", "image_text_turned",
                 "image_text_tilt_kept_straight", "turned_text_drawn_level"):
        assert TITLES[code]


# --- Reporting and editing -------------------------------------------------
def test_kept_because_curved_is_one_item_not_two():
    from ct_engine.api.findings import group_findings

    page = Page(index=0, number=1, width=612, height=792)
    page.qa = PageQA(findings=[
        QAFinding(severity="warning", code="image_text_curved", message="m", object_ids=["o1"]),
        QAFinding(severity="warning", code="original_kept", message="k", object_ids=["o1"]),
    ])
    manifest = DocumentManifest(
        document_id="d", filename="x.pdf", content_type="application/pdf",
        byte_size=1, sha256="a" * 64, source_key="k", pages=[page],
    )
    groups = group_findings(manifest)
    assert [g["code"] for g in groups] == ["image_text_curved"]
    assert groups[0]["title"] == "Curved words in a picture were left in English"


def test_the_editor_is_told_the_slant_and_can_turn_it_off():
    from ct_engine.api.routes import _block_out
    from ct_engine.pipeline.edit import EditError, _rotation

    pdf, page, _, _, (label,) = _read()
    _translate(label)
    _rebuild(pdf, page)
    out = _block_out(label, page.width, page.height)
    assert out["angle"] == pytest.approx(label.oriented.angle)
    assert out["frame"]["height"] == pytest.approx(label.oriented.height, abs=0.01)
    assert _rotation(label, "straight") == 0.0
    assert _rotation(label, "follow") is None
    with pytest.raises(EditError):
        _rotation(label, "sideways")
    label.oriented = None
    with pytest.raises(EditError):
        _rotation(label, "straight")


# --- The corpus page, read by the real Tesseract -----------------------------
@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract binary not installed"
)
def test_the_tilted_labels_corpus_page_reads_each_label_at_its_own_slant():
    import ct_engine.providers.ocr.local_tesseract  # noqa: F401  (registers)
    from conftest import CORPUS, ENGINE_ROOT
    from ct_engine.providers import get_ocr_provider

    path = CORPUS / "17_tilted_labels.pdf"
    if not path.exists():
        import sys

        sys.path.insert(0, str(ENGINE_ROOT / "corpus"))
        import build_corpus

        build_corpus.doc_tilted_labels(path)
    with PdfParser().open_bytes(path.read_bytes()) as parsed:
        page = parsed.parse_page(0)
        stats = extract_image_text(parsed, page, get_ocr_provider("local"), language="en")
    labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]

    def find(prefix):
        return next(t for t in labels if t.flat_text.startswith(prefix))

    assert find("Water Cycle").oriented is None
    for prefix, angle in (("Evaporation", 8), ("Condensation", 15), ("Science Lab", 14)):
        label = find(prefix)
        assert label.oriented is not None, prefix
        assert abs(label.oriented.angle - angle) <= 1.5, prefix
        assert not label.keep_original
    # Every slant drawn is a slant that was printed: nothing turned wrongly.
    for label in labels:
        if label.oriented is not None:
            assert 5 <= label.oriented.angle <= 28
    # The arc title, if it was read at all, is kept and reported.
    curved = [f for f in stats.findings if f.code == "image_text_curved"]
    for finding in curved:
        for oid in finding.object_ids:
            assert page.find_text_object(oid).keep_original
    assert stats.measured <= 24


# --- Review fixes: false slants, late measurement, partial covers ----------
import json
from pathlib import Path

from ct_engine.analysis.text_angle import Orientation, measure_on_page
from ct_engine.providers.ocr.base import OCRRegion, OCRWord

LEVEL_LABELS = Path(__file__).parent / "fixtures" / "level_labels"


# The level-label crops come from a real screenshot deck and are not part of
# the public source; these three tests skip without them.
needs_level_labels = pytest.mark.skipif(
    not (LEVEL_LABELS / "labels.json").exists(),
    reason="tests/fixtures/level_labels is not published",
)


def _level_label_pdfs():
    """Each real level label (a crop of a screenshot deck) on a white
    picture of its own, on a letter page: (pdf, picture rect, word box in
    page points, text)."""
    spec = json.loads((LEVEL_LABELS / "labels.json").read_text())
    for item in spec["labels"]:
        crop = Image.open(LEVEL_LABELS / item["file"]).convert("RGB")
        cw, ch = item["size"]
        pw, ph = max(cw + 40, 80), max(ch + 40, 80)
        scale = crop.width / cw
        canvas = Image.new("RGB", (round(pw * scale), round(ph * scale)), "white")
        canvas.paste(crop, (round(20 * scale), round(20 * scale)))
        raw = io.BytesIO()
        canvas.save(raw, "PNG")
        out = fitz.open()
        page = out.new_page(width=612, height=792)
        rect = fitz.Rect(60, 60, 60 + pw, 60 + ph)
        page.insert_image(rect, stream=raw.getvalue())
        x0, y0, x1, y1 = item["word"]
        word = Rect(x0=rect.x0 + 20 + x0, y0=rect.y0 + 20 + y0,
                    x1=rect.x0 + 20 + x1, y1=rect.y0 + 20 + y1)
        yield out.tobytes(), rect, word, item["text"]


@needs_level_labels
def test_real_short_level_labels_are_not_measured_as_tilted():
    """12 of 182 level labels on a real deck measured 2.7-4.9 degrees with
    most letters agreeing on an edge, and were drawn turned. The slant must
    also shrink the letters' box (and the two fits agree) to be believed."""
    count = 0
    for pdf, _, word, text in _level_label_pdfs():
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            found = measure_on_page(doc[0], word)
        if found is not None:
            assert not found.tilted, (text, found.angle, found.area_ratio)
            assert not found.sure_alone, text
        count += 1
    assert count >= 10


@needs_level_labels
def test_real_level_labels_cost_no_extra_reading_and_stay_level():
    """The per-page cost guard: a page with only level labels makes no
    single-line re-reads at all, and its labels are drawn level."""
    for pdf, rect, word, text in _level_label_pdfs():
        s = 300 / 72.0
        px = ((word.x0 - rect.x0) * s, (word.y0 - rect.y0) * s,
              (word.x1 - rect.x0) * s, (word.y1 - rect.y0) * s)
        region = OCRRegion(text=text, bbox=px, confidence=0.9,
                           words=[OCRWord(text=w, bbox=px, confidence=0.9) for w in text.split()])
        provider = ScriptedTilt([region], line=(text, 0.97))
        with PdfParser().open_bytes(pdf) as parsed:
            page = parsed.parse_page(0)
            extract_image_text(parsed, page, provider, language="en")
        labels = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
        assert provider.line_calls == 0, text
        assert all(t.oriented is None for t in labels), text


@needs_level_labels
def test_late_measurement_leaves_real_level_labels_alone():
    """A document read before slants were measured is measured on its next
    rebuild; its level labels must come back exactly as they were drawn."""
    for pdf, _, word, text in _level_label_pdfs():
        with PdfParser().open_bytes(pdf) as parsed:
            page = parsed.parse_page(0)
        from ct_engine.model.enums import ObjectKind, SemanticRole
        from ct_engine.model.objects import TextLine, TextObject, TextSpan, TextStyle

        style = TextStyle(font_size=round(word.height * 0.92, 1))
        label = TextObject(
            id="imgtxt_old", kind=ObjectKind.TEXT, bbox=word,
            norm_bbox=word.normalized(page.width, page.height), z_order=2000,
            lines=[TextLine(bbox=word, spans=[TextSpan(text=text, bbox=word, style=style)])],
            source=TextSource.IMAGE_OCR, role=SemanticRole.IMAGE_LABEL, style=style,
            confidence=0.9,
        )
        page.text_objects.append(label)
        _translate(label, "Etiqueta")
        _, _, _, stats = _rebuild(pdf, page)
        assert label.orientation_checked
        assert label.oriented is None and label.manual_rotation is None, text
        assert label.style.font_size == style.font_size
        assert not [f for f in stats.findings if f.code.startswith("image_text_t")]


def test_a_late_slant_on_a_hand_placed_block_keeps_her_box_and_says_so():
    """The real deck: her box and white fill were drawn around the upright
    guess. Read as a turned frame they drew a big white tilted shape over
    the tablet. Her box stays straight where she put it, and she is told."""
    pdf, page, _, _, (label,) = _read(frame=True, big_letter=True)
    label.oriented = None
    label.orientation_checked = False
    label.style = label.style.model_copy(update={"font_size": 40.0})
    _translate(label)
    her_box = Rect(x0=150, y0=260, x1=330, y1=300)
    label.manual_bbox = her_box
    label.manual_fill = "#ffffff"
    _, _, _, stats = _rebuild(pdf, page)
    assert label.oriented is not None
    assert label.manual_rotation == 0.0
    assert label.drawn_layout.angle == 0.0 and label.drawn_layout.quad is None
    assert label.drawn_bbox.x0 == pytest.approx(her_box.x0)
    assert label.style.font_size == 40.0
    (finding,) = [f for f in stats.findings if f.code == "image_text_tilt_kept_straight"]
    assert finding.object_ids == [label.id]
    # Reset puts it on the slant, at the letters' size.
    label.manual_bbox = None
    label.manual_fill = None
    label.manual_rotation = None
    _rebuild(pdf, page)
    assert label.drawn_layout.angle == pytest.approx(label.oriented.angle)
    assert label.drawn_layout.size <= label.oriented.height


def test_a_late_slant_on_an_automatic_block_is_reported():
    pdf, page, _, _, (label,) = _read()
    label.oriented = None
    label.orientation_checked = False
    _translate(label)
    _, _, _, stats = _rebuild(pdf, page)
    assert label.oriented is not None
    assert [f.object_ids for f in stats.findings if f.code == "image_text_turned"] == [[label.id]]


def test_a_slant_the_letters_do_not_settle_needs_the_turned_reading_to_win(monkeypatch):
    """A re-read that is merely confident is no evidence: a level word
    straightened by a false few degrees still reads. Unless the letters
    agree strongly, the words are also read as they stand."""
    monkeypatch.setattr(Orientation, "strong", property(lambda self: False))
    _, _, _, provider, (label,) = _read(as_is=(TEXT, 0.95))
    assert label.oriented is None  # reads as well level: not turned
    assert provider.slanted_calls == 1
    _, _, _, _, (label,) = _read()  # garbled as it stands: the slant is real
    assert label.oriented is not None


def test_level_scraps_are_not_rescued_at_a_slant():
    """Rejected scraps are only rescued when reading them turned is clearly
    better than reading them as they stand."""
    doc = scene("Photosynthesis Lab", 25.0, size=20.0)
    pdf, _ = picture_pdf(doc)
    provider = ScriptedTilt([region_for(doc, "rh 7j", 0.2)],
                            line=("Photosynthesis Lab", 0.95),
                            as_is=("Photosynthesis Lab", 0.94))
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        extract_image_text(parsed, page, provider, language="en")
    assert not [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]

    level = scene("Photosynthesis Lab", 0.0, size=20.0)
    pdf, _ = picture_pdf(level)
    provider = ScriptedTilt([region_for(level, "rh 7j", 0.2)], line=("Photosynthesis Lab", 0.95))
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        extract_image_text(parsed, page, provider, language="en")
    assert not [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert provider.line_calls == 0


def test_a_symbol_before_the_word_is_covered_with_it():
    """A bullet before tilted words is too small for the letters' line fit,
    so the box made from the line alone left it out. The turned cover must
    take it too, or English ink shows beside the translation."""
    text = "\u2022 HARBOR SCHOOL"
    pdf, page, _, _, (label,) = _read(13.0, text=text, line=(text, 0.96))
    assert label.oriented is not None
    assert label.oriented.width > 205  # the bullet is inside the box
    _translate(label, "ESCUELA")
    _, plate, _, _ = _rebuild(pdf, page)
    box = words_box(scene(text, 13.0))
    ox, oy = PICTURE_AT
    region = plate[int((oy + box.y0) * 2):int((oy + box.y1) * 2),
                   int((ox + box.x0) * 2):int((ox + box.x1) * 2)]
    assert region.min() > 150  # nothing of the source words is left


def test_a_letter_off_the_turned_box_keeps_the_label_level():
    """If letter-sized ink inside the reader's box falls outside the turned
    box, covering only the box would leave it showing: drawn level."""
    doc = scene(TEXT, 13.0)
    box = words_box(doc)
    doc[0].draw_rect(fitz.Rect(box.x0 + 4, box.y1 - 12, box.x0 + 11, box.y1 - 4),
                     color=None, fill=(0, 0, 0))
    pdf, _ = picture_pdf(doc)
    provider = ScriptedTilt([region_for(doc, TEXT, 0.65, box=box)], line=(TEXT, 0.96))
    with PdfParser().open_bytes(pdf) as parsed:
        page = parsed.parse_page(0)
        extract_image_text(parsed, page, provider, language="en")
    (label,) = [t for t in page.text_objects if t.source is TextSource.IMAGE_OCR]
    assert label.oriented is None


def test_a_widened_turned_block_round_trips_through_a_drag():
    """The frame widened along its line (room beside it): dragging the
    upright box the editor shows must give back the same frame."""
    pdf, page, _, _, (label,) = _read(frame=False, big_letter=False)
    _translate(label, "ESCUELA HARBOR SECUNDARIA")
    _rebuild(pdf, page)
    w0, _ = _side_lengths(label.drawn_layout.quad)
    assert w0 > label.oriented.width + 4  # it was widened
    quad, box, size = label.drawn_layout.quad, label.drawn_bbox, label.drawn_layout.size
    label.manual_bbox = Rect(x0=box.x0 - 5, y0=box.y0 + 4, x1=box.x1 - 5, y1=box.y1 + 4)
    _rebuild(pdf, page)
    assert label.drawn_layout.size == pytest.approx(size, abs=0.05)
    for (x, y), (x2, y2) in zip(quad, label.drawn_layout.quad):
        assert x2 == pytest.approx(x - 5, abs=0.1)
        assert y2 == pytest.approx(y + 4, abs=0.1)


def test_a_short_turned_translation_is_centred_where_the_word_was():
    """HARBOR's translation, left-aligned, covered the left 45% of where the
    word ran and sat over the red letter. A short one is centred."""
    pdf, page, _, _, (label,) = _read(frame=False, big_letter=False)
    _translate(label, "ESC")
    _, _, out, _ = _rebuild(pdf, page)
    with fitz.open(stream=out, filetype="pdf") as doc:
        (bbox,) = [
            line["bbox"]
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if "ESC" in "".join(s["text"] for s in line["spans"])
        ]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    ob = label.oriented
    t = math.radians(ob.angle)
    along = (cx - ob.cx) * math.cos(t) + (cy - ob.cy) * math.sin(t)
    assert abs(along) < 2.0
