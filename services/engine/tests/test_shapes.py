"""The Shape tool, Stage 1: Rectangle, Rounded rectangle and Oval.

A shape is a box she made, drawn with a different outline
(reconstruct/shapes.py). These check the engine draws it -- fill, outline,
dashes, turn, sprite -- with her words inside its text area, and that every
edit path keeps it: made empty, words taken out, Reset, undo and redo,
duplicate and paste, the format painter, and a .ctproject round trip. An
older manifest, and one naming a shape this build does not know, still load.
"""

from __future__ import annotations

import io
import math

import fitz
import numpy as np
import pytest
from PIL import Image

from ct_engine.model.document import DocumentManifest
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import TextObject
from ct_engine.pipeline.edit import (
    EditError,
    _snapshot,
    add_blocks,
    apply_bulk,
    apply_edit,
    apply_layout,
    redo,
    reset_edit,
    translate_from_english,
    undo,
)
from ct_engine.reconstruct import shapes
from ct_engine.reconstruct.builder import ASCENT_RATIO, DESCENT_RATIO, rebuild_document
from ct_engine.reconstruct.sprite import render_sprite, sprite_area
from test_edit import prepared

DPI = 144  # 2 pixels a point
# Clear of everything drawn on 01_clean_worksheet.pdf.
LOW = {"x0": 72.0, "y0": 600.0, "x1": 272.0, "y1": 700.0}
WORDS = "Lea con cuidado cada pregunta y escriba su respuesta completa aqui"


def _px(v):
    return int(round(v * DPI / 72))


def _rgb(pix):
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).astype(int)


def _render(manifest, store):
    src = store.get_bytes(manifest.source_key)
    out, _ = rebuild_document(src, manifest.pages, target_lang="es")
    with fitz.open(stream=out, filetype="pdf") as doc:
        return _rgb(doc[0].get_pixmap(dpi=DPI, alpha=False))


def _red(arr):
    return (arr[..., 0] > 190) & (arr[..., 1] < 90) & (arr[..., 2] < 90)


def _blue(arr):
    return (arr[..., 2] > 190) & (arr[..., 0] < 90) & (arr[..., 1] < 90)


def _make(manifest, store, shape="ellipse", box=LOW, text="", **spec):
    base = {
        "bbox": box, "text": text, "shape": shape, "fill": "none",
        "border_color": "#000000", "border_width": 1, "padding": 4,
        "alignment": "CENTER", "vertical_align": "middle",
    }
    _, (obj,) = add_blocks(manifest, 1, [base | spec], store=store, target_lang="es")
    return obj


def _edit(manifest, obj, store, **kw):
    return apply_layout(manifest, 1, obj.id, store=store, target_lang="es", **kw)


@pytest.fixture
def fake_translate(monkeypatch):
    from ct_engine.translate.one_off import OneOffTranslation

    def fake(text, *, target_lang, source_lang=None, provider=None):
        return OneOffTranslation(text=f"[{target_lang}] {text}", protected_intact=True, provider="fake")

    monkeypatch.setattr("ct_engine.translate.one_off.translate_one", fake)


# --- Drawing ------------------------------------------------------------------
def test_an_oval_fills_its_middle_and_leaves_the_frame_corners(store):
    manifest = prepared(store)
    obj = _make(manifest, store, fill="#0000ff", border_color="none")
    assert obj.drawn_layout.shape == "ellipse"
    assert obj.drawn_layout.frame == Rect(**LOW)
    blue = _blue(_render(manifest, store))
    r = Rect(**LOW)
    cx, cy = r.center
    assert blue[_px(cy) - 20:_px(cy) + 20, _px(cx) - 20:_px(cx) + 20].all()
    # Just inside each side's middle: filled. Just inside each corner: clear.
    for x, y in ((r.x0 + 3, cy), (r.x1 - 3, cy), (cx, r.y0 + 3), (cx, r.y1 - 3)):
        assert blue[_px(y), _px(x)]
    for x, y in ((r.x0 + 6, r.y0 + 6), (r.x1 - 6, r.y0 + 6), (r.x0 + 6, r.y1 - 6), (r.x1 - 6, r.y1 - 6)):
        assert not blue[_px(y), _px(x)]


def test_a_rounded_rectangle_has_round_corners(store):
    manifest = prepared(store)
    _make(manifest, store, shape="roundRect", fill="#0000ff", border_color="none")
    blue = _blue(_render(manifest, store))
    r = Rect(**LOW)
    # The corner radius is 1/6 of the short side (16667): ~16.7pt.
    assert not blue[_px(r.y0 + 1.5), _px(r.x0 + 1.5)]
    assert not blue[_px(r.y1 - 1.5), _px(r.x1 - 1.5)]
    assert blue[_px(r.y0 + 1.5), _px(r.x0 + 30)]
    assert blue[_px(r.y0 + 30), _px(r.x0 + 1.5)]


def test_a_rectangle_is_drawn_as_a_bordered_box_is(store):
    manifest = prepared(store)
    plain = _make(manifest, store, shape=None, box={"x0": 300.0, "y0": 600.0, "x1": 400.0, "y1": 700.0},
                  fill="none", border_color="#ff0000", border_width=4)
    rect = _make(manifest, store, shape="rect", box={"x0": 420.0, "y0": 600.0, "x1": 520.0, "y1": 700.0},
                 fill="none", border_color="#ff0000", border_width=4)
    assert plain.manual_shape is None and rect.manual_shape == "rect"
    red = _red(_render(manifest, store))
    a = red[_px(590):_px(710), _px(290):_px(410)]
    b = red[_px(590):_px(710), _px(410):_px(530)]
    assert a.sum() > 1000 and abs(int(a.sum()) - int(b.sum())) <= a.sum() * 0.02


def _flattened(cmds, steps=200):
    """Points along an outline with their distance from its start."""
    pts = []
    cur = None
    for c in cmds:
        if c[0] == "M":
            cur = (c[1], c[2])
            pts.append(cur)
        elif c[0] == "L":
            for i in range(1, steps + 1):
                t = i / steps
                pts.append((cur[0] + (c[1] - cur[0]) * t, cur[1] + (c[2] - cur[1]) * t))
            cur = (c[1], c[2])
        elif c[0] == "C":
            p0, p1, p2, p3 = cur, (c[1], c[2]), (c[3], c[4]), (c[5], c[6])
            for i in range(1, steps + 1):
                t = i / steps
                u = 1 - t
                pts.append(tuple(
                    u ** 3 * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t ** 3 * p3[k]
                    for k in range(2)
                ))
            cur = p3
    out, s = [(pts[0], 0.0)], 0.0
    for a, b in zip(pts, pts[1:]):
        s += math.hypot(b[0] - a[0], b[1] - a[1])
        out.append((b, s))
    return out


@pytest.mark.parametrize("shape", ["rect", "ellipse", "roundRect"])
@pytest.mark.parametrize("dash", ["dashed", "dotted"])
def test_dashes_fall_where_the_editors_svg_puts_them(store, shape, dash):
    """The editor strokes the same path with stroke-dasharray 3w 2w (dots:
    0 2w, round caps) from the path's start, so each dash's middle along the
    path is inked and each gap's middle is not -- sampled along the top."""
    manifest = prepared(store)
    w = 4.0
    obj = _make(manifest, store, shape=shape, fill="none", border_color="#ff0000",
                border_width=w, border_dash=dash)
    red = _red(_render(manifest, store))
    r = Rect(**LOW)
    period = 5 * w if dash == "dashed" else 2 * w
    on_at, off_at = (1.5 * w, 4.0 * w) if dash == "dashed" else (0.0, w)
    checked = 0
    path = _flattened(shapes.placed(shapes.outline(shape, r.width, r.height), r.x0, r.y0), 400)
    lengths = np.array([d for _, d in path])
    for k in range(int(lengths[-1] // period)):
        for target, inked in ((k * period + on_at, True), (k * period + off_at, False)):
            (x, y), _ = path[int(np.abs(lengths - target).argmin())]
            if y > r.y0 + r.height / 4:  # along the top only
                continue
            assert red[_px(y), _px(x)] == inked, (target, x, y, inked)
            checked += inked
    assert checked >= 3
    assert obj.drawn_layout.border_dash == dash


def test_a_turned_oval_is_turned(store):
    manifest = prepared(store)
    obj = _make(manifest, store, fill="#0000ff", border_color="none")
    _edit(manifest, obj, store, angle=30)
    assert obj.drawn_layout.quad and obj.drawn_layout.angle == 30
    assert obj.manual_bbox == Rect(**LOW)
    blue = _blue(_render(manifest, store))
    cx, cy = Rect(**LOW).center
    t = math.radians(30)
    # Along the turned long axis, 80% out: filled. The same distance along
    # the page's own x axis lies outside the turned oval.
    on = (cx + 80 * math.cos(t), cy + 80 * math.sin(t))
    off = (cx + 80, cy)
    assert blue[_px(on[1]), _px(on[0])]
    assert not blue[_px(off[1]), _px(off[0])]


@pytest.mark.parametrize("shape", ["rect", "roundRect", "ellipse"])
@pytest.mark.parametrize("angle", [None, 30])
def test_the_sprite_holds_the_whole_outline_at_6pt(store, shape, angle):
    manifest = prepared(store)
    obj = _make(manifest, store, shape=shape, border_color="#ff0000", border_width=6)
    if angle:
        _edit(manifest, obj, store, angle=angle)
    sprite = np.asarray(Image.open(io.BytesIO(render_sprite(obj, scale=2.0))).convert("RGBA"))
    area = sprite_area(obj)
    assert sprite.shape[1] == pytest.approx(area.width * 2, abs=1)
    alpha = sprite[:, :, 3]
    assert not alpha[0].any() and not alpha[-1].any()
    assert not alpha[:, 0].any() and not alpha[:, -1].any()
    assert _red(sprite[:, :, :3])[alpha > 200].sum() > 500
    # A rectangle's mitred corner reaches 0.71 of the weight out.
    reach = shapes.ink_reach(shape, 200, 100, None, 6)
    assert area.width == pytest.approx(200 + 2 * (2.0 + reach), abs=0.01)


@pytest.mark.parametrize("angle", [None, 25])
def test_words_in_an_oval_lie_inside_its_text_area(store, angle):
    manifest = prepared(store)
    obj = _make(manifest, store, text=WORDS)
    if angle:
        _edit(manifest, obj, store, angle=angle)
    layout = obj.drawn_layout
    assert layout.rows and layout.shape == "ellipse"
    r = Rect(**LOW)
    l, t, rr, b = shapes.text_area("ellipse", r.width, r.height)
    area = Rect(x0=r.x0 + l - 1, y0=r.y0 + t - 1, x1=r.x0 + rr + 1, y1=r.y0 + b + 1)
    for row in layout.rows:
        if not row.text:
            continue
        assert area.x0 <= row.x0 and row.x1 <= area.x1, row
        assert area.y0 <= row.baseline - ASCENT_RATIO * layout.size
        assert row.baseline + DESCENT_RATIO * layout.size <= area.y1
    # The frame the editor gets back is hers, never grown to the words.
    if angle:
        assert obj.manual_bbox == r
    else:
        assert obj.drawn_bbox == r


def test_a_shape_is_always_layered_off_the_clean_background(store):
    manifest = prepared(store)
    _make(manifest, store, text="Hola", fill="none", border_color="#ff0000", border_width=3)
    plates: dict = {}
    src = store.get_bytes(manifest.source_key)
    rebuild_document(
        src, manifest.pages, target_lang="es",
        on_plate=lambda i, p: plates.setdefault(i, _rgb(p.get_pixmap(dpi=DPI, alpha=False))),
    )
    assert not _red(plates[0]).any()


def test_covers_and_shapes_are_drawn_even_when_every_font_fails(store, monkeypatch):
    """rebuild_page used to return before painting covers when every
    translation on the page lost its font."""
    from ct_engine.fonts.registry import REGISTRY, FontResolutionError
    from ct_engine.reconstruct import builder

    manifest = prepared(store)
    _make(manifest, store, fill="#0000ff", border_color="none")

    def no_font(*a, **k):
        raise FontResolutionError("x", missing="x")

    monkeypatch.setattr(REGISTRY, "resolve", no_font)
    plates: dict = {}
    doc = fitz.open(stream=store.get_bytes(manifest.source_key), filetype="pdf")
    builder.rebuild_page(doc[0], manifest.page(0), target_lang="es",
                         on_plate=lambda p: plates.setdefault("p", True))
    blue = _blue(_rgb(doc[0].get_pixmap(dpi=DPI, alpha=False)))
    cx, cy = Rect(**LOW).center
    assert blue[_px(cy), _px(cx)] and plates.get("p")


# --- Edits ----------------------------------------------------------------------
def test_a_shape_can_be_made_with_no_words_and_no_colours(store):
    manifest = prepared(store)
    obj = _make(manifest, store, fill="none", border_color="none")
    assert obj.manual_shape == "ellipse" and obj.translated_text == ""
    assert obj.drawn_layout is not None and obj.drawn_layout.shape == "ellipse"
    # A plain box with nothing in it is still refused.
    with pytest.raises(EditError, match="cannot be empty"):
        _make(manifest, store, shape=None, fill="none", border_color="none")


def test_an_unknown_shape_is_refused_when_made_or_changed(store):
    manifest = prepared(store)
    with pytest.raises(EditError, match="not available"):
        _make(manifest, store, shape="star99")
    obj = _make(manifest, store)
    with pytest.raises(EditError, match="not available"):
        _edit(manifest, obj, store, shape="hexagon")


def test_a_shape_cannot_become_a_line(store):
    manifest = prepared(store)
    obj = _make(manifest, store)
    before = len(manifest.page(0).edit_history)
    with pytest.raises(EditError, match="A line cannot become a shape."):
        _edit(manifest, obj, store, shape="line")
    assert len(manifest.page(0).edit_history) == before


def test_a_block_the_engine_found_cannot_take_a_shape(store):
    manifest = prepared(store)
    found = next(t for t in manifest.page(0).text_objects if not t.is_manual_block and t.translated_text)
    with pytest.raises(EditError, match="Only a box you made"):
        _edit(manifest, found, store, shape="ellipse")
    assert found.manual_shape is None


def test_change_shape_keeps_fill_outline_and_words(store):
    manifest = prepared(store)
    obj = _make(manifest, store, text="Hola", fill="#ffff00", border_color="#ff0000", border_width=2)
    _edit(manifest, obj, store, shape="roundRect")
    assert obj.manual_shape == "roundRect" and obj.drawn_layout.shape == "roundRect"
    assert obj.manual_fill == "#ffff00" and obj.manual_border_color == "#ff0000"
    assert obj.translated_text == "Hola"


def test_words_can_be_taken_out_of_a_shape_and_it_stays(store, fake_translate):
    manifest = prepared(store)
    obj = _make(manifest, store)
    translate_from_english(manifest, 1, obj.id, "Write here", store=store, target_lang="es")
    assert obj.translated_text == "[es] Write here" and obj.drawn_layout.lines
    translate_from_english(manifest, 1, obj.id, "", store=store, target_lang="es")
    assert obj.translated_text == "" and obj.approved_source_text == ""
    assert obj.drawn_layout.shape == "ellipse" and not obj.drawn_layout.lines
    apply_edit(manifest, 1, obj.id, "Hola", store=store, target_lang="es")
    apply_edit(manifest, 1, obj.id, "  ", store=store, target_lang="es")
    assert obj.translated_text == "" and obj.manual_shape == "ellipse"


def test_other_boxes_still_refuse_empty_words(store, fake_translate):
    manifest = prepared(store)
    found = next(t for t in manifest.page(0).text_objects if not t.is_manual_block and t.translated_text)
    with pytest.raises(EditError, match="cannot be empty"):
        apply_edit(manifest, 1, found.id, "", store=store, target_lang="es")
    with pytest.raises(EditError, match="between 1 and 8,000"):
        translate_from_english(manifest, 1, found.id, "", store=store, target_lang="es")


def test_reset_keeps_the_shape(store):
    manifest = prepared(store)
    obj = _make(manifest, store, text="Hola", fill="#ffff00", border_color="#ff0000",
                border_width=3, border_dash="dashed", padding=8)
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert obj.manual_shape == "ellipse" and obj.drawn_layout.shape == "ellipse"
    # Back to a new shape's look: no fill, a black 1pt solid outline, the
    # words centred with Medium inside space; the words stay.
    assert obj.manual_fill is None and obj.manual_border_color == "#000000"
    assert obj.manual_border_width == 1.0 and obj.manual_border_dash is None
    assert obj.manual_padding == 4.0 and obj.manual_vertical_align == "middle"
    assert obj.translated_text == "Hola"


def test_reset_leaves_an_empty_unfilled_shape_visible(store):
    manifest = prepared(store)
    obj = _make(manifest, store, fill="none", border_color="#ff0000", border_width=6,
                border_dash="dotted")
    _edit(manifest, obj, store, border_color="none")
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert obj.drawn_layout.border == (0.0, 0.0, 0.0) and obj.drawn_layout.border_width == 1.0
    assert obj.drawn_layout.border_dash is None
    black = (_render(manifest, store).max(axis=2) < 90)
    r = Rect(**LOW)
    assert black[_px(r.y0), _px((r.x0 + r.x1) / 2)]


def test_undo_and_redo_step_through_shape_fill_outline_and_words(store, fake_translate):
    manifest = prepared(store)
    obj = _make(manifest, store)
    steps = [
        dict(shape="roundRect"),
        dict(fill="#ffff00"),
        dict(border_color="#ff0000", border_width=3, border_dash="dashed"),
    ]
    seen = [(obj.manual_shape, obj.manual_fill, obj.manual_border_color, obj.translated_text)]
    for step in steps:
        _edit(manifest, obj, store, **step)
        seen.append((obj.manual_shape, obj.manual_fill, obj.manual_border_color, obj.translated_text))
    translate_from_english(manifest, 1, obj.id, "Name", store=store, target_lang="es")
    seen.append((obj.manual_shape, obj.manual_fill, obj.manual_border_color, obj.translated_text))
    page = manifest.page(0)
    for want in reversed(seen[:-1]):
        undo(manifest, 1, store=store, target_lang="es")
        o = page.find_text_object(obj.id)
        assert (o.manual_shape, o.manual_fill, o.manual_border_color, o.translated_text) == want
    for want in seen[1:]:
        redo(manifest, 1, store=store, target_lang="es")
        o = page.find_text_object(obj.id)
        assert (o.manual_shape, o.manual_fill, o.manual_border_color, o.translated_text) == want


def test_undo_to_a_snapshot_from_before_shapes_makes_it_a_box(store):
    manifest = prepared(store)
    obj = _make(manifest, store, shape=None, fill="#ffffff", border_color="none")
    snap = _snapshot(manifest.page(0))
    for saved in snap["blocks"].values():
        saved.pop("manual_shape")
    for raw in snap["manual_objects"]:
        raw.pop("manual_shape")
    _edit(manifest, obj, store, shape="ellipse")
    manifest.page(0).edit_history.append(snap)
    undo(manifest, 1, store=store, target_lang="es")
    assert manifest.page(0).find_text_object(obj.id).manual_shape is None


def test_duplicate_and_batch_paste_keep_the_shape(store):
    from ct_engine.api.routes import _block_out

    manifest = prepared(store)
    a = _make(manifest, store, text="Hola", shape="roundRect")
    b = _make(manifest, store, shape="ellipse", box={"x0": 300.0, "y0": 600.0, "x1": 372.0, "y1": 672.0})
    specs = []
    for o in (a, b):
        out = _block_out(o)
        assert out["shape"] == o.manual_shape
        frame = o.manual_bbox
        specs.append({
            "bbox": {"x0": frame.x0 + 10, "y0": frame.y0 + 10, "x1": frame.x1 + 10, "y1": frame.y1 + 10},
            "text": out["translation"] or "", "shape": out["shape"],
            "border_color": out["border_color"] or "none", "border_width": out["border_width"],
            "padding": out["padding"],
        })
    _, made = add_blocks(manifest, 1, specs, store=store, target_lang="es")
    assert [m.manual_shape for m in made] == ["roundRect", "ellipse"]
    assert made[0].translated_text == "Hola" and made[1].translated_text == ""
    assert all(m.drawn_layout.shape == m.manual_shape for m in made)


def test_the_bulk_painter_leaves_each_boxs_shape_alone(store):
    manifest = prepared(store)
    oval = _make(manifest, store, text="Hola")
    box = _make(manifest, store, shape=None, text="Adios",
                box={"x0": 300.0, "y0": 600.0, "x1": 400.0, "y1": 640.0})
    apply_bulk(manifest, 1, [oval.id, box.id], fill="#ffff00", border_color="#ff0000",
               border_width=2, store=store, target_lang="es")
    assert oval.manual_shape == "ellipse" and box.manual_shape is None
    assert oval.manual_fill == box.manual_fill == "#ffff00"


def test_bulk_change_shape_skips_blocks_the_engine_found(store):
    manifest = prepared(store)
    oval = _make(manifest, store)
    found = next(t for t in manifest.page(0).text_objects if not t.is_manual_block and t.translated_text)
    apply_bulk(manifest, 1, [oval.id, found.id], shape="rect", store=store, target_lang="es")
    assert oval.manual_shape == "rect" and found.manual_shape is None


# --- Compatibility -----------------------------------------------------------
def test_an_old_manifest_loads_and_draws_unchanged(store):
    manifest = prepared(store)
    before = _render(manifest, store)
    raw = manifest.model_dump(mode="json")
    for page in raw["pages"]:
        for t in page["text_objects"]:
            t.pop("manual_shape")
            if t.get("drawn_layout"):
                t["drawn_layout"].pop("shape")
                t["drawn_layout"].pop("frame")
    old = DocumentManifest.model_validate(raw)
    assert all(t.manual_shape is None for t in old.page(0).text_objects)
    assert np.array_equal(_render(old, store), before)


def test_an_unknown_shape_name_loads_and_draws_a_rectangle(store):
    manifest = prepared(store)
    obj = _make(manifest, store, fill="#0000ff", border_color="none")
    raw = manifest.model_dump(mode="json")
    for t in raw["pages"][0]["text_objects"]:
        if t["id"] == obj.id:
            t["manual_shape"] = "star99"
    later = DocumentManifest.model_validate(raw)
    assert later.page(0).find_text_object(obj.id).manual_shape == "star99"
    blue = _blue(_render(later, store))
    r = Rect(**LOW)
    # Filled right into the corners, as a rectangle is.
    assert blue[_px(r.y0 + 3), _px(r.x0 + 3)] and blue[_px(r.y1 - 3), _px(r.x1 - 3)]


def test_a_ctproject_round_trip_keeps_every_shape_field(store):
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.pipeline.project_package import export_project, import_project

    manifest = prepared(store)
    obj = _make(manifest, store, shape="roundRect", text="Hola", fill="#ffff00")
    _edit(manifest, obj, store, angle=15)
    old_id = "c" * 32
    data = store.get_bytes(manifest.source_key)
    manifest.document_id = old_id
    manifest.source_key = f"documents/{old_id}/source.pdf"
    store.put_bytes(manifest.source_key, data)
    for page in manifest.pages:
        # Only the source travels in this test's package.
        for name in type(page.renders).model_fields:
            if name.endswith("key"):
                setattr(page.renders, name, None)
    ManifestStore(store).save_manifest(manifest)
    reopened = import_project(export_project(old_id, store), store)
    got = reopened.page(0).find_text_object(obj.id)
    assert got.manual_shape == "roundRect" and got.manual_angle == 15
    assert got.drawn_layout.shape == "roundRect" and got.drawn_layout.frame == obj.drawn_layout.frame


# --- API ----------------------------------------------------------------------
def test_the_api_makes_changes_and_lists_shapes(client):
    from test_api import _translate as translate, _upload, _wait_for_stage, _wait_terminal

    doc = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc)
    assert translate(client, doc).status_code in (200, 202)
    assert _wait_terminal(client, doc)["stage"] == "COMPLETE"
    res = client.post(f"/api/documents/{doc}/pages/1/blocks/batch", json={"blocks": [{
        "client_id": "tmp-1", "bbox": LOW, "text": "", "shape": "ellipse",
        "border_color": "#000000", "border_width": 1, "padding": 4,
        "alignment": "CENTER", "vertical_align": "middle",
    }]})
    assert res.status_code == 200, res.text
    made = res.json()["created"][0]["block"]
    assert made["shape"] == "ellipse" and made["sprite_bbox"] and made["translation"] == ""
    listed = client.get(f"/api/documents/{doc}/blocks").json()
    ids = {b["id"]: b for p in listed["pages"] for b in p["blocks"]}
    assert made["id"] in ids and ids[made["id"]]["shape"] == "ellipse"
    url = f"/api/documents/{doc}/pages/1/blocks/{made['id']}"
    res = client.patch(url, json={"shape": "rect"})
    assert res.status_code == 200 and res.json()["block"]["shape"] == "rect"
    res = client.patch(f"/api/documents/{doc}/pages/1/blocks",
                       json={"object_ids": [made["id"]], "shape": "roundRect"})
    assert res.status_code == 200, res.text
    res = client.patch(url, json={"shape": "line"})
    assert res.status_code == 400 and "line" in res.text.lower()
    # Words in, then out again: the shape stays.
    res = client.post(f"{url}/translate-english", json={"text": ""})
    assert res.status_code == 200, res.text
    assert res.json()["block"]["shape"] == "roundRect"
