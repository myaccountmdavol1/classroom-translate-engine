"""The Shape tool, Stage 2: lines, arrows and curved arrows.

A line is a box she made, `manual_shape = "line"` (or "curve"): a band as
long as the line, LINE_BAND high, turned by `manual_angle`, with its ends in
`manual_line_start` / `manual_line_end` and its colour, weight and dash in
the border fields. These check the engine draws it -- the stroke pulled back
for its ends, each end solid, a curve bulging where the editor draws it --
and that every edit path keeps it a line: no words, no fill, the band never
fattened, Reset keeping its two ends, undo, duplicate, paste, the painter, a
.ctproject round trip and the API.
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
from ct_engine.pipeline.edit import (
    EditError,
    add_blocks,
    apply_bulk,
    apply_edit,
    apply_layout,
    apply_positions,
    redo,
    reset_edit,
    translate_from_english,
    undo,
)
from ct_engine.reconstruct import shapes
from ct_engine.reconstruct.builder import rebuild_document
from ct_engine.reconstruct.sprite import render_sprite, sprite_area
from test_edit import prepared

DPI = 288  # 4 pixels a point
# Clear of everything drawn on 01_clean_worksheet.pdf.
START, END = (100.0, 650.0), (300.0, 650.0)


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
    return (arr[..., 0] > 160) & (arr[..., 1] < 110) & (arr[..., 2] < 110)


def _spec(start=START, end=END, **spec):
    (x0, y0, x1, y1), angle = shapes.line_frame(start, end)
    base = {
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "angle": angle or 0.0, "text": "",
        "shape": "line", "border_color": "#ff0000", "border_width": 2,
    }
    return base | spec


def _line(manifest, store, start=START, end=END, **spec):
    _, (obj,) = add_blocks(manifest, 1, [_spec(start, end, **spec)], store=store, target_lang="es")
    return obj


def _edit(manifest, obj, store, **kw):
    return apply_layout(manifest, 1, obj.id, store=store, target_lang="es", **kw)


def _ends(obj):
    frame = obj.manual_bbox
    return shapes.line_ends((frame.x0, frame.y0, frame.x1, frame.y1), obj.manual_angle)


@pytest.fixture
def fake_translate(monkeypatch):
    from ct_engine.translate.one_off import OneOffTranslation

    def fake(text, *, target_lang, source_lang=None, provider=None):
        return OneOffTranslation(text=f"[{target_lang}] {text}", protected_intact=True, provider="fake")

    monkeypatch.setattr("ct_engine.translate.one_off.translate_one", fake)


# --- Drawing ------------------------------------------------------------------
def test_a_line_is_a_band_from_start_to_end(store):
    manifest = prepared(store)
    obj = _line(manifest, store)
    assert obj.manual_shape == "line" and obj.manual_angle is None
    assert obj.manual_bbox == Rect(x0=100.0, y0=646.0, x1=300.0, y1=654.0)
    assert obj.translated_text == "" and obj.manual_fill is None
    assert obj.drawn_layout.shape == "line" and obj.drawn_layout.frame == obj.manual_bbox
    red = _red(_render(manifest, store))
    y = _px(650)
    assert red[y, _px(110):_px(290)].all()
    # 2pt thick: one point either side, nothing at two and a half.
    assert red[_px(649.3), _px(200)] and not red[_px(652.5), _px(200)]
    assert not red[y, _px(95)] and not red[y, _px(305)]


@pytest.mark.parametrize("weight", [0.5, 1.0, 6.0])
@pytest.mark.parametrize("kind", shapes.LINE_ENDS)
def test_each_end_at_each_weight(store, kind, weight):
    """Every end ends at the end point (an open arrow's mitre a little
    past it), a filled one is filled, an open one is hollow, and the line
    never shows past a filled arrow."""
    manifest = prepared(store)
    obj = _line(manifest, store, border_width=weight, line_end=kind)
    assert obj.manual_line_end == kind and obj.drawn_layout.line_end == kind
    red = _red(_render(manifest, store))
    s = shapes.head_size(weight)
    x, y = END
    base = kind.removeprefix("open_")
    # How far the end's ink reaches past the end point along the line.
    past = {"arrow": 0.0, "circle": s / 2, "square": s / 2, "diamond": s / 2}[base]
    if kind.startswith("open_"):
        past += weight * (1.2 if base == "arrow" else 0.75)
    assert not red[_px(y), _px(x + past) + 3:_px(x + past) + 12].any(), "ink past the end"
    if base in ("arrow", "diamond"):
        # Right up to the end point (an arrow's or a diamond's tip).
        assert red[_px(y), _px(x - 0.6 - (0 if base == "arrow" else s / 2 - 0.6))]
    if base == "arrow":
        # Filled arrow: half its length back, it is solid across most of
        # its width; open: hollow there.
        cx = x - s / 2
        across = red[_px(y - s / 5):_px(y + s / 5) + 1, _px(cx)]
        if kind == "arrow":
            assert across.all()
        else:
            # An open arrow: its arms, the line running between them, and
            # nothing just outside an arm.
            assert red[_px(y), _px(cx)]
            assert not red[_px(y - s / 4 - weight - 1), _px(cx)]
    else:
        # Circle, square, diamond centred on the end: filled through the
        # middle, or hollow there with the line stopped at its near edge.
        if kind.startswith("open_"):
            if s - 2 * weight > 3:
                assert not red[_px(y), _px(x)], "an open end is hollow"
        else:
            assert red[_px(y), _px(x)]


def test_a_dashed_line_has_solid_ends(store):
    manifest = prepared(store)
    obj = _line(manifest, store, border_width=2, border_dash="dashed", line_end="arrow",
                line_start="open_square")
    assert obj.manual_border_dash == "dashed"
    red = _red(_render(manifest, store))
    row = red[_px(650), _px(110):_px(290)]
    # The stroke dashes 6pt on, 4pt off...
    assert 0.45 < row.mean() < 0.75
    # ...and the arrow is solid: filled across its back half.
    s = shapes.head_size(2)
    assert red[_px(650) - 2:_px(650) + 3, _px(300 - s / 3)].all()


def test_the_line_stops_at_an_arrows_base(store):
    """No stroke inside a filled arrow, so a translucent or thin arrow tip
    is never blunted: rendered without its head, the stroke ends at the
    base."""
    manifest = prepared(store)
    obj = _line(manifest, store, border_width=6, line_end="arrow")
    stroke, heads = shapes.line_parts(200.0, 6.0, None, "arrow")
    s = shapes.head_size(6)
    assert stroke[-1] == ("L", 200.0 - s, 4.0)
    assert obj.drawn_layout.line_end == "arrow"


def test_a_right_to_left_arrow_points_left(store):
    manifest = prepared(store)
    obj = _line(manifest, store, start=(300.0, 650.0), end=(100.0, 650.0), line_end="arrow")
    assert obj.manual_angle == 180.0
    (sx, sy), (ex, ey) = _ends(obj)
    assert math.isclose(ex, 100.0, abs_tol=0.01) and math.isclose(sx, 300.0, abs_tol=0.01)
    red = _red(_render(manifest, store))
    s = shapes.head_size(2)
    # The arrow's wide base is near x=100+s, its tip at 100.
    assert red[_px(650 + s / 2 - 1), _px(100 + s - 0.7)]
    assert not red[_px(650 + s / 2 - 1), _px(300 - s + 0.7)]


def test_a_line_drawn_upward_points_up(store):
    manifest = prepared(store)
    obj = _line(manifest, store, start=(200.0, 700.0), end=(200.0, 560.0), line_end="arrow")
    assert obj.manual_angle == -90.0
    red = _red(_render(manifest, store))
    s = shapes.head_size(2)
    assert red[_px(560 + s - 0.7), _px(200 + s / 2 - 1)]
    assert not red[_px(700 - s + 0.7), _px(200 + s / 2 - 1)]


def test_a_curve_bulges_to_its_bend_and_its_arrow_ends_at_the_end(store):
    manifest = prepared(store)
    obj = _line(manifest, store, shape="curve", shape_adjust=[25000], line_end="arrow")
    assert obj.manual_shape == "curve" and obj.manual_shape_adjust == [25000.0]
    assert obj.drawn_layout.shape_adjust == [25000.0]
    red = _red(_render(manifest, store))
    # The middle sits a quarter of 200pt above the straight line.
    assert red[_px(600), _px(200)]
    assert not red[_px(650), _px(200)]
    # Near the end the curve comes down to the end point.
    assert red[_px(650) - 2:_px(650) + 3, _px(299):_px(300)].any()
    # A turned curve is the same curve turned: drawn downward it bulges right
    # (its band's top is the page's right when it runs down).
    other = _line(manifest, store, start=(450.0, 560.0), end=(450.0, 760.0), shape="curve",
                  shape_adjust=[25000])
    assert other.manual_angle == 90.0
    red = _red(_render(manifest, store))
    assert red[_px(660), _px(500)] and not red[_px(660), _px(400)]


@pytest.mark.parametrize("angle_end", [END, (240.0, 560.0)])
@pytest.mark.parametrize("kind", ["arrow", "open_circle", "square"])
def test_the_sprite_holds_the_heads(store, kind, angle_end):
    manifest = prepared(store)
    obj = _line(manifest, store, end=angle_end, border_width=6, line_end=kind, line_start=kind)
    sprite = np.asarray(Image.open(io.BytesIO(render_sprite(obj, scale=2.0))).convert("RGBA"))
    area = sprite_area(obj)
    assert sprite.shape[1] == pytest.approx(area.width * 2, abs=1)
    alpha = sprite[:, :, 3]
    assert not alpha[0].any() and not alpha[-1].any()
    assert not alpha[:, 0].any() and not alpha[:, -1].any()
    assert _red(sprite[:, :, :3])[alpha > 200].sum() > 500


def test_the_sprite_holds_a_curves_bulge(store):
    manifest = prepared(store)
    obj = _line(manifest, store, shape="curve", shape_adjust=[-50000], line_end="arrow",
                border_width=4)
    sprite = np.asarray(Image.open(io.BytesIO(render_sprite(obj, scale=2.0))).convert("RGBA"))
    alpha = sprite[:, :, 3]
    assert not alpha[0].any() and not alpha[-1].any()
    assert not alpha[:, 0].any() and not alpha[:, -1].any()
    assert alpha.any()


def test_word_pictures_reach_takes_in_the_heads(store):
    from ct_engine.reconstruct.shapes import layout_ink_reach

    manifest = prepared(store)
    obj = _line(manifest, store, border_width=6, line_end="arrow")
    assert layout_ink_reach(obj.drawn_layout) == pytest.approx(shapes.head_size(6) / 2)


# --- Edits ----------------------------------------------------------------------
def test_a_line_has_no_words_and_no_fill(store, fake_translate):
    manifest = prepared(store)
    with pytest.raises(EditError, match="A line has no words."):
        _line(manifest, store, text="Hola")
    obj = _line(manifest, store, fill="#ffff00")
    assert obj.manual_fill is None
    with pytest.raises(EditError, match="A line has no words."):
        apply_edit(manifest, 1, obj.id, "Hola", store=store, target_lang="es")
    with pytest.raises(EditError, match="A line has no words."):
        translate_from_english(manifest, 1, obj.id, "Hello", store=store, target_lang="es")
    _edit(manifest, obj, store, fill="#ffff00")
    assert obj.manual_fill is None


def test_a_line_always_has_a_colour(store):
    manifest = prepared(store)
    obj = _line(manifest, store, border_color="none")
    assert obj.manual_border_color == "#000000"
    before = len(manifest.page(0).edit_history)
    with pytest.raises(EditError, match="A line needs a colour."):
        _edit(manifest, obj, store, border_color="none")
    assert len(manifest.page(0).edit_history) == before


def test_ends_belong_to_lines_and_a_bend_to_curves(store):
    manifest = prepared(store)
    with pytest.raises(EditError, match="not available"):
        _line(manifest, store, line_end="star")
    obj = _line(manifest, store)
    with pytest.raises(EditError, match="nothing to adjust"):
        _edit(manifest, obj, store, shape_adjust=[10000])
    _, (oval,) = add_blocks(manifest, 1, [{
        "bbox": {"x0": 400.0, "y0": 600.0, "x1": 472.0, "y1": 672.0}, "text": "",
        "shape": "ellipse", "border_color": "#000000",
    }], store=store, target_lang="es")
    with pytest.raises(EditError, match="Only a line has ends."):
        _edit(manifest, oval, store, line_end="arrow")
    with pytest.raises(EditError, match="A line cannot become a shape."):
        _edit(manifest, obj, store, shape="ellipse")
    with pytest.raises(EditError, match="A line cannot become a shape."):
        _edit(manifest, oval, store, shape="curve")
    curve = _line(manifest, store, shape="curve")
    with pytest.raises(EditError, match="bent too far"):
        _edit(manifest, curve, store, shape_adjust=[60000])
    # Straightened, a curve loses its bend; curved again, it takes the
    # default one.
    _edit(manifest, curve, store, shape_adjust=[-30000])
    _edit(manifest, curve, store, shape="line")
    assert curve.manual_shape == "line" and curve.manual_shape_adjust is None


def test_changing_ends_bend_and_look(store):
    manifest = prepared(store)
    obj = _line(manifest, store, shape="curve", shape_adjust=[10000])
    _edit(manifest, obj, store, line_start="circle", line_end="open_arrow", shape_adjust=[-20000],
          border_color="#0000ff", border_width=4, border_dash="dotted")
    assert (obj.manual_line_start, obj.manual_line_end) == ("circle", "open_arrow")
    assert obj.manual_shape_adjust == [-20000.0]
    assert obj.drawn_layout.border == (0.0, 0.0, 1.0) and obj.drawn_layout.border_width == 4
    _edit(manifest, obj, store, line_end="none")
    assert obj.manual_line_end is None


def test_a_band_is_never_fattened(store):
    manifest = prepared(store)
    obj = _line(manifest, store, start=(100.0, 650.0), end=(250.0, 600.0))
    # A group resize hands back a taller frame: the line keeps its band.
    apply_positions(manifest, 1, [{"id": obj.id, "bbox": {"x0": 90.0, "y0": 600.0, "x1": 290.0, "y1": 640.0}}],
                    store=store, target_lang="es")
    assert obj.manual_bbox == Rect(x0=90.0, y0=616.0, x1=290.0, y1=624.0)
    _edit(manifest, obj, store, bbox={"x0": 90.0, "y0": 600.0, "x1": 290.0, "y1": 700.0},
          angle=obj.manual_angle)
    assert obj.manual_bbox.height == pytest.approx(shapes.LINE_BAND)
    _, (copy,) = add_blocks(manifest, 1, [_spec() | {"bbox": {"x0": 100.0, "y0": 600.0, "x1": 300.0, "y1": 700.0}}],
                            store=store, target_lang="es")
    assert copy.manual_bbox == Rect(x0=100.0, y0=646.0, x1=300.0, y1=654.0)


def test_reset_keeps_the_lines_ends_and_gives_it_a_new_lines_look(store):
    manifest = prepared(store)
    obj = _line(manifest, store, shape="curve", shape_adjust=[-30000], line_end="arrow",
                line_start="diamond", border_dash="dashed", border_width=6)
    _edit(manifest, obj, store, bbox={"x0": 120.0, "y0": 600.0, "x1": 320.0, "y1": 608.0}, angle=30)
    frame, angle = obj.manual_bbox, obj.manual_angle
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    assert (obj.manual_bbox, obj.manual_angle) == (frame, angle)
    assert (obj.manual_shape, obj.manual_shape_adjust) == ("curve", [-30000.0])
    assert (obj.manual_line_start, obj.manual_line_end) == ("diamond", "arrow")
    assert obj.manual_border_color == "#000000" and obj.manual_border_width == 1.0
    assert obj.manual_border_dash is None and obj.manual_fill is None
    assert obj.drawn_layout is not None and obj.drawn_layout.shape == "curve"


def test_undo_and_redo_step_through_ends_and_bend(store):
    manifest = prepared(store)
    obj = _line(manifest, store, shape="curve")
    page = manifest.page(0)
    seen = [(obj.manual_line_end, obj.manual_shape_adjust, obj.manual_angle)]
    _edit(manifest, obj, store, line_end="arrow")
    seen.append((obj.manual_line_end, obj.manual_shape_adjust, obj.manual_angle))
    _edit(manifest, obj, store, shape_adjust=[-40000])
    seen.append((obj.manual_line_end, obj.manual_shape_adjust, obj.manual_angle))
    _edit(manifest, obj, store, bbox={"x0": 100.0, "y0": 600.0, "x1": 300.0, "y1": 608.0}, angle=180)
    seen.append((obj.manual_line_end, obj.manual_shape_adjust, obj.manual_angle))
    for want in reversed(seen[:-1]):
        undo(manifest, 1, store=store, target_lang="es")
        o = page.find_text_object(obj.id)
        assert (o.manual_line_end, o.manual_shape_adjust, o.manual_angle) == want
    for want in seen[1:]:
        redo(manifest, 1, store=store, target_lang="es")
        o = page.find_text_object(obj.id)
        assert (o.manual_line_end, o.manual_shape_adjust, o.manual_angle) == want


def test_duplicate_and_paste_keep_ends_and_bend(store):
    from ct_engine.api.routes import _block_out

    manifest = prepared(store)
    a = _line(manifest, store, line_end="arrow", line_start="open_circle", angle=0)
    b = _line(manifest, store, start=(100.0, 700.0), end=(300.0, 600.0), shape="curve",
              shape_adjust=[-20000], line_end="open_arrow")
    specs = []
    for o in (a, b):
        out = _block_out(o)
        assert (out["shape"], out["line_start"], out["line_end"], out["shape_adjust"]) == (
            o.manual_shape, o.manual_line_start, o.manual_line_end, o.manual_shape_adjust)
        frame = o.manual_bbox
        specs.append({
            "bbox": {"x0": frame.x0 + 10, "y0": frame.y0 + 10, "x1": frame.x1 + 10, "y1": frame.y1 + 10},
            "angle": o.manual_angle or 0, "text": "", "shape": out["shape"],
            "shape_adjust": out["shape_adjust"], "line_start": out["line_start"],
            "line_end": out["line_end"], "border_color": out["border_color"],
            "border_width": out["border_width"],
        })
    _, made = add_blocks(manifest, 1, specs, store=store, target_lang="es")
    got = [(m.manual_shape, m.manual_shape_adjust, m.manual_line_start, m.manual_line_end,
            m.manual_angle) for m in made]
    assert got == [("line", None, "open_circle", "arrow", None),
                   ("curve", [-20000.0], None, "open_arrow", b.manual_angle)]


def test_the_bulk_painter_gives_ends_only_to_lines(store):
    manifest = prepared(store)
    line = _line(manifest, store)
    _, (box,) = add_blocks(manifest, 1, [{
        "bbox": {"x0": 400.0, "y0": 600.0, "x1": 500.0, "y1": 640.0}, "text": "Hola",
        "border_color": "#00ff00", "border_width": 1,
    }], store=store, target_lang="es")
    apply_bulk(manifest, 1, [line.id, box.id], line_end="arrow", border_color="none",
               fill="#ffff00", store=store, target_lang="es")
    assert line.manual_line_end == "arrow" and box.manual_line_end is None
    # The box loses its border; the line keeps its colour and takes no fill.
    assert box.manual_border_color is None and line.manual_border_color == "#ff0000"
    assert line.manual_fill is None and box.manual_fill == "#ffff00"


# --- Compatibility -----------------------------------------------------------
def test_an_old_manifest_without_line_fields_loads(store):
    manifest = prepared(store)
    obj = _line(manifest, store, line_end="arrow")
    raw = manifest.model_dump(mode="json")
    for t in raw["pages"][0]["text_objects"]:
        for name in ("manual_shape_adjust", "manual_line_start", "manual_line_end"):
            t.pop(name)
        if t.get("drawn_layout"):
            for name in ("shape_adjust", "line_start", "line_end"):
                t["drawn_layout"].pop(name)
    old = DocumentManifest.model_validate(raw)
    got = old.page(0).find_text_object(obj.id)
    assert got.manual_shape == "line" and got.manual_line_end is None
    # Drawn as a plain line.
    red = _red(_render(old, store))
    assert red[_px(650), _px(200)]


def test_a_ctproject_round_trip_keeps_every_line_field(store):
    from ct_engine.pipeline.manifest_store import ManifestStore
    from ct_engine.pipeline.project_package import export_project, import_project

    manifest = prepared(store)
    obj = _line(manifest, store, start=(300.0, 700.0), end=(100.0, 600.0), shape="curve",
                shape_adjust=[-15000], line_start="square", line_end="arrow")
    old_id = "d" * 32
    data = store.get_bytes(manifest.source_key)
    manifest.document_id = old_id
    manifest.source_key = f"documents/{old_id}/source.pdf"
    store.put_bytes(manifest.source_key, data)
    for page in manifest.pages:
        for name in type(page.renders).model_fields:
            if name.endswith("key"):
                setattr(page.renders, name, None)
    ManifestStore(store).save_manifest(manifest)
    reopened = import_project(export_project(old_id, store), store)
    got = reopened.page(0).find_text_object(obj.id)
    assert (got.manual_shape, got.manual_shape_adjust, got.manual_line_start, got.manual_line_end,
            got.manual_angle) == ("curve", [-15000.0], "square", "arrow", obj.manual_angle)
    assert got.drawn_layout.line_end == "arrow"


# --- API ----------------------------------------------------------------------
def test_the_api_makes_and_changes_lines(client):
    from test_api import _translate as translate, _upload, _wait_for_stage, _wait_terminal

    doc = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc)
    assert translate(client, doc).status_code in (200, 202)
    assert _wait_terminal(client, doc)["stage"] == "COMPLETE"
    spec = _spec(line_end="arrow") | {"client_id": "tmp-1"}
    res = client.post(f"/api/documents/{doc}/pages/1/blocks/batch", json={"blocks": [spec]})
    assert res.status_code == 200, res.text
    made = res.json()["created"][0]["block"]
    assert made["shape"] == "line" and made["line_end"] == "arrow" and made["line_start"] is None
    assert made["sprite_bbox"] and made["translation"] == ""
    listed = client.get(f"/api/documents/{doc}/blocks").json()
    ids = {b["id"]: b for p in listed["pages"] for b in p["blocks"]}
    assert ids[made["id"]]["line_end"] == "arrow"
    url = f"/api/documents/{doc}/pages/1/blocks/{made['id']}"
    res = client.patch(url, json={"line_start": "circle", "shape": "curve", "shape_adjust": [20000]})
    assert res.status_code == 200, res.text
    out = res.json()["block"]
    assert (out["shape"], out["line_start"], out["shape_adjust"]) == ("curve", "circle", [20000.0])
    res = client.patch(f"/api/documents/{doc}/pages/1/blocks",
                       json={"object_ids": [made["id"]], "line_end": "open_diamond"})
    assert res.status_code == 200, res.text
    res = client.post(f"{url}/translate-english", json={"text": "Hello"})
    assert res.status_code == 400 and "no words" in res.text
