"""Developer overlay renderer (spec sec. 37).

Draws the page object model on top of the rendered page so a developer can
see exactly what the engine believes is where. This is the fastest way to
catch a coordinate regression -- a misplaced box is obvious in one glance and
invisible in a JSON dump.

Not shown in the teacher-facing UI.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from ..model.document import Page

# Per-layer colors, chosen to stay distinguishable over printed text.
LAYER_STYLE: dict[str, tuple[tuple[int, int, int], int]] = {
    "text": ((0, 110, 220), 2),
    "image": ((200, 60, 0), 3),
    "vector": ((120, 120, 120), 1),
    "table": ((150, 0, 150), 2),
    "form": ((0, 150, 90), 2),
    "response": ((0, 160, 60), 3),
    "protected": ((220, 30, 30), 2),
    "ocr": ((230, 140, 0), 2),
    "collision": ((255, 0, 0), 3),
}

ALL_LAYERS = tuple(LAYER_STYLE.keys())


def render_overlay(
    page_png: bytes,
    page: Page,
    *,
    layers: tuple[str, ...] = ALL_LAYERS,
    labels: bool = True,
) -> bytes:
    """Draw the object model over a rendered page image."""
    with Image.open(io.BytesIO(page_png)) as base:
        im = base.convert("RGB")

    # The preview may be rendered at any DPI, so derive scale from the actual
    # pixel size rather than assuming the configured DPI was used.
    sx = im.width / page.width if page.width else 1.0
    sy = im.height / page.height if page.height else 1.0

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    def box(rect, layer: str, text: str | None = None) -> None:
        color, w = LAYER_STYLE[layer]
        xy = [rect.x0 * sx, rect.y0 * sy, rect.x1 * sx, rect.y1 * sy]
        if xy[2] - xy[0] < 1:
            xy[2] = xy[0] + 1
        if xy[3] - xy[1] < 1:
            xy[3] = xy[1] + 1
        d.rectangle(xy, outline=(*color, 255), width=w)
        if labels and text:
            ty = max(0, xy[1] - 11)
            d.rectangle([xy[0], ty, xy[0] + 7.2 * len(text), ty + 11],
                        fill=(*color, 210))
            d.text((xy[0] + 2, ty), text, fill=(255, 255, 255, 255))

    if "vector" in layers:
        for v in page.vector_objects:
            box(v.bbox, "vector")
    if "image" in layers:
        for i, img in enumerate(page.image_objects):
            box(img.bbox, "image", f"img{i}")
    if "table" in layers:
        for t in page.table_objects:
            box(t.bbox, "table", f"table {t.row_count}x{t.column_count}")
    if "response" in layers:
        for r in page.response_areas:
            box(r.bbox, "response", r.kind)
    if "ocr" in layers:
        # Text read out of a picture, which is a different thing from text
        # the PDF declared and is worth seeing separately.
        for t in page.text_objects:
            if t.source.value == "IMAGE_OCR":
                box(t.bbox, "ocr", "label")

    if "text" in layers:
        for i, t in enumerate(page.text_objects):
            tag = f"{i}:{t.role.value[:4]}" if labels else None
            box(t.bbox, "text", tag)
    if "protected" in layers:
        by_id = {t.id: t for t in page.text_objects}
        for pr in page.protected_regions:
            owner = by_id.get(pr.object_id)
            if owner:
                box(owner.bbox, "protected")
    if "form" in layers:
        for f in page.form_fields:
            box(f.bbox, "form", f.field_type or "field")
    if "collision" in layers:
        for finding in page.qa.findings:
            if finding.bbox is not None and finding.code.startswith("collision"):
                box(finding.bbox, "collision", finding.code[:18])

    composed = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    out = io.BytesIO()
    composed.save(out, format="PNG", optimize=True)
    return out.getvalue()
