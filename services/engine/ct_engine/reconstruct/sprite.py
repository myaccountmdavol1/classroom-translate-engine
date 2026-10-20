"""One translated block, drawn on its own with a transparent background.

The live editor places these over the page's clean background, so moving a
block is moving a picture: instant, with nothing to re-render. Each sprite
is drawn from the block's DrawnLayout by MuPDF, with the font file the PDF
embeds, so it is the same ink the download will have -- in every script,
with no fonts shipped to the browser.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from ..fonts.registry import REGISTRY
from ..model.geometry import Rect
from ..model.objects import DrawnLayout, TextObject
from . import shapes
from .builder import draw_decorations, paint_border, paint_fill, turn_morph, turned_frame_of

# Room around the frame for ascenders, accents and anti-aliasing that fall
# just outside the measured box.
PAD = 2.0


def sprite_pad(layout: DrawnLayout) -> float:
    """Room around the frame: PAD, plus how far her outline reaches past it."""
    return PAD + shapes.layout_ink_reach(layout)


def sprite_area(obj: TextObject) -> Rect:
    """The sprite's area in page points, in the block's UNTURNED frame.

    Upright: drawn_bbox grown by sprite_pad. Turned: the unturned frame
    (builder.turned_frame_of) grown by sprite_pad; the editor turns the
    picture with the box about the frame's centre, which draws the same
    quad as the PDF (see turned_frame_of).
    """
    layout, box = obj.drawn_layout, obj.drawn_bbox
    if layout is None or box is None:
        raise ValueError("This block has not been drawn yet.")
    pad = sprite_pad(layout)
    frame = turned_frame_of(layout)[0] if layout.quad else box
    return frame.expanded(pad)


def render_sprite(obj: TextObject, *, scale: float = 2.0) -> bytes:
    """PNG of the block's drawn text over `sprite_area`, transparent.

    Upright, the image covers drawn_bbox grown by sprite_pad on each side.
    Turned, it is drawn UNTURNED: the frame grown by sprite_pad, which the
    editor turns in CSS together with the box. Either way a client places
    it at sprite_area's top-left in page points.
    """
    layout, box = obj.drawn_layout, obj.drawn_bbox
    if layout is None or box is None:
        raise ValueError("This block has not been drawn yet.")
    # A cover has a fill and no words, so no font to load.
    font = (
        REGISTRY._load(
            builtin=layout.font_builtin,
            path=Path(layout.font_path) if layout.font_path else None,
        )
        if layout.lines
        else None
    )
    if layout.lines and font is None:
        raise ValueError("The font this block was drawn with is not available.")
    area = sprite_area(obj)
    doc = fitz.open()
    try:
        page = doc.new_page(width=area.width, height=area.height)
        # Page points to sprite points for the frame (fill and border) and
        # for the lines: the same upright, while a turned layout's lines
        # also move into its frame.
        fx, fy = -area.x0, -area.y0
        dx, dy = fx, fy
        if layout.quad:
            frame, sx, sy = turned_frame_of(layout)
            dx, dy = fx + sx, fy + sy
            box = frame
            layout = layout.model_copy(update={"angle": 0.0, "pivot": None, "quad": None})
        paint_fill(page, layout, box, fx, fy)
        # Highlight and underline between the fill and the words, as in the
        # PDF. Justified rows need nothing here: they are already one entry
        # per word in `lines`.
        draw_decorations(page, layout, dx, dy)
        if layout.lines:
            writer = fitz.TextWriter(page.rect)
            for x, y, text in layout.lines:
                writer.append(
                    fitz.Point(x + dx, y + dy),
                    text,
                    font=font,
                    fontsize=layout.size,
                )
            # List markers, each in the font it was drawn in on the page,
            # in the same writer so they take the same colour and turn.
            for mark in layout.marks:
                mfont = REGISTRY._load(
                    builtin=mark.font_builtin,
                    path=Path(mark.font_path) if mark.font_path else None,
                )
                if mfont is None:
                    raise ValueError("The font a list marker was drawn with is not available.")
                writer.append(
                    fitz.Point(mark.x + dx, mark.baseline + dy), mark.text,
                    font=mfont, fontsize=layout.size,
                )
            writer.write_text(page, color=layout.color, morph=turn_morph(layout, dx, dy))
        paint_border(page, layout, box, fx, fy)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
        return pix.tobytes("png")
    finally:
        doc.close()
