"""Page preview and thumbnail generation.

Previews are what the review viewer shows, so they must match the source
page exactly in aspect ratio -- a preview that crops or letterboxes would
make the side-by-side comparison lie about layout.
"""

from __future__ import annotations

import io

from PIL import Image

from ..config import get_settings
from ..ingest.base import OpenDocument
from ..model.document import Page, PageRenderRefs
from ..storage import keys as K
from ..storage.base import ObjectStore
from ..util.logging import get_logger
from .base import RenderError

log = get_logger(__name__)


def make_thumbnail(png_bytes: bytes, width_px: int) -> tuple[bytes, int, int]:
    """Downscale a preview to thumbnail width, preserving aspect ratio.

    The fallback for when a page cannot simply be rendered small. Kept cheap
    on purpose: `optimize=True` and a full LANCZOS pass cost twice the page
    render on the free host and save almost nothing at ~220 px wide.
    """
    with Image.open(io.BytesIO(png_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w <= width_px:
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue(), w, h
        ratio = width_px / float(w)
        new_h = max(1, int(round(h * ratio)))
        # reducing_gap shrinks by whole factors first, then filters the
        # last step: near-LANCZOS quality at a fraction of the work.
        thumb = im.resize((width_px, new_h), Image.BILINEAR, reducing_gap=2.0)
        out = io.BytesIO()
        thumb.save(out, format="PNG")
        return out.getvalue(), width_px, new_h


def render_source_page(
    doc: OpenDocument,
    page: Page,
    *,
    document_id: str,
    store: ObjectStore,
    dpi: int | None = None,
) -> PageRenderRefs:
    """Rasterize one source page and persist preview + thumbnail."""
    settings = get_settings()
    dpi = dpi or settings.preview_dpi

    try:
        png = doc.render_page_png(page.index, dpi=dpi)
    except Exception as exc:
        raise RenderError(
            f"Page {page.number} could not be turned into a preview image.",
            diagnostic=repr(exc),
        ) from exc

    # Opening reads the PNG header only; the pixels are never decoded.
    with Image.open(io.BytesIO(png)) as im:
        width_px, height_px = im.size

    preview_key = K.page_source_render_key(document_id, page.index)
    store.put_bytes(preview_key, png, content_type="image/png")

    # The thumbnail is rendered again, small, rather than decoded from the
    # preview and shrunk: about 1 ms against 35 ms a page.
    thumb_bytes: bytes | None = None
    if width_px > settings.thumbnail_width_px:
        # Rounded down: a thumbnail is never wider than its slot.
        thumb_dpi = max(1, int(dpi * settings.thumbnail_width_px / width_px))
        try:
            thumb_bytes = doc.render_page_png(page.index, dpi=thumb_dpi)
            with Image.open(io.BytesIO(thumb_bytes)) as im:
                if im.width > settings.thumbnail_width_px:
                    thumb_bytes = None  # rounding overshot; shrink instead
        except Exception as exc:
            log.warning(
                "Small render of page %s failed; shrinking the preview: %r",
                page.number, exc,
            )
    if thumb_bytes is None:
        thumb_bytes, _, _ = make_thumbnail(png, settings.thumbnail_width_px)
    del png
    thumb_key = K.page_source_thumb_key(document_id, page.index)
    store.put_bytes(thumb_key, thumb_bytes, content_type="image/png")

    refs = page.renders
    refs.source_preview_key = preview_key
    refs.source_thumb_key = thumb_key
    refs.preview_dpi = dpi
    refs.preview_width_px = width_px
    refs.preview_height_px = height_px
    return refs
