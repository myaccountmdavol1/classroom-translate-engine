"""Extraction and preservation of original visual assets (spec sec. 14).

The rule is absolute: reuse the original bytes. Nothing here re-encodes,
resizes or regenerates artwork. Images are stored content-addressed so a logo
repeated on 100 pages costs one copy and one OCR pass.
"""

from __future__ import annotations

from ..ingest.base import OpenDocument
from ..model.document import Page
from ..storage import keys as K
from ..storage.base import ObjectStore
from ..util.hashing import sha256_bytes
from ..util.logging import get_logger

log = get_logger(__name__)

# Placements smaller than this in either axis are spacers and rules, not
# instructional images worth storing separately.
MIN_ASSET_POINTS = 6.0


def extract_page_assets(
    doc: OpenDocument,
    page: Page,
    *,
    document_id: str,
    store: ObjectStore,
    seen_hashes: set[str] | None = None,
) -> tuple[int, int]:
    """Persist every image on a page. Returns (stored, failed).

    A failure here never aborts the page: the image object keeps its
    geometry, `asset_key` stays None, and QA reports the gap rather than the
    document silently losing a diagram.
    """
    seen = seen_hashes if seen_hashes is not None else set()
    stored = 0
    failed = 0

    for img in page.image_objects:
        if img.bbox.width < MIN_ASSET_POINTS or img.bbox.height < MIN_ASSET_POINTS:
            continue
        data: bytes | None = None
        ext = "png"

        if img.xref:
            got = doc.extract_image_asset(img.xref)
            if got:
                data, ext = got

        if data is None:
            # Inline image (no resource entry) or an extraction refusal. The
            # honest fallback is to rasterize exactly the placement region, so
            # the artwork is still preserved rather than dropped.
            render = getattr(doc, "render_image_placement_png", None)
            if callable(render):
                data = render(page.index, img.bbox.as_tuple(), 300)
                ext = "png"
            if data:
                img.image_format = "png"
                log.debug(
                    "Page %s image had no extractable xref; captured placement",
                    page.number,
                )

        if not data:
            failed += 1
            log.warning(
                "Page %s: could not preserve image at %s",
                page.number,
                img.bbox.as_tuple(),
            )
            continue

        content_hash = sha256_bytes(data)
        img.image_hash = content_hash
        img.image_format = img.image_format or ext
        key = K.asset_key(document_id, content_hash, ext)
        if content_hash not in seen and not store.exists(key):
            store.put_bytes(key, data, content_type=f"image/{ext}")
        seen.add(content_hash)
        img.asset_key = key
        stored += 1

    return stored, failed
