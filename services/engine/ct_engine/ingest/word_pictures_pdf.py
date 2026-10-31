"""A Word file's pictures as pages, for when nothing here can lay it out.

In the browser there is no LibreOffice, so a Word file cannot be shown as
pages. Its words do not need pages: at download every paragraph is
translated straight from the file (pipeline/word_export.py). The words
inside its pictures do: they are read, translated and redrawn on a page
(pipeline/image_text.py). So each picture the document shows gets a page of
its own, at its own pixels, and from there the file takes the path a Word
file laid out by LibreOffice already takes: the translated pictures go back
into word/media/ (pipeline/word_pictures.py).
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from dataclasses import dataclass

import fitz
import numpy as np
from PIL import Image

from .docx_walk import parse_xml
from .image_import import image_to_pdf

# Points of the picture's own background kept around its page. The engine
# reads a label's background from a ring just outside its box, clipped to
# the page; on a page that IS the picture, a label spanning the picture has
# that ring clipped onto the picture's edge, where big letters touch it, and
# is refused. A margin of the edge colour gives the ring plain paper.
MARGIN = 6.0
_EDGE_STRIP = 2  # pixels
_EDGE_SHARE = 0.7  # of the edge pixels that must match its colour...
_EDGE_TOLERANCE = 24  # ...within this many levels, on every channel
_DOCUMENT = "word/document.xml"
_RELS = "word/_rels/document.xml.rels"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


@dataclass
class PicturesPdf:
    # None when no picture could be given a page.
    pdf: bytes | None
    # Distinct pictures the document's body shows, and how many have a page.
    # The rest (not PNG or JPEG, or too large) stay exactly as they are.
    shown: int
    paged: int


def _image_targets(z: zipfile.ZipFile) -> dict[str, str]:
    """Relationship id -> package path, for the body's pictures."""
    targets: dict[str, str] = {}
    for rel in parse_xml(z.read(_RELS)).iter(f"{_PKG}Relationship"):
        if rel.get("Type") != _IMAGE or rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target") or ""
        path = target.lstrip("/") if target.startswith("/") else posixpath.join("word", target)
        targets[rel.get("Id") or ""] = posixpath.normpath(path)
    return targets


def _plain_edge(image: bytes) -> tuple[float, float, float] | None:
    """The colour (0-1) of the picture's edge when it is plain, else None."""
    with Image.open(io.BytesIO(image)) as im:
        arr = np.asarray(im.convert("RGB"))
    n = _EDGE_STRIP
    edge = np.concatenate([
        arr[:n].reshape(-1, 3), arr[-n:].reshape(-1, 3),
        arr[:, :n].reshape(-1, 3), arr[:, -n:].reshape(-1, 3),
    ]).astype(int)
    median = np.median(edge, axis=0)
    near = (np.abs(edge - median) <= _EDGE_TOLERANCE).all(axis=1)
    if near.mean() < _EDGE_SHARE:
        return None
    return tuple(float(c) / 255.0 for c in median)


def _add_page(out: fitz.Document, flat: bytes) -> None:
    """Add the flattened picture's page, with a margin when its edge is plain."""
    with fitz.open(stream=flat, filetype="pdf") as one:
        src = one[0]
        xref = src.get_images(full=True)[0][0]
        image = one.extract_image(xref)["image"]
        colour = _plain_edge(image)
        if colour is None:
            out.insert_pdf(one)
            return
        w, h = src.rect.width, src.rect.height
        page = out.new_page(width=w + 2 * MARGIN, height=h + 2 * MARGIN)
        page.draw_rect(page.rect, color=None, fill=colour)
        page.insert_image(fitz.Rect(MARGIN, MARGIN, MARGIN + w, MARGIN + h), stream=image)


def pictures_pdf(docx: bytes) -> PicturesPdf:
    """One page per distinct picture the body shows, in reading order."""
    with zipfile.ZipFile(io.BytesIO(docx)) as z:
        names = set(z.namelist())
        if _DOCUMENT not in names or _RELS not in names:
            return PicturesPdf(None, 0, 0)
        targets = _image_targets(z)
        order: list[str] = []
        for el in parse_xml(z.read(_DOCUMENT)).iter():
            # r:embed in a drawing, r:id in an older VML picture.
            for attr in (f"{_R}embed", f"{_R}id"):
                path = targets.get(el.get(attr) or "")
                if path and path in names and path not in order:
                    order.append(path)

        out = fitz.open()
        try:
            for path in order:
                try:
                    page, _ = image_to_pdf(z.read(path))
                except ValueError:
                    continue
                _add_page(out, page)
            paged = out.page_count
            return PicturesPdf(out.tobytes() if paged else None, len(order), paged)
        finally:
            out.close()
