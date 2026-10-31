"""Putting translated pictures into the teacher's Word file.

The words inside a screenshot are read, translated and drawn on the pages
(pipeline/image_text.py, reconstruct/builder.py). For a Word file the
download also needs them in the .docx, where the picture is a file of its
own (word/media/... or ppt/media/...). So for each picture with translated words:

  * the picture's own pixels are the starting point, scaled up so the new
    words are readable (a 70 dpi screenshot cannot hold 5pt Spanish);
  * only the areas of its translated labels are copied in from the finished
    page, so no other text or picture on the page is baked into it;
  * the result replaces the .docx picture it was made from, found by size
    and appearance -- LibreOffice re-encodes pictures, so the bytes differ.

A picture is left as it is, and counted with its page, when it cannot be
matched, was cropped in Word (the page shows only part of it), or is stored
in a format other than PNG or JPEG. The pages are reported to the teacher:
the editor showed that picture translated, and the saved file does not.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from dataclasses import dataclass, field

from ..model.enums import TextSource
from ..model.geometry import Rect
from ..model.objects import grow_quad
from ..reconstruct.shapes import erase_margin, layout_ink_reach as ink_reach
from ..util.logging import get_logger
from ..util.memory import release_page_memory

log = get_logger(__name__)

# Sharp enough for small translated labels; Word shows it at the same size.
OUTPUT_DPI = 200
# Two pictures are the same when their small grey thumbnails differ by less
# than this (0-255 mean absolute difference) and their shapes agree.
MATCH_DIFFERENCE = 18.0
MATCH_ASPECT = 0.03
_THUMB = (24, 24)


@dataclass
class PictureStats:
    translated: int = 0
    unmatched: int = 0
    names: list[str] = field(default_factory=list)
    # Page number of each picture left with its original words, repeated
    # once per picture. Empty when pictures were given without pages.
    kept_pages: list[int] = field(default_factory=list)

    def keep(self, page_number: int | None) -> None:
        self.unmatched += 1
        if page_number is not None:
            self.kept_pages.append(page_number)


def _thumb(img):
    from PIL import Image

    return list(img.convert("L").resize(_THUMB, Image.BILINEAR).getdata())


def _difference(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _labels_in(page, image):
    """The translated words drawn inside this picture, and the areas they
    cover: upright Rects, or corner lists (perimeter order) for turned ones."""
    areas = []
    # The page's own text, which the file keeps as text: a box she made
    # over any of it is not copied into the picture (see _made_box_in).
    native = [
        t.drawn_bbox or t.bbox
        for t in page.text_objects
        if t.source is TextSource.NATIVE and not t.is_manual_block and not t.suppressed
    ]
    for t in page.text_objects:
        if t.is_manual_block:
            areas.extend(_made_box_in(t, image, native))
            continue
        # Words read from the picture's pixels, by either reader: a slide that
        # is mostly picture is read as a scanned page, and its labels come
        # back as OCR rather than IMAGE_OCR ("Roots grow down" was left out).
        if t.source not in (TextSource.IMAGE_OCR, TextSource.OCR):
            continue
        cx = (t.bbox.x0 + t.bbox.x1) / 2
        cy = (t.bbox.y0 + t.bbox.y1) / 2
        if not (image.bbox.x0 <= cx <= image.bbox.x1 and image.bbox.y0 <= cy <= image.bbox.y1):
            continue
        if t.keep_original or (t.drawn_layout is None and not t.suppressed):
            continue
        layout = t.drawn_layout
        # Past the drawn frame by the seam allowance, and by her border's
        # weight, whose outer half lies outside the frame.
        reach = 2.5 + (ink_reach(layout) if layout is not None else 0.0)
        if t.oriented is not None:
            # Tilted words: the turned strip they were covered in. Their
            # upright boxes hold the artwork they pass over, which must
            # stay the picture's own pixels.
            areas.append(grow_quad(t.oriented.corners(), 2.5))
        else:
            # Where the original words were covered: it stays there however
            # the translation was moved or turned.
            boxes = [t.bbox, *t.cover_boxes]
            boxes += [s.bbox for line in t.lines for s in line.spans]
            # A little past each box, so the seam of the copied patch falls
            # on plain background rather than leaving a hairline at its edge.
            # A hand-lettered word was erased well past its box (erase_margin),
            # so the copy goes that much further, to land on plain background.
            areas.extend(
                b.expanded(2.5 if not t.erase_grown else 2.5 + erase_margin(t, b))
                for b in boxes
            )
        if layout is not None and layout.quad:
            # Drawn turned (along the tilt, or by her): the turned frame.
            areas.append(grow_quad(layout.quad, reach))
        elif t.drawn_bbox is not None:
            areas.append(t.drawn_bbox.expanded(reach))
    return areas


def _made_box_in(t, image, native=()) -> list:
    """The area of a box she made (added, duplicated or pasted) when it is
    drawn wholly inside this picture, else nothing.

    Such a box is pixels on the picture, exactly like a translated label --
    a duplicate of a picture label, or a box pasted onto a diagram -- so it
    goes into the Word or PowerPoint picture too. Only where it is DRAWN:
    it covers nothing of the original, so unlike a label it has no source
    boxes to copy. A box outside every picture, or across a picture's edge,
    has no place in the file's pictures and no paragraph of its own, so
    it is left out of the file on purpose (see word_export); the editor
    says so.

    Nor one whose area overlaps the page's own text (`native`): that text
    is written into the file as text, so copying the finished page there
    would bake it into the picture as well -- on a slide whose background
    is one full-slide picture, every box she made sits "inside" it.
    """
    layout = t.drawn_layout
    if t.suppressed or t.keep_original or layout is None or t.drawn_bbox is None:
        return []
    # Her outline's ink, mitres included: a shape is wholly inside a
    # picture by its ink, not its frame.
    reach = 2.5 + ink_reach(layout)
    if layout.quad:
        xs, ys = zip(*layout.quad)
        extent = (min(xs), min(ys), max(xs), max(ys))
        area = grow_quad(layout.quad, reach)
    else:
        extent = t.drawn_bbox.as_tuple()
        area = t.drawn_bbox.expanded(reach)
    tol = 1.0
    b = image.bbox
    inside = (
        extent[0] >= b.x0 - tol and extent[1] >= b.y0 - tol
        and extent[2] <= b.x1 + tol and extent[3] <= b.y1 + tol
    )
    if not inside:
        return []
    box = Rect(x0=extent[0], y0=extent[1], x1=extent[2], y1=extent[3]).expanded(reach)
    if any(box.intersection_area(r) > 0 for r in native):
        return []
    return [area]


def pages_with_labelled_pictures(manifest) -> list[int]:
    """The page number of each picture whose words were translated, once
    per picture, like PictureStats.kept_pages."""
    return [
        page.number
        for page in manifest.pages
        for image in page.image_objects
        if _labels_in(page, image)
    ]


def translated_pictures(manifest, translated_pdf: bytes, store):
    """(page number, original picture, translated picture) for each picture
    with labels, made one at a time as they are asked for. The translated
    picture is None when it cannot be made, and the original None when it
    could not be opened: both are kept as they are, and reported.

    One at a time because the browser engine has little memory: a long
    book's pictures, all made before any went into the file, did not fit."""
    import fitz

    pdf = fitz.open(stream=translated_pdf, filetype="pdf")
    try:
        for page in manifest.pages:
            for image in page.image_objects:
                areas = _labels_in(page, image)
                if not areas or not image.asset_key:
                    continue
                # Made in a function of its own, so nothing of this picture
                # is still held here while the next one is made.
                yield _translated_picture(pdf, page, image, areas, store)
    finally:
        pdf.close()


def _translated_picture(pdf, page, image, areas, store):
    """One (page number, original, translated) of translated_pictures."""
    import fitz
    from PIL import Image, ImageDraw

    try:
        original = Image.open(io.BytesIO(store.get_bytes(image.asset_key)))
        original.load()
    except Exception as exc:
        log.warning("Picture on page %s could not be opened: %r", page.number, exc)
        return (page.number, None, None)
    box = image.bbox
    width = max(original.width, round(box.width / 72 * OUTPUT_DPI))
    height = round(width * box.height / box.width)
    if abs(original.width / original.height - box.width / box.height) > MATCH_ASPECT * (box.width / box.height):
        return (page.number, original, None)  # cropped in Word
    clip = fitz.Rect(*box.as_tuple())
    zoom = width / box.width
    pix = pdf[page.index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    finished = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).resize((width, height))
    # MuPDF keeps each picture it decoded for the page: dropped, so a long
    # book's pages do not pile up while its pictures are made.
    del pix
    release_page_memory()
    base = original.convert("RGB").resize((width, height), Image.LANCZOS)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for a in areas:
        if isinstance(a, list):
            draw.polygon(
                [((x - box.x0) * zoom, (y - box.y0) * zoom) for x, y in a],
                fill=255,
            )
            continue
        draw.rectangle(
            [(a.x0 - box.x0) * zoom, (a.y0 - box.y0) * zoom,
             (a.x1 - box.x0) * zoom, (a.y1 - box.y0) * zoom],
            fill=255,
        )
    base.paste(finished, (0, 0), mask)
    return (page.number, original, base)


def replace_pictures(docx: bytes, pictures) -> tuple[bytes, PictureStats]:
    """The .docx with each matched picture swapped for its translation.

    `pictures` holds (page number, original, translated) triples, or
    (original, translated) pairs when no page is known: a list, or an
    iterator such as translated_pictures, taken one picture at a time.
    """
    from PIL import Image

    stats = PictureStats()
    zin = zipfile.ZipFile(io.BytesIO(docx))
    # For each picture in the file only what matching needs -- its size,
    # format and small thumbnail -- not the decoded picture: a long book's
    # pictures, all decoded at once, did not fit in the browser's memory.
    media = {}
    for name in zin.namelist():
        # word/media/ in a .docx, ppt/media/ in a .pptx.
        if "/media/" not in name:
            continue
        try:
            img = Image.open(io.BytesIO(zin.read(name)))
            img.load()
        except Exception:
            continue
        with img:
            media[name] = (img.width, img.height, img.format, _thumb(img))

    # Each translation is kept only as the bytes written into the file.
    replacement: dict[str, bytes] = {}
    for item in pictures:
        _replace_one(item, media, replacement, stats)
        # Let go of this picture before the next one is made.
        del item

    if not replacement:
        return docx, stats
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = replacement.get(info.filename) or zin.read(info.filename)
            zout.writestr(info, data)
    return out.getvalue(), stats


def _replace_one(item, media, replacement, stats) -> None:
    """Match one picture to the file's pictures, and keep its encoded bytes."""
    page_number = item[0] if len(item) == 3 else None
    original, translated = item[-2], item[-1]
    if original is None or translated is None:
        stats.keep(page_number)
        return
    want = _thumb(original)
    aspect = original.width / original.height
    best, best_diff = None, MATCH_DIFFERENCE
    for name, (width, height, _, thumb) in media.items():
        if name in replacement or abs(width / height - aspect) > MATCH_ASPECT * aspect:
            continue
        diff = _difference(want, thumb)
        if diff < best_diff:
            best, best_diff = name, diff
    if best is None:
        stats.keep(page_number)
        return
    fmt = media[best][2]
    if fmt not in ("PNG", "JPEG"):
        # Written in its own format or not at all: Word reads the file
        # by its declared type, and a PNG under a .gif name may not open.
        stats.keep(page_number)
        return
    buf = io.BytesIO()
    if fmt == "JPEG":
        translated.save(buf, "JPEG", quality=92)
    else:
        translated.save(buf, "PNG", optimize=True)
    replacement[best] = buf.getvalue()
    stats.translated += 1
    stats.names.append(posixpath.basename(best))
