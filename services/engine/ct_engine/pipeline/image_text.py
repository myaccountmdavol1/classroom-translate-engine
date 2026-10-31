"""Translating text baked into images (spec sec. 18 / 19).

Diagram labels -- "Cell Wall", "Nucleus", "Atlantic Ocean", "Step 1" -- are
instructional content that happens to be stored as pixels. Leaving them in
the source language means a student gets a translated worksheet wrapped
around an untranslated diagram.

APPROACH: the original image is never edited or re-encoded.

The image region is rendered, read with sparse-text OCR, and each label
becomes an ordinary TextObject in page coordinates with
`source = IMAGE_OCR`. From there the existing machinery does the work: the
protection layer, the translator, the fitter, and the same
cover-with-sampled-background renderer built for scanned pages. That means:

  * the original asset is reused byte-for-byte (spec sec. 14),
  * translated labels are selectable text, not baked-in pixels,
  * and the background-uniformity check already refuses to paint over
    artwork, which is exactly the protection a diagram needs.

CONFIDENCE IS DELIBERATELY STRICTER HERE than for a scanned page. Replacing
a label means painting over artwork; a wrong label on a diagram is worse
than an untranslated one, so anything we cannot read confidently is left
alone and reported.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace

from ..analysis.text_angle import Orientation, measure_region
from ..config import get_settings
from ..ingest.base import OpenDocument
from ..model.document import Page, QAFinding
from ..model.enums import ObjectKind, SemanticRole, TextSource
from ..model.geometry import Rect
from ..model.objects import ImageObject, TextLine, TextObject, TextSpan, TextStyle, _new_id
from ..providers.ocr.base import OCRProvider, OCRRegion, OCRResult, OCRWord
from ..util.logging import get_logger

log = get_logger(__name__)

# A label must clear this to be replaced. Higher than page OCR: see module docstring.
MIN_LABEL_CONFIDENCE = 0.62
# Images smaller than this in either axis carry no readable label.
MIN_IMAGE_POINTS = 48.0
# An image covering this much of a scanned page is the scan itself, already
# handled by the page-level OCR stage.
FULL_PAGE_FRACTION = 0.70
# Rendering density for reading labels. Always this, even for a picture
# stored at a lower density: a screenshot pasted into a handout is often
# ~70 dpi, and read at 150 its 5pt captions were 10px tall and came back as
# scraps ("also be", "RL whe") on the real installation guide. At 300 the
# same pictures read as whole lines. 400 was measurably worse.
LABEL_OCR_DPI = 300
# A word read below this is a guess. One guess in a line makes the whole
# line uncertain: its translation would be built on a misreading.
UNSURE_WORD_CONFIDENCE = 0.50
# Pieces of one printed line closer than this many line-heights are one
# line: sparse reading splits a heading at a wide letter gap ("Add Chrome
# app or ext" | "sion by ID"), and each half was translated alone.
SAME_LINE_GAP = 0.8
# Words further apart than this many line-heights are in different boxes:
# Tesseract read a planning guide's row of six course boxes as one line,
# with 0.4 line-heights between words in a box and 2.5 to 5 between boxes.
# Justified text stretches a space to about one.
BOX_GAP = 1.5
# A sentence read below MIN_LABEL_CONFIDENCE but above this, with at least
# SENTENCE_WORDS real words, is offered to her to check rather than thrown
# away: on the installation guide that was both sentences of a dialog.
CHECKABLE_CONFIDENCE = 0.50
SENTENCE_WORDS = 3
# Measuring a line's slant costs a crop and a labelling pass (~20 ms); a
# re-read of a tilted line one more Tesseract call (~50-200 ms). Bounded per
# picture and per page so a busy collage cannot run the free host long.
MAX_MEASURES_PER_PICTURE = 8
MAX_MEASURES_PER_PAGE = 24
# Tilted lines that sparse reading turned to scraps, given one more chance.
MAX_RESCUES_PER_PICTURE = 4
# Single-line re-reads of measured slants across one whole document: each
# is a Tesseract call (50-200 ms), so a 300-page deck cannot spend minutes.
MAX_REREADS_PER_DOCUMENT = 120
# A slant the letters alone do not settle is confirmed by reading the words
# turned level and as they stand: a real slant reads clearly better turned.
# A level word measured a few degrees off reads the same both ways.
SLANT_READ_MARGIN = 0.05

_WORDLIKE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


@dataclass
class ImageTextStats:
    images_examined: int = 0
    images_with_text: int = 0
    labels_found: int = 0
    labels_accepted: int = 0
    labels_rejected: int = 0
    # Lines found printed at a slant and redrawn along it; and lines left
    # in English because they curve or are turned too steeply to redraw.
    labels_tilted: int = 0
    labels_curved: int = 0
    labels_steep: int = 0
    labels_rescued: int = 0
    measured: int = 0
    findings: list[QAFinding] = field(default_factory=list)


def looks_instructional(text: str) -> bool:
    """Is this label worth translating, and confidently readable?

    Rejects OCR debris ("NN", "oN", "@"), bare axis ticks, and measurements
    that should stay as they are. A label has to look like a word.
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    words = _WORDLIKE.findall(stripped)
    if not words:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    if letters / len(stripped) < 0.55:
        return False
    # A single short "word" is usually garbage from a stray mark.
    if len(words) == 1 and len(words[0]) < 4:
        return False
    return True


def _skip_image(image: ImageObject, page: Page) -> bool:
    if image.bbox.width < MIN_IMAGE_POINTS or image.bbox.height < MIN_IMAGE_POINTS:
        return True
    page_area = page.width * page.height
    if page_area and image.bbox.area / page_area >= FULL_PAGE_FRACTION:
        # The page scan itself; the page OCR stage owns this.
        return page.is_scanned or page.needs_ocr
    return False


# The paper behind the words is found by closing (a grey-level max filter)
# with a window this share of the picture's shorter side: wider than any
# letter stroke, narrower than a shaded panel.
FLATTEN_WINDOW = 0.06
# How much the paper's brightness must vary across the picture (0-255,
# 5th to 95th percentile) before a flattened copy is worth a second read.
FLATTEN_MIN_SPREAD = 30
# A dark panel is pixels darker than this (0-255) ...
PANEL_INK = 110
# ... joined across gaps narrower than this (inches: wider than a letter's
# stroke at label sizes, narrower than the white inside a frame) ...
PANEL_JOIN = 0.05
# ... at least this share of the picture ...
PANEL_MIN_AREA = 0.002
# ... mostly dark (share of the panel) ...
PANEL_MIN_DARK = 0.6
# ... with light strokes in it (half-width, pixels): a frame or a solid
# block has none ...
PANEL_MIN_STROKE = 1.5
# ... its dark parts at least this many times thicker than them. Big
# black lettering joins up into one dark shape too, but its strokes are
# thinner than the white inside its letters (6.6 against 8.2 pixels); the
# "Collection Pond" panel is 20 against 4 ...
PANEL_THICKER = 2.0
# ... and one flat colour: the middle half of its dark greys within this
# (0-255). A dark patch of a photograph varies by 8 to 33.
PANEL_MAX_SPREAD = 6
# A panel turned further than this is not straightened (degrees).
PANEL_MAX_TURN = 40.0
# Panels read on their own, per picture: one read each.
MAX_PANELS_PER_PICTURE = 6


def _with_second_read(
    result: OCRResult, png: bytes, provider: OCRProvider, language: str | None,
    *, dpi: int = LABEL_OCR_DPI,
) -> OCRResult:
    """Add the labels `result` missed on shaded backgrounds and dark panels.

    Tesseract turns the picture black-and-white with one threshold, so dark
    words on a gradient, a shaded panel or a texture go with the background
    and are never read. Dividing the picture by its own paper -- the
    brightest nearby grey -- evens the background out to white and leaves
    the words.

    White words on a navy panel are lost the other way: the panel is ink to
    Tesseract, and nothing on it was read ("Collection Pond", on a navy
    label tilted 12 degrees). Each such panel is cut out, flipped to dark on
    light, turned level by its own edges and read on its own; flipped in
    place, the tilted label was still missed by the read of the whole
    picture.

    A plain picture has nothing to clean up and is read once.
    """
    import cv2
    import numpy as np

    grey = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return result

    def overlaps(a, b) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    regions = list(result.regions)
    for panel in _lettered_panels(grey, dpi)[:MAX_PANELS_PER_PICTURE]:
        for r in _read_panel(grey, panel, provider, language):
            # The panel's own reading replaces a weaker one of the same
            # words: read as they stand, white on navy came back as
            # "Collectio, Pong" (0.48) and was thrown away.
            rivals = [f for f in regions if overlaps(r.bbox, f.bbox)]
            if all(f.confidence < r.confidence for f in rivals):
                regions = [f for f in regions if f not in rivals] + [r]
    extra: list[OCRRegion] = []

    # The paper is found on a quarter-size copy: closing at full size with a
    # window this wide is slow, and the paper varies slowly anyway.
    small = cv2.resize(grey, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(3, int(min(small.shape) * FLATTEN_WINDOW) | 1)
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    lo, hi = np.percentile(paper, (5, 95))
    if hi - lo >= FLATTEN_MIN_SPREAD:
        paper = cv2.resize(paper, (grey.shape[1], grey.shape[0]), interpolation=cv2.INTER_LINEAR)
        flat = np.clip(grey.astype(np.float32) / np.maximum(paper, 1) * 255,
                       0, 255).astype(np.uint8)
        second = provider.recognize(cv2.imencode(".png", flat)[1].tobytes(),
                                    language=language, sparse=True)
        if not second.error:
            extra += second.regions

    for r in extra:
        if not any(overlaps(r.bbox, f.bbox) for f in regions):
            regions.append(r)
    if regions == result.regions:
        return result
    return replace(result, regions=regions)


def _lettered_panels(grey, dpi: int) -> list:
    """The picture's dark panels with light lettering on them, biggest
    first, as (x, y, mask) with the mask covering the panel's own box."""
    import cv2
    import numpy as np

    dark = (grey < PANEL_INK).astype(np.uint8)
    k = max(3, int(dpi * PANEL_JOIN) | 1)
    joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    min_area = PANEL_MIN_AREA * grey.size
    panels = []
    for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            break
        x, y, w, h = (int(v) for v in stats[i, :4])
        here = labels[y:y + h, x:x + w] == i
        ink = here & (dark[y:y + h, x:x + w] > 0)
        if ink.sum() < PANEL_MIN_DARK * here.sum():
            continue
        # Distance to the nearest edge: twice the largest is how thick the
        # dark parts, and the light strokes, are at their thickest.
        thick_dark = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3).max()
        stroke = cv2.distanceTransform((here & ~ink).astype(np.uint8), cv2.DIST_L2, 3).max()
        if stroke < PANEL_MIN_STROKE or thick_dark < PANEL_THICKER * stroke:
            continue
        core = cv2.erode(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        lo, hi = np.percentile(grey[y:y + h, x:x + w][core], (25, 75))
        if hi - lo > PANEL_MAX_SPREAD:
            continue
        panels.append((x, y, here))
    return panels


def _read_panel(grey, panel, provider: OCRProvider, language: str | None) -> list[OCRRegion]:
    """The lines on one dark panel, read flipped and turned level, with
    upright boxes in the picture's pixels."""
    import cv2
    import numpy as np

    x, y, here = panel
    ys, xs = np.nonzero(here)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(
        np.column_stack([xs, ys]).astype(np.float32)
    )
    # The turn that lays the panel's long side level.
    if rw < rh:
        angle -= 90.0
    angle = (angle + 90.0) % 180.0 - 90.0
    if abs(angle) > PANEL_MAX_TURN:
        return []
    h, w = here.shape
    crop = np.full((h, w), 255, np.uint8)
    crop[here] = 255 - grey[y:y + h, x:x + w][here]
    # Turned on a canvas big enough for any turn, so no corner is cut off.
    side = int(np.ceil(np.hypot(w, h))) + 8
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[:, 2] += (side / 2 - cx, side / 2 - cy)
    level = cv2.warpAffine(crop, m, (side, side), flags=cv2.INTER_CUBIC,
                           borderValue=255)
    read = provider.recognize(cv2.imencode(".png", level)[1].tobytes(),
                              language=language, sparse=True)
    if read.error:
        return []
    back = cv2.invertAffineTransform(m)

    def upright(b):
        corners = np.array([[b[0], b[1], 1], [b[2], b[1], 1],
                            [b[0], b[3], 1], [b[2], b[3], 1]], float) @ back.T
        return (float(corners[:, 0].min() + x), float(corners[:, 1].min() + y),
                float(corners[:, 0].max() + x), float(corners[:, 1].max() + y))

    return [
        replace(r, bbox=upright(r.bbox),
                words=[replace(wd, bbox=upright(wd.bbox)) for wd in r.words])
        for r in read.regions
    ]


def extract_image_text(
    doc: OpenDocument,
    page: Page,
    provider: OCRProvider,
    *,
    language: str | None,
    seen_hashes: dict[str, OCRResult] | None = None,
) -> ImageTextStats:
    """Find translatable labels inside this page's images.

    `seen_hashes` is one dict per document, shared across its pages: a
    school logo or a slide-template picture repeated on every page is read
    once. It is keyed by the rendered pixels, not the picture's own hash, so
    a picture with different page text drawn over it is read again rather
    than given another page's labels. Rendering costs milliseconds; the
    reading it saves costs seconds on the free host.
    """
    stats = ImageTextStats()
    if not page.image_objects:
        return stats

    settings = get_settings()
    cache = seen_hashes if seen_hashes is not None else {}

    for image in page.image_objects:
        if _skip_image(image, page):
            continue
        stats.images_examined += 1
        # The page-wide tally runs on; this picture's own rejections are what
        # its finding reports. Using the tally flagged every later picture on
        # a page once any one had an unreadable label (20 flags, real packet).
        rejected_before = stats.labels_rejected

        dpi = LABEL_OCR_DPI

        try:
            png = doc.render_region_png(page.index, image.bbox.as_tuple(), dpi=dpi)
        except Exception as exc:
            log.warning("Could not render image on page %s: %r", page.number, exc)
            image.contains_text = None
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="image_text_not_read",
                    message=(
                        f"Words in a picture on page {page.number} could not be "
                        "read, so they were left unchanged."
                    ),
                    object_ids=[image.id],
                    bbox=image.bbox,
                    details={"reason": repr(exc)[:200]},
                )
            )
            continue

        key = hashlib.sha256(png).hexdigest()
        result = cache.get(key)
        if result is None:
            result = provider.recognize(png, language=language, sparse=True)
            # A failed read is not remembered, so the next copy tries again.
            if not result.error:
                result = _with_second_read(result, png, provider, language, dpi=dpi)
                cache[key] = result
        if result.error:
            del png
            image.contains_text = None
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="image_text_unreadable",
                    message=(
                        f"Text inside a picture on page {page.number} could not "
                        "be read, so it was left unchanged."
                    ),
                    object_ids=[image.id],
                    bbox=image.bbox,
                )
            )
            continue

        # Pixel space of the rendered region -> page points.
        scale_x = image.bbox.width / max(1.0, (image.bbox.width / 72.0) * dpi)
        scale_y = image.bbox.height / max(1.0, (image.bbox.height / 72.0) * dpi)

        # The slant of each line is measured on a small crop rendered the
        # same way, never on a decoded copy of the whole picture.
        del png
        tilt = _TiltReader(
            doc, page.index, image, provider, language, cache, stats,
            dpi=dpi, scale_x=scale_x, scale_y=scale_y,
            page_budget=MAX_MEASURES_PER_PAGE,
        )

        accepted: list[TextObject] = []
        unsure: list[str] = []
        curved: list[str] = []
        steep: list[str] = []
        rejected_regions: list[OCRRegion] = []
        for region in _join_line_pieces(_split_at_wide_gaps(result.regions)):
            stats.labels_found += 1
            sentence = len(_WORDLIKE.findall(region.text)) >= SENTENCE_WORDS
            readable = region.confidence >= MIN_LABEL_CONFIDENCE or (
                sentence and region.confidence >= CHECKABLE_CONFIDENCE
            )
            if not readable or not looks_instructional(region.text):
                stats.labels_rejected += 1
                rejected_regions.append(region)
                continue

            x0 = image.bbox.x0 + region.bbox[0] * scale_x
            y0 = image.bbox.y0 + region.bbox[1] * scale_y
            x1 = image.bbox.x0 + region.bbox[2] * scale_x
            y1 = image.bbox.y0 + region.bbox[3] * scale_y
            bbox = Rect(x0=x0, y0=y0, x1=x1, y1=y1).clamped_to(image.bbox)
            if bbox.is_degenerate(epsilon=1.5):
                continue

            found = tilt.measure(region.bbox)
            obj = _label_object(
                region, image, page, bbox, scale_x, scale_y, z=2000 + len(accepted)
            )
            obj.orientation_checked = True
            if found is not None and found.curved:
                # Words around an arc cannot be redrawn the same way, and a
                # straight translation over a curved title is wrong in a
                # way the teacher should choose, not the engine.
                obj.keep_original = True
                curved.append(obj.id)
                stats.labels_curved += 1
            elif found is not None and found.steep and found.consistent:
                obj.keep_original = True
                steep.append(obj.id)
                stats.labels_steep += 1
            elif found is not None and found.tilted:
                reread = tilt.confirm(found)
                if reread is not None or found.strong:
                    if (
                        reread is not None
                        and reread.confidence >= region.confidence
                        and reread.confidence >= MIN_LABEL_CONFIDENCE
                    ):
                        # The straightened re-read is the better reading.
                        region = reread
                    _turn(obj, region, found, image, page)
                    stats.labels_tilted += 1
                # Otherwise nothing backs the slant -- the letters do not
                # agree on it strongly and reading the words turned is no
                # better than reading them as they stand: drawn level.
            # A measurement that is level, or not consistent with the
            # letters' shape (a short level word measured a few degrees
            # off), leaves the block exactly as it was always drawn.

            # One guessed word makes the line a guess. It is kept as a block
            # she can check, with its English left on the picture until she
            # confirms or corrects it: translated as read, "Hf iis outside"
            # became nonsense painted over a sentence that was fine.
            guessed = region.confidence < MIN_LABEL_CONFIDENCE or any(
                w.confidence < UNSURE_WORD_CONFIDENCE for w in region.words
            )
            if guessed:
                obj.keep_original = True
            accepted.append(obj)
            if guessed and obj.id not in curved and obj.id not in steep:
                unsure.append(obj.id)

        # Tilted lines sparse reading turned to scraps ("Photosynthesis Lab"
        # at 10-40 degrees: fragments at confidence 0-51). Each cluster of
        # rejected scraps is measured; a clean slant is cut out, turned
        # level and read once more, and kept only if that reading is sure.
        for cluster in _clusters(rejected_regions)[:MAX_RESCUES_PER_PICTURE]:
            obj = _rescue(cluster, tilt, image, page, accepted, scale_x, scale_y)
            if obj is None:
                continue
            obj.z_order = 2000 + len(accepted)
            accepted.append(obj)
            stats.labels_rejected -= len(cluster)
            stats.labels_rescued += 1
            stats.labels_tilted += 1
        tilt.close()

        if curved:
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="image_text_curved",
                    message=(
                        f"Words in a picture on page {page.number} follow a "
                        "curve, so they can't be redrawn the same way. They "
                        "were left in English. Check them, or press "
                        "'Translate it' to put a straight translation there."
                    ),
                    object_ids=curved,
                    bbox=image.bbox,
                )
            )
        if steep:
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="image_text_steep",
                    message=(
                        f"Words in a picture on page {page.number} are turned "
                        "too steeply to redraw reliably. They were left in "
                        "English. Check them, or press 'Translate it' to put "
                        "a level translation there."
                    ),
                    object_ids=steep,
                    bbox=image.bbox,
                )
            )

        if unsure:
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="ocr_low_confidence",
                    message=(
                        f"{len(unsure)} line(s) in a picture on page {page.number} "
                        "were hard to read, so they were left in English. Check "
                        "the English and translate them."
                    ),
                    object_ids=unsure,
                    bbox=image.bbox,
                )
            )

        image.contains_text = bool(accepted) or bool(result.regions)
        if accepted:
            stats.images_with_text += 1
            stats.labels_accepted += len(accepted)
            page.text_objects.extend(accepted)

        rejected_here = stats.labels_rejected - rejected_before
        if rejected_here:
            stats.findings.append(
                QAFinding(
                    severity="warning",
                    code="image_label_unreadable",
                    message=(
                        f"Some text inside a picture on page {page.number} was "
                        "too unclear to translate and was left as it is."
                    ),
                    object_ids=[image.id],
                    bbox=image.bbox,
                    details={"rejected": rejected_here},
                )
            )

    page.recompute_diagnostics()
    return stats


def _label_object(
    region: OCRRegion,
    image: ImageObject,
    page: Page,
    bbox: Rect,
    scale_x: float,
    scale_y: float,
    *,
    z: int,
) -> TextObject:
    """One read line as a block in page coordinates, drawn level."""
    style = TextStyle(font_size=round(max(4.0, bbox.height * 0.92), 1))
    spans: list[TextSpan] = []
    for word in region.words:
        wx0 = image.bbox.x0 + word.bbox[0] * scale_x
        wy0 = image.bbox.y0 + word.bbox[1] * scale_y
        wx1 = image.bbox.x0 + word.bbox[2] * scale_x
        wy1 = image.bbox.y0 + word.bbox[3] * scale_y
        spans.append(
            TextSpan(
                text=word.text + " ",
                bbox=Rect(x0=wx0, y0=wy0, x1=wx1, y1=wy1).clamped_to(image.bbox),
                style=style,
                confidence=word.confidence,
            )
        )
    if spans:
        spans[-1] = spans[-1].model_copy(update={"text": spans[-1].text.rstrip()})
    else:
        spans = [
            TextSpan(
                text=region.text, bbox=bbox, style=style, confidence=region.confidence,
            )
        ]
    return TextObject(
        id=_new_id("imgtxt"),
        kind=ObjectKind.TEXT,
        bbox=bbox,
        norm_bbox=bbox.normalized(page.width, page.height),
        z_order=z,
        lines=[TextLine(bbox=bbox, spans=spans)],
        source=TextSource.IMAGE_OCR,
        role=SemanticRole.IMAGE_LABEL,
        role_confidence=0.8,
        style=style,
        confidence=region.confidence,
        cover_boxes=[
            Rect(
                x0=image.bbox.x0 + b[0] * scale_x,
                y0=image.bbox.y0 + b[1] * scale_y,
                x1=image.bbox.x0 + b[2] * scale_x,
                y1=image.bbox.y0 + b[3] * scale_y,
            ).clamped_to(image.bbox)
            for b in region.discarded_boxes
        ],
    )


def _turn(
    obj: TextObject, region: OCRRegion, found: Orientation, image: ImageObject, page: Page
) -> None:
    """Give a block the slant its letters were measured at.

    The upright box Tesseract reported takes in whatever the turned word
    passes over, so the box becomes the turned rectangle's own extent and
    the size comes from the letters' height across the line, not the
    upright box's (HARBOR: 8.5pt tall letters in a 17pt box, drawn at
    twice their size before).
    """
    ob = found.page_box()
    bbox = ob.aabb().clamped_to(image.bbox)
    style = TextStyle(font_size=round(max(4.0, ob.height * 0.92), 1))
    if region.text.strip() != obj.flat_text.strip():
        spans = [
            TextSpan(text=region.text, bbox=bbox, style=style, confidence=region.confidence)
        ]
    else:
        spans = [s.model_copy(update={"style": style}) for s in obj.lines[0].spans]
    obj.bbox = bbox
    obj.norm_bbox = bbox.normalized(page.width, page.height)
    obj.lines = [TextLine(bbox=bbox, spans=spans, rotation=ob.angle)]
    obj.style = style
    obj.confidence = region.confidence
    obj.oriented = ob
    obj.rotation = ob.angle
    # Scraps Tesseract located but could not read are upright boxes too;
    # the turned box already covers the line's letters.
    obj.cover_boxes = []


class _TiltReader:
    """Measures slants on one picture, within the work limits."""

    def __init__(self, doc, page_index, image, provider, language, cache, stats,
                 *, dpi, scale_x, scale_y, page_budget):
        self._doc = doc
        self._page_index = page_index
        self._image = image
        self._provider = provider
        self._language = language
        self._cache = cache
        self._stats = stats
        self._dpi = dpi
        self._sx = scale_x
        self._sy = scale_y
        self._page_budget = page_budget
        self._used = 0

    def _render(self, clip, dpi) -> bytes:
        return self._doc.render_region_png(self._page_index, clip, dpi=dpi)

    def measure(self, bbox_px) -> Orientation | None:
        """The slant of the words at `bbox_px` (pixels of the reading
        render), or None when it cannot be told or the budget is spent."""
        if self._used >= MAX_MEASURES_PER_PICTURE or self._stats.measured >= self._page_budget:
            return None
        self._used += 1
        self._stats.measured += 1
        b = self._image.bbox
        box = (
            b.x0 + bbox_px[0] * self._sx, b.y0 + bbox_px[1] * self._sy,
            b.x0 + bbox_px[2] * self._sx, b.y0 + bbox_px[3] * self._sy,
        )
        try:
            return measure_region(
                self._render, box, dpi=self._dpi, bounds=b.as_tuple(),
            )
        except Exception as exc:  # a measurement is never worth a failed page
            log.warning("Could not measure a line's slant: %r", exc)
            return None

    def confirm(self, found: Orientation) -> OCRRegion | None:
        """The words read turned level, when that reading backs the slant;
        None when it does not, or no re-read could be made.

        A reading of the straightened words that is merely confident is not
        evidence: a level word straightened by a false 3 degrees still reads
        fine. So unless the letters agree on the slant strongly, the same
        words are also read as they stand, and the slant is only backed when
        the turned reading is clearly the better one.
        """
        turned = self._reread(found, level=False)
        if turned is None or turned.confidence < MIN_LABEL_CONFIDENCE:
            return None
        if found.strong:
            return turned
        as_is = self._reread(found, level=True)
        if as_is is not None and as_is.confidence + SLANT_READ_MARGIN > turned.confidence:
            return None
        return turned

    def _reread(self, found: Orientation, *, level: bool) -> OCRRegion | None:
        """The line cut out (turned level unless `level`) and read as one
        line, if it reads as words; None when it does not or the provider
        cannot."""
        try:
            png = found.straightened_png(level=level)
        except Exception as exc:
            log.warning("Could not straighten a tilted line: %r", exc)
            return None
        if png is None:
            return None
        key = "line:" + hashlib.sha256(png).hexdigest()
        result = self._cache.get(key)
        if result is None:
            spent = self._cache.get(_REREADS_KEY, 0)
            if spent >= MAX_REREADS_PER_DOCUMENT:
                return None
            self._cache[_REREADS_KEY] = spent + 1
            result = self._provider.recognize_line(png, language=self._language)
            if result is None:
                return None
            if not result.error:
                self._cache[key] = result
        if result.error or not result.regions:
            return None
        text = " ".join(r.text for r in result.regions).strip()
        if not looks_instructional(text):
            return None
        words: list[OCRWord] = [w for r in result.regions for w in r.words]
        chars = sum(len(r.text) for r in result.regions) or 1
        confidence = sum(r.confidence * len(r.text) for r in result.regions) / chars
        return OCRRegion(
            text=text, bbox=(0.0, 0.0, 0.0, 0.0), confidence=round(confidence, 4),
            words=words,
        )

    def close(self) -> None:
        self._doc = None


# Where the document-wide re-read count is kept in the shared label cache.
_REREADS_KEY = "tilt:rereads"


def _clusters(regions: list[OCRRegion]) -> list[list[OCRRegion]]:
    """Rejected scraps near one another, biggest group first. A tilted line
    read sparsely comes back as several overlapping or adjacent scraps, each
    within about its own height of the next; two lines of a diagram are
    usually further apart than that, and a cluster spanning two lines does
    not measure as one."""
    groups: list[list[OCRRegion]] = []
    for region in sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0])):
        x0, y0, x1, y1 = region.bbox
        h = y1 - y0
        for group in groups:
            if any(
                max(gx0 - x1, x0 - gx1, gy0 - y1, y0 - gy1) < max(h, gy1 - gy0)
                for gx0, gy0, gx1, gy1 in (g.bbox for g in group)
            ):
                group.append(region)
                break
        else:
            groups.append([region])
    return sorted(groups, key=lambda g: -sum(len(r.text) for r in g))


def _rescue(cluster, tilt: _TiltReader, image, page, accepted, scale_x, scale_y):
    """A block for a tilted line found among rejected scraps, or None."""
    import statistics

    x0 = min(r.bbox[0] for r in cluster)
    y0 = min(r.bbox[1] for r in cluster)
    x1 = max(r.bbox[2] for r in cluster)
    y1 = max(r.bbox[3] for r in cluster)
    h = statistics.median(r.bbox[3] - r.bbox[1] for r in cluster)
    found = tilt.measure((x0 - 1.5 * h, y0 - 0.75 * h, x1 + 1.5 * h, y1 + 0.75 * h))
    if found is None or not found.tilted or found.glyphs < 4:
        return None
    # Always both readings here, however well the letters agree: these
    # scraps were rejected when read level, and a level line that psm 11
    # garbled reads as well level as "straightened" -- it must not come
    # back as a new block drawn at a false slant.
    turned = tilt._reread(found, level=False)
    if turned is None or turned.confidence < MIN_LABEL_CONFIDENCE:
        return None
    as_is = tilt._reread(found, level=True)
    if as_is is not None and as_is.confidence + SLANT_READ_MARGIN > turned.confidence:
        return None
    reread = turned
    ob = found.page_box()
    bbox = ob.aabb().clamped_to(image.bbox)
    if bbox.is_degenerate(epsilon=1.5):
        return None
    if any(
        a.bbox.overlap_fraction_of_self(bbox) > 0.3 or bbox.overlap_fraction_of_self(a.bbox) > 0.3
        for a in accepted
    ):
        return None  # already read, level or tilted
    if any(w.confidence < UNSURE_WORD_CONFIDENCE for w in reread.words):
        return None
    level = OCRRegion(text=reread.text, bbox=(0, 0, 0, 0), confidence=reread.confidence)
    obj = _label_object(level, image, page, bbox, scale_x, scale_y, z=0)
    obj.orientation_checked = True
    _turn(obj, reread, found, image, page)
    return obj


def _split_at_wide_gaps(regions):
    """Cut a line where its words jump a gap no sentence has: from one box
    to the next. Each piece keeps its own words and unreadable glyphs."""
    out: list = []
    for region in regions:
        height = max(region.bbox[3] - region.bbox[1], 1.0)
        groups: list[list] = []
        for word in sorted(region.words, key=lambda w: w.bbox[0]):
            if groups and word.bbox[0] - max(w.bbox[2] for w in groups[-1]) <= BOX_GAP * height:
                groups[-1].append(word)
            else:
                groups.append([word])
        if len(groups) < 2:
            out.append(region)
            continue
        spans = [(min(w.bbox[0] for w in g), max(w.bbox[2] for w in g)) for g in groups]
        glyphs: list[list] = [[] for _ in groups]
        for box in region.discarded_boxes:
            middle = (box[0] + box[2]) / 2
            nearest = min(range(len(spans)),
                          key=lambda i: max(spans[i][0] - middle, middle - spans[i][1], 0))
            glyphs[nearest].append(box)
        for group, (x0, x1), boxes in zip(groups, spans, glyphs):
            chars = sum(len(w.text) for w in group)
            out.append(OCRRegion(
                text=" ".join(w.text for w in group),
                bbox=(x0, region.bbox[1], x1, region.bbox[3]),
                confidence=round(
                    sum(w.confidence * len(w.text) for w in group) / max(1, chars), 4
                ),
                rotation=region.rotation,
                handwritten=region.handwritten,
                line_index=region.line_index,
                words=group,
                discarded_boxes=boxes,
            ))
    return out


def _join_line_pieces(regions):
    """Rejoin pieces of one printed line that sparse reading split apart."""
    out: list = []
    for region in sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0])):
        x0, y0, x1, y1 = region.bbox
        height = y1 - y0
        for i, prev in enumerate(out):
            px0, py0, px1, py1 = prev.bbox
            overlap = min(y1, py1) - max(y0, py0)
            if (
                overlap > 0.6 * min(height, py1 - py0)
                and 0 <= x0 - px1 < SAME_LINE_GAP * max(height, py1 - py0)
            ):
                chars = len(prev.text) + len(region.text)
                out[i] = OCRRegion(
                    text=f"{prev.text} {region.text}",
                    bbox=(px0, min(y0, py0), x1, max(y1, py1)),
                    confidence=round(
                        (prev.confidence * len(prev.text)
                         + region.confidence * len(region.text)) / max(1, chars),
                        4,
                    ),
                    line_index=prev.line_index,
                    words=[*prev.words, *region.words],
                    discarded_boxes=[*prev.discarded_boxes, *region.discarded_boxes],
                )
                break
        else:
            out.append(region)
    return out
