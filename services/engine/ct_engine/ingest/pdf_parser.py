"""PyMuPDF-backed PDF parser.

COORDINATE CONTRACT -- the single most important thing in this file.

PyMuPDF reports `page.rect` with the page's /Rotate already applied (a 90
degree-rotated Letter page reports 792x612), but `get_text`, `get_drawings`
and `get_image_info` all return geometry in the *unrotated* space (612x792).
Mixing the two silently misplaces every object on every rotated page, and the
error is invisible on the portrait pages most test documents are made of.

So: every rect and every point coming out of PyMuPDF is multiplied by
`page.rotation_matrix` exactly once, here at the boundary. Nothing downstream
ever sees unrotated coordinates, and `Page.width/height` always agree with the
object boxes.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from ..config import get_settings
from ..model.document import DocumentMetadata, Page
from ..model.enums import (
    ObjectKind,
    Orientation,
    PageTextMode,
    TextAlignment,
    TextSource,
)
from ..model.geometry import Rect
from ..model.objects import (
    FormFieldObject,
    ImageObject,
    TextLine,
    TextObject,
    TextSpan,
    TextStyle,
    VectorObject,
    _new_id,
)
from ..util.hashing import sha256_bytes
from ..util.logging import get_logger
from .base import DocumentParser, OpenDocument, PageParseError, ParseError
from .math_runs import normalize_line_spans

log = get_logger(__name__)

# PyMuPDF span flag bits.
_FLAG_SUPERSCRIPT = 1 << 0
_FLAG_ITALIC = 1 << 1
_FLAG_SERIF = 1 << 2
_FLAG_MONOSPACE = 1 << 3
_FLAG_BOLD = 1 << 4

# Subset prefixes look like "ABCDEE+". Six uppercase letters then a plus.
_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
_STYLE_SUFFIX_RE = re.compile(
    r"[-,_]?(bold|italic|oblique|regular|light|medium|semibold|demibold|black|"
    r"heavy|thin|book|roman|condensed|narrow|extrabold|ultralight|mt|ms)+$",
    re.IGNORECASE,
)

# A page with more than this many vector paths is almost certainly clip art or
# a chart rendered as thousands of tiny strokes. We keep the aggregate and a
# capped sample: retaining every path would balloon the manifest for no QA
# benefit, but dropping them silently would hide real artwork from collision
# detection, so the truncation is recorded on the page.
_MAX_VECTOR_PATHS = 3000

# Characters that indicate a broken /ToUnicode map. Text made mostly of these
# extracts as gibberish and must be OCR'd instead of trusted.
_PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))


def _is_pua(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _PUA_RANGES)


def garbled_ratio(text: str) -> float:
    """Fraction of characters that suggest a broken text encoding."""
    if not text:
        return 0.0
    bad = sum(1 for ch in text if ch == "�" or _is_pua(ch))
    return bad / len(text)


def normalize_font_family(raw: str | None) -> str | None:
    """Reduce an embedded font name to a family name.

    "ABCDEE+Calibri-BoldItalic" -> "Calibri". The raw name is kept separately
    on the style; this is only for font matching and grouping.
    """
    if not raw:
        return None
    name = _SUBSET_RE.sub("", raw.strip())
    name = name.split("+")[-1]
    prev = None
    while prev != name:
        prev = name
        name = _STYLE_SUFFIX_RE.sub("", name).strip("-,_ ")
    return name or None


def color_to_hex(value: Any) -> str | None:
    """Convert a PyMuPDF color to #rrggbb.

    Spans carry a packed sRGB int (sometimes negative because the alpha byte
    makes it exceed 2^31); drawings carry a float tuple.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) < 3:
            return None
        try:
            r, g, b = (max(0, min(255, int(round(float(c) * 255)))) for c in value[:3])
        except (TypeError, ValueError):
            return None
        return f"#{r:02x}{g:02x}{b:02x}"
    try:
        packed = int(value) & 0xFFFFFF
    except (TypeError, ValueError):
        return None
    return f"#{packed:06x}"


def _style_from_span(span: dict[str, Any]) -> TextStyle:
    flags = int(span.get("flags", 0) or 0)
    raw_name = span.get("font")
    family = normalize_font_family(raw_name)
    lowered = (raw_name or "").lower()
    return TextStyle(
        font_family=family,
        font_name_raw=raw_name,
        font_size=round(float(span.get("size", 0.0) or 0.0), 2),
        # The flag is authoritative when set, but many producers leave it at
        # zero and encode the weight only in the font name.
        bold=bool(flags & _FLAG_BOLD) or "bold" in lowered or "black" in lowered,
        italic=bool(flags & _FLAG_ITALIC)
        or "italic" in lowered
        or "oblique" in lowered,
        serif=bool(flags & _FLAG_SERIF),
        monospace=bool(flags & _FLAG_MONOSPACE),
        color=color_to_hex(span.get("color")),
    )


def _dominant_style(spans: list[TextSpan]) -> TextStyle:
    """Pick the style covering the most characters in a block.

    Character-weighted rather than span-count-weighted: a 400-character
    paragraph with one bold word must not be classified as bold.
    """
    if not spans:
        return TextStyle()
    buckets: dict[tuple, list[int | TextStyle]] = {}
    for sp in spans:
        st = sp.style
        key = (st.font_family, round(st.font_size, 1), st.bold, st.italic, st.color)
        entry = buckets.setdefault(key, [0, st])
        entry[0] = int(entry[0]) + max(1, len(sp.text))
    best = max(buckets.values(), key=lambda e: int(e[0]))
    return best[1]  # type: ignore[return-value]


def _detect_alignment(lines: list[TextLine], tolerance: float = 1.5) -> TextAlignment:
    """Infer paragraph alignment from line edge geometry."""
    rows = [ln for ln in lines if ln.bbox.width > 0]
    if len(rows) < 2:
        # One line tells us nothing: a centered single line and a left-aligned
        # single line are geometrically identical without column context.
        return TextAlignment.UNKNOWN

    lefts = [ln.bbox.x0 for ln in rows]
    rights = [ln.bbox.x1 for ln in rows]
    centers = [ln.bbox.center[0] for ln in rows]

    left_flush = (max(lefts) - min(lefts)) <= tolerance
    # The final line of a justified paragraph is short by design, so the
    # right-edge test excludes it when there is more than one line to judge.
    body_rights = rights[:-1] if len(rows) > 2 else rights
    right_flush = (max(body_rights) - min(body_rights)) <= tolerance
    center_flush = (max(centers) - min(centers)) <= tolerance

    if left_flush and right_flush:
        return TextAlignment.JUSTIFY
    if left_flush:
        return TextAlignment.LEFT
    if right_flush:
        return TextAlignment.RIGHT
    if center_flush:
        return TextAlignment.CENTER
    return TextAlignment.UNKNOWN


def _mean_line_spacing(lines: list[TextLine], angle_deg: float) -> float | None:
    """Mean baseline-to-baseline distance, measured across the text flow."""
    if len(lines) < 2:
        return None
    horizontal = abs(math.cos(math.radians(angle_deg))) >= 0.5
    positions = sorted(
        (ln.bbox.y0 if horizontal else ln.bbox.x0) for ln in lines
    )
    deltas = [b - a for a, b in zip(positions, positions[1:]) if b - a > 0.1]
    if not deltas:
        return None
    return round(sum(deltas) / len(deltas), 2)


class PyMuPDFDocument(OpenDocument):
    """An open PDF, parseable and renderable page by page."""

    def __init__(self, doc: "fitz.Document") -> None:
        self._doc = doc
        self._closed = False
        self._settings = get_settings()

    # -- lifecycle ---------------------------------------------------------
    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def close(self) -> None:
        if not self._closed:
            try:
                self._doc.close()
            finally:
                self._closed = True

    def __enter__(self) -> "PyMuPDFDocument":
        return self

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> DocumentMetadata:
        md = self._doc.metadata or {}

        def clean(v: Any) -> str | None:
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        has_form = False
        try:
            has_form = bool(self._doc.is_form_pdf)
        except Exception:
            pass

        return DocumentMetadata(
            title=clean(md.get("title")),
            author=clean(md.get("author")),
            subject=clean(md.get("subject")),
            keywords=clean(md.get("keywords")),
            creator=clean(md.get("creator")),
            producer=clean(md.get("producer")),
            creation_date=clean(md.get("creationDate")),
            modification_date=clean(md.get("modDate")),
            is_encrypted=bool(self._doc.is_encrypted),
            is_form=has_form,
            pdf_version=clean(md.get("format")),
        )

    # -- geometry helpers --------------------------------------------------
    @staticmethod
    def _to_visual(raw: Any, matrix: "fitz.Matrix") -> Rect:
        """Map a PyMuPDF rect/tuple from unrotated to visual page space."""
        r = fitz.Rect(raw) * matrix
        return Rect(x0=float(r.x0), y0=float(r.y0), x1=float(r.x1), y1=float(r.y1))

    @staticmethod
    def _visual_angle(direction: Any, matrix: "fitz.Matrix") -> float:
        """Visual writing angle in degrees, clockwise from horizontal.

        The line direction vector is in unrotated space too, so it needs the
        linear part of the rotation matrix applied (translation excluded).
        """
        try:
            dx, dy = float(direction[0]), float(direction[1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        ndx = dx * matrix.a + dy * matrix.c
        ndy = dx * matrix.b + dy * matrix.d
        if abs(ndx) < 1e-9 and abs(ndy) < 1e-9:
            return 0.0
        angle = math.degrees(math.atan2(ndy, ndx))
        # Snap near-axis angles so trivial float noise does not present as a
        # rotated line downstream.
        for target in (-180.0, -90.0, 0.0, 90.0, 180.0):
            if abs(angle - target) < 0.75:
                return 0.0 if abs(target) == 180.0 else target
        return round(angle, 2)

    # -- extraction --------------------------------------------------------
    def _extract_text(
        self, page: "fitz.Page", matrix: "fitz.Matrix"
    ) -> tuple[list[TextObject], int]:
        """Extract text blocks. Returns (objects, highest_block_number)."""
        objects: list[TextObject] = []
        max_number = 0
        raw = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_WHITESPACE
            | fitz.TEXT_MEDIABOX_CLIP,
        )
        for block in raw.get("blocks", []):
            number = int(block.get("number", 0) or 0)
            max_number = max(max_number, number)
            if block.get("type") != 0:
                continue

            lines: list[TextLine] = []
            all_spans: list[TextSpan] = []
            angles: list[float] = []

            for raw_line in block.get("lines", []):
                spans: list[TextSpan] = []
                for raw_span in raw_line.get("spans", []):
                    text = raw_span.get("text", "")
                    if not text:
                        continue
                    span = TextSpan(
                        text=text,
                        bbox=self._to_visual(raw_span["bbox"], matrix),
                        style=_style_from_span(raw_span),
                    )
                    spans.append(span)
                    all_spans.append(span)
                if not spans:
                    continue
                kept = [s for s in raw_line.get("spans", []) if s.get("text")]
                normalize_line_spans(
                    spans,
                    origins=[float(s.get("origin", (0.0, 0.0))[1]) for s in kept],
                    flags=[int(s.get("flags", 0) or 0) for s in kept],
                )
                angle = self._visual_angle(raw_line.get("dir", (1.0, 0.0)), matrix)
                angles.append(angle)
                lines.append(
                    TextLine(
                        bbox=self._to_visual(raw_line["bbox"], matrix),
                        spans=spans,
                        rotation=angle,
                    )
                )

            # Whitespace-only blocks carry no instructional content and would
            # otherwise inflate object counts and QA expectations.
            if not lines or not any(ln.text.strip() for ln in lines):
                continue

            bbox = Rect.bounding([ln.bbox for ln in lines])
            if bbox is None:
                continue
            dominant_angle = max(set(angles), key=angles.count) if angles else 0.0

            objects.append(
                TextObject(
                    id=_new_id("text"),
                    kind=ObjectKind.TEXT,
                    bbox=bbox,
                    norm_bbox=bbox.normalized(page.rect.width, page.rect.height),
                    z_order=number,
                    rotation=dominant_angle,
                    lines=lines,
                    source=TextSource.NATIVE,
                    alignment=_detect_alignment(lines),
                    style=_dominant_style(all_spans),
                    line_spacing=_mean_line_spacing(lines, dominant_angle),
                )
            )
        return objects, max_number

    def _extract_images(
        self, page: "fitz.Page", matrix: "fitz.Matrix"
    ) -> list[ImageObject]:
        objects: list[ImageObject] = []
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception as exc:
            log.warning("get_image_info failed on page %s: %r", page.number, exc)
            infos = []

        for idx, info in enumerate(infos):
            try:
                bbox = self._to_visual(info["bbox"], matrix)
            except Exception:
                continue
            # Sub-point placements are rendering artifacts (hairline spacers),
            # not instructional images.
            if bbox.is_degenerate(epsilon=1.0):
                continue

            xref = int(info.get("xref", 0) or 0)
            digest = info.get("digest")
            image_hash = digest.hex() if isinstance(digest, (bytes, bytearray)) else None

            objects.append(
                ImageObject(
                    id=_new_id("img"),
                    kind=ObjectKind.IMAGE,
                    bbox=bbox,
                    norm_bbox=bbox.normalized(page.rect.width, page.rect.height),
                    # Images and text share the content-stream numbering from
                    # get_text, but get_image_info numbers restart at 0, so a
                    # separate band keeps them from colliding. Relative order
                    # within images is preserved.
                    z_order=idx,
                    pixel_width=int(info.get("width", 0) or 0),
                    pixel_height=int(info.get("height", 0) or 0),
                    image_hash=image_hash,
                    xref=xref or None,
                    colorspace=info.get("cs-name"),
                    has_alpha=bool(info.get("has-alpha", 0)),
                )
            )
        return objects

    def _extract_vectors(
        self, page: "fitz.Page", matrix: "fitz.Matrix"
    ) -> tuple[list[VectorObject], bool]:
        objects: list[VectorObject] = []
        truncated = False
        try:
            drawings = page.get_drawings()
        except Exception as exc:
            log.warning("get_drawings failed on page %s: %r", page.number, exc)
            return [], False

        if len(drawings) > _MAX_VECTOR_PATHS:
            truncated = True
            drawings = drawings[:_MAX_VECTOR_PATHS]

        for idx, path in enumerate(drawings):
            try:
                bbox = self._to_visual(path["rect"], matrix)
            except Exception:
                continue

            items = path.get("items", []) or []
            line_count = sum(1 for it in items if it and it[0] == "l")
            rect_count = sum(1 for it in items if it and it[0] == "re")
            curve_count = sum(1 for it in items if it and it[0] == "c")

            width = path.get("width")
            stroke_w = float(width) if isinstance(width, (int, float)) else None
            filled = path.get("fill") is not None

            # A "rule" is a writing line or a table border: a single stroke
            # that is long in one axis and flat in the other. These are the
            # primary evidence for student response areas (spec sec. 28).
            thin = bbox.height <= 2.5 or bbox.width <= 2.5
            looks_like_rule = (
                thin
                and max(bbox.width, bbox.height) >= 24.0
                and (line_count + rect_count) <= 2
                and curve_count == 0
            )
            looks_like_box = (
                rect_count >= 1
                and bbox.width >= 18.0
                and bbox.height >= 12.0
                and curve_count == 0
                and not filled
            )

            objects.append(
                VectorObject(
                    id=_new_id("vec"),
                    kind=ObjectKind.VECTOR,
                    bbox=bbox,
                    norm_bbox=bbox.normalized(page.rect.width, page.rect.height),
                    z_order=idx,
                    line_count=line_count,
                    rect_count=rect_count,
                    curve_count=curve_count,
                    stroke_color=color_to_hex(path.get("color")),
                    fill_color=color_to_hex(path.get("fill")),
                    stroke_width=stroke_w,
                    is_filled=filled,
                    looks_like_rule=looks_like_rule,
                    looks_like_box=looks_like_box,
                )
            )
        return objects, truncated

    def _extract_form_fields(
        self, page: "fitz.Page", matrix: "fitz.Matrix"
    ) -> list[FormFieldObject]:
        fields: list[FormFieldObject] = []
        try:
            widgets = list(page.widgets())
        except Exception:
            return []
        for idx, w in enumerate(widgets):
            try:
                bbox = self._to_visual(w.rect, matrix)
            except Exception:
                continue
            fields.append(
                FormFieldObject(
                    id=_new_id("field"),
                    kind=ObjectKind.FORM_FIELD,
                    bbox=bbox,
                    norm_bbox=bbox.normalized(page.rect.width, page.rect.height),
                    z_order=idx,
                    field_name=getattr(w, "field_name", None),
                    field_type=str(getattr(w, "field_type_string", "") or "") or None,
                    field_value=(
                        str(w.field_value) if getattr(w, "field_value", None) else None
                    ),
                )
            )
        return fields

    # -- page assembly -----------------------------------------------------
    def parse_page(self, index: int) -> Page:
        if index < 0 or index >= self.page_count:
            raise PageParseError(index, f"Page {index + 1} does not exist.")
        try:
            page = self._doc.load_page(index)
        except Exception as exc:
            raise PageParseError(
                index,
                f"Page {index + 1} could not be opened.",
                diagnostic=repr(exc),
            ) from exc

        try:
            matrix = page.rotation_matrix
            rect = page.rect
            width = round(float(rect.width), 2)
            height = round(float(rect.height), 2)

            if width <= 0 or height <= 0:
                raise PageParseError(
                    index,
                    f"Page {index + 1} reports an invalid size "
                    f"({width} x {height} points).",
                    diagnostic=f"page.rect={rect}",
                )

            text_objects, _ = self._extract_text(page, matrix)
            image_objects = self._extract_images(page, matrix)
            vector_objects, vectors_truncated = self._extract_vectors(page, matrix)
            form_fields = self._extract_form_fields(page, matrix)

            if width > height * 1.02:
                orientation = Orientation.LANDSCAPE
            elif height > width * 1.02:
                orientation = Orientation.PORTRAIT
            else:
                orientation = Orientation.SQUARE

            page_area = width * height
            text_area = sum(t.bbox.area for t in text_objects)
            coverage = min(text_area / page_area, 1.0) if page_area else 0.0

            all_text = "".join(t.text for t in text_objects)
            stripped_len = len("".join(all_text.split()))
            garbled = garbled_ratio(all_text)

            text_mode, needs_ocr = self._classify_text_mode(
                char_count=stripped_len,
                coverage=coverage,
                garbled=garbled,
                images=image_objects,
                vectors=vector_objects,
                page_area=page_area,
            )

            result = Page(
                index=index,
                number=index + 1,
                width=width,
                height=height,
                rotation=int(page.rotation or 0),
                orientation=orientation,
                text_mode=text_mode,
                text_coverage=round(coverage, 5),
                is_scanned=text_mode == PageTextMode.SCANNED,
                needs_ocr=needs_ocr,
                text_objects=text_objects,
                image_objects=image_objects,
                vector_objects=vector_objects,
                form_fields=form_fields,
                analyzed=True,
            )
            result.recompute_diagnostics()
            if vectors_truncated:
                result.diagnostics.extra["vectors_truncated_at"] = _MAX_VECTOR_PATHS
            if garbled > 0.2:
                result.diagnostics.extra["garbled_text_ratio"] = round(garbled, 3)
            return result
        except PageParseError:
            raise
        except Exception as exc:
            raise PageParseError(
                index,
                f"Page {index + 1} could not be read.",
                diagnostic=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _classify_text_mode(
        self,
        *,
        char_count: int,
        coverage: float,
        garbled: float,
        images: list[ImageObject],
        vectors: list[VectorObject],
        page_area: float,
    ) -> tuple[PageTextMode, bool]:
        """Decide how this page's text must be obtained.

        Getting this wrong is expensive in both directions: a false SCANNED
        runs needless OCR over perfectly good text, and a false NATIVE ships a
        page with no translated content at all. So the rules stay conservative
        and explicit rather than clever.
        """
        s = self._settings
        largest_image_fraction = 0.0
        if page_area > 0 and images:
            largest_image_fraction = max(i.bbox.area for i in images) / page_area

        # Broken encoding: characters exist but do not mean anything.
        if char_count > 0 and garbled > 0.35:
            return PageTextMode.SCANNED, True

        if char_count >= s.scanned_text_char_threshold:
            # A full-page image with only a trace of real text is a scan with
            # a stray header (page number, watermark) drawn over it.
            if largest_image_fraction >= 0.6 and coverage < 0.015:
                return PageTextMode.MIXED, True
            return PageTextMode.NATIVE, False

        # Below the text threshold from here on.
        has_visual_content = bool(images) or len(vectors) > 4
        if not has_visual_content:
            return PageTextMode.EMPTY, False
        if largest_image_fraction >= 0.35 or coverage < s.text_coverage_scanned_threshold:
            return PageTextMode.SCANNED, True
        return PageTextMode.MIXED, True

    # -- rasterization -----------------------------------------------------
    def render_page_png(
        self, index: int, *, dpi: int = 144, gray: bool = False
    ) -> bytes:
        if index < 0 or index >= self.page_count:
            raise PageParseError(index, f"Page {index + 1} does not exist.")
        try:
            page = self._doc.load_page(index)
            pix = page.get_pixmap(
                dpi=dpi,
                alpha=False,
                colorspace=fitz.csGRAY if gray else fitz.csRGB,
            )
            out = pix.tobytes("png")
            # Drop the pixmap before returning. At OCR DPI it is the largest
            # single allocation the process makes, and it is native memory
            # that a later garbage collection will not help with.
            pix = None
            return out
        except Exception as exc:
            raise PageParseError(
                index,
                f"Page {index + 1} could not be converted to an image.",
                diagnostic=repr(exc),
            ) from exc

    def render_region_png(
        self, index: int, bbox: tuple[float, float, float, float], *, dpi: int = 300
    ) -> bytes:
        try:
            page = self._doc.load_page(index)
            # `get_pixmap` clips in the same visual space as `page.rect` --
            # i.e. rotation already applied -- so the bbox is passed through
            # unchanged. (Verified against /Rotate 90 and 270 pages; applying
            # the inverse rotation here produces a blank crop.)
            clip = fitz.Rect(*bbox) & page.rect
            pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
            return pix.tobytes("png")
        except Exception as exc:
            raise PageParseError(
                index,
                f"A region of page {index + 1} could not be converted to an image.",
                diagnostic=repr(exc),
            ) from exc

    def extract_image_asset(self, xref: int) -> tuple[bytes, str] | None:
        """Return the original embedded bytes for an image xref."""
        if not xref:
            return None
        try:
            info = self._doc.extract_image(xref)
        except Exception as exc:
            log.warning("extract_image(%s) failed: %r", xref, exc)
            return None
        data = info.get("image")
        if not data:
            return None
        return bytes(data), str(info.get("ext") or "png")

    def render_image_placement_png(
        self, index: int, bbox: tuple[float, float, float, float], dpi: int = 300
    ) -> bytes | None:
        """Fallback asset capture for inline images with no xref.

        Inline images live in the content stream and have no resource entry to
        extract, so the only faithful way to preserve them is to rasterize
        their placement.
        """
        try:
            return self.render_region_png(index, bbox, dpi=dpi)
        except Exception:
            return None


class PdfParser(DocumentParser):
    name = "pymupdf"
    supported_content_types = frozenset(
        {"application/pdf", "application/x-pdf", "application/acrobat"}
    )
    supported_extensions = frozenset({"pdf"})

    def open_path(self, path: Path) -> PyMuPDFDocument:
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ParseError(
                "That PDF could not be opened.", diagnostic=repr(exc)
            ) from exc
        return self._prepare(doc)

    def open_bytes(self, data: bytes) -> PyMuPDFDocument:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ParseError(
                "That PDF could not be opened.", diagnostic=repr(exc)
            ) from exc
        return self._prepare(doc)

    @staticmethod
    def _prepare(doc: "fitz.Document") -> PyMuPDFDocument:
        if doc.is_encrypted:
            # Validation already handled the empty-password case; reaching
            # here means the file really is locked.
            if not doc.authenticate(""):
                doc.close()
                raise ParseError(
                    "That PDF is password-protected. Please remove the password "
                    "and upload it again.",
                    diagnostic="authenticate('') failed",
                )
        if doc.page_count == 0:
            doc.close()
            raise ParseError("That PDF has no pages.", diagnostic="page_count=0")
        return PyMuPDFDocument(doc)


def sha256_of(data: bytes) -> str:
    return sha256_bytes(data)
