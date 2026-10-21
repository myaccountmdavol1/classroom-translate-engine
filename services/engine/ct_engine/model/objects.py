"""Page objects: the typed contents of a single page.

Design note -- why text is modelled at three levels (block / line / span):

  * A *span* is a run of characters sharing one font, size and color. It is
    the only level at which PDF typography is actually well-defined, so it is
    what the reconstruction engine uses to pick fonts.
  * A *line* is a visual row. Needed to measure original line spacing and to
    re-wrap translated text sensibly.
  * A *block* is the semantic unit: a whole question, a whole direction, a
    whole paragraph. Translation happens here, never on spans, because
    translating fragments in isolation destroys meaning (spec sec. 15/22).

Collapsing these into one flat "text with a box" loses either the typography
or the meaning, and we need both.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from .enums import (
    ObjectKind,
    SemanticRole,
    TextAlignment,
    TextSource,
    TranslationStatus,
)
from .geometry import NormRect, Rect


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TextStyle(BaseModel):
    """Typography of a run of text, as recovered from the source document."""

    font_family: str | None = None
    # The raw embedded font name (e.g. "ABCDEE+Calibri-Bold"). Kept verbatim
    # because the subset prefix and suffix are the best signal we have for
    # matching a fallback face later.
    font_name_raw: str | None = None
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False
    serif: bool = False
    monospace: bool = False
    # sRGB hex, e.g. "#1a1a1a". None when the producer left it unset.
    color: str | None = None

    def is_similar_to(self, other: "TextStyle", size_tolerance: float = 0.6) -> bool:
        return (
            self.font_family == other.font_family
            and abs(self.font_size - other.font_size) <= size_tolerance
            and self.bold == other.bold
            and self.italic == other.italic
        )


class TextSpan(BaseModel):
    """A run of characters sharing a single style."""

    id: str = Field(default_factory=lambda: _new_id("span"))
    text: str
    bbox: Rect
    style: TextStyle = Field(default_factory=TextStyle)
    # Per-character OCR/extraction confidence in [0, 1]; None for native text,
    # which is exact by construction.
    confidence: float | None = None
    # "super" or "sub" when the source drew this run raised or lowered. When
    # script_converted is False the text had no Unicode script form and is
    # shown on the baseline, which the page reports.
    script: str | None = None
    script_converted: bool = False


Point2 = tuple[float, float]


def grow_quad(quad: list[Point2], d: float) -> list[Point2]:
    """A rectangle given as corners in perimeter order (ul, ur, lr, ll),
    grown by `d` points on every side, along its own edges."""
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = quad
    lu = math.hypot(bx - ax, by - ay) or 1.0
    lv = math.hypot(dx - ax, dy - ay) or 1.0
    ux, uy = (bx - ax) / lu * d, (by - ay) / lu * d
    vx, vy = (dx - ax) / lv * d, (dy - ay) / lv * d
    return [
        (ax - ux - vx, ay - uy - vy),
        (bx + ux - vx, by + uy - vy),
        (cx + ux + vx, cy + uy + vy),
        (dx - ux + vx, dy - uy + vy),
    ]


class OrientedBox(BaseModel):
    """A turned rectangle: where tilted words sit inside a picture.

    `angle` is degrees clockwise from horizontal on the page, the same
    convention as TextLine.rotation and the PDF parser's visual angle. A
    word printed on a drawing of a tablet held at a slant has an upright
    box (its `aabb`) far bigger than the word: it takes in the tablet's
    frame and the artwork beside it, which is why covering or measuring
    that box goes wrong.
    """

    cx: float
    cy: float
    width: float
    height: float
    angle: float

    def _axes(self) -> tuple[float, float, float, float]:
        t = math.radians(self.angle)
        # u runs along the writing, v down through the letters.
        return math.cos(t), math.sin(t), -math.sin(t), math.cos(t)

    def corners(self) -> list[Point2]:
        """(ul, ur, lr, ll): perimeter order, for drawing a polygon."""
        ux, uy, vx, vy = self._axes()
        hw, hh = self.width / 2.0, self.height / 2.0
        return [
            (self.cx + sx * hw * ux + sy * hh * vx, self.cy + sx * hw * uy + sy * hh * vy)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def quad(self) -> tuple[Point2, Point2, Point2, Point2]:
        """(ul, ur, ll, lr): the corner order fitz.Quad expects. Passing the
        perimeter order instead draws a bow-tie that covers almost nothing."""
        ul, ur, lr, ll = self.corners()
        return ul, ur, ll, lr

    def frame(self) -> Rect:
        """The box before turning, centred on the same point."""
        return Rect(
            x0=self.cx - self.width / 2.0, y0=self.cy - self.height / 2.0,
            x1=self.cx + self.width / 2.0, y1=self.cy + self.height / 2.0,
        )

    def aabb(self) -> Rect:
        xs, ys = zip(*self.corners())
        return Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    def expanded(self, d: float) -> "OrientedBox":
        return self.model_copy(
            update={"width": self.width + 2 * d, "height": self.height + 2 * d}
        )

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        ux, uy, vx, vy = self._axes()
        du, dv = x - self.cx, y - self.cy
        return (
            abs(du * ux + dv * uy) <= self.width / 2.0 + pad
            and abs(du * vx + dv * vy) <= self.height / 2.0 + pad
        )


class DrawnRow(BaseModel):
    """One wrapped row of a drawn block, blank rows included.

    The editor draws the words itself for a moment after each change, and
    with these rows it breaks the lines exactly where the picture will.
    """

    # The inked extent, in the unturned frame. A justified row spans
    # exactly the box's width; a blank row has x0 == x1.
    x0: float
    x1: float
    baseline: float
    # Logical order, as wrapped.
    text: str
    # True when the row ends its paragraph, so it is never spread.
    end: bool
    # True when the words were spread to the box (justified). The editor
    # follows this rather than guessing from the text, so the two agree
    # even on a row the engine could not spread (words wider than the box).
    spread: bool = False
    # Lists and indents (reconstruct/lists.py). `indent` is where the text
    # column starts, in points from the inner box's start edge (the left,
    # or the right for right-to-left text); `hang` is from the marker's
    # start edge to the text's. `marker` (logical order) is drawn on the
    # first row of a paragraph only, its ink at marker_x0..marker_x1 in
    # the unturned frame, in `marker_face`: "text" (the words' own face),
    # "marks" (Noto Sans Symbols 2) or "sans" (Noto Sans). x0/x1 stay the
    # words' ink alone, so an underline or highlight never covers it.
    indent: float = 0.0
    hang: float = 0.0
    marker: str | None = None
    marker_x0: float | None = None
    marker_x1: float | None = None
    marker_face: Literal["text", "marks", "sans"] | None = None


class DrawnMark(BaseModel):
    """A list marker as drawn: its own font, since a bullet the words'
    face lacks is drawn in another."""

    x: float
    baseline: float
    # Visual order (".1" for a right-to-left "1.").
    text: str
    font_builtin: str | None = None
    font_path: str | None = None


class DrawnLayout(BaseModel):
    """Exactly how the rebuild drew a block: enough to draw it again alone.

    The live editor shows each translated block as its own picture over a
    clean page, so it can move with no server round trip. That picture is
    drawn from this record by the same renderer as the PDF, so what the
    teacher drags is what downloads.
    """

    font_builtin: str | None = None
    font_path: str | None = None
    size: float
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # (x, baseline y, visual text) in page points, one per drawn line --
    # except a justified row, which is one entry per word at its spread
    # position. The sprite and the PDF draw the entries as they are, so
    # neither needs to know about justifying.
    lines: list[tuple[float, float, str]] = Field(default_factory=list)
    # Background painted under the text over the whole frame, if any.
    fill: tuple[float, float, float] | None = None
    # Turned text. `lines` stay in the unturned frame; drawing turns them
    # `angle` degrees clockwise about `pivot`, so the sprite and the PDF
    # turn the same record the same way. `quad` is the drawn frame's
    # corners in perimeter order (ul, ur, lr, ll), for anything that has to
    # cover or copy exactly that area.
    angle: float = 0.0
    pivot: Point2 | None = None
    quad: list[Point2] | None = None
    # Baseline-to-baseline distance in points, and the wrapped rows, so the
    # editor can stand in for the picture with the same spacing and breaks.
    # Missing on layouts drawn before these existed.
    leading: float | None = None
    rows: list[DrawnRow] = Field(default_factory=list)
    # Drawn under each row's words (see builder.draw_decorations): a rule
    # in the ink colour, and a band of colour behind the words only.
    underline: bool = False
    highlight: tuple[float, float, float] | None = None
    # A line around the drawn frame (the quad when turned), on top of the
    # words: colour, weight in points, and None (solid), "dashed" or
    # "dotted". See builder.paint_border.
    border: tuple[float, float, float] | None = None
    border_width: float = 0.0
    border_dash: str | None = None
    # A shape (reconstruct/shapes.py) drawn instead of the plain frame by
    # the fill and the border: its name, and the unturned frame it is
    # drawn in, in page points. None draws the frame as it always was.
    shape: str | None = None
    frame: Rect | None = None
    # The shape's adjust values (a curve's bend) and a line's ends, as on
    # TextObject: None for defaults and plain ends.
    shape_adjust: list[float] | None = None
    line_start: str | None = None
    line_end: str | None = None
    # True when a turned picture label was centred along its frame rather
    # than drawn with its own alignment, so the editor centres it too.
    centred: bool = False
    # List markers, drawn after the lines with each one's own font, in the
    # same colour and turned the same way.
    marks: list[DrawnMark] = Field(default_factory=list)
    # The catalog font (fonts/catalog.py) the words were drawn in: None
    # when automatic, an older style, or when her font lacked the letters
    # and `face_fallback` is True.
    face: str | None = None
    face_fallback: bool = False


class TextLine(BaseModel):
    """A visual row of text made of one or more spans."""

    id: str = Field(default_factory=lambda: _new_id("line"))
    bbox: Rect
    spans: list[TextSpan] = Field(default_factory=list)
    # Rotation of the line's writing direction, degrees clockwise from
    # horizontal. Non-zero for rotated table headers, vertical axis labels etc.
    rotation: float = 0.0

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


class BaseObject(BaseModel):
    """Fields every page object carries."""

    id: str
    kind: ObjectKind
    bbox: Rect
    # Page-relative box, so the UI can position overlays at any zoom without
    # knowing point dimensions.
    norm_bbox: NormRect
    # Paint order within the page; lower paints first. Recovered from the
    # content stream sequence, so it is a faithful z-order for native PDFs.
    z_order: int = 0
    rotation: float = 0.0


class TextObject(BaseObject):
    kind: Literal[ObjectKind.TEXT] = ObjectKind.TEXT
    lines: list[TextLine] = Field(default_factory=list)

    source: TextSource = TextSource.NATIVE
    role: SemanticRole = SemanticRole.OTHER
    # How confident the classifier is about `role`, in [0, 1]. Low-confidence
    # roles are still recorded but must not drive destructive decisions.
    role_confidence: float = 0.0

    alignment: TextAlignment = TextAlignment.UNKNOWN
    # Dominant style across the block, used as the reconstruction default.
    style: TextStyle = Field(default_factory=TextStyle)
    # Mean baseline-to-baseline distance in points; None for single-line blocks.
    line_spacing: float | None = None

    translation_status: TranslationStatus = TranslationStatus.PENDING
    # True when the block must survive translation unchanged (formula, URL,
    # bare question number). Set by the protection layer, honoured by the
    # translation engine and re-verified during QA.
    protected: bool = False
    protected_reason: str | None = None

    # Set once translation runs; links this object to its translation segment
    # so a single manual edit can re-render exactly one object (spec sec. 36).
    segment_id: str | None = None
    translated_text: str | None = None
    # What the machine produced, kept separately so "Reset translation" can
    # restore it after a human edit without re-running the translator.
    machine_translated_text: str | None = None
    # The translation WITHOUT bilingual pairing. In bilingual modes
    # `translated_text` carries both languages and is ~2.4x the original
    # length; when a page cannot be reflowed to fit that, reconstruction
    # falls back to this rather than overlapping text.
    translation_only_text: str | None = None
    # True when a person edited this block. Manual text is never overwritten
    # by a later automated pass.
    manually_edited: bool = False

    # Mean extraction confidence across spans. None for native text.
    confidence: float | None = None

    # True when this block is a student response area (answer lines, boxes)
    # that translated text must not consume.
    is_response_area: bool = False

    # Position in human reading order on the page, assigned by
    # analysis.reading_order. -1 until computed.
    reading_index: int = -1

    # Extra regions of the source raster this object replaces: places where
    # OCR found glyphs but could not read them. Covered during
    # reconstruction so a scanned line is never half original, half
    # translation.
    cover_boxes: list[Rect] = Field(default_factory=list)

    # --- Reflowable formats (DOCX / PPTX) --------------------------------
    # Where this block's characters live inside the source package, as an
    # opaque token produced by the parser (e.g. "word/document.xml#12#0").
    # Reflowable formats are reconstructed by editing the original file in
    # place rather than by redrawing at coordinates, so the anchor -- not the
    # bbox -- is what identifies the text to replace. None for PDFs.
    anchor: str | None = None

    # sRGB hex of the page behind this block, measured during
    # reconstruction. The editor paints this behind a block while its text
    # is being edited, so a white heading on a dark banner does not turn
    # into black-on-white mid-edit. None until the page has been rebuilt.
    background_color: str | None = None
    # Explicitly approved recovery, preserved in projects and undo history.
    retry_background: bool = False
    keep_original: bool = False

    # --- Tilted words in a picture ----------------------------------------
    # Where the source words really are, when they are printed at a slant
    # (pipeline/text_angle.py). Only picture text gets one: covering and
    # drawing follow it, and everything without one is drawn as before.
    oriented: OrientedBox | None = None
    # True once the slant was measured, whatever the answer, so a document
    # read before this existed is measured once on its next rebuild.
    orientation_checked: bool = False
    # Her choice: 0 draws the translation straight, None follows the slant.
    manual_rotation: float | None = None

    # --- Manual layout overrides (spec sec. 36, extended) ---------------
    # Automatic layout cannot be right on every document, especially
    # graphic-heavy ones. These let a person place and style a block exactly,
    # and they always win over the automatic result.
    #
    # `manual_bbox` is where the TRANSLATION is drawn. The original bbox is
    # still what gets redacted or covered, because that is where the source
    # text physically is -- moving a block must not leave the original behind.
    approved_source_text: str | None = None
    manual_bbox: Rect | None = None
    manual_color: str | None = None
    manual_font_size: float | None = None
    # True when the person fixed the size, which stops the fitter shrinking
    # it. A human choice is not a suggestion.
    manual_size_locked: bool = False
    # Size the words to fill the box, again whenever the box changes.
    manual_fit: bool = False
    manual_alignment: TextAlignment | None = None
    # Weight the reviewer chose, overriding the source. None means "use
    # whatever the source had". False is meaningful and distinct from None:
    # it turns bold OFF on a block the source had bold.
    manual_bold: bool | None = None
    manual_italic: bool | None = None
    # A type style she chose ("sans", "serif", "mono"), or a font from the
    # catalog by id ("andika", see fonts/catalog.py); None is automatic.
    manual_font_family: str | None = None
    # A background colour she chose for the box ("#ffffff"), painted under
    # the text to cover what is beneath -- "HW" that would not translate.
    manual_fill: str | None = None
    # Where the words sit up and down in the box. Top is how text has always
    # been drawn, so it is never stored: None means top.
    manual_vertical_align: Literal["middle", "bottom"] | None = None
    # Line spacing she chose, as a multiple of the size: one of
    # fitting.LINE_SPACING_PRESETS. None keeps the automatic spacing (the
    # source's own, or the default, tightened when space is short).
    manual_line_spacing: float | None = None
    # A plain bool: the source's underline is never read, so there is no
    # "the source had it" state for a None to stand for.
    manual_underline: bool = False
    # "#rrggbb" painted behind the words only, row by row. Not manual_fill,
    # which paints the whole box.
    manual_highlight: str | None = None
    # --- Box styles (Google Slides' border, inside space and turning) ----
    # Her free angle, degrees clockwise, in (-180, 180]; never 0 -- a
    # straight box is None. While it is set, manual_bbox is the UNTURNED
    # frame, turned about its own centre, and it wins over manual_rotation
    # and `oriented`. While it is None, manual_bbox keeps its older meaning
    # for a block drawn along a tilt: the turned frame's upright extent.
    manual_angle: float | None = None
    # Space inside the box, in points: one of fitting.PADDING_PRESETS.
    # None is none. The words lay out inside it; fill and border do not.
    manual_padding: float | None = None
    # A line around the box: "#rrggbb", or None for no border. The weight
    # (fitting.BORDER_WEIGHTS; None draws 1pt) and the dash ("dashed",
    # "dotted"; None is solid) are kept while the colour is off, as Slides
    # keeps them, so turning the border back on brings them back.
    manual_border_color: str | None = None
    manual_border_width: float | None = None
    manual_border_dash: Literal["dashed", "dotted"] | None = None
    # --- Shapes (Google Slides' Insert > Shape) ------------------------
    # A preset name from reconstruct/shapes.SHAPES ("rect", "roundRect",
    # "ellipse"); None is a plain box. Only a box she made has one. A plain
    # str, not a Literal: a manifest written by a later build with more
    # shapes must still load here, and an unknown name draws as "rect".
    # The fill is manual_fill and the outline the border fields, so after
    # a rollback to a build without this field a shape is a box with her
    # words, fill and outline. Reset keeps it.
    manual_shape: str | None = None
    # The shape's adjust values in its own units (100000ths); None is its
    # defaults. A curve ("curve") has one: how far its middle sits off the
    # straight line, as a share of its length, positive to the band's top.
    manual_shape_adjust: list[float] | None = None
    # A line ("line" or "curve") is a band as long as the line, LINE_BAND
    # high, turned by manual_angle (which carries its direction: right to
    # left is 180). Its ends, start and end: "arrow", "open_arrow",
    # "circle", "open_circle", "square", "open_square", "diamond",
    # "open_diamond"; None is a plain end. Plain str for the same reason as
    # manual_shape. Its colour, weight and dash are the border fields.
    manual_line_start: str | None = None
    manual_line_end: str | None = None
    # --- Lists (Google Slides' bulleted and numbered lists, indents) ----
    # See reconstruct/lists.py. "bullet" or "number"; None is no list.
    manual_list: Literal["bullet", "number"] | None = None
    # The style of level 0 ("disc", "decimal", "lower-alpha-paren"...);
    # deeper levels follow from it. None with a list is "disc"/"decimal".
    manual_list_style: str | None = None
    # The first number at level 0, 1 to 999; None is 1.
    manual_list_start: int | None = None
    # Each paragraph's level, 0 to 3, by index. A paragraph past the end
    # takes the last level, and the list is never rewritten when the
    # translation changes, so a line that goes and comes back keeps its
    # level. None is level 0 throughout. For a paragraph not in a list
    # the level is a plain indent.
    manual_indents: list[int] | None = None
    # True when a person removed this block. The original text is still
    # redacted or covered -- it is removed from the page -- but nothing is
    # drawn in its place. Engine-found blocks cannot simply be deleted,
    # because the source glyphs exist and still have to be taken out.
    suppressed: bool = False

    @property
    def has_manual_layout(self) -> bool:
        return any(
            v is not None
            for v in (
                self.manual_bbox,
                self.manual_color,
                self.manual_font_size,
                self.manual_alignment,
                self.manual_bold,
                self.manual_italic,
                self.manual_font_family,
                self.manual_fill,
                self.manual_rotation,
                self.manual_vertical_align,
                self.manual_line_spacing,
                self.manual_highlight,
                self.manual_angle,
                self.manual_padding,
                self.manual_border_color,
                self.manual_border_width,
                self.manual_border_dash,
                self.manual_shape,
                self.manual_shape_adjust,
                self.manual_line_start,
                self.manual_line_end,
                self.manual_list,
                self.manual_list_style,
                self.manual_list_start,
                self.manual_indents,
            )
        ) or self.manual_underline

    @property
    def is_manual_block(self) -> bool:
        return self.source is TextSource.MANUAL

    @property
    def drawn_alignment(self) -> TextAlignment:
        """The alignment the words are drawn with.

        Only a JUSTIFY she chose is spread. One the parser detected keeps
        drawing from the left, as it always has: otherwise every justified
        worksheet she has already checked would re-space on its next
        rebuild.
        """
        if self.manual_alignment is not None:
            return self.manual_alignment
        if self.alignment is TextAlignment.JUSTIFY:
            return TextAlignment.LEFT
        return self.alignment

    @property
    def render_bold(self) -> bool:
        """Weight to draw with: the reviewer's choice, else the source's."""
        return self.style.bold if self.manual_bold is None else self.manual_bold

    @property
    def render_italic(self) -> bool:
        return self.style.italic if self.manual_italic is None else self.manual_italic

    @property
    def render_angle(self) -> float:
        """Degrees clockwise to draw the translation at: her free angle,
        else for tilted words her Straight (0) or the measured slant."""
        if self.manual_angle is not None:
            return self.manual_angle
        if self.oriented is None:
            return 0.0
        if self.manual_rotation is not None:
            return self.manual_rotation
        return self.oriented.angle

    @property
    def border_weight(self) -> float:
        """The border's weight in points, when it has a colour."""
        return self.manual_border_width or 1.0

    @property
    def render_bbox(self) -> Rect:
        """Where the translation is drawn (manual placement wins)."""
        return self.manual_bbox or self.bbox

    # Font size actually used when the translation was rendered. Differs
    # from style.font_size when the fitter had to shrink it, and QA
    # reports the ratio.
    translated_font_size: float | None = None
    # Where the last rebuild actually drew the translation: the fitted box
    # after any widening into empty space or reflow shift. The editor frames
    # this, so the handles a teacher grabs sit around the words on the page.
    drawn_bbox: Rect | None = None
    drawn_layout: DrawnLayout | None = None
    # How far right (page x, points) the last rebuild could have widened
    # this block's words into empty space, whether or not it needed to.
    # The editor's live words widen into it the same way, so a bullet or
    # an indent does not squeeze them before the new picture arrives. None
    # for a block placed by hand, turned, or in a picture.
    widen_x1: float | None = None

    @property
    def text(self) -> str:
        """Block text with newlines between visual lines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def flat_text(self) -> str:
        """Block text as one logical string, with soft line breaks joined.

        PDF line breaks inside a paragraph are a layout artifact, not
        punctuation. Joining them with a space (and repairing hyphenation)
        gives the translator a coherent sentence instead of shredded clauses.
        """
        if self.approved_source_text is not None:
            return self.approved_source_text
        parts: list[str] = []
        lines = [ln.text.strip() for ln in self.lines if ln.text.strip()]
        for idx, ln in enumerate(lines):
            if idx == 0:
                parts.append(ln)
                continue
            prev = parts[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha():
                # Soft hyphen from justification: "photo-\nsynthesis".
                parts[-1] = prev[:-1] + ln
            else:
                parts.append(" " + ln)
        return "".join(parts).strip()

    @property
    def char_count(self) -> int:
        return len(self.text)


class ImageObject(BaseObject):
    kind: Literal[ObjectKind.IMAGE] = ObjectKind.IMAGE

    # Storage key of the extracted original asset. The original bytes are
    # always preserved and reused; we never regenerate artwork (spec sec. 14).
    asset_key: str | None = None
    # Native pixel dimensions of the embedded image, before page scaling.
    pixel_width: int = 0
    pixel_height: int = 0
    # sha256 of the raw asset bytes; used to dedupe repeated logos across
    # pages and to cache OCR results per unique image.
    image_hash: str | None = None
    image_format: str | None = None
    # PDF xref of the image object, when the asset came from the resource
    # dictionary rather than a rasterized region.
    xref: int | None = None
    colorspace: str | None = None
    has_alpha: bool = False

    # --- Embedded-text pipeline (phase 4) ---
    # None = not yet examined; True/False = examined with this result. The
    # three-state value matters: "not checked" must never read as "no text".
    contains_text: bool | None = None
    ocr_regions: list[TextObject] = Field(default_factory=list)
    # Storage key of the rebuilt image with translated labels, when produced.
    translated_asset_key: str | None = None

    @property
    def aspect_ratio(self) -> float:
        return self.bbox.width / self.bbox.height if self.bbox.height else 0.0

    @property
    def pixel_aspect_ratio(self) -> float:
        return self.pixel_width / self.pixel_height if self.pixel_height else 0.0


class VectorObject(BaseObject):
    """Vector artwork: rules, boxes, arrows, hand-drawn diagram strokes.

    We deliberately do not try to reconstruct vector paths. They are left
    untouched in the output page, which is both the highest-fidelity and the
    safest option. What we record here is geometry, so the layout engine can
    avoid drawing translated text on top of a diagram and can recognize answer
    boxes and writing lines.
    """

    kind: Literal[ObjectKind.VECTOR] = ObjectKind.VECTOR
    # Counts of primitive path ops, used to tell a rule from real artwork.
    line_count: int = 0
    rect_count: int = 0
    curve_count: int = 0
    stroke_color: str | None = None
    fill_color: str | None = None
    stroke_width: float | None = None
    is_filled: bool = False
    # Heuristic labels; see analysis.vectors for how these are derived.
    looks_like_rule: bool = False
    looks_like_box: bool = False


class TableObject(BaseObject):
    kind: Literal[ObjectKind.TABLE] = ObjectKind.TABLE
    row_count: int = 0
    column_count: int = 0
    # Ids of the TextObjects that fall inside this table's cells. The text
    # itself stays in Page.text_objects -- a table is an overlay grouping,
    # not a container, so nothing is duplicated or lost.
    cell_object_ids: list[str] = Field(default_factory=list)
    cell_boxes: list[Rect] = Field(default_factory=list)


class FormFieldObject(BaseObject):
    kind: Literal[ObjectKind.FORM_FIELD] = ObjectKind.FORM_FIELD
    field_name: str | None = None
    field_type: str | None = None
    field_value: str | None = None


class ResponseArea(BaseModel):
    """A region reserved for student work. Treated as inviolable.

    Detected from writing rules, empty boxes and large whitespace gaps. The
    layout engine may not place translated text here; if it must, the page is
    flagged for human review rather than silently overwriting the space
    (spec sec. 28).
    """

    id: str = Field(default_factory=lambda: _new_id("resp"))
    bbox: Rect
    norm_bbox: NormRect
    # "writing_lines" | "answer_box" | "blank_space" | "grid"
    kind: str = "blank_space"
    confidence: float = 0.0
    line_count: int = 0
    source_object_ids: list[str] = Field(default_factory=list)


class ProtectedRegion(BaseModel):
    """A span of source text that must survive translation byte-identical."""

    id: str = Field(default_factory=lambda: _new_id("prot"))
    object_id: str
    # Character offsets into the owning object's flat_text.
    start: int
    end: int
    text: str
    # "math" | "chemistry" | "url" | "email" | "question_number" |
    # "answer_letter" | "page_number" | "citation" | "identifier" | "measurement"
    category: str
    placeholder: str


AnyObject = TextObject | ImageObject | VectorObject | TableObject | FormFieldObject


class PageDiagnostics(BaseModel):
    """Counts surfaced in the developer panel (spec sec. 56)."""

    text_objects: int = 0
    image_objects: int = 0
    vector_objects: int = 0
    table_objects: int = 0
    form_fields: int = 0
    ocr_regions: int = 0
    protected_regions: int = 0
    response_areas: int = 0
    translated_regions: int = 0
    potential_collisions: int = 0
    native_chars: int = 0
    ocr_chars: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)
