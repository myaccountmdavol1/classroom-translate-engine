"""Manual correction of a single translation (spec sec. 36).

Machine translation gets things wrong, especially on short technical labels,
so a teacher has to be able to fix a block and see the result immediately.
The constraint is that a one-word correction must not reprocess a 100-page
document.

What one edit does and does not do:

    re-run translation        NO   the corrected text IS the translation
    re-run OCR                NO   the pixels did not change
    re-render every preview    NO   only the edited page
    re-run document-wide QA    NO   only the edited page
    rebuild the export PDF     NOT NOW -- it is marked stale and rebuilt
                                    when someone downloads. Measured, a
                                    100-page export takes ~16s to rebuild
                                    versus 0.06s to re-render one page, so
                                    doing it per edit would make the editor
                                    unusable and would redo the same work
                                    for every correction

Protected content is re-verified against the edited text, so a human
correction cannot quietly remove a formula that the automated pass preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model.document import DocumentManifest, Page
from ..model.geometry import Rect
from ..model.enums import (
    ObjectKind,
    PageStatus,
    SemanticRole,
    TextAlignment,
    TextSource,
    TranslationStatus,
)
from ..model.objects import TextLine, TextObject, TextSpan, TextStyle, _new_id
from ..storage.base import ObjectStore
from ..util.logging import get_logger
from .reconstruct_stage import rebuild_single_page, verify_page

log = get_logger(__name__)

MAX_EDIT_LENGTH = 8000
# A block smaller than this cannot hold legible text.
MIN_BOX_POINTS = 8.0
# Font-size bounds for a hand-set size. Below 4pt is unreadable; above 96pt
# is almost certainly a mistake.
MIN_FONT_SIZE, MAX_FONT_SIZE = 4.0, 96.0
_HEX_COLOR = __import__("re").compile(r"^#[0-9a-fA-F]{6}$")


class EditError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic


@dataclass
class EditResult:
    page: Page
    object_id: str
    status: PageStatus
    layout_confidence: float
    # Protected values that are no longer present after the edit. Non-empty
    # means the page is flagged; the edit is still kept, because overriding a
    # teacher would be worse than warning them.
    lost_protected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# The only statuses a person may change. PENDING is deliberately absent: a
# block still waiting for the translator has nothing settled to correct, and
# an edit landing on it would be overwritten the moment the translation did.
_EDITABLE_STATUSES = (
    TranslationStatus.TRANSLATED,
    TranslationStatus.MANUAL_EDIT,
    TranslationStatus.FAILED,
)

# Internal role names, in words a teacher can act on. `protected_reason` is
# developer text ("role=FORMULA") and must never reach the screen.
_LOCK_WORDS = {
    "FORMULA": "Equations are kept exactly as they are.",
    "PAGE_NUMBER": "Page numbers are kept as they are.",
}


def is_editable(obj: TextObject) -> bool:
    """The single test for "may a person change this block".

    There used to be three of these and they disagreed: the list the editor
    greys out excluded PENDING and NOT_REQUIRED, the single-block mutators
    allowed PENDING, and the bulk path allowed both. So a block the UI drew
    as locked could still be changed through a bulk selection, and a block
    the UI thought editable could be refused -- with the refusal arriving as
    a bare "That change could not be saved." Everything that asks the
    question now asks it here.
    """
    return not obj.protected and obj.translation_status in _EDITABLE_STATUSES


def lock_reason(obj: TextObject) -> str:
    """Why a block cannot be corrected, in words a teacher can act on.

    Shown beside a locked block in the manuscript, and appended to the
    refusal when a mutation reaches one anyway, so the two say the same
    thing.
    """
    role = obj.role.value
    if role in _LOCK_WORDS:
        return _LOCK_WORDS[role]
    # Before the status branches: a protected block's status is not always
    # PROTECTED (a formula whose translation failed is both), and "this could
    # not be translated" would be the wrong reason to give for it.
    if obj.protected:
        return "Numbers and formulas here are kept exactly as they are."
    status = obj.translation_status.value
    if status == "NOT_REQUIRED":
        return "Nothing here needs translating."
    if status == "FAILED":
        return "This could not be translated, so the original was kept."
    if status == "PENDING":
        return "Not translated yet."
    return "Kept exactly as it was."


def _editable_object(page: Page, object_id: str) -> TextObject:
    obj = page.find_text_object(object_id)
    if obj is None:
        raise EditError("That text could not be found on this page.")
    if not is_editable(obj):
        raise EditError(f"That text cannot be edited. {lock_reason(obj)}")
    return obj


def editable_blocks(page: Page) -> list[TextObject]:
    """Blocks a teacher may correct, in reading order."""
    return [t for t in page.text_objects if is_editable(t)]


def apply_edit(
    manifest: DocumentManifest,
    page_number: int,
    object_id: str,
    new_text: str,
    *,
    list_levels: list[int] | None = None,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Replace one block's translation, then re-render and re-check the page.

    `list_levels` are the lines' indents sent with the text (the editor's
    text panel shows them as leading tabs), saved in the same step.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")

    obj = _editable_object(page, object_id)
    _no_words_on_a_line(obj)
    levels = _levels(list_levels) if list_levels is not None else None
    cleaned = (new_text or "").strip()
    # A shape keeps its outline with no words in it, so she can take the
    # words out; any other box with nothing in it is a mistake.
    if not cleaned and not obj.manual_shape:
        raise EditError("The translation cannot be empty.")
    if len(cleaned) > MAX_EDIT_LENGTH:
        raise EditError("That translation is too long.")
    push_history(page)

    # Preserve the machine output on the first edit so Reset has something to
    # restore without calling the translator again.
    if obj.machine_translated_text is None:
        obj.machine_translated_text = obj.translated_text

    obj.translated_text = cleaned
    if not cleaned:
        # No words in the shape: no English either.
        obj.approved_source_text = ""
    _forget_translation_only(obj, cleaned)
    obj.manually_edited = True
    obj.translation_status = TranslationStatus.MANUAL_EDIT
    if list_levels is not None:
        obj.manual_indents = levels

    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


def reset_edit(
    manifest: DocumentManifest,
    page_number: int,
    object_id: str,
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Put the machine translation back for one block."""
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    obj = _editable_object(page, object_id)

    if obj.machine_translated_text is None and not obj.has_manual_layout:
        raise EditError("There is no earlier version to go back to.")
    # After the refusal above, so a reset with nothing to reset does not
    # spend a history slot -- and before any field is touched, which is the
    # only point a snapshot is still of the state Reset is about to replace.
    # This was the one mutator without it: Reset cleared a manual box and
    # recorded nothing, so pressing Undo next popped the snapshot from the
    # drag BEFORE the last one and put the block back at an intermediate
    # position she had already moved on from. (This is also the real cause
    # of what was written up earlier as "the undo stack is client-side only,
    # so a reload disables Undo" -- undo is server-side and rides along in
    # the manifest; it was this.)
    push_history(page)
    # A line's place and turn are its two ends, which she drew: Reset keeps
    # them, and gives it a new line's look.
    held = (obj.manual_bbox, obj.manual_angle) if _is_line(obj) else None
    if obj.machine_translated_text is None:
        # Layout-only changes: keep the text, drop the placement overrides.
        obj.manual_bbox = None
        obj.manual_color = None
        obj.manual_font_size = None
        obj.manual_size_locked = False
        obj.manual_fit = False
        obj.manual_alignment = None
        obj.manual_bold = None
        obj.manual_italic = None
        obj.manual_font_family = None
        obj.manual_fill = None
        obj.manual_rotation = None
        _clear_text_styles(obj)
        _clear_box_styles(obj)
        _shape_look(obj)
        if held is not None:
            obj.manual_bbox, obj.manual_angle = held
        obj.manually_edited = False
        return _rerender(manifest, page, obj, store=store, target_lang=target_lang)

    obj.translated_text = obj.machine_translated_text
    _forget_translation_only(obj, obj.translated_text)
    obj.manually_edited = False
    obj.translation_status = TranslationStatus.TRANSLATED
    # Reset means "put it back as it was", which includes placement.
    obj.manual_bbox = None
    obj.manual_color = None
    obj.manual_font_size = None
    obj.manual_size_locked = False
    obj.manual_fit = False
    obj.manual_alignment = None
    obj.manual_bold = None
    obj.manual_italic = None
    obj.manual_font_family = None
    obj.manual_fill = None
    obj.manual_rotation = None
    _clear_text_styles(obj)
    _clear_box_styles(obj)
    _shape_look(obj)

    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


# How a new shape looks (the editor's SHAPE_LOOK): no fill, a black 1pt
# solid outline, words centred both ways with Medium inside space.
SHAPE_DEFAULTS = {
    "manual_fill": None,
    "manual_border_color": "#000000",
    "manual_border_width": 1.0,
    "manual_border_dash": None,
    "manual_padding": 4.0,
    "manual_alignment": TextAlignment.CENTER,
    "manual_vertical_align": "middle",
}


# How a new line looks (the editor's LINE_LOOK): black, 1pt, solid. Its
# ends and a curve's bend are kept, as the shape is.
LINE_DEFAULTS = {
    "manual_fill": None,
    "manual_border_color": "#000000",
    "manual_border_width": 1.0,
    "manual_border_dash": None,
    "manual_padding": None,
}


def _shape_look(obj: TextObject) -> None:
    """Reset on a shape puts back the look a new shape gets, not a bare
    box: cleared to no outline, an empty unfilled shape would vanish."""
    if not obj.manual_shape:
        return
    for name, value in (LINE_DEFAULTS if _is_line(obj) else SHAPE_DEFAULTS).items():
        setattr(obj, name, value)


def _is_line(obj) -> bool:
    """A line or a curve she drew (reconstruct/shapes.LINES)."""
    from ..reconstruct.shapes import is_line

    return is_line(obj.manual_shape)


def _no_words_on_a_line(obj: TextObject) -> None:
    if _is_line(obj):
        raise EditError("A line has no words.")


def _line_band(box: Rect) -> Rect:
    """A line's band: its length kept, LINE_BAND high about its centre, so
    a group resize or a stretched paste never makes it a fat box."""
    from ..reconstruct.shapes import LINE_BAND

    cy = (box.y0 + box.y1) / 2.0
    return Rect(x0=box.x0, y0=cy - LINE_BAND / 2.0, x1=box.x1, y1=cy + LINE_BAND / 2.0)


def _clear_text_styles(obj: TextObject) -> None:
    """Back to how the words were drawn before she styled them: top of the
    box, automatic spacing, no underline, no highlight, no list and no
    indents."""
    obj.manual_vertical_align = None
    obj.manual_line_spacing = None
    obj.manual_underline = False
    obj.manual_highlight = None
    obj.manual_list = None
    obj.manual_list_style = None
    obj.manual_list_start = None
    obj.manual_indents = None


def _clear_box_styles(obj: TextObject) -> None:
    """No turn, no inside space, no border: the box as the engine drew it.
    A shape stays a shape: Reset never turns her oval into a rectangle, and
    _shape_look gives it a new shape's look."""
    obj.manual_angle = None
    obj.manual_padding = None
    obj.manual_border_color = None
    obj.manual_border_width = None
    obj.manual_border_dash = None


def _hold_rows(obj: TextObject) -> None:
    """A box she placed, at a size she set, that the engine had to grow to
    hold its rows (builder: "size set by hand") keeps that height as its
    own frame, as a Slides text box grows round its text. The editor shows
    her frame, so without this its outline stopped rows short of the words
    the page drew below it. The next rebuild lays out the same rows in it.
    Turned boxes keep their frame: growing one would move its centre."""
    layout, drawn, frame = obj.drawn_layout, obj.drawn_bbox, obj.manual_bbox
    # A shape is never grown round its words: they fit its text area.
    if obj.suppressed or not obj.manual_size_locked or layout is None or obj.manual_shape:
        return
    if frame is None or drawn is None or obj.manual_angle is not None or layout.quad:
        return
    # Less than half a row past it is the usual allowance under the last
    # row, which the builder draws without growing anything.
    if drawn.y1 - frame.y1 > layout.leading / 2:
        obj.manual_bbox = frame.model_copy(update={"y1": round(drawn.y1, 3)})


def _rerender(
    manifest: DocumentManifest,
    page: Page,
    obj: TextObject | None,
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Rebuild the page and re-check it. `obj` is the block the change was
    about: only its warnings are reported. None (a change to several
    blocks, or one that removed the last of them) reports the page's."""
    try:
        findings = rebuild_single_page(
            manifest, page, store=store, target_lang=target_lang
        )
    except Exception as exc:
        raise EditError(
            "That page could not be rebuilt with your change.",
            diagnostic=f"{type(exc).__name__}: {exc}",
        ) from exc
    for t in page.text_objects:
        _hold_rows(t)

    # A changed page must be reviewed again; do not carry a prior approval
    # over a new layout merely because its object IDs survived.
    manifest.reviewed_findings = [key for key in manifest.reviewed_findings
        if len(key.split(":", 2)) < 2 or key.split(":", 2)[1] != str(page.number)]
    qa = verify_page(page, findings)
    page.revision += 1

    lost = sorted(
        {
            str(f.details.get("category", "value"))
            for f in qa.findings
            if f.code == "protected_content_lost"
        }
    )
    warnings = [
        f.message
        for f in qa.findings
        if f.severity in ("warning", "error")
        and (obj is None or not f.object_ids or obj.id in f.object_ids)
    ]

    return EditResult(
        page=page,
        object_id=obj.id if obj is not None else "",
        status=qa.status,
        layout_confidence=qa.layout_confidence,
        lost_protected=lost,
        warnings=warnings[:5],
    )


def _restack(page, objs: list[TextObject], layer: str) -> None:
    """Bring blocks to the front or send them to the back, keeping their
    order among themselves."""
    if layer not in ("front", "back"):
        raise EditError("That layer could not be read.")
    others = [t.z_order for t in page.text_objects if t not in objs]
    ordered = sorted(objs, key=lambda t: t.z_order)
    if layer == "front":
        start = max(others, default=0) + 1
        for i, t in enumerate(ordered):
            t.z_order = start + i
    else:
        start = min(others, default=0) - len(ordered)
        for i, t in enumerate(ordered):
            t.z_order = start + i


def _keep_frame(obj: TextObject, page, boxed: dict) -> None:
    """Hold a block the engine placed in the frame it is drawn in now,
    before its size is set by hand or to Fit.

    Left unplaced, the block would be laid out afresh in its narrower
    source box -- without the width it was widened into -- so on the page
    it shrank sideways, grew taller and re-wrapped the moment the new
    picture arrived, while the editor (which keeps the frame it shows)
    had drawn it one row wide. Pinned, the engine wraps the words in the
    very frame the editor drew them in.

    Space inside the box set in the same change grows the frame outward
    by the difference, as it does for a block left unplaced, so the words
    keep the room they had. Turned blocks and boxes she made already keep
    their frame, and a block never drawn has no frame to keep.
    """
    if obj.manual_bbox is not None or obj.is_manual_block or obj.drawn_bbox is None:
        return
    if obj.render_angle or (obj.drawn_layout is not None and obj.drawn_layout.quad):
        return
    d = 0.0
    if "manual_padding" in boxed:
        d = (boxed["manual_padding"] or 0.0) - (obj.manual_padding or 0.0)
    r = obj.drawn_bbox
    frame = Rect(x0=r.x0 - d, y0=r.y0 - d, x1=r.x1 + d, y1=r.y1 + d)
    obj.manual_bbox = frame.clamped_to(page.bounds)


def apply_layout(
    manifest: DocumentManifest,
    page_number: int,
    object_id: str,
    *,
    bbox: dict | None = None,
    color: str | None = None,
    font_size: float | None = None,
    alignment: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    font_family: str | None = None,
    fill: str | None = None,
    fit: bool | None = None,
    layer: str | None = None,
    rotation: str | None = None,
    vertical_align: str | None = None,
    line_spacing: float | None = None,
    underline: bool | None = None,
    highlight: str | None = None,
    border_color: str | None = None,
    border_width: float | None = None,
    border_dash: str | None = None,
    padding: float | None = None,
    angle: float | None = None,
    list: str | None = None,
    list_style: str | None = None,
    list_start: int | None = None,
    indent: int | None = None,
    list_levels: list[int] | None = None,
    shape: str | None = None,
    shape_adjust: list[float] | None = None,
    line_start: str | None = None,
    line_end: str | None = None,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Move, resize, recolour, realign or resize the type of one block.

    Automatic layout cannot be right on every document, particularly
    graphic-heavy ones, so a person can place a block exactly. Manual values
    always win over the automatic result, and the size is locked so the
    fitter will not quietly shrink a chosen size.

    The block's ORIGINAL box still governs what gets redacted or covered:
    moving the translation must not leave the source text showing.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    obj = _editable_object(page, object_id)
    # Read before history is recorded, so a value that cannot be read
    # leaves no empty undo step behind.
    styles = _text_styles(vertical_align, line_spacing, underline, highlight)
    boxed = _box_styles(border_color, border_width, border_dash, padding)
    turn = _angle(angle) if angle is not None else None
    # A list that would show the text's own numbers twice is refused here,
    # before history, so the refusal leaves no undo step.
    listed = _list_styles(list, list_style, list_start, indent, list_levels, obj)
    shaped = _shape_change(obj, shape) if shape is not None else None
    lined = _line_styles(
        obj, shaped or obj.manual_shape, shape_adjust, line_start, line_end,
        changed=shaped is not None and shaped != obj.manual_shape,
    )
    line = is_line_name(shaped or obj.manual_shape)
    if line and boxed.get("manual_border_color", "") is None:
        raise EditError("A line needs a colour.")
    push_history(page)

    if bbox is not None:
        try:
            box = Rect(
                x0=float(bbox["x0"]),
                y0=float(bbox["y0"]),
                x1=float(bbox["x1"]),
                y1=float(bbox["y1"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EditError("That position could not be read.") from exc
        if line:
            box = _line_band(box)
        if box.width < MIN_BOX_POINTS or box.height < MIN_BOX_POINTS - 1e-6:
            raise EditError("That box is too small to hold any text.")
        if angle is not None:
            # The box comes with its angle: a turned box is its frame.
            obj.manual_bbox = _clamped(box, page, turned=turn is not None)
        elif rotation is not None:
            # Straight or Follow the tilt decide below; the box keeps the
            # meaning it had before free turning existed.
            obj.manual_bbox = box.clamped_to(page.bounds)
        else:
            _place(obj, box, page)

    if color is not None:
        if not _HEX_COLOR.match(color):
            raise EditError("That colour could not be read.")
        obj.manual_color = color.lower()

    if font_size is not None:
        if not (MIN_FONT_SIZE <= float(font_size) <= MAX_FONT_SIZE):
            raise EditError(
                f"Text size must be between {MIN_FONT_SIZE:.0f} and "
                f"{MAX_FONT_SIZE:.0f} points."
            )
        obj.manual_font_size = round(float(font_size), 2)
        obj.manual_size_locked = True
        obj.manual_fit = False
    if fit:
        obj.manual_fit = True
        obj.manual_font_size = None
        obj.manual_size_locked = False
    if layer is not None:
        _restack(page, [obj], layer)

    if alignment is not None:
        if alignment not in {a.value for a in TextAlignment}:
            raise EditError("That alignment could not be read.")
        obj.manual_alignment = TextAlignment(alignment)

    if bold is not None:
        obj.manual_bold = bold
    if italic is not None:
        obj.manual_italic = italic
    if font_family is not None:
        obj.manual_font_family = _font_family(font_family)
    if fill is not None:
        # A line has no inside to fill.
        obj.manual_fill = None if line else _fill(fill)
    if rotation is not None:
        was_frame = obj.manual_angle is not None
        obj.manual_rotation = _rotation(obj, rotation)
        obj.manual_angle = None
        if bbox is None and not (rotation == "straight" and was_frame):
            # The upright box she may have dragged was drawn around the
            # other shape; it is read afresh rather than reinterpreted. A
            # box she had turned herself was its own frame, which drawn
            # Straight is simply the same box, level.
            obj.manual_bbox = None
    if angle is not None:
        _turn(obj, turn, pinned=bbox is not None)
    if (font_size is not None or fit) and bbox is None and angle is None and rotation is None:
        _keep_frame(obj, page, boxed)
    for field, value in styles.items():
        setattr(obj, field, value)
    _set_box_styles(obj, boxed)
    for field, value in listed.items():
        setattr(obj, field, value)
    if shaped is not None:
        obj.manual_shape = shaped
    for field, value in lined.items():
        setattr(obj, field, value)

    obj.manually_edited = True
    if obj.translation_status is TranslationStatus.TRANSLATED:
        obj.translation_status = TranslationStatus.MANUAL_EDIT

    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


def _levels(values) -> list[int] | None:
    """Per-line indents, each 0 to 3, in their stored form (None when all
    are 0)."""
    from ..reconstruct.lists import LEVELS, normalise_levels

    try:
        levels = [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise EditError("Each line's indent must be from 0 to 3.") from exc
    if any(v < 0 or v >= LEVELS for v in levels) or any(
        isinstance(v, float) and not float(v).is_integer() for v in values
    ):
        raise EditError("Each line's indent must be from 0 to 3.")
    return normalise_levels(levels)


def _list_styles(
    kind: str | None,
    style: str | None,
    start: int | None,
    indent: int | None,
    levels: list[int] | None,
    obj: TextObject,
) -> dict:
    """The model fields a list choice sets, each checked; only the ones
    given. Read before history, as _text_styles is.

    Turning a list on runs lists.check_list over the translation: text that
    already numbers itself in a way a list cannot take over is refused,
    so no number is shown twice. Turning it off keeps the indents, as
    Slides keeps them. `indent` (+1 or -1) moves every line of the box a
    level, from 0 to 3; `levels` sets each line's level outright.
    """
    from ..reconstruct import lists

    out: dict = {}
    if kind is not None and kind not in ("bullet", "number", "none"):
        raise EditError("That list could not be read.")
    if style is not None:
        style_kind = lists.style_kind(style)
        if style_kind is None or (kind is not None and kind != style_kind):
            raise EditError("That list style is not available.")
        kind = style_kind
    if start is not None:
        if isinstance(start, bool) or not isinstance(start, int) or not 1 <= start <= 999:
            raise EditError("Numbering can start from 1 to 999.")
    if indent is not None and indent not in (-1, 1):
        raise EditError("Indent can only go in or out one step.")
    given_levels = _levels(levels) if levels is not None else None

    if kind == "none":
        out.update(manual_list=None, manual_list_style=None, manual_list_start=None)
    elif kind is not None:
        if obj.manual_list == kind:
            # The list is already on, so only its style or start changes:
            # its own markers already take the place of typed ones
            # (list_number_changed says when a number differs), and text
            # changed since it was turned on is no reason to refuse.
            check = lists.ListCheck(ok=True)
        else:
            check = lists.check_list(
                obj.translated_text, kind, lists.source_hang(obj)
            )
        if not check.ok:
            raise EditError(check.message)
        if style is None:
            if obj.manual_list == kind and obj.manual_list_style:
                style = obj.manual_list_style
            else:
                style = check.style
        if style is not None and lists.style_kind(style) != kind:
            style = None  # typed bullets suggest nothing for numbers
        if start is None:
            start = check.start if check.start is not None else obj.manual_list_start
        out.update(manual_list=kind, manual_list_style=style)
    if start is not None:
        out["manual_list_start"] = None if start == 1 else start

    if levels is not None or indent is not None:
        current = given_levels if levels is not None else obj.manual_indents
        if indent is not None:
            count = len(lists.paragraphs_of(obj.translated_text))
            current = [
                max(0, min(lists.LEVELS - 1, v + indent))
                for v in lists.effective_levels(current, count)
            ]
        out["manual_indents"] = lists.normalise_levels(current)
    return out


def _vertical_align(value: str) -> str | None:
    """Where the words sit up and down in the box. Top is how text has
    always been drawn, so it is stored as None."""
    if value == "top":
        return None
    if value in ("middle", "bottom"):
        return value
    raise EditError("That vertical alignment could not be read.")


def _line_spacing(value: float) -> float | None:
    """One of the line spacing choices, or None (0) for automatic."""
    from ..reconstruct.fitting import LINE_SPACING_PRESETS

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EditError("That line spacing is not available.") from exc
    if number == 0:
        return None
    for preset in LINE_SPACING_PRESETS:
        if abs(number - preset) <= 1e-6:
            return preset
    raise EditError("That line spacing is not available.")


def _text_styles(
    vertical_align: str | None,
    line_spacing: float | None,
    underline: bool | None,
    highlight: str | None,
) -> dict:
    """The model fields these choices set, each checked; only the ones
    given. Shared by one block's edit and the bulk edit."""
    out: dict = {}
    if vertical_align is not None:
        out["manual_vertical_align"] = _vertical_align(vertical_align)
    if line_spacing is not None:
        out["manual_line_spacing"] = _line_spacing(line_spacing)
    if underline is not None:
        out["manual_underline"] = bool(underline)
    if highlight is not None:
        out["manual_highlight"] = _colour_or_none(
            highlight, "That highlight colour could not be read."
        )
    return out


def _border_color(value: str) -> str | None:
    """A border colour "#rrggbb", or None for "none"."""
    return _colour_or_none(value, "That border colour could not be read.")


def _border_width(value: float) -> float:
    """One of the border weights (fitting.BORDER_WEIGHTS)."""
    from ..reconstruct.fitting import BORDER_WEIGHTS

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EditError("That border weight is not available.") from exc
    for weight in BORDER_WEIGHTS:
        if abs(number - weight) <= 1e-6:
            return weight
    raise EditError("That border weight is not available.")


def _border_dash(value: str) -> str | None:
    """"dashed" or "dotted"; solid, the usual line, is stored as None."""
    if value == "solid":
        return None
    if value in ("dashed", "dotted"):
        return value
    raise EditError("That border style is not available.")


def is_line_name(shape: str | None) -> bool:
    from ..reconstruct.shapes import is_line

    return is_line(shape)


def _shape(value, *, lines: bool = False) -> str:
    """A closed shape the engine draws (reconstruct/shapes.SHAPES), or with
    `lines`, a line or a curve too (shapes.LINES) -- only a new box can be
    made a line."""
    from ..reconstruct.shapes import LINES, SHAPES

    if value in LINES and not lines:
        raise EditError("A line cannot become a shape.")
    if not isinstance(value, str) or (value not in SHAPES and value not in LINES):
        raise EditError("That shape is not available.")
    return value


def _shape_change(obj: TextObject, value) -> str:
    """Her choice of shape for a box, checked: only a box she made can be a
    shape, and a shape can change to another shape ("Change shape"). A line
    stays a line: it can be straightened or curved, never closed."""
    shape = _shape(value, lines=_is_line(obj))
    if not obj.is_manual_block:
        raise EditError("Only a box you made can be a shape.")
    if _is_line(obj) != is_line_name(shape):
        raise EditError("A line cannot become a shape.")
    return shape


def _line_end(value) -> str | None:
    """One of a line's ends (shapes.LINE_ENDS), or None for "none"."""
    from ..reconstruct.shapes import LINE_ENDS

    if value == "none":
        return None
    if value not in LINE_ENDS:
        raise EditError("That line end is not available.")
    return value


def _shape_adjust(shape: str | None, values) -> list[float] | None:
    """A shape's adjust values, checked against what it has: a curve has
    one, its bend, at most half its length either way (in 100000ths). The
    other shapes have none to set yet; [] puts back the defaults."""
    from ..reconstruct.shapes import MAX_BEND

    if values is None or (isinstance(values, list) and not values):
        return None
    if shape != "curve":
        raise EditError("That shape has nothing to adjust.")
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise EditError("That adjustment could not be read.") from exc
    if len(numbers) != 1 or not all(math.isfinite(v) for v in numbers):
        raise EditError("That adjustment could not be read.")
    if abs(numbers[0]) > MAX_BEND + 1e-6:
        raise EditError("That curve is bent too far.")
    return [round(numbers[0], 1)]


def _line_styles(
    obj, shape: str | None, adjust, start, end, *, changed: bool = False
) -> dict:
    """The line fields these choices set, each checked; only the ones
    given. Ends belong to a line, a bend to a curve. A line that changes
    kind (straight to curve) takes the new kind's own bend."""
    out: dict = {}
    line = is_line_name(shape)
    if (start is not None or end is not None) and not line:
        raise EditError("Only a line has ends.")
    if start is not None:
        out["manual_line_start"] = _line_end(start)
    if end is not None:
        out["manual_line_end"] = _line_end(end)
    if adjust is not None:
        out["manual_shape_adjust"] = _shape_adjust(shape, adjust)
    elif changed:
        out["manual_shape_adjust"] = None
    return out


def _padding(value: float) -> float | None:
    """One of the inside-space choices (fitting.PADDING_PRESETS), or None
    for 0."""
    from ..reconstruct.fitting import PADDING_PRESETS

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EditError("That inside space is not available.") from exc
    if number == 0:
        return None
    for preset in PADDING_PRESETS:
        if abs(number - preset) <= 1e-6:
            return preset
    raise EditError("That inside space is not available.")


def _angle(value: float) -> float | None:
    """Degrees clockwise in (-180, 180], to 2 places; None when straight."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EditError("That angle could not be read.") from exc
    if not math.isfinite(number):
        raise EditError("That angle could not be read.")
    number = math.fmod(number, 360.0)
    if number > 180.0:
        number -= 360.0
    elif number <= -180.0:
        number += 360.0
    number = round(number, 2)
    if number <= -180.0:
        number = 180.0
    return None if abs(number) < 0.005 else number


def _box_styles(
    border_color: str | None,
    border_width: float | None,
    border_dash: str | None,
    padding: float | None,
) -> dict:
    """The model fields these box choices set, each checked; only the ones
    given. Shared by one block's edit and the bulk edit."""
    out: dict = {}
    if border_color is not None:
        out["manual_border_color"] = _border_color(border_color)
    if border_width is not None:
        out["manual_border_width"] = _border_width(border_width)
    if border_dash is not None:
        out["manual_border_dash"] = _border_dash(border_dash)
    if padding is not None:
        out["manual_padding"] = _padding(padding)
    return out


def _set_box_styles(obj: TextObject, styles: dict) -> None:
    """Apply `_box_styles`. Choosing a weight or a dash on a box with no
    border gives it a black one, as Slides does: otherwise the choice would
    show nothing. Turning the border off keeps the weight and dash."""
    for field, value in styles.items():
        setattr(obj, field, value)
    if (
        ("manual_border_width" in styles or "manual_border_dash" in styles)
        and "manual_border_color" not in styles
        and obj.manual_border_color is None
    ):
        obj.manual_border_color = "#000000"


def _drawn_angle(obj: TextObject) -> float:
    """The angle the block is drawn at now (or would be, before its first
    drawing)."""
    if obj.drawn_layout is not None:
        return obj.drawn_layout.angle
    return obj.render_angle


def current_frame(obj: TextObject) -> Rect:
    """The box the editor shows for a block: the unturned frame of a turned
    one (builder.turned_frame_of), else her own box when she placed it
    (as _block_out shows it, even when the words run past it), else where
    it was drawn."""
    from ..reconstruct.builder import turned_frame_of

    layout = obj.drawn_layout
    if layout is not None and layout.quad:
        return turned_frame_of(layout)[0]
    return obj.manual_bbox or obj.drawn_bbox or obj.render_bbox


def _clamped(box: Rect, page: Page, *, turned: bool) -> Rect:
    """Keep a box on the page. Clamp rather than reject: a drag that ends
    slightly off-page should snap back, not throw the edit away. A turned
    frame is moved so its centre is on the page, never shrunk: cutting
    its sides would change the shape it turns to."""
    if not turned:
        return box.clamped_to(page.bounds)
    b = page.bounds
    cx, cy = box.center
    dx = min(max(cx, b.x0), b.x1) - cx
    dy = min(max(cy, b.y0), b.y1) - cy
    return Rect(x0=box.x0 + dx, y0=box.y0 + dy, x1=box.x1 + dx, y1=box.y1 + dy)


def _place(obj: TextObject, box: Rect, page: Page) -> None:
    """Store a box she dragged or resized, with no angle given.

    Every box the editor sends is a frame. A block drawn turned that has no
    angle of her own yet -- words following a picture's tilt -- takes the
    angle it is drawn at as hers, so the box she dragged is read as that
    turned frame and the slant stays exactly as it was.
    """
    if obj.manual_angle is None:
        drawn = _drawn_angle(obj)
        if drawn:
            obj.manual_angle = _angle(drawn)
    obj.manual_bbox = _clamped(box, page, turned=obj.manual_angle is not None)


def _turn(obj: TextObject, angle: float | None, *, pinned: bool) -> None:
    """Set her free angle (None is straight). `pinned` means a box came in
    the same request and was already stored as the frame."""
    if angle is None:
        if obj.manual_angle is None and obj.oriented is not None and not pinned:
            # Straight on words following a tilt: their upright box was
            # drawn around the turned shape and is read afresh, as Straight
            # always did. A frame she turned herself is kept, now level.
            obj.manual_bbox = None
        if obj.oriented is not None:
            obj.manual_rotation = 0.0
        obj.manual_angle = None
        return
    if obj.manual_angle is None and not pinned:
        # Pin the box where it is drawn, as a drag does, so it turns about
        # its own centre instead of jumping back to where it was read.
        obj.manual_bbox = current_frame(obj)
    obj.manual_angle = angle


def _rotation(obj: TextObject, value: str) -> float | None:
    """Her choice for tilted picture words: 0.0 to draw the translation
    straight, None to follow the measured slant again. The measured slant
    can be wrong; this is the way out."""
    if value not in ("straight", "follow"):
        raise EditError("That rotation could not be read.")
    if obj.oriented is None:
        raise EditError("This text is not tilted.")
    return 0.0 if value == "straight" else None


def _colour_or_none(value: str, what: str) -> str | None:
    """A chosen colour "#rrggbb", lower-cased, or None for "none". `what`
    is the message when it cannot be read."""
    if value == "none":
        return None
    if not _HEX_COLOR.match(value):
        raise EditError(what)
    return value.lower()


def _fill(value: str) -> str | None:
    """A chosen background colour, or None for no background."""
    return _colour_or_none(value, "That background colour could not be read.")


def _font_family(value: str) -> str | None:
    """A chosen type style ("sans", "serif", "mono") or catalog font
    ("andika", fonts/catalog.py), or None for automatic."""
    from ..fonts import catalog
    from ..fonts.registry import FONT_STYLES

    if value == "auto":
        return None
    if value not in FONT_STYLES and value not in catalog.family_ids():
        raise EditError("That font is not available.")
    return value


# --------------------------------------------------------------------------
# Undo
# --------------------------------------------------------------------------
# How many steps back a person can go. Each entry is small (only editable
# fields), but the history rides along in the manifest, so it is bounded.
MAX_HISTORY = 25

# The fields a person can change. Anything not listed here is engine-owned
# and is therefore not snapshotted or restored.
_EDITABLE_FIELDS = (
    "retry_background",
    "keep_original",
    "approved_source_text",
    "machine_translated_text",
    "translated_text",
    "manual_bbox",
    "manual_color",
    "manual_font_size",
    "manual_size_locked",
    "manual_fit",
    "manual_alignment",
    "manual_bold",
    "manual_italic",
    "manual_font_family",
    "manual_fill",
    "manual_rotation",
    "manual_vertical_align",
    "manual_line_spacing",
    "manual_underline",
    "manual_highlight",
    "manual_angle",
    "manual_padding",
    "manual_border_color",
    "manual_border_width",
    "manual_border_dash",
    "manual_shape",
    "manual_shape_adjust",
    "manual_line_start",
    "manual_line_end",
    "manual_list",
    "manual_list_style",
    "manual_list_start",
    "manual_indents",
    "z_order",
    "suppressed",
    "manually_edited",
    "translation_status",
)


# Only her style choices fall back to their defaults when a snapshot lacks
# them. The others' defaults are not "unset": translation_status would
# become PENDING (locking the block) and z_order would lose its place.
_DEFAULT_WHEN_MISSING = tuple(f for f in _EDITABLE_FIELDS if f.startswith("manual_"))


def _coerce(field: str, value):
    """Rebuild a typed value from a snapshot.

    Snapshots ride along in the manifest, so after a save/load cycle a Rect
    has become a plain dict and an enum a plain string. Assigning those with
    setattr bypasses Pydantic validation, and the dict then reaches the
    renderer and fails on attribute access -- so they are coerced back here.
    """
    if value is None:
        return None
    if field == "manual_bbox":
        return value if isinstance(value, Rect) else Rect(**value)
    if field == "manual_alignment":
        return value if isinstance(value, TextAlignment) else TextAlignment(value)
    if field == "translation_status":
        return (
            value
            if isinstance(value, TranslationStatus)
            else TranslationStatus(value)
        )
    return value


# How a manual block was last drawn: recomputed by every rebuild.
_NOT_SNAPSHOTTED = {"drawn_layout", "drawn_bbox", "translated_font_size"}


def _snapshot(page: Page) -> dict:
    blocks = {}
    for obj in page.text_objects:
        blocks[obj.id] = {
            field: getattr(obj, field) for field in _EDITABLE_FIELDS
        }
    return {
        "blocks": blocks,
        # Person-created ids, so undoing an "add" removes the block again.
        "manual_ids": [o.id for o in page.text_objects if o.is_manual_block],
        # Whole, so undoing a cut or a delete brings each box back with its
        # id. Not the drawing: every rebuild redraws every block from
        # scratch (builder resets drawn_layout/drawn_bbox first) and a
        # restore always rebuilds, and leaving it out keeps a page of
        # pasted copies from making each of the 25 snapshots several times
        # larger. Older snapshots that do hold it still load.
        "manual_objects": [
            o.model_dump(mode="json", exclude=_NOT_SNAPSHOTTED)
            for o in page.text_objects if o.is_manual_block
        ],
    }


def push_history(page: Page) -> None:
    """Record the page's editable state before it is changed."""
    page.redo_history.clear()
    page.edit_history.append(_snapshot(page))
    if len(page.edit_history) > MAX_HISTORY:
        del page.edit_history[: len(page.edit_history) - MAX_HISTORY]


def can_undo(page: Page) -> bool:
    return bool(page.edit_history)


def undo(
    manifest: DocumentManifest,
    page_number: int,
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Step back one change on this page."""
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    if not page.edit_history:
        raise EditError("There is nothing to undo on this page.")

    page.redo_history.append(_snapshot(page))
    snapshot = page.edit_history.pop()
    return _restore_snapshot(manifest, page, snapshot, store=store, target_lang=target_lang)


def redo(manifest: DocumentManifest, page_number: int, *, store: ObjectStore, target_lang: str) -> EditResult:
    page = manifest.page(page_number - 1)
    if page is None or not page.redo_history:
        raise EditError("There is nothing to redo on this page.")
    page.edit_history.append(_snapshot(page))
    snapshot = page.redo_history.pop()
    return _restore_snapshot(manifest, page, snapshot, store=store, target_lang=target_lang)


def _restore_snapshot(manifest, page, snapshot, *, store, target_lang):
    keep = set(snapshot["manual_ids"])

    # Remove person-created blocks that did not exist at that point.
    page.text_objects = [
        o for o in page.text_objects if not o.is_manual_block or o.id in keep
    ]
    existing = {obj.id for obj in page.text_objects}
    for raw in snapshot.get("manual_objects", []):
        if raw["id"] not in existing:
            page.text_objects.append(TextObject.model_validate(raw))
    for obj in page.text_objects:
        saved = snapshot["blocks"].get(obj.id)
        if saved is None:
            continue
        for field, value in saved.items():
            setattr(obj, field, _coerce(field, value))
        # A snapshot taken before a field existed has no entry for it. Its
        # value then was the field's default; leaving today's value instead
        # would keep an underline she added after that snapshot.
        for field in _DEFAULT_WHEN_MISSING:
            if field not in saved:
                setattr(obj, field, TextObject.model_fields[field].get_default())

    # Rebuilt even when no block is left (undoing the only box added to an
    # empty page), or the page would still show it. The undo itself must
    # not be re-recorded.
    anchor = page.text_objects[0] if page.text_objects else None
    return _rerender(manifest, page, anchor, store=store, target_lang=target_lang)


# --------------------------------------------------------------------------
# Adding and removing blocks
# --------------------------------------------------------------------------
def add_block(
    manifest: DocumentManifest,
    page_number: int,
    *,
    bbox: dict,
    text: str,
    color: str | None = None,
    font_size: float | None = None,
    alignment: str | None = None,
    source_text: str | None = None,
    fill: str | None = None,
    store: ObjectStore,
    target_lang: str,
) -> tuple[EditResult, TextObject]:
    """Create a new text block on the page.

    Needed because OCR misses things: stylised headings on dark backgrounds
    ("JOIN THE ADVENTURE" on a flyer) are never detected, so without this a
    document cannot be finished at all. The block has no source text beneath
    it, so reconstruction draws it and removes nothing.
    """
    # Typed in English (source_text): translate it, and keep the English as
    # the block's source so it can be corrected in English later. No size
    # given: the words fill the box she drew.
    english = (source_text or "").strip()
    spec = {
        "bbox": bbox, "text": text, "color": color, "font_size": font_size,
        "fit": font_size is None, "alignment": alignment, "fill": fill or "none",
        "source_text": english or None, "translate": bool(english),
    }
    result, (obj,) = add_blocks(
        manifest, page_number, [spec], store=store, target_lang=target_lang
    )
    return result, obj


def _spec_object(
    page: Page,
    spec: dict,
    *,
    english: str | None,
    translated: str,
    z_order: int,
    target_lang: str,
) -> TextObject:
    """One box she made -- added, duplicated or pasted -- from a spec (the
    batch endpoint's BoxSpecIn as a dict; missing keys take its defaults).

    Shared by every way a box is made, so they cannot drift. Every value is
    read here and an unreadable one raises EditError; nothing on the page
    changes.

    Nothing is taken from the block a copy was made of: no tilt measured in
    a picture, no cover boxes, no Word anchor, no "keep original". A copy is
    a new box drawn where it is put; it never takes anything off the page,
    and never matches a paragraph of a Word file.
    """
    try:
        b = spec["bbox"]
        box = Rect(
            x0=float(b["x0"]), y0=float(b["y0"]),
            x1=float(b["x1"]), y1=float(b["y1"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EditError("That position could not be read.") from exc
    shape = _shape(spec["shape"], lines=True) if spec.get("shape") is not None else None
    line = is_line_name(shape)
    if line:
        # A line is its band, whatever height came with it.
        box = _line_band(box)
    if box.width < MIN_BOX_POINTS or box.height < MIN_BOX_POINTS - 1e-6:
        raise EditError("That box is too small to hold any text.")
    turn = _angle(spec.get("angle") or 0.0)

    color = spec.get("color")
    if color is not None and not _HEX_COLOR.match(color):
        raise EditError("That colour could not be read.")
    fit = bool(spec.get("fit"))
    size = spec.get("font_size")
    if size is not None and fit:
        raise EditError("Choose a text size or Fit, not both.")
    if size is not None:
        try:
            size = round(float(size), 2)
        except (TypeError, ValueError) as exc:
            raise EditError("That text size could not be read.") from exc
        if not (MIN_FONT_SIZE <= size <= MAX_FONT_SIZE):
            raise EditError(
                f"Text size must be between {MIN_FONT_SIZE:.0f} and "
                f"{MAX_FONT_SIZE:.0f} points."
            )
    alignment = spec.get("alignment")
    if alignment is not None and alignment not in {a.value for a in TextAlignment}:
        raise EditError("That alignment could not be read.")
    look = spec.get("font_look") or "sans"
    if look not in ("sans", "serif", "mono"):
        raise EditError("That font style is not available.")
    family = _font_family(spec.get("font_family") or "auto")
    # A line has no inside to fill.
    fill = None if line else _fill(spec.get("fill") or "none")
    styles = _text_styles(
        spec.get("vertical_align") or "top",
        spec.get("line_spacing") or 0,
        bool(spec.get("underline")),
        spec.get("highlight") or "none",
    )
    boxed = _box_styles(
        spec.get("border_color") or "none",
        spec.get("border_width"),
        spec.get("border_dash") or "solid",
        spec.get("padding") or 0,
    )
    if line:
        if translated:
            raise EditError("A line has no words.")
        # A line always shows: black unless she chose a colour.
        boxed.setdefault("manual_border_color", "#000000")
        if boxed["manual_border_color"] is None:
            boxed["manual_border_color"] = "#000000"
    lined = _line_styles(
        None, shape, spec.get("shape_adjust"), spec.get("line_start") or "none",
        spec.get("line_end") or "none",
    ) if line else _line_styles(None, shape, spec.get("shape_adjust"), None, None)
    if len(translated) > MAX_EDIT_LENGTH:
        raise EditError("That text is too long.")
    # A box with a background or a border and no words is a cover: it
    # whites out (or frames) what did not translate. A shape with no words
    # is still the shape she drew. Without any of these, empty text is a
    # mistake.
    if not translated and not fill and not boxed.get("manual_border_color") and not shape:
        raise EditError("The text cannot be empty.")

    # Read against a stand-in holding only the text, so a list that would
    # show its own numbers twice is refused before anything is made.
    listed = _list_styles(
        None if (spec.get("list") or "none") == "none" else spec.get("list"),
        spec.get("list_style"), spec.get("list_start"), None,
        spec.get("list_levels"),
        TextObject.model_construct(translated_text=translated, manual_list=None,
                                   manual_list_style=None, manual_list_start=None,
                                   manual_indents=None),
    )
    if spec.get("fit_height") and translated:
        box = _fitted_height(
            box, translated, spec | {"_listed": listed}, size or 12.0, look, family, target_lang
        )
    # A turned frame is moved onto the page, never cut: cutting its sides
    # would change the shape it turns to.
    box = _clamped(box, page, turned=turn is not None)

    bold, italic = spec.get("bold"), spec.get("italic")
    machine = (spec.get("machine_text") or "").strip() or None
    if spec.get("translate"):
        machine = translated
    # The face the original was drawn in when its type is automatic
    # (builder._source_look reads it back from these two flags).
    style = TextStyle(
        font_size=size or (min(MAX_FONT_SIZE, max(8.0, box.height * 0.7)) if fit else 12.0),
        color=color.lower() if color else None,
        bold=bool(bold), italic=bool(italic),
        serif=look == "serif", monospace=look == "mono",
    )
    obj = TextObject(
        id=_new_id("manual"),
        kind=ObjectKind.TEXT,
        bbox=box,
        norm_bbox=box.normalized(page.width, page.height),
        z_order=z_order,
        lines=[
            TextLine(
                bbox=box,
                spans=[TextSpan(text=english or translated, bbox=box, style=style)],
            )
        ],
        source=TextSource.MANUAL,
        role=SemanticRole.OTHER,
        style=style,
        translated_text=translated,
        translation_only_text=translated,
        machine_translated_text=machine,
        approved_source_text=english,
        translation_status=TranslationStatus.MANUAL_EDIT,
        manually_edited=True,
        manual_bbox=box,
        manual_color=color.lower() if color else None,
        manual_font_size=size,
        manual_size_locked=size is not None,
        manual_fit=fit,
        manual_alignment=TextAlignment(alignment) if alignment else None,
        manual_bold=bold,
        manual_italic=italic,
        manual_font_family=family,
        manual_fill=fill,
        manual_angle=turn,
        manual_shape=shape,
    )
    for name, value in styles.items():
        setattr(obj, name, value)
    _set_box_styles(obj, boxed)
    for name, value in listed.items():
        setattr(obj, name, value)
    for name, value in lined.items():
        setattr(obj, name, value)
    return obj


def _fitted_height(
    box: Rect, text: str, spec: dict, size: float, look: str,
    family: str | None, target_lang: str,
) -> Rect:
    """The box made exactly as tall as the rows the engine will draw.

    For text pasted from outside, whose length in the translation nobody
    knew when the box was made: wrapped at the box's inner width with the
    same font lay_out resolves, so the box holds what is drawn, no more.
    """
    from types import SimpleNamespace

    from ..fonts.registry import REGISTRY, FontResolutionError
    from ..languages import get_language
    from ..reconstruct.fitting import DEFAULT_LEADING, inset, wrap_rows
    from ..reconstruct.lists import MarkerFonts, marker_visual, plan_paragraphs

    language = get_language(target_lang)
    try:
        rfont = REGISTRY.resolve(
            text, script=language.script if language else "Latin",
            bold=bool(spec.get("bold")), italic=bool(spec.get("italic")),
            family=family, look=look,
        )
    except FontResolutionError:
        return box
    padding = _padding(spec.get("padding") or 0) or 0.0
    inner, _, _ = inset(box, padding)
    # Its list and indents as the box will have them, so indented rows
    # are counted where they wrap.
    listed = spec.get("_listed") or {}
    stub = SimpleNamespace(
        source=TextSource.MANUAL, lines=[],
        **{name: listed.get(name) for name in (
            "manual_list", "manual_list_style", "manual_list_start", "manual_indents",
        )},
    )
    rtl = bool(language and language.rtl)
    rows = len(wrap_rows(
        plan_paragraphs(text, stub, REGISTRY.font_for(rfont)), rfont, size,
        max(1.0, inner.width), MarkerFonts(rfont, bool(spec.get("bold")), bool(spec.get("italic"))),
        text=text, visual=(lambda m: marker_visual(m, True)) if rtl else None,
    )) or 1
    ratio = _line_spacing(spec.get("line_spacing") or 0) or DEFAULT_LEADING
    height = rows * size * ratio + 2 * padding + 2
    return Rect(x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y0 + max(MIN_BOX_POINTS, height))


def add_blocks(
    manifest: DocumentManifest,
    page_number: int,
    specs: list[dict],
    *,
    store: ObjectStore,
    target_lang: str,
) -> tuple[EditResult, list[TextObject]]:
    """Make several boxes at once: one undo step, one re-render.

    Duplicate, Paste and text pasted from outside the app land here. All or
    nothing: every box is read, and every English one translated, before
    anything is recorded, so a box that cannot be made -- or a translation
    that fails -- leaves the page and its undo history as they were.

    Returns the boxes in the order of `specs`. They are stacked above
    everything on the page, in that order, so copies keep their stacking
    among themselves.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    if not specs:
        raise EditError("Nothing was added.")

    base = max((t.z_order for t in page.text_objects), default=0) + 1
    created: list[TextObject] = []
    for i, spec in enumerate(specs):
        english = (spec.get("source_text") or "").strip() or None
        if english and len(english) > MAX_EDIT_LENGTH:
            raise EditError("That text is too long.")
        if spec.get("translate"):
            # Pasted from outside: teachers work in English, so what she
            # pasted is the English and its translation is drawn -- made
            # beforehand by translate_specs where the caller could, so the
            # document is not held while the translator answers.
            if not english:
                raise EditError("There is no English to translate.")
            translated = spec.get("translated")
            if translated is None:
                translated = _translate_english(
                    english, manifest=manifest, target_lang=target_lang
                )
        else:
            translated = (spec.get("text") or "").strip()
        created.append(
            _spec_object(
                page, spec, english=english, translated=translated,
                z_order=base + i, target_lang=target_lang,
            )
        )

    push_history(page)
    page.text_objects.extend(created)
    result = _rerender(manifest, page, created[0], store=store, target_lang=target_lang)
    return result, created


def translate_specs(manifest: DocumentManifest, specs: list[dict], *, target_lang: str) -> None:
    """Translate the English of every pasted box (`translate`) into its
    spec's "translated", ready for add_blocks.

    Run BEFORE the document is locked: a slow translator then holds up no
    other save on it. Nothing is changed here, so all or nothing still
    holds -- a translation that fails raises EditError before any box is
    made.
    """
    for spec in specs:
        if not spec.get("translate"):
            continue
        english = (spec.get("source_text") or "").strip()
        if not english:
            raise EditError("There is no English to translate.")
        if len(english) > MAX_EDIT_LENGTH:
            raise EditError("That text is too long.")
        spec["translated"] = _translate_english(
            english, manifest=manifest, target_lang=target_lang
        )


def remove_block(
    manifest: DocumentManifest,
    page_number: int,
    object_id: str,
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Remove a block from the page.

    A person-created block is deleted outright. An engine-found block is
    SUPPRESSED instead: its original glyphs exist on the page and still have
    to be taken out, so it is removed and simply not redrawn.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    obj = _editable_object(page, object_id)

    push_history(page)

    if obj.is_manual_block:
        page.text_objects = [o for o in page.text_objects if o.id != object_id]
        # The last box on a page that had no text of its own can go too:
        # the page is then drawn with nothing on it.
        anchor = page.text_objects[0] if page.text_objects else None
        return _rerender(manifest, page, anchor, store=store, target_lang=target_lang)

    obj.suppressed = True
    obj.manually_edited = True
    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


def remove_blocks(
    manifest: DocumentManifest,
    page_number: int,
    object_ids: list[str],
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Remove several blocks as one undo step: Cut, or Delete on a
    selection.

    Each goes the way remove_block takes one: a box she made is deleted, a
    block the engine found is suppressed. Blocks that are gone already or
    locked are skipped, as a bulk edit skips them, so a stale selection
    does not throw the whole cut away.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    found = [o for o in (page.find_text_object(i) for i in object_ids) if o is not None]
    removable = [o for o in found if is_editable(o)]
    if not removable:
        if len(object_ids) == 1 and len(found) == 1:
            raise EditError(f"That text cannot be removed. {lock_reason(found[0])}")
        raise EditError("None of the selected text could be removed.")

    push_history(page)
    gone = {o.id for o in removable if o.is_manual_block}
    for obj in removable:
        if not obj.is_manual_block:
            obj.suppressed = True
            obj.manually_edited = True
    page.text_objects = [o for o in page.text_objects if o.id not in gone]
    anchor = page.text_objects[0] if page.text_objects else None
    return _rerender(manifest, page, anchor, store=store, target_lang=target_lang)


def restore_block(
    manifest: DocumentManifest,
    page_number: int,
    object_id: str,
    *,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Put a suppressed block back."""
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    obj = page.find_text_object(object_id)
    if obj is None:
        raise EditError("That text could not be found on this page.")
    push_history(page)
    obj.suppressed = False
    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


# --------------------------------------------------------------------------
# Bulk changes
# --------------------------------------------------------------------------
def apply_bulk(
    manifest: DocumentManifest,
    page_number: int,
    object_ids: list[str],
    *,
    color: str | None = None,
    font_size: float | None = None,
    alignment: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    font_family: str | None = None,
    fill: str | None = None,
    fit: bool | None = None,
    layer: str | None = None,
    vertical_align: str | None = None,
    line_spacing: float | None = None,
    underline: bool | None = None,
    highlight: str | None = None,
    border_color: str | None = None,
    border_width: float | None = None,
    border_dash: str | None = None,
    padding: float | None = None,
    angle: float | None = None,
    list: str | None = None,
    list_style: str | None = None,
    list_start: int | None = None,
    indent: int | None = None,
    shape: str | None = None,
    line_start: str | None = None,
    line_end: str | None = None,
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Apply the same colour, size or alignment to several blocks at once.

    One re-render for the whole set rather than one per block: a page with 48
    shrunken blocks is otherwise 48 round trips.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    if not object_ids:
        raise EditError("No text was selected.")
    if color is not None and not _HEX_COLOR.match(color):
        raise EditError("That colour could not be read.")
    if font_size is not None and not (
        MIN_FONT_SIZE <= float(font_size) <= MAX_FONT_SIZE
    ):
        raise EditError(
            f"Text size must be between {MIN_FONT_SIZE:.0f} and "
            f"{MAX_FONT_SIZE:.0f} points."
        )
    if alignment is not None and alignment not in {a.value for a in TextAlignment}:
        raise EditError("That alignment could not be read.")
    family = _font_family(font_family) if font_family is not None else None
    fill_value = _fill(fill) if fill is not None else None
    styles = _text_styles(vertical_align, line_spacing, underline, highlight)
    boxed = _box_styles(border_color, border_width, border_dash, padding)
    turn = _angle(angle) if angle is not None else None
    shape_value = _shape(shape) if shape is not None else None
    ends = {}
    if line_start is not None:
        ends["manual_line_start"] = _line_end(line_start)
    if line_end is not None:
        ends["manual_line_end"] = _line_end(line_end)
    # Each box's list, read before history: a box whose own numbers a
    # list would show twice keeps them and is reported, and the others
    # still get the list. If every box would be refused, nothing changes.
    wants_list = any(v is not None for v in (list, list_style, list_start, indent))
    listed: dict[str, dict] = {}
    refusals: dict[str, str] = {}
    if wants_list:
        for object_id in object_ids:
            obj = page.find_text_object(object_id)
            if obj is None or not is_editable(obj):
                continue
            try:
                listed[obj.id] = _list_styles(list, list_style, list_start, indent, None, obj)
            except EditError as exc:
                refusals[obj.id] = exc.message
        if refusals and not listed:
            raise EditError(next(iter(refusals.values())))

    push_history(page)

    touched: list[TextObject] = []
    for object_id in object_ids:
        obj = page.find_text_object(object_id)
        # Skip silently rather than failing the batch: a stale selection
        # should not throw away the whole operation. The test is the same
        # `is_editable` the editor greys blocks out by, so a bulk selection
        # can no longer reach a block a single edit would have refused.
        if obj is None or not is_editable(obj) or obj.id in refusals:
            continue
        if shape_value is not None and (not obj.is_manual_block or _is_line(obj)):
            # Only a box she made takes a shape, and never a line; the rest
            # of the selection still changes.
            continue
        if color is not None:
            obj.manual_color = color.lower()
        if font_size is not None:
            obj.manual_font_size = round(float(font_size), 2)
            obj.manual_size_locked = True
            obj.manual_fit = False
        if fit:
            obj.manual_fit = True
            obj.manual_font_size = None
            obj.manual_size_locked = False
        if alignment is not None:
            obj.manual_alignment = TextAlignment(alignment)
        if bold is not None:
            obj.manual_bold = bold
        if italic is not None:
            obj.manual_italic = italic
        if font_family is not None:
            obj.manual_font_family = family
        line = _is_line(obj)
        if fill is not None and not line:
            # A line has no inside to fill.
            obj.manual_fill = fill_value
        if (font_size is not None or fit) and angle is None:
            _keep_frame(obj, page, boxed)
        for field, value in styles.items():
            setattr(obj, field, value)
        # A line keeps its colour when the others lose their border, and
        # takes ends that only a line has.
        _set_box_styles(obj, {
            k: v for k, v in boxed.items()
            if not (line and k == "manual_border_color" and v is None)
        })
        if line:
            for field, value in ends.items():
                setattr(obj, field, value)
        if angle is not None:
            # Each box turns about its own centre.
            _turn(obj, turn, pinned=False)
        for field, value in listed.get(obj.id, {}).items():
            setattr(obj, field, value)
        if shape_value is not None:
            obj.manual_shape = shape_value
        obj.manually_edited = True
        if obj.translation_status is TranslationStatus.TRANSLATED:
            obj.translation_status = TranslationStatus.MANUAL_EDIT
        touched.append(obj)

    if touched and layer is not None:
        _restack(page, touched, layer)

    if not touched:
        # With one block selected there is a specific reason, and saying it
        # is the difference between "that change could not be saved" and
        # "page numbers are kept as they are".
        blocked = [
            o for o in (page.find_text_object(i) for i in object_ids) if o is not None
        ]
        if len(blocked) == 1:
            raise EditError(f"That text cannot be changed. {lock_reason(blocked[0])}")
        raise EditError("None of the selected text could be changed.")
    result = _rerender(manifest, page, touched[0], store=store, target_lang=target_lang)
    if refusals:
        result.warnings[:0] = [
            next(iter(refusals.values())), "Some boxes kept their own numbers.",
        ]
        del result.warnings[5:]
    return result



def apply_positions(
    manifest: DocumentManifest,
    page_number: int,
    moves: list[dict],
    *,
    alignment: str | None = None,
    align_ids: list[str] | tuple[str, ...] = (),
    store: ObjectStore,
    target_lang: str,
) -> EditResult:
    """Move several blocks at once: one undo step, one re-render.

    A group drag in the editor lands here. Saving each block on its own
    would be N rebuilds of the page and N undo steps for one gesture.

    `alignment` is given to the blocks in `align_ids` in the same step:
    lining boxes up by their left edges also lines up their words, and
    Undo puts back both at once.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    if not moves:
        raise EditError("Nothing was moved.")
    if alignment is not None and alignment not in {a.value for a in TextAlignment}:
        raise EditError("That alignment could not be read.")
    parsed: list[tuple[TextObject, Rect]] = []
    for move in moves:
        obj = page.find_text_object(str(move.get("id", "")))
        if obj is None or not is_editable(obj):
            continue
        try:
            b = move["bbox"]
            box = Rect(x0=float(b["x0"]), y0=float(b["y0"]), x1=float(b["x1"]), y1=float(b["y1"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EditError("That position could not be read.") from exc
        if _is_line(obj):
            box = _line_band(box)
        if box.width < MIN_BOX_POINTS or box.height < MIN_BOX_POINTS - 1e-6:
            raise EditError("That box is too small to hold any text.")
        parsed.append((obj, box))
    if not parsed:
        raise EditError("None of the selected text could be moved.")

    push_history(page)
    for obj, box in parsed:
        # Frames, as from a single drag: a block following a tilt keeps it.
        _place(obj, box, page)
        obj.manually_edited = True
        if obj.translation_status is TranslationStatus.TRANSLATED:
            obj.translation_status = TranslationStatus.MANUAL_EDIT
    if alignment is not None:
        for object_id in align_ids:
            obj = page.find_text_object(object_id)
            if obj is None or not is_editable(obj):
                continue
            obj.manual_alignment = TextAlignment(alignment)
            obj.manually_edited = True
            if obj.translation_status is TranslationStatus.TRANSLATED:
                obj.translation_status = TranslationStatus.MANUAL_EDIT
    return _rerender(manifest, page, parsed[0][0], store=store, target_lang=target_lang)


def _forget_translation_only(obj, text) -> None:
    """Drop the translation-only copy made from the old wording.

    A tight box is drawn from `translation_only_text`; an edit that left it
    alone was saved but never shown -- English corrected in the editor, and
    the page (and the Word file's picture) kept the misread translation.
    A block she added has only her text, so it is that.
    """
    obj.translation_only_text = text if obj.is_manual_block else None


def _translate_english(text: str, *, manifest, target_lang: str):
    """Translate a teacher's English, keeping each of her lines a line.

    She cannot read the translation, so a new line is something she can only
    place in the English. Sent whole, the translator runs her lines together;
    sent line by line, each comes back where she put it.
    """
    from ..translate.one_off import translate_one, OneOffError
    out: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        try:
            result = translate_one(
                line.strip(), target_lang=target_lang,
                source_lang=manifest.analysis.detected_language,
            )
        except OneOffError as exc:
            raise EditError(str(exc)) from exc
        if not result.protected_intact:
            raise EditError(
                "The translation changed a number or formula. Check the English, "
                "or edit the translation directly."
            )
        out.append(result.text.strip())
    return "\n".join(out)


def translate_from_english(
    manifest, page_number, object_id, text, *, store, target_lang, levels=None
):
    """Replace a block's English and translate it again.

    The way a teacher who cannot read the target language fixes a block:
    she changes what it should SAY. Unlike correct_source this always
    re-translates, even over a hand edit -- asking for it is the point.

    `levels` are her lines' indents, from the leading tabs she typed in the
    English. Each English line comes back as one translated line, so they
    index the translation's lines too.
    """
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError("That page could not be found.")
    obj = _editable_object(page, object_id)
    _no_words_on_a_line(obj)
    cleaned = (text or "").strip()
    # A shape may have its words taken out: it keeps its outline.
    if (not cleaned and not obj.manual_shape) or len(cleaned) > MAX_EDIT_LENGTH:
        raise EditError("Enter English text between 1 and 8,000 characters.")
    stored_levels = _levels(levels) if levels is not None else None
    translated = (
        _translate_english(cleaned, manifest=manifest, target_lang=target_lang)
        if cleaned else ""
    )
    push_history(page)
    if levels is not None:
        obj.manual_indents = stored_levels
    obj.approved_source_text = cleaned
    obj.translated_text = translated
    obj.machine_translated_text = translated
    # Asking for a translation is asking to see it: a line held back as
    # hard to read, or left alone over a busy background, is drawn now.
    obj.keep_original = False
    obj.retry_background = True
    _forget_translation_only(obj, translated)
    if not obj.is_manual_block:
        obj.manually_edited = False
        obj.translation_status = TranslationStatus.TRANSLATED
    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)


def correct_source(manifest, page_number, object_id, text, *, store, target_lang):
    """Correct OCR wording without changing the original pixels or mask boxes.

    Teacher-edited target text stays pinned. Only an untouched machine
    translation is regenerated, and only after successful protected translation.
    """
    from ..translate.one_off import translate_one, OneOffError
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError("That page could not be found.")
    obj = _editable_object(page, object_id)
    cleaned = text.strip()
    if not cleaned or len(cleaned) > MAX_EDIT_LENGTH:
        raise EditError("Enter source text between 1 and 8,000 characters.")
    translated = None
    if not obj.manually_edited:
        try:
            translated = translate_one(cleaned, target_lang=target_lang,
                source_lang=manifest.analysis.detected_language)
        except OneOffError as exc:
            raise EditError(str(exc)) from exc
        if not translated.protected_intact:
            raise EditError("The translation changed a number or formula. Please edit the translation directly.")
    push_history(page)
    obj.approved_source_text = cleaned
    obj.keep_original = False
    if translated is not None:
        obj.translated_text = translated.text
        obj.machine_translated_text = translated.text
        _forget_translation_only(obj, translated.text)
        obj.translation_status = TranslationStatus.TRANSLATED
    result = _rerender(manifest, page, obj, store=store, target_lang=target_lang)
    if obj.manually_edited:
        result.warnings.append("Your translation edit was kept. Update it directly if the correction changes its meaning.")
    return result
