"""Bulleted and numbered lists, indents, and numbers a worksheet typed itself.

One module for the drawing (builder), the edit checks (pipeline/edit) and
what the editor is told (api/routes._block_out), so the three cannot
disagree about where a marker goes or whether numbering is allowed.

A box's paragraphs are its translation split at each line break. Each can
carry a marker -- a bullet or a number -- that hangs to the left of its
text, so the wrapped lines line up under the words and not under the
marker, as in Google Slides. A marker is one of two kinds:

  * Managed: she turned on a list (manual_list), and the engine draws and
    counts the markers. The translation itself never holds them.
  * Typed: the text itself starts with "1." or "a)" or "•", as worksheets
    usually do. It is split off and hung the same way, which is what
    keeps a wrapped question's second line under its words -- on every
    block, list or not. Turning numbering on over typed numbers either
    takes them over (consecutive "3." "4." "5." becomes a list starting at
    3) or is refused, so a number is never shown twice.

Every distance is in em of the drawn size, so Fit scales the indents with
the words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..fonts import catalog
from ..fonts.registry import REGISTRY, ResolvedFont

LEVELS = 4
# Each level's marker starts this far in from the one above.
LEVEL_STEP_EM = 2.0
# Marker start to text start, at least, for a list she turned on.
MIN_HANG_EM = 1.25
# Room between the widest marker of a level and the text.
MARKER_GAP_EM = 0.5
# The words always keep at least this much width beside their marker.
MIN_COLUMN_EM = 2.0

# The square is the small one (as Slides draws it): the full-size ■ stood as
# tall as a capital and outweighed the words.
BULLET_GLYPHS = {"disc": "•", "circle": "◦", "square": "▪", "dash": "–", "check": "✓"}
# Level 0's style picks the bullet at each deeper level, as Slides does.
BULLET_LEVELS = {
    "disc": ("disc", "circle", "square", "disc"),
    "circle": ("circle", "square", "disc", "circle"),
    "square": ("square", "disc", "circle", "square"),
    "dash": ("dash", "circle", "square", "disc"),
    "check": ("check", "circle", "square", "disc"),
}
# Number styles by base; a "-paren" style ("1)") keeps the bracket at
# every level.
NUMBER_LEVELS = {
    "decimal": ("decimal", "lower-alpha", "lower-roman", "decimal"),
    "lower-alpha": ("lower-alpha", "lower-roman", "decimal", "lower-alpha"),
    "upper-alpha": ("upper-alpha", "decimal", "lower-alpha", "lower-roman"),
    "lower-roman": ("lower-roman", "decimal", "lower-alpha", "lower-roman"),
    "upper-roman": ("upper-roman", "upper-alpha", "decimal", "lower-alpha"),
}
BULLET_STYLES = tuple(BULLET_GLYPHS)
NUMBER_STYLES = tuple(NUMBER_LEVELS) + tuple(f"{s}-paren" for s in NUMBER_LEVELS)
DEFAULT_STYLE = {"bullet": "disc", "number": "decimal"}

# A marker typed at the start of a paragraph: a number, a letter or a
# roman numeral with "." or ")" (optionally opened with "("), or a bullet
# character, then a space and more text. The delimiter and the space are
# both required, so "3 apples", "1.5 cm", "e.g. this", "Mr. Smith" and
# "U.S." never match.
TYPED_MARKER = re.compile(
    r"^(\(?(\d{1,3}|[A-Za-z]|[ivxlcdm]{2,6}|[IVXLCDM]{2,6})[.)]|[•●◦▪■□–\-*✓✔])\s+(?=\S)"
)
# A printed row's whole text is a marker (the source drew "3." apart from
# its words), for measuring the source's own hang.
_WHOLE_MARKER = re.compile(
    r"^(\(?(\d{1,3}|[A-Za-z]|[ivxlcdm]{2,6}|[IVXLCDM]{2,6})[.)]|[•●◦▪■□–\-*✓✔])$"
)
_TYPED_BULLETS = {
    "•": "disc", "●": "disc", "◦": "circle", "▪": "square", "■": "square",
    "□": "square", "–": "dash", "-": "dash", "*": "dash", "✓": "check", "✔": "check",
}
# The source's own hang is believed up to this many times its size.
MAX_SOURCE_HANG_SIZES = 6.0

REFUSE_NUMBERS = (
    "This text already has its own numbers or letters, so numbering it here "
    "would show them twice. They are kept as written."
)
REFUSE_BULLETS = (
    "These lines are already numbered. Choose Numbered list instead, so no "
    "number is lost."
)
NUMBER_CHANGED = (
    "A number written in this text was replaced by the list's own number. "
    "Check it still matches the answer key."
)

_ROMAN = (
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
)


def to_roman(n: int) -> str:
    out = []
    for value, letters in _ROMAN:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def from_roman(text: str) -> int | None:
    """The value of a well-formed roman numeral, else None ("civil")."""
    low = text.lower()
    i, total = 0, 0
    for value, letters in _ROMAN:
        while low.startswith(letters, i):
            total += value
            i += len(letters)
    if i != len(low) or not total or to_roman(total) != low:
        return None
    return total


def to_alpha(n: int) -> str:
    """a..z, then aa, ab... (1 is "a")."""
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def style_kind(style: str | None) -> str | None:
    """"bullet", "number", or None for a style that does not exist."""
    if style in BULLET_STYLES:
        return "bullet"
    if style in NUMBER_STYLES:
        return "number"
    return None


def level_style(kind: str, style: str | None, level: int) -> str:
    """The style a list's paragraph at `level` is drawn in."""
    style = style or DEFAULT_STYLE[kind]
    level = max(0, min(LEVELS - 1, level))
    if kind == "bullet":
        return BULLET_LEVELS.get(style, BULLET_LEVELS["disc"])[level]
    paren = style.endswith("-paren")
    base = style[: -len("-paren")] if paren else style
    picked = NUMBER_LEVELS.get(base, NUMBER_LEVELS["decimal"])[level]
    return f"{picked}-paren" if paren else picked


def format_number(style: str, n: int) -> str:
    paren = style.endswith("-paren")
    base = style[: -len("-paren")] if paren else style
    if base == "lower-alpha":
        body = to_alpha(n)
    elif base == "upper-alpha":
        body = to_alpha(n).upper()
    elif base == "lower-roman":
        body = to_roman(n)
    elif base == "upper-roman":
        body = to_roman(n).upper()
    else:
        body = str(n)
    return body + (")" if paren else ".")


def paragraphs_of(text: str | None) -> list[str]:
    """A box's paragraphs: its text, trimmed, split at each line break.
    The same split the drawing makes, so levels index the same lines."""
    return (text or "").strip().split("\n")


def effective_levels(indents: list[int] | None, count: int) -> list[int]:
    """Each paragraph's level. Levels follow the paragraph's index; extra
    paragraphs (a translation that came back with more lines) take the
    last level given, and no levels at all is level 0 everywhere."""
    if not indents:
        return [0] * count
    out = []
    for i in range(count):
        level = indents[i] if i < len(indents) else indents[-1]
        out.append(max(0, min(LEVELS - 1, int(level))))
    return out


def normalise_levels(levels: list[int] | None) -> list[int] | None:
    """Stored form: None when every level is 0."""
    if not levels or not any(levels):
        return None
    return [max(0, min(LEVELS - 1, int(v))) for v in levels]


@dataclass(frozen=True)
class Typed:
    """A marker typed at a paragraph's start."""

    marker: str          # as typed, e.g. "3." or "(a)" or "•"
    rest: str            # the text after it and its space
    kind: str            # "number" or "bullet"
    # Numbers: "decimal", "lower-alpha", "upper-alpha", "lower-roman" or
    # "upper-roman", the delimiter and the value. Bullets: the style id.
    cls: str
    delim: str = ""
    value: int = 0


def typed_marker(paragraph: str) -> Typed | None:
    """The marker typed at the start of a paragraph, if any. A lone i, v
    or x reads as a letter here; check_list reads a whole sequence of
    them as roman numerals when only that makes sense of it."""
    m = TYPED_MARKER.match(paragraph)
    if m is None:
        return None
    marker = m.group(1)
    rest = paragraph[m.end():]
    token = m.group(2)
    if token is None:
        return Typed(marker=marker, rest=rest, kind="bullet", cls=_TYPED_BULLETS[marker])
    delim = marker[-1]
    if token.isdigit():
        return Typed(marker, rest, "number", "decimal", delim, int(token))
    if len(token) == 1:
        cls = "lower-alpha" if token.islower() else "upper-alpha"
        return Typed(marker, rest, "number", cls, delim, ord(token.lower()) - ord("a") + 1)
    value = from_roman(token)
    if value is None:
        return None
    cls = "lower-roman" if token.islower() else "upper-roman"
    return Typed(marker, rest, "number", cls, delim, value)


def _weak(t: Typed) -> bool:
    """A marker that ordinary prose can also start with: an initial
    ("A. Lincoln", "J. K. Rowling") or a dialogue dash ("- Hola, dijo").
    Brackets ("a)", "(a)"), digits and real bullet characters are not."""
    if t.kind == "bullet":
        return t.marker in ("-", "*")
    return t.delim == "." and not t.marker.startswith("(") and len(t.marker) == 2 and t.marker[0].isalpha()


def _family(t: Typed) -> tuple:
    """Markers that make a list together: bullets with bullets, letters
    of one case and delimiter with each other ("i." beside "ii.")."""
    if t.kind == "bullet":
        return ("bullet",)
    letter = t.marker.strip("(").rstrip(".)")
    case = "digit" if letter.isdigit() else "lower" if letter.islower() else "upper"
    return ("number", case, t.delim)


def typed_markers(paras: list[str], first_hang: float | None = None) -> list[Typed | None]:
    """The typed marker of each paragraph, as lists and the drawing read
    them. A weak marker (see _weak) only counts when another paragraph of
    the box has a marker of its family, or when the source itself set the
    first paragraph's words apart from it (`first_hang`); alone, it is the
    start of a sentence and stays in the words."""
    found = [typed_marker(p.strip()) if p.strip() else None for p in paras]
    families: dict[tuple, int] = {}
    for t in found:
        if t is not None:
            families[_family(t)] = families.get(_family(t), 0) + 1
    out: list[Typed | None] = []
    for i, t in enumerate(found):
        if t is not None and _weak(t) and families[_family(t)] < 2 and not (i == 0 and first_hang):
            t = None
        out.append(t)
    return out


def _as_roman(t: Typed) -> Typed | None:
    """A single-letter marker read as a roman numeral (i, v, x...)."""
    if t.cls not in ("lower-alpha", "upper-alpha"):
        return None
    letter = t.marker.strip("().")
    value = from_roman(letter)
    if value is None:
        return None
    cls = "lower-roman" if t.cls == "lower-alpha" else "upper-roman"
    return Typed(t.marker, t.rest, "number", cls, t.delim, value)


def _typed_matches(t: Typed, style: str, n: int) -> bool:
    """Whether a typed number says what the list's own marker says."""
    base = style[: -len("-paren")] if style.endswith("-paren") else style
    if base.endswith("roman") and t.cls.endswith("alpha"):
        t = _as_roman(t) or t
    return t.cls == base and t.value == n


def _consecutive(items: list[Typed]) -> bool:
    first = items[0]
    return all(
        t.cls == first.cls and t.delim == first.delim and t.value == first.value + i
        for i, t in enumerate(items)
    )


@dataclass(frozen=True)
class ListCheck:
    ok: bool
    style: str | None = None
    start: int | None = None
    message: str | None = None


def check_list(text: str | None, kind: str, first_hang: float | None = None) -> ListCheck:
    """Whether a list of `kind` may be turned on over this text, and the
    style and start its own markers suggest. One predicate for the edit
    and for what the editor is told (own_markers), so the editor knows a
    refusal before it asks. `first_hang` is the source's own hang
    (source_hang), which makes a lone initial-like marker count."""
    all_paras = paragraphs_of(text)
    typed = typed_markers(all_paras, first_hang)
    paras = [p for p in all_paras if p.strip()]
    found = [t for t in typed if t is not None]
    if not found:
        return ListCheck(ok=True)
    numbers = [t for t in found if t.kind == "number"]
    if kind == "bullet":
        if numbers:
            return ListCheck(ok=False, message=REFUSE_BULLETS)
        return ListCheck(ok=True, style=found[0].cls)
    if not numbers:
        # Bullets only: numbering takes their place.
        return ListCheck(ok=True)
    if len(numbers) == len(paras):
        candidates = [numbers]
        # "i." alone reads as a letter; beside "ii." and "iii." it is a
        # roman numeral, so the sequence is also read that way.
        romans = [t if t.cls.endswith("roman") else _as_roman(t) for t in numbers]
        if all(romans):
            candidates.append(romans)
        # When both readings count on ("I." alone is the 9th letter or the
        # first roman numeral; "C." the 3rd letter or a hundred), the one
        # that starts lower is meant.
        fits = sorted(
            (items for items in candidates if _consecutive(items)),
            key=lambda items: items[0].value,
        )
        if fits:
            first = fits[0][0]
            style = first.cls + ("-paren" if first.delim == ")" else "")
            return ListCheck(
                ok=True, style=style,
                start=first.value if first.value != 1 else None,
            )
    return ListCheck(ok=False, message=REFUSE_NUMBERS)


def own_markers(text: str | None, first_hang: float | None = None) -> dict:
    """What the editor needs to know about the markers a text typed, with
    the style and start a list turned on over them would take."""
    found = [t for t in typed_markers(paragraphs_of(text), first_hang) if t is not None]
    kinds = {t.kind for t in found}
    kind = (
        "none" if not found
        else "numbers" if kinds == {"number"}
        else "bullets" if kinds == {"bullet"}
        else "mixed"
    )
    number = check_list(text, "number", first_hang)
    bullet = check_list(text, "bullet", first_hang)
    return {
        "kind": kind,
        "number_ok": number.ok,
        "bullet_ok": bullet.ok,
        "message": number.message or bullet.message,
        "number_message": number.message,
        "bullet_message": bullet.message,
        "number_style": number.style,
        "number_start": number.start,
        "bullet_style": bullet.style,
    }


@dataclass
class ParaSpec:
    """How one paragraph is drawn."""

    # The words, without any marker.
    body: str
    # The marker in logical order ("3."), or None.
    marker: str | None
    # Which font draws the marker: "text" (the words' own), "marks" (Noto
    # Sans Symbols 2) or "sans" (the bundled Noto Sans). None without one.
    face: str | None
    level: int
    # True when the marker was typed in the text itself.
    typed: bool = False
    # The source's own marker-to-text distance in points (paragraph 0 of
    # an engine-found block only). For a list's own marker, set when it
    # took the place of the source's typed one, and the least hang of its
    # level.
    source_hang: float | None = None
    # A typed number replaced by a list's own number: what was typed.
    typed_value: int | None = None
    replaced: bool = False


def source_hang(obj) -> float | None:
    """How far the source's first words start after its marker, in points.

    Only for a block the engine found: when its first printed row is a
    marker on its own ("3.") with the words beginning further right on the
    same row. Corpus 01's questions give 93.6 - 72.0 = 21.6.
    """
    from ..model.enums import TextSource

    if obj is None or getattr(obj, "source", None) is TextSource.MANUAL:
        return None
    lines = [ln for ln in getattr(obj, "lines", []) or [] if ln.text.strip()]
    if not lines:
        return None
    first = min(lines, key=lambda ln: (ln.bbox.y0, ln.bbox.x0))
    mid = (first.bbox.y0 + first.bbox.y1) / 2
    row = sorted(
        (ln for ln in lines if ln.bbox.y0 - 0.5 <= mid <= ln.bbox.y1 + 0.5),
        key=lambda ln: ln.bbox.x0,
    )
    hang = None
    if len(row) >= 2 and _WHOLE_MARKER.match(row[0].text.strip()):
        hang = row[1].bbox.x0 - row[0].bbox.x0
    else:
        spans = [s for s in row[0].spans if s.text.strip()]
        if len(spans) >= 2 and _WHOLE_MARKER.match(spans[0].text.strip()):
            hang = spans[1].bbox.x0 - spans[0].bbox.x0
    size = (getattr(obj, "style", None) and obj.style.font_size) or 10.0
    if hang is None or hang <= 0 or hang > MAX_SOURCE_HANG_SIZES * size:
        return None
    return round(hang, 3)


# Bullets drawn in the marks face whatever the text face has, so the
# browser can predict the face from the glyph alone.
_MARKS_ONLY = {"◦", "▪", "■", "✓"}


def marker_face(marker: str, kind: str, has_glyph: Callable[[str], bool] | None) -> str:
    """Which face draws a managed marker (see ParaSpec.face).

    • and – use the text face when it has them (else • the marks face and
    – Noto Sans); ◦ ■ ✓ always the marks face; numbers the text face when
    it has every character, else Noto Sans. Without a text font to ask,
    the text face is assumed to have what it usually has.
    """
    if kind == "bullet":
        if marker in _MARKS_ONLY:
            return "marks"
        if has_glyph is None or has_glyph(marker):
            return "text"
        return "marks" if marker == "•" else "sans"
    if has_glyph is None or all(has_glyph(c) for c in marker):
        return "text"
    return "sans"


def _glyph_test(font) -> Callable[[str], bool] | None:
    if font is None:
        return None

    def has(ch: str) -> bool:
        try:
            return bool(font.has_glyph(ord(ch)))
        except Exception:
            return False

    return has


def plan_paragraphs(text: str | None, obj, font=None) -> list[ParaSpec]:
    """Each paragraph's body, marker, level and face, as the box is drawn.

    `obj` supplies the list fields (manual_list, manual_list_style,
    manual_list_start, manual_indents) and, for the source's own hang, its
    printed lines. `font` is the text's fitz.Font, to pick marker faces;
    None predicts them.
    """
    paras = paragraphs_of(text)
    kind = getattr(obj, "manual_list", None)
    style = getattr(obj, "manual_list_style", None)
    start = getattr(obj, "manual_list_start", None) or 1
    levels = effective_levels(getattr(obj, "manual_indents", None), len(paras))
    has = _glyph_test(font)
    counters = [0] * LEVELS
    counters[0] = start - 1
    hang0 = source_hang(obj)
    all_typed = typed_markers(paras, hang0)
    out: list[ParaSpec] = []
    for i, (para, level) in enumerate(zip(paras, levels)):
        stripped = para.strip()
        if not stripped:
            out.append(ParaSpec(body="", marker=None, face=None, level=level))
            continue
        typed = all_typed[i]
        if kind is None:
            if typed is None:
                out.append(ParaSpec(body=stripped, marker=None, face=None, level=level))
            else:
                out.append(ParaSpec(
                    body=typed.rest, marker=typed.marker, face="text", level=level,
                    typed=True, source_hang=hang0 if i == 0 else None,
                ))
            continue
        body = stripped
        typed_value = None
        kept_hang = None
        if typed is not None and (typed.kind == "bullet" or kind == "number"):
            body = typed.rest
            if typed.kind == "number":
                typed_value = typed.value
            # The list takes over the source's own marker: its words stay
            # where the source put them (in line with the worksheet's
            # other questions and its answer lines).
            kept_hang = hang0 if i == 0 else None
        counters[level] += 1
        for deeper in range(level + 1, LEVELS):
            counters[deeper] = 0
        lstyle = level_style(kind, style, level)
        if kind == "bullet":
            marker = BULLET_GLYPHS[lstyle]
        else:
            marker = format_number(lstyle, counters[level])
        replaced = (
            kind == "number" and typed is not None and typed.kind == "number"
            and not _typed_matches(typed, lstyle, counters[level])
        )
        out.append(ParaSpec(
            body=body, marker=marker, face=marker_face(marker, kind, has),
            level=level, source_hang=kept_hang, typed_value=typed_value,
            replaced=replaced,
        ))
    return out


def has_layout(paras: list[ParaSpec]) -> bool:
    """Whether any paragraph has a marker or an indent; without either the
    box wraps exactly as it always has."""
    return any(p.marker or p.level for p in paras)


@dataclass
class MarkerFonts:
    """The fonts a box's markers are measured and drawn in, loaded only
    when a marker needs them. A face whose file is missing stands in with
    the text face, so a marker is never a missing-glyph box."""

    text: ResolvedFont
    bold: bool = False
    italic: bool = False
    _cache: dict[str, ResolvedFont] = field(default_factory=dict)

    def resolved(self, face: str | None) -> ResolvedFont:
        if face in (None, "text"):
            return self.text
        if face not in self._cache:
            path: Path | None = (
                catalog.marks_face() if face == "marks" else catalog.noto_sans(self.bold, self.italic)
            )
            found = None
            if path is not None and REGISTRY._load(path=path) is not None:
                found = ResolvedFont(key=str(path), path=path, display_name=path.stem)
            self._cache[face] = found or self.text
        return self._cache[face]

    def font(self, face: str | None):
        return REGISTRY.font_for(self.resolved(face))

    def measure(self, face: str | None, text: str, size: float) -> float:
        return REGISTRY.measure(self.resolved(face), text, size)


def drawable_marker(p: ParaSpec, fonts: MarkerFonts) -> tuple[str, str]:
    """(marker, face) to draw, after any stand-in: a marks face that could
    not be loaded draws a bullet in the text face, as "•" when the text
    face has it, else "-"."""
    marker, face = p.marker or "", p.face or "text"
    if face == "text" or fonts.resolved(face) is not fonts.text:
        return marker, face
    if p.marker and not p.marker[0].isalnum():
        has = _glyph_test(fonts.font("text"))
        marker = "•" if has is None or has("•") else "-"
    return marker, "text"


@dataclass(frozen=True)
class Offsets:
    marker_offset: float
    text_offset: float
    hang: float


def offsets(
    paras: list[ParaSpec],
    fonts: MarkerFonts,
    size: float,
    width: float,
    visual: Callable[[str], str] | None = None,
) -> list[Offsets]:
    """Where each paragraph's marker and text start, in points from the
    box's start edge, at `size` in a box `width` wide.

    A list's markers share one hang per level, so the words line up from
    item to item; a typed marker hangs at least as far as the source hung
    it. However deep, the words keep MIN_COLUMN_EM of width.
    """
    show = visual or (lambda s: s)
    widest: dict[int, float] = {}
    least: dict[int, float] = {}
    for p in paras:
        if p.marker and not p.typed:
            marker, face = drawable_marker(p, fonts)
            w = fonts.measure(face, show(marker), size)
            widest[p.level] = max(widest.get(p.level, 0.0), w)
            if p.source_hang:
                least[p.level] = max(least.get(p.level, 0.0), p.source_hang)
    out: list[Offsets] = []
    for p in paras:
        start = p.level * LEVEL_STEP_EM * size
        if not p.marker:
            hang = 0.0
        elif p.typed:
            hang = max(
                p.source_hang or 0.0,
                fonts.measure("text", show(p.marker), size) + fonts.measure("text", " ", size),
            )
        else:
            hang = max(
                MIN_HANG_EM * size,
                widest.get(p.level, 0.0) + MARKER_GAP_EM * size,
                least.get(p.level, 0.0),
            )
        text_at = start + hang
        floor = MIN_COLUMN_EM * size
        if width - text_at < floor:
            text_at = max(0.0, width - floor)
            hang = min(hang, text_at)
            start = max(0.0, text_at - hang)
        out.append(Offsets(marker_offset=start, text_offset=text_at, hang=hang))
    return out


def paragraphs_out(paras: list[ParaSpec]) -> list[dict[str, Any]]:
    """The plan as the editor reads it (_block_out's "paragraphs")."""
    return [
        {
            "marker": p.marker, "face": p.face, "level": p.level,
            "typed": p.typed, "source_hang": p.source_hang,
        }
        for p in paras
    ]


def marker_visual(marker: str, rtl: bool) -> str:
    """A marker in drawing order: right to left, "1." shows as ".1"."""
    if not rtl or not marker:
        return marker
    from ..fonts.shaping import _bidi

    get_display = _bidi()
    if get_display is None:
        return marker
    try:
        return get_display(marker, base_dir="R")
    except Exception:
        return marker
