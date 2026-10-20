"""Fitting translated text into the space the original text occupied
(spec sec. 25).

Different languages need different amounts of room: Spanish and French run
~25% longer than English, Chinese and Japanese considerably shorter. The
layout engine must absorb that without covering anything that matters.

Order of concessions, least damaging first:
  1. Keep the original typography and re-wrap.
  2. Tighten line spacing, within a floor.
  3. Reduce the font size, down to a floor.
  4. Grow the box into verifiably empty space below it.
  5. Give up: report that it does not fit, so the page is flagged for review
     rather than silently clipped or overlapped.

Nothing here is allowed to grow a box over a response area, an image, or
another text block -- that check lives in the caller, which knows the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..fonts.registry import REGISTRY, ResolvedFont
from ..model.geometry import Rect

if TYPE_CHECKING:
    from .lists import MarkerFonts, ParaSpec

# Typography floors. Below roughly 6pt a worksheet stops being readable by a
# student, which defeats the purpose of translating it.
MIN_FONT_SIZE = 6.0
# Never shrink below this fraction of the source size without flagging.
SOFT_SHRINK_FLOOR = 0.80
HARD_SHRINK_FLOOR = 0.62
# Line spacing as a multiple of font size.
DEFAULT_LEADING = 1.18
MIN_LEADING = 1.02
# Line spacing she can choose in the editor, as a multiple of the font size
# (Google Slides' Single, 1.15, 1.5 and Double). Without a choice the
# automatic leading above is used, tightened when space is short.
LINE_SPACING_PRESETS = (1.0, 1.15, 1.5, 2.0)
# Step used when searching for a size that fits.
SIZE_STEP = 0.25
# Space inside the box she can choose, in points (Small, Medium, Large;
# none is not stored). The words lay out inside the box shrunk by it on
# every side; her background and border keep the whole box.
PADDING_PRESETS = (2.0, 4.0, 8.0)
# Border weights she can choose, in points, as Google Slides offers them.
BORDER_WEIGHTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
# The inside space never leaves less room than this for the words: a short
# box with Large space keeps a sliver rather than turning inside out. The
# editor clamps the same way, so its stand-in words sit where these do.
MIN_INNER = 4.0


def inset(box: Rect, padding: float | None) -> tuple[Rect, float, float]:
    """(inner box, px, py): `box` shrunk by `padding` on each side, by no
    more than leaves MIN_INNER points across and down."""
    p = max(0.0, float(padding or 0.0))
    px = min(p, max(0.0, (box.width - MIN_INNER) / 2.0))
    py = min(p, max(0.0, (box.height - MIN_INNER) / 2.0))
    if not px and not py:
        return box, 0.0, 0.0
    return Rect(x0=box.x0 + px, y0=box.y0 + py, x1=box.x1 - px, y1=box.y1 - py), px, py


# Scripts written without spaces between words (Thai, Lao, Myanmar, Khmer,
# Chinese, Japanese) or spread letter by letter when justified (Korean is
# spaced, but browsers justify Hangul between syllables). Spreading the gaps
# between words, which is what justify does here, cannot match that, so a
# justified block in one of these scripts is drawn from its start instead.
NO_SPACE_SCRIPTS = re.compile(
    "[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff"
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)


def justifiable(text: str) -> bool:
    """Whether a justified block in `text` is spread to the box's width.

    One predicate for the drawing and for the editor (sent as `justifiable`),
    so the browser makes the same choice without its own copy of the list.
    """
    return not NO_SPACE_SCRIPTS.search(text)


@dataclass
class RowSpec:
    """One wrapped row and where it sits in its paragraph's column.

    `indent` is the text's start in points from the box's start edge (left,
    or right for right-to-left text), `hang` the marker's start to the
    text's, and `marker`/`face` the marker drawn on a paragraph's first
    row (see reconstruct/lists.py).
    """

    text: str
    end: bool
    para: int = 0
    first: bool = True
    indent: float = 0.0
    hang: float = 0.0
    marker: str | None = None
    face: str | None = None


def wrap_rows(
    paras: "list[ParaSpec] | None",
    font: ResolvedFont,
    size: float,
    width: float,
    mfonts: "MarkerFonts | None" = None,
    *,
    text: str | None = None,
    visual=None,
) -> list[RowSpec]:
    """Every row of a box, each paragraph wrapped in its own column: the
    box's width less its text offset, so wrapped lines sit under the words
    and not under the marker.

    With no paragraph plan, or one with no marker and no indent anywhere,
    the rows are exactly wrap_with_ends' over `text`, so a box without a
    list wraps as it always has.
    """
    from .lists import has_layout, offsets

    if not paras or mfonts is None or not has_layout(paras):
        joined = text if text is not None else "\n".join(p.body for p in paras or [])
        lines, ends = wrap_with_ends(joined, font, size, width)
        return [RowSpec(text=t, end=e) for t, e in zip(lines, ends)]
    out: list[RowSpec] = []
    for i, (p, at) in enumerate(zip(paras, offsets(paras, mfonts, size, width, visual))):
        if not p.body.strip():
            out.append(RowSpec(text="", end=True, para=i))
            continue
        column = max(1.0, width - at.text_offset)
        lines, ends = wrap_with_ends(p.body, font, size, column)
        for j, (t, e) in enumerate(zip(lines, ends)):
            out.append(RowSpec(
                text=t, end=e, para=i, first=j == 0,
                indent=at.text_offset, hang=at.hang if j == 0 and p.marker else 0.0,
                marker=p.marker if j == 0 else None, face=p.face if j == 0 else None,
            ))
    return out


@dataclass
class FitResult:
    lines: list[str]
    font_size: float
    leading: float          # baseline-to-baseline distance, in points
    used_height: float
    fits: bool
    # Set when the box had to grow downward to accommodate the text.
    extra_height: float = 0.0
    shrink_ratio: float = 1.0
    overflowed_by: float = 0.0
    notes: list[str] = field(default_factory=list)
    # The rows with their columns and markers (lines are their texts).
    rows: list[RowSpec] = field(default_factory=list)


def _wrap(text, paras, font, size, width, mfonts, visual=None) -> list[RowSpec]:
    return wrap_rows(paras, font, size, width, mfonts, text=text, visual=visual)


def _keeps_words_whole(
    text: str, font: ResolvedFont, size: float, width: float,
    rows: list[RowSpec] | None = None, paras=None,
) -> bool:
    """No Latin word is wider than the box -- or than its own paragraph's
    column, beside a marker or an indent -- so none is broken mid-word."""
    if rows and paras and any(r.indent for r in rows):
        column = {r.para: width - r.indent for r in rows}
        return all(
            _keeps_words_whole(p.body, font, size, column.get(i, width))
            for i, p in enumerate(paras)
        )
    for word in text.split():
        if all(ord(ch) < 0x2E80 for ch in word) and REGISTRY.measure(font, word, size) > width:
            return False
    return True


def fit_to_box(
    text: str,
    rect: Rect,
    font: ResolvedFont,
    *,
    max_size: float = 96.0,
    leading: float = DEFAULT_LEADING,
    paras=None,
    mfonts=None,
    visual=None,
) -> FitResult:
    """The largest size at which `text` fills `rect` without overflowing.

    For a box a teacher drew or chose to fit: she sizes the box, the words
    follow. Guessing a point size for a small starburst ("HW" -> 家庭作业) took
    several tries and still wrapped one character per line.

    `leading` is the line spacing as a multiple of the size: her chosen
    spacing, or the default. `paras` and `mfonts` are the box's paragraph
    plan and marker fonts (reconstruct/lists.py): each attempt then wraps
    every paragraph in its own column, which scales with the size.
    """
    width, height = max(1.0, rect.width), max(1.0, rect.height)

    def attempt(size: float) -> FitResult | None:
        rows = _wrap(text, paras, font, size, width, mfonts, visual)
        used = size * leading * len(rows)
        if used > height + 1e-6 or not _keeps_words_whole(
            text, font, size, width, rows, paras
        ):
            return None
        return FitResult(
            lines=[r.text for r in rows], font_size=round(size, 2),
            leading=round(size * leading, 2), used_height=round(used, 2),
            fits=True, notes=["sized to fit its box"], rows=rows,
        )

    lo, hi = MIN_FONT_SIZE, max(MIN_FONT_SIZE, min(max_size, height / leading))
    best = attempt(lo)
    if best is None:
        return fit_text(
            text, rect, font, base_size=lo,
            leading=None if leading == DEFAULT_LEADING else leading,
            paras=paras, mfonts=mfonts, visual=visual,
        )
    # Wrapping makes this only nearly monotonic; a bisection to a quarter
    # point is plenty, and stays fast on a page-sized box.
    while hi - lo > SIZE_STEP:
        mid = (lo + hi) / 2
        found = attempt(mid)
        if found:
            best, lo = found, mid
        else:
            hi = mid
    return best


def wrap_text(
    text: str, font: ResolvedFont, size: float, width: float
) -> list[str]:
    """Greedy word wrap using real glyph advances, not character counts.

    Character-count wrapping is wildly wrong for CJK (wide glyphs) and for
    proportional Latin faces, and produces overflow that only shows up in the
    rendered PDF.
    """
    return wrap_with_ends(text, font, size, width)[0]


def wrap_with_ends(
    text: str, font: ResolvedFont, size: float, width: float
) -> tuple[list[str], list[bool]]:
    """`wrap_text`'s rows, and for each whether it ends its paragraph.

    Justified text spreads every row but the last of each paragraph (the
    row before a line break she typed, and the block's last row), so the
    drawing needs to know which rows those are.
    """
    if width <= 0:
        return [text], [True]
    lines: list[str] = []
    ends: list[bool] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            ends.append(True)
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if REGISTRY.measure(font, candidate, size) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                ends.append(False)
                current = word
        if current:
            lines.append(current)
            ends.append(True)

    # Scripts without spaces (Chinese, Japanese) never wrap on the pass above,
    # so any line still too wide is broken by glyph advance. Only the last
    # piece of a row can end a paragraph.
    out: list[str] = []
    out_ends: list[bool] = []
    for line, end in zip(lines, ends):
        if REGISTRY.measure(font, line, size) <= width or not line:
            out.append(line)
            out_ends.append(end)
            continue
        current = ""
        for ch in line:
            if REGISTRY.measure(font, current + ch, size) > width and current:
                out.append(current)
                out_ends.append(False)
                current = ch
            else:
                current += ch
        if current:
            out.append(current)
            out_ends.append(end)
    return out, out_ends


def fit_text(
    text: str,
    rect: Rect,
    font: ResolvedFont,
    *,
    base_size: float,
    source_leading: float | None = None,
    max_extra_height: float = 0.0,
    leading: float | None = None,
    paras=None,
    mfonts=None,
    visual=None,
) -> FitResult:
    """Find the largest typography that fits `text` into `rect`.

    `max_extra_height` is how far the caller has verified the box may grow
    downward without hitting anything. `leading` is line spacing she chose,
    as a multiple of the size: it is not a starting point to tighten, so
    only the size gives way, as when she locks a size.
    """
    width = max(1.0, rect.width)
    height = max(1.0, rect.height)
    base_size = max(base_size or 10.0, 1.0)

    # Preserve the source's own line spacing where we recovered it.
    leading_ratio = DEFAULT_LEADING
    if source_leading and base_size > 0:
        observed = source_leading / base_size
        if MIN_LEADING <= observed <= 2.2:
            leading_ratio = observed
    ratios = (leading_ratio, max(MIN_LEADING, leading_ratio * 0.92), MIN_LEADING)
    if leading:
        leading_ratio = leading
        ratios = (leading,)

    best: FitResult | None = None
    size = base_size
    # The floor can never exceed the starting size. OCR on a small line box
    # can yield a base size already below MIN_FONT_SIZE, and without this
    # clamp the search loop never ran and the function crashed.
    floor = min(base_size, max(MIN_FONT_SIZE, base_size * HARD_SHRINK_FLOOR))
    while size >= floor - 1e-9:
        for ratio in ratios:
            rows = _wrap(text, paras, font, size, width, mfonts, visual)
            lines = [r.text for r in rows]
            step = size * ratio
            used = step * len(lines)
            if used <= height + 1e-6:
                return FitResult(
                    lines=lines,
                    font_size=round(size, 2),
                    leading=round(step, 2),
                    used_height=round(used, 2),
                    fits=True,
                    shrink_ratio=round(size / base_size, 3),
                    notes=(
                        []
                        if size >= base_size * SOFT_SHRINK_FLOOR
                        else ["font reduced noticeably to fit"]
                    ),
                    rows=rows,
                )
            if best is None or used < best.used_height:
                best = FitResult(
                    lines=lines,
                    font_size=round(size, 2),
                    leading=round(step, 2),
                    used_height=round(used, 2),
                    fits=False,
                    shrink_ratio=round(size / base_size, 3),
                    overflowed_by=round(used - height, 2),
                    rows=rows,
                )
        size -= SIZE_STEP

    if best is None:
        # Defensive: the loop above always runs at least once now, but a
        # caller must never receive None. Report a non-fitting result rather
        # than raising, so one odd block cannot fail a whole page.
        rows = _wrap(text, paras, font, size, width, mfonts, visual)
        lines = [r.text for r in rows]
        best = FitResult(
            lines=lines,
            font_size=round(max(size, MIN_FONT_SIZE), 2),
            leading=round(max(size, MIN_FONT_SIZE) * leading_ratio, 2),
            used_height=round(max(size, MIN_FONT_SIZE) * leading_ratio * len(lines), 2),
            fits=False,
            shrink_ratio=1.0,
            overflowed_by=0.0,
            notes=["could not measure a fitting size"],
            rows=rows,
        )

    # Last resort: grow downward into space the caller confirmed is empty.
    if max_extra_height > 0 and best.overflowed_by <= max_extra_height:
        best.fits = True
        best.extra_height = round(best.overflowed_by, 2)
        best.overflowed_by = 0.0
        best.notes.append("text box extended downward into empty space")
    return best


def available_growth(
    rect: Rect,
    page_bounds: Rect,
    obstacles: list[Rect],
    *,
    limit: float = 48.0,
) -> float:
    """How far `rect` may grow downward before hitting anything.

    Obstacles are every other object and every response area on the page.
    Response areas are inviolable, so they are passed in like any other
    obstacle and simply stop the growth (spec sec. 28).
    """
    ceiling = min(page_bounds.y1, rect.y1 + limit)
    for other in obstacles:
        if other.y0 < rect.y1:
            continue  # above or overlapping already; not a downward blocker
        horizontally_clear = other.x1 <= rect.x0 + 0.5 or other.x0 >= rect.x1 - 0.5
        if horizontally_clear:
            continue
        ceiling = min(ceiling, other.y0)
    return max(0.0, round(ceiling - rect.y1, 2))


def available_growth_right(
    rect: Rect,
    page_bounds: Rect,
    obstacles: list[Rect],
    *,
    limit_ratio: float = 1.8,
    absolute_allowance: float = 110.0,
    margin: float = 6.0,
    right_margin: float | None = None,
) -> float:
    """How far `rect` may widen to the right before hitting anything.

    Short blocks (a footer, a heading) frequently need width rather than
    height: "Lab 7" becoming "Laboratorio 7" wraps to three tiny lines if the
    only concession available is shrinking. Widening into verified empty
    space preserves the original type size, which reads better and stays
    closer to the source.

    `right_margin` is where the document's own text stops. Blank margin is
    not empty space to fill: a paragraph widened into it runs to within a
    few points of the paper's edge, where printers clip.
    """
    # A ratio alone is useless for a narrow box: "Lab 7" is 21pt wide, so
    # 1.8x buys 17pt while "Laboratorio 7" needs ~50pt. Allow whichever is
    # larger, a proportional or an absolute allowance -- obstacles and the
    # page margin still bound the result.
    allowance = max(rect.width * limit_ratio, rect.width + absolute_allowance)
    ceiling = min(page_bounds.x1 - margin, rect.x0 + allowance)
    if right_margin is not None:
        ceiling = min(ceiling, right_margin)
    for other in obstacles:
        # Touching counts as in the way: an equation that starts 0.1pt inside
        # the prose box before it was skipped, and the prose grew across it.
        if other.x0 < rect.x1 - 2.0:
            continue  # to the left or genuinely underneath
        vertically_clear = other.y1 <= rect.y0 + 0.5 or other.y0 >= rect.y1 - 0.5
        if vertically_clear:
            continue
        ceiling = min(ceiling, other.x0 - 2.0)
    return max(0.0, round(ceiling - rect.x1, 2))
