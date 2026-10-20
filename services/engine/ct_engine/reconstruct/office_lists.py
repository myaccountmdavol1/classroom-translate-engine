"""Lists and fonts in the Word and PowerPoint files (docx/pptx_builder).

The page editor's lists become real Word numbering and PowerPoint bullets
where a box is exactly one paragraph of the file: one w:p or a:p whose
only words are this box's, holding one line. Anywhere else -- a box of
several lines, or a paragraph shared with other boxes -- each line is
written starting with its marker and a space, so no number is ever lost,
but without the hanging indent. The editor tells her which happens.

A font from the catalog is named on the paragraph's words, never embedded:
a computer without it shows a similar one, and Google Docs and Slides have
all of them. It is named only where it has the language's letters
(word_export decides), since naming Andika over Arabic would ask Word for
letters it does not have.
"""

from __future__ import annotations

from ..fonts import catalog
from ..fonts.registry import REGISTRY
from . import lists

# Word's numFmt for each number style's base.
WORD_FORMATS = {
    "decimal": "decimal", "lower-alpha": "lowerLetter", "upper-alpha": "upperLetter",
    "lower-roman": "lowerRoman", "upper-roman": "upperRoman",
}
# PowerPoint's a:buAutoNum types.
PPTX_AUTONUM = {
    "decimal": "arabicPeriod", "decimal-paren": "arabicParenR",
    "lower-alpha": "alphaLcPeriod", "lower-alpha-paren": "alphaLcParenR",
    "upper-alpha": "alphaUcPeriod", "upper-alpha-paren": "alphaUcParenR",
    "lower-roman": "romanLcPeriod", "lower-roman-paren": "romanLcParenR",
    "upper-roman": "romanUcPeriod", "upper-roman-paren": "romanUcParenR",
}
# Bullets most text fonts lack, named in the face the pages draw them in.
SYMBOL_FONT = "Noto Sans Symbols 2"
SYMBOL_BULLETS = {"◦", "▪", "■", "✓"}


def plan(obj) -> list[lists.ParaSpec] | None:
    """The box's lines as the pages draw them, when she turned a list on."""
    if not getattr(obj, "manual_list", None) or obj.translated_text is None:
        return None
    return lists.plan_paragraphs(obj.translated_text, obj)


def single_item(paras: list[lists.ParaSpec]) -> lists.ParaSpec | None:
    """The box's one line, when it has exactly one: only then can it be a
    real list paragraph in the file."""
    inked = [p for p in paras if p.body.strip()]
    return inked[0] if len(inked) == 1 else None


def marker_text(paras: list[lists.ParaSpec]) -> str:
    """Every line starting with its marker and a space."""
    return "\n".join(f"{p.marker} {p.body}" if p.marker else p.body for p in paras)


def word_level(kind: str, style: str | None, level: int) -> tuple[str, str]:
    """(numFmt, lvlText) for one level of a Word list."""
    lstyle = lists.level_style(kind, style, level)
    if kind == "bullet":
        return "bullet", lists.BULLET_GLYPHS[lstyle]
    paren = lstyle.endswith("-paren")
    base = lstyle[: -len("-paren")] if paren else lstyle
    return WORD_FORMATS[base], f"%{level + 1}" + (")" if paren else ".")


def hang_points(item: lists.ParaSpec, size: float) -> tuple[float, float]:
    """(marker offset, hang) in points for a PowerPoint list paragraph at
    `size`, measured as the pages measure them (in Helvetica's widths: the
    slide's own font is not known here)."""
    fonts = lists.MarkerFonts(REGISTRY.resolve("1.", script="Latin"))
    at = lists.offsets([item], fonts, size, 10_000.0)[0]
    return at.marker_offset, at.hang


def font_name(face_id: str | None) -> str | None:
    entry = catalog.face(face_id)
    return entry.name if entry is not None else None
