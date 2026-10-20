"""Recover inline math that a PDF encodes as layout rather than characters.

A PDF has no "superscript 2". Word draws a smaller "2" a little higher, and
draws π as the letter "p" in the Symbol font. Read naively, "x²" becomes "x2"
and "2πr" becomes "2pr": the translator then treats them as words, and the
rebuilt page draws them flat. Fixing it here, at extraction, means protection,
translation, the editor and the rebuild all see the text a reader sees.

Only what has an exact Unicode form is converted. A raised glyph with no
superscript codepoint ("q") is left as-is and marked, so the page can say it
was flattened instead of pretending it was kept.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..model.objects import TextSpan

_SUPER = dict(zip(
    "0123456789+-=()abcdefghijklmnoprstuvwxyzABDEGHIJKLMNOPRTUVW",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ",
))
_SUPER["−"] = "⁻"
_SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
_SUB["−"] = "₋"

# Raised letters after a number are ordinals ("1st", "2ème", "3º"). They are
# words the translator must localise, not exponents.
_ORDINAL = re.compile(r"^(st|nd|rd|th|er|re|e|ème|o|a|º|ª)$", re.IGNORECASE)

# Adobe Symbol encoding: the byte Word writes for the glyph Symbol draws.
_SYMBOL = {
    "a": "α", "b": "β", "c": "χ", "d": "δ", "e": "ε", "f": "φ", "g": "γ",
    "h": "η", "i": "ι", "j": "ϕ", "k": "κ", "l": "λ", "m": "μ", "n": "ν",
    "o": "ο", "p": "π", "q": "θ", "r": "ρ", "s": "σ", "t": "τ", "u": "υ",
    "v": "ϖ", "w": "ω", "x": "ξ", "y": "ψ", "z": "ζ",
    "A": "Α", "B": "Β", "C": "Χ", "D": "Δ", "E": "Ε", "F": "Φ", "G": "Γ",
    "H": "Η", "I": "Ι", "K": "Κ", "L": "Λ", "M": "Μ", "N": "Ν", "O": "Ο",
    "P": "Π", "Q": "Θ", "R": "Ρ", "S": "Σ", "T": "Τ", "U": "Υ", "W": "Ω",
    "X": "Ξ", "Y": "Ψ", "Z": "Ζ",
    "\x22": "∀", "\x24": "∃", "\x27": "∋", "\x2a": "∗", "\x2d": "−",
    "\x40": "≅", "\x5c": "∴", "\x5e": "⊥",
    "\xa1": "ϒ", "\xa2": "′", "\xa3": "≤", "\xa5": "∞", "\xb0": "°",
    "\xb1": "±", "\xb2": "″", "\xb3": "≥", "\xb4": "×", "\xb5": "∝",
    "\xb6": "∂", "\xb7": "•", "\xb8": "÷", "\xb9": "≠", "\xba": "≡",
    "\xbb": "≈", "\xc5": "⊕", "\xc6": "∅", "\xc7": "∩", "\xc8": "∪",
    "\xc9": "⊃", "\xca": "⊇", "\xcb": "⊄", "\xcc": "⊂", "\xcd": "⊆",
    "\xce": "∈", "\xcf": "∉", "\xd0": "∠", "\xd1": "∇", "\xd5": "∏",
    "\xd6": "√", "\xd7": "⋅", "\xd8": "¬", "\xd9": "∧", "\xda": "∨",
    "\xdb": "⇔", "\xdc": "⇐", "\xdd": "⇑", "\xde": "⇒", "\xdf": "⇓",
    "\xe5": "∑", "\xf2": "∫", "\xae": "→", "\xac": "←", "\xad": "↑",
    "\xaf": "↓", "\xab": "↔",
}

# Fonts that only ever draw mathematics. Their text layer is not trusted:
# Word's Cambria Math subsets extract a fraction as '! " + # $'. Symbol is
# not here -- its encoding is known and mapped above.
_MATH_FONT = re.compile(
    r"(cambria\s*-?math|stix.*math|latin\s*-?modern\s*-?math|.*-math$|"
    r"^cm(mi|sy|ex|bsy|mib)\d*|^msam|^msbm|^eufm|^rsfs|mathematicalpi|"
    r"^euclid|mt\s*extra|asana|xits\s*math|tex\s*gyre.*math|newcm.*math|"
    r"^lmmath|^lmsy|^lmmi|^lmex)",
    re.IGNORECASE,
)
MATH_FONT_REASON = "drawn in a math font"


def is_math_font(font: str | None) -> bool:
    if not font:
        return False
    return bool(_MATH_FONT.search(font.split("+")[-1].replace(" ", "")))


# Below this fraction of the line's body size, smaller text is a script.
# Word sets scripts at about 0.64; drop caps and small caps sit far from it.
_SCRIPT_RATIO = 0.8


def _is_symbol_font(font: str | None) -> bool:
    if not font:
        return False
    name = font.split("+")[-1].lower()
    return name.startswith("symbol")


def symbol_to_unicode(text: str, font: str | None) -> str:
    """Map Symbol-font bytes to the characters they draw.

    Word sometimes writes the byte directly ("p") and sometimes in the
    private-use block (U+F070); both mean π. Anything unknown is kept.
    """
    if not _is_symbol_font(font):
        return text
    out = []
    for ch in text:
        if 0xF020 <= ord(ch) <= 0xF0FF:
            ch = chr(ord(ch) - 0xF000)
        out.append(_SYMBOL.get(ch, ch))
    return "".join(out)


def _body(spans: Sequence[TextSpan], origins: Sequence[float]) -> tuple[float, float]:
    """The line's body size and baseline: those of the most characters."""
    weight: dict[tuple[float, float], int] = {}
    for span, y in zip(spans, origins):
        key = (round(span.style.font_size, 1), round(y, 1))
        weight[key] = weight.get(key, 0) + len(span.text.strip())
    return max(weight, key=weight.get) if weight else (0.0, 0.0)


def normalize_line_spans(
    spans: list[TextSpan], *, origins: Sequence[float], flags: Sequence[int]
) -> None:
    """Rewrite one line's spans in place. `origins` are baseline y values."""
    for span in spans:
        span.text = symbol_to_unicode(span.text, span.style.font_name_raw)
    if len(spans) < 2:
        return
    body_size, baseline = _body(spans, origins)
    if body_size <= 0:
        return
    for i, (span, y, flag) in enumerate(zip(spans, origins, flags)):
        size = span.style.font_size
        if not span.text.strip() or size > body_size * _SCRIPT_RATIO:
            continue
        rise = baseline - y
        if flag & 1 or rise > body_size * 0.15:
            script, table = "super", _SUPER
        elif rise < -body_size * 0.05:
            script, table = "sub", _SUB
        else:
            continue
        before = spans[i - 1].text if i else ""
        if script == "super" and before[-1:].isdigit() and _ORDINAL.match(span.text.strip()):
            continue
        span.script = script
        core = span.text.strip()
        if all(ch in table for ch in core):
            span.text = span.text.replace(core, "".join(table[ch] for ch in core))
            span.script_converted = True
