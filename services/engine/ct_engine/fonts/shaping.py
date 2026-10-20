"""Complex-script shaping for right-to-left languages (spec sec. 27).

PyMuPDF's TextWriter reverses character order for RTL but does not apply
OpenType contextual shaping, so Arabic renders as disconnected isolated
letterforms -- legible to nobody. Urdu and Persian have the same problem.

The pipeline is therefore:

    logical text
      -> arabic_reshaper   (contextual forms: initial/medial/final/isolated)
      -> wrap to lines     (measured on the SHAPED text, whose widths differ)
      -> bidi per line     (visual order)
      -> draw LTR          (the string is already in visual order)

Note the last step: once bidi has reordered a line, it must be drawn
left-to-right. Passing right_to_left=True as well would reverse it twice.

Both libraries are optional. Without them, RTL text is still drawn, but the
caller is told shaping is unavailable so the page can be flagged rather than
quietly shipping unreadable Arabic.
"""

from __future__ import annotations

from functools import lru_cache

from ..util.logging import get_logger

log = get_logger(__name__)

# Scripts that need contextual shaping before they can be drawn.
SHAPED_SCRIPTS = frozenset({"Arabic"})
# Scripts that need bidi reordering (shaping not required).
BIDI_SCRIPTS = frozenset({"Arabic", "Hebrew"})


@lru_cache(maxsize=1)
def _reshaper():
    try:
        import arabic_reshaper

        return arabic_reshaper
    except ImportError:
        log.warning(
            "arabic_reshaper is not installed; Arabic and Urdu will render "
            "with disconnected letterforms"
        )
        return None


@lru_cache(maxsize=1)
def _bidi():
    # python-bidi moved get_display to the package root in 0.6.
    try:
        from bidi import get_display

        return get_display
    except ImportError:
        pass
    try:
        from bidi.algorithm import get_display

        return get_display
    except ImportError:
        log.warning("python-bidi is not installed; RTL text order will be wrong")
        return None


def shaping_available(script: str) -> bool:
    if script in SHAPED_SCRIPTS and _reshaper() is None:
        return False
    if script in BIDI_SCRIPTS and _bidi() is None:
        return False
    return True


def shape_logical(text: str, script: str) -> str:
    """Apply contextual letterforms, keeping logical character order."""
    if script not in SHAPED_SCRIPTS:
        return text
    reshaper = _reshaper()
    if reshaper is None:
        return text
    try:
        return reshaper.reshape(text)
    except Exception as exc:
        log.warning("Arabic reshaping failed: %r", exc)
        return text


def to_visual(line: str, script: str) -> str:
    """Reorder one already-shaped line into visual order for drawing."""
    if script not in BIDI_SCRIPTS:
        return line
    get_display = _bidi()
    if get_display is None:
        return line
    try:
        return get_display(line)
    except Exception as exc:
        log.warning("Bidi reordering failed: %r", exc)
        return line
