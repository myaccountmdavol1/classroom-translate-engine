"""Whether a PDF's text was set in a serif face.

The PDF says so in a flag on each font, and the flag can be wrong: Canva
marked every font of a slide serif, Tuffy and Futura (sans faces)
included, and the translation of words in a face the engine could not
match was drawn in Times. An embedded font says what it is itself, in its
OS/2 table (its family class, or its PANOSE serif style), so that is
believed first, then a known face's name, then the flag -- unless the same
PDF was caught marking a sans face serif, when an unknown one is sans.
"""
from __future__ import annotations

import io
import re

# OS/2 sFamilyClass, high byte: 1-5 and 7 are serif families, 8 sans.
_SERIF_CLASSES = {1, 2, 3, 4, 5, 7}
_SANS_CLASS = 8
# PANOSE (Latin text, family type 2) serif style: 2-10 have serifs, 11-15
# are sans (normal, obtuse, perpendicular), flared and rounded.
_LATIN_TEXT = 2
_SANS_STYLES = range(11, 16)
_SERIF_STYLES = range(2, 11)

_SANS_NAMES = re.compile(
    r"sans|arial|helvetica|futura|gothic|grotesk|verdana|tahoma|calibri|segoe|roboto|"
    r"lato|montserrat|poppins|inter|comic|lexend|nunito|quicksand|avenir|gill|trebuchet|"
    r"franklin|myriad|frutiger|univers|raleway|tuffy|andika|aptos|carlito",
    re.I,
)
_SERIF_NAMES = re.compile(
    r"serif|times|georgia|garamond|baskerville|cambria|palatino|bookman|antiqua|caslon|"
    r"bodoni|didot|minion|merriweather|lora|playfair|schoolbook|caladea|constantia",
    re.I,
)


def _face_is_serif(data: bytes) -> bool | None:
    from fontTools.ttLib import TTFont

    try:
        font = TTFont(io.BytesIO(data), lazy=True)
        os2 = font["OS/2"] if "OS/2" in font else None
    except Exception:
        return None
    if os2 is None:
        return None
    family_class = os2.sFamilyClass >> 8
    if family_class in _SERIF_CLASSES:
        return True
    if family_class == _SANS_CLASS:
        return False
    panose = getattr(os2, "panose", None)
    if panose is None or panose.bFamilyType != _LATIN_TEXT:
        return None
    if panose.bSerifStyle in _SANS_STYLES:
        return False
    if panose.bSerifStyle in _SERIF_STYLES:
        return True
    return None


def _bare(name: str) -> str:
    """A font's name without its subset prefix ("ABCDEF+Tuffy-Bold")."""
    return name.split("+", 1)[1] if "+" in name[:7] else name


def key(name: str | None) -> str:
    """A font's name to match by: the page's font list says "Noto Sans
    Regular" where its spans say "NotoSans-Regular"."""
    return re.sub(r"[\s\-_]", "", _bare(name or "")).lower()


def own_looks(doc, page, cache: dict[int, tuple[str, bool | None]] | None = None) -> dict[str, bool | None]:
    """What each font embedded on the page says it is, by its key(): True
    serif, False sans, None unsaid or not embedded. `cache` keeps a
    document's fonts read once, by xref."""
    cache = {} if cache is None else cache
    out: dict[str, bool | None] = {}
    for entry in page.get_fonts(full=True):
        xref = entry[0]
        if xref not in cache:
            look = None
            try:
                _name, ext, _kind, data = doc.extract_font(xref)
                if data and ext in ("ttf", "otf", "cff", "ttc"):
                    look = _face_is_serif(data)
            except Exception:
                look = None
            cache[xref] = (key(entry[3]), look)
        name, look = cache[xref]
        if out.get(name) is None:
            out[name] = look
    return out


def _google_category(name: str) -> bool | None:
    """A Google family's category, as its index lists it: serif, sans, or
    None for display, handwriting and anything not in it."""
    from ..fonts import google

    found = google.match(name)
    if found is None:
        return None
    category = google._index()["families"][found.key][2]
    return {"r": True, "s": False}.get(category)


def serif(name: str | None, flagged: bool, own: bool | None, flags_lie: bool) -> bool:
    """Whether to treat a span's face as serif."""
    if own is not None:
        return own
    listed = _google_category(name or "")
    if listed is not None:
        return listed
    bare = _bare(name or "")
    if _SANS_NAMES.search(bare):
        return False
    if _SERIF_NAMES.search(bare):
        return True
    return flagged and not flags_lie
