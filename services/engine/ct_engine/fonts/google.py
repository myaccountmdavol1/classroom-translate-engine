"""A worksheet's own font, when it is a Google Font.

Many worksheets are made in Canva or Google Docs, in Google Fonts (Lexend,
Poppins, Roboto Serif). The PDF carries only the letters the English used,
so the translation cannot be drawn in it; it used to be drawn in the
nearest of the eight faces the engine ships. Now the font's own name is
looked up in google_fonts.json (every Google Fonts family at one commit,
scripts/build_google_fonts_index.py) and that one file is fetched -- from
jsDelivr's copy of the commit, not from Google -- once per session.

Most families are now one variable file. MuPDF draws a variable file at
its default weight only, so a bold heading would come out regular: the
weight asked for is cut out of it with fontTools first.

Two Office fonts have Google twins of the same letter widths, so a line
translated from Calibri fits as Calibri did: Carlito and Caladea.

Nothing here can stop a page: no match, a failed download or a font too
large to fetch gives None, and the registry chooses as it always has.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..util.logging import get_logger

log = get_logger(__name__)

INDEX_FILE = Path(__file__).parent / "google_fonts.json"
CDN = "https://cdn.jsdelivr.net/gh/google/fonts@{commit}/{folder}/{file}"
# Larger files (whole CJK families) are not fetched for one worksheet: the
# engine's own faces cover those scripts.
MAX_BYTES = 6_000_000
TIMEOUT = 20.0
# Office fonts with a Google twin of the same letter widths.
TWINS = {"calibri": "carlito", "cambria": "caladea"}
WEIGHTS = {
    "thin": 100, "hairline": 100, "extralight": 200, "ultralight": 200,
    "light": 300, "regular": 400, "normal": 400, "book": 400, "roman": 400,
    "medium": 500, "semibold": 600, "demibold": 600, "bold": 700,
    "extrabold": 800, "ultrabold": 800, "heavy": 900, "black": 900,
}
# Headers httpx adds that a browser's XMLHttpRequest refuses (with a console
# error each) -- as ct_browser.relay strips them.
_FORBIDDEN_HEADERS = ("host", "accept-encoding", "connection", "content-length")
CACHE_DIR = Path(tempfile.gettempdir()) / "ct-google-fonts"

_lock = threading.Lock()
_failed: set[str] = set()


def enabled() -> bool:
    return os.environ.get("CT_GOOGLE_FONTS", "1") != "0"


@lru_cache(maxsize=1)
def _index() -> dict:
    try:
        return json.loads(INDEX_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        log.warning("Google Fonts index missing or unreadable")
        return {"commit": "", "families": {}}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class Match:
    key: str
    name: str
    weight: int
    italic: bool


def match(font_name: str | None, bold: bool = False, italic: bool = False) -> Match | None:
    """The Google family a PDF font name is, with the weight and slant its
    name gives ("ABCDEF+Poppins-SemiBoldItalic"), or None."""
    if not font_name:
        return None
    raw = font_name.split("+")[-1]
    family, _, style = raw.partition("-" if "-" in raw else ",")
    families = _index()["families"]
    key = _key(re.sub(r"(PSMT|MT|PS)$", "", family))
    key = TWINS.get(key, key)
    if key not in families:
        # The weight glued on: "PoppinsBold".
        for word in sorted(WEIGHTS, key=len, reverse=True):
            stem = TWINS.get(key[: -len(word)], key[: -len(word)])
            if key.endswith(word) and stem in families:
                key, style = stem, word + style
                break
        else:
            return None
    style = _key(style)
    slanted = italic or "italic" in style or "oblique" in style
    style = style.replace("italic", "").replace("oblique", "")
    weight = WEIGHTS.get(style, 400)
    if bold and weight < 600:
        weight = 700
    return Match(key, families[key][1], weight, slanted)


def _transport():
    """In the browser, httpx's transport minus the headers it refuses."""
    if sys.platform != "emscripten":
        return None
    import httpx

    class BrowserSafe(httpx.BaseTransport):
        def __init__(self) -> None:
            self._inner = httpx.HTTPTransport()

        def handle_request(self, request):
            for name in _FORBIDDEN_HEADERS:
                request.headers.pop(name, None)
            return self._inner.handle_request(request)

        def close(self) -> None:
            self._inner.close()

    return BrowserSafe()


def fetch(url: str) -> bytes | None:
    """The file at `url`, or None (and not asked for again this session)."""
    import httpx

    if url in _failed:
        return None
    try:
        with httpx.Client(timeout=TIMEOUT, transport=_transport(), follow_redirects=True) as client:
            r = client.get(url)
        r.raise_for_status()
        return r.content
    except Exception as exc:  # noqa: BLE001 - any failure means "not this font"
        log.warning("could not fetch %s: %s", url, exc)
        with _lock:
            _failed.add(url)
        return None


def _instance(data: bytes, axes: list, weight: int, italic: bool) -> bytes:
    """A variable font fixed at `weight` (and italic, by an ital or slnt
    axis), every other axis at its default."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(io.BytesIO(data))
    limits: dict = {tag: None for tag, _, _ in axes}
    for tag, lo, hi in axes:
        if tag == "wght":
            limits[tag] = max(lo, min(hi, weight))
        elif tag == "ital" and italic:
            limits[tag] = hi
        elif tag == "slnt" and italic:
            limits[tag] = lo
    try:
        # Named for the weight ("Lexend Bold"), so the PDF and a Word file
        # made from it say which face it is.
        made = instancer.instantiateVariableFont(font, limits, updateFontNames=True)
    except Exception:  # noqa: BLE001 - no STAT table to name it from
        made = instancer.instantiateVariableFont(TTFont(io.BytesIO(data)), limits)
    out = io.BytesIO()
    made.save(out)
    return out.getvalue()


WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
                600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}


def _cut_name(family: str, weight: int, italic: bool) -> str:
    """"Roboto Serif" at 700, italic: "RobotoSerif-BoldItalic.ttf", as a
    static file of it would be called."""
    style = WEIGHT_NAMES.get(round(weight / 100) * 100, "Regular")
    if italic:
        style = "Italic" if style == "Regular" else style + "Italic"
    return f"{family.replace(' ', '')}-{style}.ttf"


def face_file(m: Match) -> tuple[Path, bool] | None:
    """The file to draw `m` in, fetched and made once, and whether it is the
    weight and slant asked for; None when it cannot be had."""
    index = _index()
    entry = index["families"].get(m.key)
    if entry is None or not index["commit"]:
        return None
    folder, _, _, files, axes = entry
    fixed = [f for f in files if "[" not in f[2]]
    variable = [f for f in files if "[" in f[2]]
    if fixed:
        same_slant = [f for f in fixed if bool(f[1]) == m.italic] or fixed
        weight, slant, name, size = min(same_slant, key=lambda f: (abs(f[0] - m.weight), f[0]))
        exact = weight == m.weight and bool(slant) == m.italic
        cut = None
    elif variable:
        same_slant = [f for f in variable if bool(f[1]) == m.italic] or variable
        _, slant, name, size = same_slant[0]
        tags = {tag for tag, _, _ in axes}
        slanted = bool(slant) or (m.italic and bool(tags & {"ital", "slnt"}))
        wght = next(((lo, hi) for tag, lo, hi in axes if tag == "wght"), None)
        exact = slanted == m.italic and (wght is None or wght[0] <= m.weight <= wght[1])
        cut = m.weight
    else:
        return None
    if size > MAX_BYTES or name in _failed:
        return None
    path = CACHE_DIR / index["commit"][:12] / (_cut_name(m.name, cut, m.italic) if cut else name)
    if path.is_file():
        return path, exact
    data = fetch(CDN.format(commit=index["commit"], folder=folder,
                            file=name.replace("[", "%5B").replace("]", "%5D")))
    if data is None:
        with _lock:
            _failed.add(name)
        return None
    if cut is not None:
        try:
            data = _instance(data, axes, cut, m.italic and not slant)
        except Exception as exc:  # noqa: BLE001 - a font fontTools cannot cut is not used
            log.warning("could not make %s at %s: %s", name, cut, exc)
            with _lock:
                _failed.add(name)
            return None
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(data)
    part.replace(path)
    return path, exact
