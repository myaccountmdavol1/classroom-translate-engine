"""Font resolution with glyph-coverage verification (spec sec. 26).

The hard rule: never render a missing-character box. So a font is only used
after every character of the actual string is confirmed present in it, and if
nothing covers the text the caller is told so and the page is flagged --
rather than shipping a worksheet full of tofu.

Sources, in order:
  1. Fonts bundled with the deployment (``fonts/bundled/``), if present.
  2. PyMuPDF built-ins. Notably ``china-s``/``japan``/``korea`` all resolve to
     Droid Sans Fallback (~50k glyphs), which covers Latin, Cyrillic, Greek
     and all CJK in one face -- so most target languages need nothing else.
  3. Platform system fonts, for the scripts MuPDF does not bundle (Arabic,
     Hebrew, Bengali, Devanagari).

DENYLIST: macOS ships ``LastResort.otf``, which reports a glyph for every
codepoint and draws a labelled box. A naive coverage check picks it for every
script and produces exactly the output this module exists to prevent.
"""

from __future__ import annotations

import platform
import threading
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import fitz

from ..util.logging import get_logger
from . import catalog

log = get_logger(__name__)

BUNDLED_DIR = Path(__file__).parent / "bundled"

# Fonts that claim universal coverage but draw placeholder glyphs.
DENYLIST = {"lastresort", "adobeblank", "notdef"}

# PyMuPDF built-in names, tried in this order after the face matching the
# source's look (see `_builtin_order`). The Base-14 faces cover Latin,
# Greek and Cyrillic; "china-s" (Droid Sans Fallback, which all four CJK
# aliases map to) covers far more but embeds as a 1.7 MB file, so it comes
# last -- only for text the small faces cannot draw.
_BUILTIN_CANDIDATES = ("helv", "tiro", "cour", "china-s")

_SYSTEM_DIRS = {
    "Darwin": [
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ],
    "Linux": [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ],
    "Windows": [Path("C:/Windows/Fonts")],
}

# Preferred faces per script, tried before a broad filesystem scan so we get
# a well-designed face rather than the first thing that happens to match.
_PREFERRED: dict[str, tuple[str, ...]] = {
    "Arabic": ("GeezaPro.ttc", "NotoSansArabic-Regular.ttf", "NotoNaskhArabic-Regular.ttf",
               "Arial Unicode.ttf", "DejaVuSans.ttf", "SFArabic.ttf", "arial.ttf"),
    "Hebrew": ("ArialHB.ttc", "NotoSansHebrew-Regular.ttf", "Arial Unicode.ttf",
               "DejaVuSans.ttf", "SFHebrew.ttf"),
    "Bengali": ("KohinoorBangla.ttc", "NotoSansBengali-Regular.ttf",
                "Bangla Sangam MN.ttc", "Arial Unicode.ttf"),
    "Devanagari": ("Kohinoor.ttc", "NotoSansDevanagari-Regular.ttf",
                   "Devanagari Sangam MN.ttc", "Arial Unicode.ttf"),
}

# Faces with wide symbol coverage (checkmarks, arrows, boxes, shapes). When
# the only characters nothing earlier covered are symbols, these are the
# only files tried: loading every font on the server to look for one emoji
# took the engine past 1.3 GB, and the symbol is left out instead (the
# caller reports it). DejaVu Sans comes first because it also has Latin,
# Greek, Cyrillic, Arabic and Hebrew, so a symbol beside words can be drawn
# with them.
_SYMBOL_FACES = ("DejaVuSans.ttf", "NotoSansSymbols2-Regular.ttf",
                 "NotoSansSymbols-Regular.ttf", "NotoSansMath-Regular.ttf",
                 "Apple Symbols.ttf", "seguisym.ttf", "Arial Unicode.ttf")
# The same faces as name fragments, for picking their bold/italic siblings.
_SYMBOL_FAMILIES = ("dejavusans", "notosanssymbols", "notosansmath",
                    "applesymbols", "seguisym", "arialunicode")


def is_symbol(ch: str) -> bool:
    """Whether `ch` is an emoji, symbol, mark or private-use character --
    something a page can lose without losing words."""
    return (
        unicodedata.category(ch) in ("So", "Sk", "Mn", "Cf", "Co")
        or ord(ch) >= 0x1F000
    )


def stands_alone(ch: str) -> bool:
    """A symbol whose absence never changes a word around it.

    Combining marks are left out (bar the emoji variation selectors): a
    Thai vowel or a Devanagari vowel sign is a mark too, and its letters
    still need a full search for a face that has them.
    """
    if not is_symbol(ch):
        return False
    return unicodedata.category(ch) != "Mn" or 0xFE00 <= ord(ch) <= 0xFE0F


@dataclass(frozen=True)
class ResolvedFont:
    """A font proven to cover the text it was resolved for."""

    key: str
    # Exactly one of these is set. `builtin` names a MuPDF base font;
    # `path` points at a file that must be embedded in the output PDF.
    builtin: str | None = None
    path: Path | None = None
    display_name: str = ""
    # False when a bold or italic face was asked for and none was found, so
    # the caller can flag the page instead of quietly shipping regular text
    # where the source was bold.
    weight_matched: bool = True
    # The catalog face (fonts/catalog.py) the text is drawn in, when she
    # chose one and it has the letters. `fell_back` is True when she chose
    # one that does not, and the automatic face for the script stands in.
    face: str | None = None
    fell_back: bool = False

    @property
    def is_builtin(self) -> bool:
        return self.builtin is not None


class FontResolutionError(RuntimeError):
    """No available font covers the requested text."""

    def __init__(self, message: str, *, missing: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.missing = missing


# Builtin faces MuPDF always has, per look. These are why a weight request
# can usually be honoured even with no font files installed at all, and why
# a Helvetica-Oblique line stays a sans italic and a Times-Italic one a
# serif italic instead of whatever italic file happens to sort first.
_BUILTIN_FACES: dict[str, dict[tuple[bool, bool], str]] = {
    "sans": {(False, False): "helv", (True, False): "hebo",
             (False, True): "heit", (True, True): "hebi"},
    "serif": {(False, False): "tiro", (True, False): "tibo",
              (False, True): "tiit", (True, True): "tibi"},
    "mono": {(False, False): "cour", (True, False): "cobo",
             (False, True): "coit", (True, True): "cobi"},
}
_BUILTIN_WEIGHTS = _BUILTIN_FACES["sans"]

# File-name fragments of faces that are a poor stand-in for an ordinary
# weight: narrower, lighter or decorative cuts, and the CJK collections
# (tens of megabytes to load for text a plain face covers).
_ODD_CUTS = ("condensed", "narrow", "light", "thin", "display", "cjk")


def _builtin_order(look: str) -> tuple[str, ...]:
    """Built-in regular faces to try, the source's look first."""
    own = _BUILTIN_FACES.get(look, _BUILTIN_FACES["sans"])[(False, False)]
    return (own, *(n for n in _BUILTIN_CANDIDATES if n != own))


def _normalized(stem: str) -> str:
    return stem.lower().replace(" ", "").replace("-", "").replace("_", "")

# Filename fragments marking a face's weight or slant, matched
# case-insensitively against a font file's stem and ranked best (index 0)
# to worst. A true "Bold" face must always outrank a heavier relative like
# Black -- they are not the same weight -- and the heavier faces are kept
# only as fallbacks for families with no Bold at all. Likewise a true
# "Italic" must outrank a synthetic "Oblique".
#
# Compounds such as "Semibold", "Demibold", "Extrabold" and "Ultrabold"
# contain the literal substring "bold" and would otherwise tie with a true
# Bold face; they are named explicitly and excluded from tier 0 so they
# rank as the (lighter- or heavier-than-bold) fallbacks they actually are.
_BOLD_TIER0 = ("bold", "-bd")
_BOLD_TIER0_EXCLUDE = ("semibold", "demibold", "extrabold", "ultrabold")
_BOLD_TIER1 = ("black", "heavy", "extrabold", "ultrabold", "semibold", "demibold")
_BOLD_RANKS: tuple[tuple[str, ...], ...] = (_BOLD_TIER0, _BOLD_TIER1)

# Italic has no equivalent substring trap -- "oblique" and "italic" share no
# common substring, and there is no "semi-italic" style in common use -- but
# a true Italic face must still outrank a synthetic Oblique.
_ITALIC_TIER0 = ("italic", "-it")
_ITALIC_TIER1 = ("oblique",)
_ITALIC_RANKS: tuple[tuple[str, ...], ...] = (_ITALIC_TIER0, _ITALIC_TIER1)


def _style_rank(
    stem: str,
    ranks: tuple[tuple[str, ...], ...],
    tier0_exclude: tuple[str, ...] = (),
) -> int | None:
    """The best (lowest) tier `stem` matches in `ranks`, or None for no match.

    `tier0_exclude` is checked before tier 0 only, so a compound like
    "Semibold" -- which contains the substring "bold" -- cannot tie with a
    true "Bold" face; it falls through to whichever later tier actually
    names it.
    """
    low = stem.lower()
    for index, markers in enumerate(ranks):
        if index == 0 and any(bad in low for bad in tier0_exclude):
            continue
        if any(marker in low for marker in markers):
            return index
    return None


# Type styles a teacher can choose, as (name fragments to look for in
# preference order, fragments that rule a file out). Fragments are matched
# against the lower-cased file name with spaces, hyphens and underscores
# removed. Each has server (Noto, DejaVu) and macOS faces, CJK included.
FONT_STYLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "sans": (
        ("notosanscjk", "notosans", "dejavusans", "pingfang", "helvetica", "arial"),
        ("mono", "serif", "condensed", "narrow", "display"),
    ),
    "serif": (
        ("notoserifcjk", "notoserif", "dejavuserif", "songti", "stsong", "times", "georgia", "baskerville"),
        ("display", "condensed"),
    ),
    "mono": (
        ("dejavusansmono", "notosansmono", "menlo", "courier", "monaco"),
        (),
    ),
}


def _can_draw_with(font) -> bool:
    """Whether MuPDF can actually write text in this font."""
    try:
        doc = fitz.open()
        try:
            page = doc.new_page(width=100, height=40)
            writer = fitz.TextWriter(page.rect)
            sample = next(
                (ch for ch in "Aa1\u4e00\u0627" if font.has_glyph(ord(ch))), None
            )
            if sample is None:
                return True  # nothing to probe with; coverage checks still apply
            writer.append((10, 25), sample, font=font, fontsize=10)
            writer.write_text(page)
        finally:
            doc.close()
        return True
    except Exception:
        return False


class FontRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Only fonts that were chosen for some text are kept: a loaded font
        # holds its whole file in memory, and the server has hundreds.
        self._font_cache: dict[str, fitz.Font] = {}
        self._resolved: dict[str, ResolvedFont] = {}
        # Files `_resolve_uncached` chose for plain text. Only these are
        # tried again for other plain text: the font cache also holds the
        # bold, italic and teacher-chosen faces, and regular words must
        # never be drawn in one of those just because it was loaded first.
        self._plain_chosen: list[Path] = []
        # Character sets no font covers, so a block repeated on every page
        # does not repeat the search that already failed.
        self._unresolvable: dict[str, FontResolutionError] = {}
        # A character each tried file is known to lack, so a later search
        # can rule the file out without opening it again.
        self._lacks: dict[str, set[str]] = {}
        self._undrawable: set[str] = set()
        self._scanned_dirs: dict[Path, list[Path]] = {}

    # -- loading -----------------------------------------------------------
    def _load(
        self,
        *,
        builtin: str | None = None,
        path: Path | None = None,
        cache: bool = True,
    ):
        """The font, or None if it cannot be opened or drawn with.

        `cache=False` returns the font without keeping it, for candidates
        that may not cover the text: the caller keeps it only if it does.
        """
        key = builtin or str(path)
        with self._lock:
            font = self._font_cache.get(key)
            if font is not None:
                return font
            if key in self._undrawable:
                return None
            try:
                font = (
                    fitz.Font(builtin)
                    if builtin
                    else fitz.Font(fontfile=str(path))
                )
            except Exception:
                self._undrawable.add(key)
                return None
            if not builtin and not _can_draw_with(font):
                # Loads and reports glyphs, but MuPDF cannot draw with it
                # (macOS Songti.ttc: "substitute font creation is not
                # implemented"). Chosen once, it made a page's text vanish
                # from the PDF, so it is never offered.
                log.info("Font %s cannot be drawn with; skipped", path)
                self._undrawable.add(key)
                return None
            if cache or builtin:
                self._font_cache[key] = font
            return font

    def _covering(self, path: Path, text: str):
        """The font at `path` if it covers `text`, else None.

        A file that falls short is dropped straight away (and the character
        it lacked remembered), so a search through many files holds one at
        a time instead of all of them.
        """
        key = str(path)
        with self._lock:
            lacked = self._lacks.get(key)
            if lacked and not lacked.isdisjoint(text):
                return None
        font = self._load(path=path, cache=False)
        if font is None:
            return None
        gap = self._first_missing(font, text)
        with self._lock:
            if gap is not None:
                self._lacks.setdefault(key, set()).add(gap)
                return None
            return self._font_cache.setdefault(key, font)

    @staticmethod
    def _denied(path: Path) -> bool:
        # A leading dot marks Apple's private system faces
        # (".NewYork-RegularItalic"): their text copies out of the PDF as
        # the wrong characters.
        if path.name.startswith("."):
            return True
        stem = path.stem.lower().replace(" ", "").replace("-", "")
        return any(bad in stem for bad in DENYLIST)

    def _search_dirs(self) -> list[Path]:
        """Every directory a face may be found in, best first.

        Extracted so the weighted and unweighted searches look in the same
        places -- if they diverged, a bold face could be found where the
        regular one was not, or the reverse.
        """
        return [BUNDLED_DIR, *_SYSTEM_DIRS.get(platform.system(), [])]

    def _scan_dir(self, directory: Path) -> list[Path]:
        """Every non-denied font file under `directory`, recursive, covering
        every extension a face might use.

        This is the *one* traversal implementation every caller reads from
        -- `_all_font_files` (the full search), the bundled-only lookup in
        `_resolve_uncached`, and anything else -- so they cannot drift on
        recursion depth or extensions the way `_resolve_weighted` and
        `_system_fonts` once did (one used `rglob`, the other a top-level
        `glob`; Debian's font packages install a directory *below* the
        search path, e.g. ``/usr/share/fonts/truetype/dejavu/``, so the
        shallow walk found nothing there at all).

        Cached per directory, so a caller that only needs one directory --
        the bundled-fonts check, which in most deployments has no bundled
        directory to scan at all -- never pays for a walk of every system
        font directory it doesn't need. The cache means a font installed on
        disk *after* the first scan of a given directory will not be seen
        for the lifetime of this registry; correct for a deployment with
        fonts baked into the image at build time (the case here), not for
        one where fonts appear on disk while the process is already
        running.
        """
        cached = self._scanned_dirs.get(directory)
        if cached is not None:
            return cached
        found: list[Path] = []
        if directory.is_dir():
            for ext in ("*.ttf", "*.ttc", "*.otf"):
                try:
                    found.extend(p for p in directory.rglob(ext) if not self._denied(p))
                except OSError:
                    pass
        self._scanned_dirs[directory] = found
        return found

    def _all_font_files(self) -> list[Path]:
        """Every non-denied font file under every search directory.

        This is the enumeration `_system_fonts` and `_resolve_weighted`
        read from when they genuinely need the full set. See `_scan_dir`
        for the traversal rules and the per-directory caching (and its
        staleness caveat) this delegates to -- a caller that needs only one
        directory, such as the bundled-fonts check, should call `_scan_dir`
        directly rather than this method, to avoid scanning directories it
        has no use for.
        """
        found: list[Path] = []
        for directory in self._search_dirs():
            found.extend(self._scan_dir(directory))
        return found

    def _system_fonts(self) -> list[Path]:
        """The subset of `_all_font_files()` outside the bundled directory."""
        return [p for p in self._all_font_files() if not p.is_relative_to(BUNDLED_DIR)]

    # -- coverage ----------------------------------------------------------
    @staticmethod
    def _first_missing(font: "fitz.Font", text: str) -> str | None:
        """The first character of `text` the font has no glyph for."""
        for ch in text:
            if ch.isspace() or ch in "\n\r\t":
                continue
            try:
                if not font.has_glyph(ord(ch)):
                    return ch
            except Exception:
                return ch
        return None

    @classmethod
    def _covers(cls, font: "fitz.Font", text: str) -> bool:
        return cls._first_missing(font, text) is None

    def missing_characters(self, font: "fitz.Font", text: str) -> str:
        out = []
        for ch in text:
            if ch.isspace():
                continue
            try:
                if not font.has_glyph(ord(ch)):
                    out.append(ch)
            except Exception:
                out.append(ch)
        return "".join(dict.fromkeys(out))

    # -- resolution --------------------------------------------------------
    def resolve(
        self,
        text: str,
        *,
        script: str = "Latin",
        bold: bool = False,
        italic: bool = False,
        family: str | None = None,
        look: str | None = None,
    ) -> ResolvedFont:
        """Find a font that covers every character of `text`.

        `bold` and `italic` are preferences, not requirements. If no
        matching face covers the text, the regular face is returned with
        `weight_matched=False` so the caller can flag it -- dropping the
        text because a weight is missing would be far worse than drawing it
        in the regular face.

        `look` ("sans", "serif" or "mono") is the source text's kind of
        face, so a Times line is redrawn in a serif and a Helvetica one in a
        sans. `family` is the teacher's explicit choice and wins over it.

        Raises FontResolutionError if no font at all covers the text. Its
        `missing` lists the characters in the way; when every one of them
        is a symbol, the text without them may still resolve.
        """
        if not text.strip():
            return ResolvedFont(key="helv", builtin="helv", display_name="Helvetica")
        look = look if look in _BUILTIN_FACES else "sans"

        # Cache on the distinct characters, not the string: a 100-page
        # document has thousands of strings but few character sets. The
        # weight/slant flags must be part of the key too, or a bold request
        # would be served a cached regular face.
        cache_key = (
            f"{script}:{int(bold)}{int(italic)}:{family or ''}:{look}:"
            f"{''.join(sorted(set(text)))[:180]}"
        )
        with self._lock:
            hit = self._resolved.get(cache_key)
            failed = self._unresolvable.get(cache_key)
        if hit is not None:
            return hit
        if failed is not None:
            raise FontResolutionError(failed.message, missing=failed.missing)

        try:
            entry = catalog.face(family) if family else None
            styled = (
                self._resolve_catalog(text, family, bold, italic) if entry is not None
                else self._resolve_family(text, family, bold, italic) if family
                else None
            )
            if entry is not None and styled is None:
                # Her font has no letters for this text (Arabic in
                # Andika): the automatic face for the script, in that
                # font's look, and the page says so.
                look = entry.look
            if styled is not None:
                resolved = styled
            elif bold or italic:
                resolved = self._resolve_weighted(text, script, bold, italic, look)
                if resolved is None:
                    resolved = replace(
                        self._resolve_uncached(text, script, look), weight_matched=False
                    )
            else:
                resolved = self._resolve_uncached(text, script, look)
            if entry is not None and styled is None:
                resolved = replace(resolved, face=None, fell_back=True)
        except FontResolutionError as exc:
            with self._lock:
                self._unresolvable[cache_key] = exc
            raise

        with self._lock:
            self._resolved[cache_key] = resolved
        return resolved

    def _resolve_catalog(
        self, text: str, face_id: str, bold: bool, italic: bool
    ) -> ResolvedFont | None:
        """A catalog face she chose by name, if it covers the text: the file
        of the asked weight, else its regular file with weight_matched
        False (Lexend has no italic); None when it lacks the letters.

        The files are opened one at a time through _covering and kept only
        once chosen, and never join the plain-text faces (_plain_chosen),
        so automatic text elsewhere is never drawn in them.
        """
        found = catalog.face_path(face_id, bold, italic)
        if found is None:
            return None
        path, matched = found
        tries = [(path, matched)]
        regular = catalog.face_path(face_id)
        if regular is not None and regular[0] != path:
            tries.append((regular[0], False))
        for candidate, exact in tries:
            if candidate.is_file() and self._covering(candidate, text) is not None:
                return ResolvedFont(
                    key=str(candidate), path=candidate, display_name=candidate.stem,
                    weight_matched=exact or not (bold or italic), face=face_id,
                )
        return None

    def _resolve_family(
        self, text: str, family: str, bold: bool, italic: bool
    ) -> ResolvedFont | None:
        """The best font of a chosen style that covers the text, or None.

        Styles name a look, not a file: the fonts present differ between a
        Mac and the server, and each style must work in every script. A
        style no installed font can draw the text in returns None, and the
        caller falls back to the automatic choice.
        """
        patterns = FONT_STYLES.get(family)
        if not patterns:
            return None
        include, exclude = patterns
        candidates: list[tuple[int, int, int, str, Path]] = []
        for path in self._all_font_files():
            stem = _normalized(path.stem)
            if not any(p in stem for p in include) or any(x in stem for x in exclude):
                continue
            is_bold = _style_rank(path.stem, _BOLD_RANKS, _BOLD_TIER0_EXCLUDE) is not None
            is_italic = _style_rank(path.stem, _ITALIC_RANKS) is not None
            mismatch = int(is_bold != bold) + int(is_italic != italic)
            rank = next(i for i, p in enumerate(include) if p in stem)
            # A CJK collection is tens of megabytes; it is used only for
            # text the plain face of the style cannot draw.
            candidates.append((mismatch, rank, int("cjk" in stem), str(path), path))
        candidates.sort(key=lambda c: c[:4])
        for mismatch, _, _, _, path in candidates:
            if self._covering(path, text) is not None:
                return ResolvedFont(
                    key=str(path),
                    path=path,
                    display_name=path.stem,
                    weight_matched=mismatch == 0 or not (bold or italic),
                )
        return None

    def _resolve_weighted(
        self, text: str, script: str, bold: bool, italic: bool, look: str = "sans"
    ) -> ResolvedFont | None:
        """The best face matching the requested weight, or None if there is none.

        The built-in face of the source's look comes first, as the built-in
        regular face does for plain text: it covers Latin, Greek and
        Cyrillic, matches the regular words beside it, and embeds small.
        Files are searched only for text it cannot draw.

        Reads `_all_font_files()` -- the same enumeration `_resolve_uncached`
        reads through `_system_fonts()` -- rather than walking the
        filesystem itself, so the two paths cannot drift on which
        directories, what depth or which extensions they search, and so a
        weighted resolution costs no extra directory I/O beyond the first
        call of either kind.

        Every candidate is gathered before any choice is made -- a
        directory earlier in `_search_dirs()` must never win just because it
        happened to contain *a* match; the best-ranked match anywhere wins,
        and candidates of equal rank are broken on the path string so the
        result cannot depend on filesystem or OS directory-listing order.
        """
        builtin = _BUILTIN_FACES[look][(bold, italic)]
        face = self._load(builtin=builtin)
        if face is not None:
            missing = self.missing_characters(face, text)
            if not missing:
                return ResolvedFont(
                    key=builtin, builtin=builtin, display_name=face.name,
                    weight_matched=True,
                )
        else:
            missing = text
        # Only symbols are missing: their bold or italic is looked for in
        # the symbol faces alone, not in every styled file on the server.
        symbols_only = all(stands_alone(c) for c in missing)

        requests: list[tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]] = []
        if bold:
            requests.append((_BOLD_RANKS, _BOLD_TIER0_EXCLUDE))
        if italic:
            requests.append((_ITALIC_RANKS, ()))
        include, exclude = FONT_STYLES[look]

        candidates: list[tuple[tuple[int, ...], int, int, int, str, Path]] = []
        for path in self._all_font_files():
            stem = _normalized(path.stem)
            if symbols_only and not any(f in stem for f in _SYMBOL_FAMILIES):
                continue
            tiers: list[int] = []
            for ranks, rank_exclude in requests:
                tier = _style_rank(path.stem, ranks, rank_exclude)
                if tier is None:
                    break
                tiers.append(tier)
            else:
                # A face that also carries a style nobody asked for
                # (e.g. "Bold Italic" when only bold was requested) is a
                # worse match than one that carries only what was asked --
                # otherwise it can win purely because its filename happens
                # to sort earlier.
                extraneous = 0
                if not bold and _style_rank(path.stem, _BOLD_RANKS, _BOLD_TIER0_EXCLUDE) is not None:
                    extraneous += 1
                if not italic and _style_rank(path.stem, _ITALIC_RANKS) is not None:
                    extraneous += 1
                # Then a face of the source's look, then an ordinary cut
                # over a condensed, light or CJK one.
                other_look = int(
                    not any(p in stem for p in include) or any(x in stem for x in exclude)
                )
                odd = int(any(x in stem for x in _ODD_CUTS))
                candidates.append(
                    (tuple(tiers), extraneous, other_look, odd, str(path), path)
                )

        candidates.sort(key=lambda c: c[:5])
        for *_, path in candidates:
            if self._covering(path, text) is not None:
                return ResolvedFont(
                    key=str(path),
                    path=path,
                    display_name=path.stem,
                    weight_matched=True,
                )
        return None

    def _plain_face(self, path: Path) -> ResolvedFont:
        """`path` as the face for plain text, remembered for the next block."""
        with self._lock:
            if path not in self._plain_chosen:
                self._plain_chosen.append(path)
        return ResolvedFont(key=str(path), path=path, display_name=path.stem)

    def _resolve_uncached(self, text: str, script: str, look: str = "sans") -> ResolvedFont:
        # 1. Bundled fonts ship with the deployment and are the only fully
        #    portable option, so they win when present. Scoped to just
        #    `BUNDLED_DIR` via `_scan_dir` -- the same traversal rules
        #    `_resolve_weighted` and `_system_fonts` use, so a bundled
        #    `.ttc` is found consistently either way, but without forcing a
        #    recursive scan of every system font directory first. Most
        #    deployments have no bundled directory at all, and this is the
        #    first thing every `resolve()` call touches, including plain
        #    Latin text that will be answered by a builtin a moment later.
        bundled = sorted(self._scan_dir(BUNDLED_DIR), key=str)
        for path in bundled:
            if self._covering(path, text) is not None:
                return self._plain_face(path)

        # 2. MuPDF built-ins, the source's look first. The small Base-14
        #    faces come before Droid Sans Fallback, which embeds as 1.7 MB.
        for name in _builtin_order(look):
            font = self._load(builtin=name)
            if font and self._covers(font, text):
                return ResolvedFont(key=name, builtin=name, display_name=font.name)

        # 3. Preferred system faces for this script. The first one present
        #    is also what missing characters are measured against below.
        system = self._system_fonts()
        by_name = {p.name: p for p in system}
        reference = None
        for preferred in _PREFERRED.get(script, ()):
            path = by_name.get(preferred)
            if path is None:
                continue
            if self._covering(path, text) is not None:
                return self._plain_face(path)
            if reference is None:
                reference = self._load(path=path, cache=False)

        # 4. Files already chosen for other plain text: the same script
        #    turns up block after block, and its face is already in memory.
        #    Not every cached file -- a bold heading's face would then draw
        #    the regular paragraphs below it, here and in later documents.
        with self._lock:
            chosen = list(self._plain_chosen)
        for path in chosen:
            if self._covering(path, text) is not None:
                return self._plain_face(path)

        reference = reference or self._load(builtin="china-s")
        missing = self.missing_characters(reference, text) if reference else text[:12]

        # 5. Only symbols are missing (an emoji, a private-use character
        #    from a damaged text layer): try the symbol faces, not every
        #    font on the server -- that search loaded hundreds of files for
        #    one emoji -- and report the symbols, so the caller can draw the
        #    words without them.
        if missing and all(stands_alone(c) for c in missing):
            for name in _SYMBOL_FACES:
                path = by_name.get(name)
                if path is not None and self._covering(path, text) is not None:
                    return self._plain_face(path)
            raise FontResolutionError(
                "No font on this server has these symbols.", missing=missing
            )

        # 6. Broad scan as a last resort before giving up: letters of a
        #    script with no preferred face. Each file is opened, checked and
        #    let go unless it covers the text.
        for path in system:
            if self._covering(path, text) is not None:
                return self._plain_face(path)

        # Letters and a symbol were both missing and no face has them all.
        # Name the symbols first: without them the letters alone may still
        # resolve, and the caller retries without them.
        symbols = "".join(c for c in missing if stands_alone(c))
        raise FontResolutionError(
            "No font on this server can display that language's characters.",
            missing=symbols or missing,
        )

    def font_for(self, resolved: ResolvedFont):
        return self._load(builtin=resolved.builtin, path=resolved.path)

    def measure(self, resolved: ResolvedFont, text: str, size: float) -> float:
        """Rendered width of `text` in points."""
        font = self.font_for(resolved)
        if font is None:
            # Conservative fallback: assume a wide average advance so the
            # fitter errs toward shrinking rather than overflowing.
            return len(text) * size * 0.6
        return font.text_length(text, fontsize=size)


REGISTRY = FontRegistry()
