"""The fonts a teacher can choose by name, shipped inside the package.

`faces/catalog.json` is the one list: the engine reads it here, and the
editor has a byte-identical copy (apps/web/src/lib/font-catalog.json, with
the same files under apps/web/public/fonts/v1/) so the menu, the words it
draws while a picture is on its way, and the page all use the same faces.
Every face is under the SIL Open Font License, with its OFL.txt beside it.

The faces directory is deliberately not fonts/bundled/ and not one of the
registry's search directories: automatic, weighted and broad font searches
never look in it, so no document drawn before these existed changes face,
and a server never loads these files unless she picks one (or a list
needs its marks).

None of these covers Arabic, Urdu, Bengali, Chinese, Japanese or Korean.
Text in those is drawn in the automatic face for its script instead
(registry.resolve), and the editor says so beside the font's name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FACES_DIR = Path(__file__).parent / "faces"
CATALOG_FILE = FACES_DIR / "catalog.json"


@dataclass(frozen=True)
class Face:
    id: str
    name: str
    look: str
    files: dict[str, str | None]
    covers: tuple[str, ...]


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict[str, Face], str | None]:
    try:
        raw = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, None
    faces = {
        f["id"]: Face(
            id=f["id"], name=f["name"], look=f.get("look", "sans"),
            files=dict(f["files"]), covers=tuple(f.get("covers", ())),
        )
        for f in raw.get("faces", [])
    }
    return faces, raw.get("marks")


def faces() -> dict[str, Face]:
    return _catalog()[0]


def face(face_id: str | None) -> Face | None:
    return faces().get(face_id or "")


def family_ids() -> tuple[str, ...]:
    return tuple(faces())


def face_path(face_id: str, bold: bool = False, italic: bool = False) -> tuple[Path, bool] | None:
    """(file, matched) for a face in a weight. `matched` is False when that
    weight does not exist (Lexend has no italic) and the regular file, or
    the bold one for bold italic, stands in."""
    entry = face(face_id)
    if entry is None:
        return None
    wanted = {
        (False, False): "regular", (True, False): "bold",
        (False, True): "italic", (True, True): "bold_italic",
    }[(bool(bold), bool(italic))]
    name = entry.files.get(wanted)
    if name:
        return FACES_DIR / name, True
    fallback = entry.files.get("bold") if bold else None
    return FACES_DIR / (fallback or entry.files["regular"]), False


def marks_face() -> Path | None:
    """Noto Sans Symbols 2: the ◦ ■ ✓ bullets, which few text faces have."""
    name = _catalog()[1]
    if not name:
        return None
    path = FACES_DIR / name
    return path if path.is_file() else None


def noto_sans(bold: bool = False, italic: bool = False) -> Path | None:
    """The bundled Noto Sans in a weight: list numbers and dashes a text
    face cannot draw are drawn in it."""
    found = face_path("noto-sans", bold, italic)
    if found is None or not found[0].is_file():
        return None
    return found[0]


def covers(face_id: str, lang: str | None) -> bool:
    """Whether a face has the letters of a language ("pt-BR" reads "pt")."""
    entry = face(face_id)
    if entry is None or not lang:
        return False
    base = lang.split("-")[0].lower()
    return base in entry.covers
