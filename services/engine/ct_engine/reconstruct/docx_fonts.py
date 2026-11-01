"""Fonts carried inside a Word download.

A translated worksheet is drawn in faces the teacher's computer may not
have: a Google family matched to the PDF's own (Tuffy), or one of the
bundled faces (Andika, Lexend). Word put a stand-in in their place, so the
Word file looked unlike the PDF. Word reads fonts embedded in the file
(ECMA-376: a font part per face, its first 32 bytes obfuscated with a key
named in the font table, and embedTrueTypeFonts set), so each face the
document names that we hold is put in it.

Only the faces we serve are embedded -- the bundled ones and Google's, all
under licences that allow it (SIL OFL, Apache, Ubuntu Font Licence) --
never a font found on the computer, and nothing too big to send.
"""
from __future__ import annotations

import io
import re
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path

from ..fonts import google
from ..fonts.catalog import FACES_DIR

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
FONT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/font"
FONT_TYPE = "application/vnd.openxmlformats-officedocument.obfuscatedFont"
# One face larger than this (a CJK family) is left out, and the faces stop
# once the file would grow by more than the total.
MAX_FACE = 4 * 1024 * 1024
MAX_TOTAL = 12 * 1024 * 1024
_SLOTS = (("embedRegular", False, False), ("embedBold", True, False),
          ("embedItalic", False, True), ("embedBoldItalic", True, True))
# settings.xml children that come before w:embedTrueTypeFonts.
_BEFORE_EMBED = ("writeProtection", "view", "zoom", "removePersonalInformation",
                 "removeDateAndTime", "doNotDisplayPageBoundaries", "displayBackgroundShape",
                 "printPostScriptOverText", "printFractionalCharacterWidth", "printFormsData")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _describe(path: Path) -> tuple[str, bool, bool] | None:
    """A face file's family, and whether it is bold and italic; None for a
    variable font, which Word does not take embedded."""
    from fontTools.ttLib import TTFont

    try:
        font = TTFont(str(path), lazy=True)
        if "fvar" in font:
            return None
        names = font["name"]
        family = names.getDebugName(16) or names.getDebugName(1)
        sub = (names.getDebugName(17) or names.getDebugName(2) or "").lower()
        os2 = font["OS/2"] if "OS/2" in font else None
    except Exception:
        return None
    if not family:
        return None
    bold = "bold" in sub or "black" in sub or (os2 is not None and os2.usWeightClass >= 600)
    italic = "italic" in sub or "oblique" in sub or (os2 is not None and bool(os2.fsSelection & 1))
    return family, bold, italic


def _face_files() -> tuple[str, ...]:
    out: list[str] = []
    for folder in (FACES_DIR, google.CACHE_DIR):
        if folder.is_dir():
            out += [str(p) for p in folder.rglob("*")
                    if p.suffix.lower() in (".ttf", ".otf") and p.stat().st_size <= MAX_FACE]
    return tuple(sorted(out))


@lru_cache(maxsize=4)
def _faces(files: tuple[str, ...]) -> dict[str, dict[tuple[bool, bool], Path]]:
    out: dict[str, dict[tuple[bool, bool], Path]] = {}
    for name in files:
        found = _describe(Path(name))
        if found is not None:
            family, bold, italic = found
            out.setdefault(_key(family), {}).setdefault((bold, italic), Path(name))
    return out


def known_faces() -> dict[str, dict[tuple[bool, bool], Path]]:
    """The faces we may embed, by family; read again when one more has been
    fetched into the Google cache."""
    return _faces(_face_files())


def _obfuscated(data: bytes, font_key: str) -> bytes:
    """ECMA-376 font obfuscation: the first 32 bytes XORed with the key's
    16 bytes, taken from its GUID in reverse order."""
    hexes = re.sub(r"[{}-]", "", font_key)
    key = bytes(int(hexes[i:i + 2], 16) for i in range(0, 32, 2))[::-1]
    head = bytes(b ^ key[i % 16] for i, b in enumerate(data[:32]))
    return head + data[32:]


def _used_families(parts: dict[str, bytes]) -> list[str]:
    names: list[str] = []
    for name, data in parts.items():
        if re.fullmatch(r"word/(document|header\d*|footer\d*|styles)\.xml", name):
            for found in re.findall(rb'w:ascii="([^"]+)"', data):
                text = found.decode("utf-8", "replace")
                if text not in names:
                    names.append(text)
    return names


def _xml(element) -> bytes:
    from lxml import etree

    return etree.tostring(element, xml_declaration=True, encoding="UTF-8", standalone=True)


def embed_fonts(docx: bytes) -> bytes:
    """The Word file with the faces it names that we hold inside it."""
    from lxml import etree

    with zipfile.ZipFile(io.BytesIO(docx)) as zin:
        parts = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    faces = known_faces()
    wanted = [(name, faces[_key(name)]) for name in _used_families(parts) if _key(name) in faces]
    if not wanted or "word/fontTable.xml" not in parts:
        return docx

    table = etree.fromstring(parts["word/fontTable.xml"])
    rels_name = "word/_rels/fontTable.xml.rels"
    rels = etree.fromstring(parts.get(rels_name) or f'<Relationships xmlns="{PKG_NS}"/>'.encode())
    added = total = 0
    for family, styles in wanted:
        font = next((f for f in table.findall(f"{{{W_NS}}}font")
                     if f.get(f"{{{W_NS}}}name") == family), None)
        if font is None:
            font = etree.SubElement(table, f"{{{W_NS}}}font")
            font.set(f"{{{W_NS}}}name", family)
        for slot, bold, italic in _SLOTS:
            path = styles.get((bold, italic))
            if path is None:
                continue
            data = path.read_bytes()
            if total + len(data) > MAX_TOTAL:
                continue
            added += 1
            total += len(data)
            font_key = "{" + str(uuid.uuid4()).upper() + "}"
            target = f"fonts/font{added}.odttf"
            parts[f"word/{target}"] = _obfuscated(data, font_key)
            rel = etree.SubElement(rels, f"{{{PKG_NS}}}Relationship")
            rel.set("Id", f"rIdFont{added}")
            rel.set("Type", FONT_REL)
            rel.set("Target", target)
            embed = etree.SubElement(font, f"{{{W_NS}}}{slot}")
            embed.set(f"{{{R_NS}}}id", f"rIdFont{added}")
            embed.set(f"{{{W_NS}}}fontKey", font_key)
    if not added:
        return docx
    parts["word/fontTable.xml"] = _xml(table)
    parts[rels_name] = _xml(rels)

    types = parts["[Content_Types].xml"]
    if b'Extension="odttf"' not in types:
        types = types.replace(
            b"</Types>", f'<Default Extension="odttf" ContentType="{FONT_TYPE}"/></Types>'.encode())
    parts["[Content_Types].xml"] = types

    if "word/settings.xml" in parts:
        settings = etree.fromstring(parts["word/settings.xml"])
        if settings.find(f"{{{W_NS}}}embedTrueTypeFonts") is None:
            flag = etree.Element(f"{{{W_NS}}}embedTrueTypeFonts")
            before = [c for c in settings if etree.QName(c).localname in _BEFORE_EMBED]
            if before:
                before[-1].addnext(flag)
            else:
                settings.insert(0, flag)
            parts["word/settings.xml"] = _xml(settings)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in parts.items():
            zout.writestr(name, data)
    return out.getvalue()
