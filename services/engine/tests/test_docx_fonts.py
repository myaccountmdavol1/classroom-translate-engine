"""The faces a Word download is drawn in go inside it.

Word put a stand-in for Tuffy (a Google family matched to a slide's font)
because the teacher's computer did not have it; embedded, Word shows it.
"""
from __future__ import annotations

import io
import re
import zipfile

from docx import Document
from fontTools.ttLib import TTFont

from ct_engine.reconstruct.docx_fonts import _obfuscated, embed_fonts, known_faces


def _docx(*faces: str) -> bytes:
    doc = Document()
    for face in faces:
        run = doc.add_paragraph().add_run(f"Texto en {face}")
        run.font.name = face
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _parts(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist()}


def test_a_bundled_face_is_embedded_and_a_system_one_is_not():
    assert "andika" in known_faces()
    parts = _parts(embed_fonts(_docx("Andika", "Arial")))

    table = parts["word/fontTable.xml"].decode()
    andika = re.search(r'<w:font w:name="Andika">.*?</w:font>', table, re.S).group(0)
    assert "embedRegular" in andika
    assert not re.search(r'w:name="Arial">[^/]*?embedRegular', table)
    assert any(n.startswith("word/fonts/") and n.endswith(".odttf") for n in parts)
    assert b"embedTrueTypeFonts" in parts["word/settings.xml"]
    assert b'Extension="odttf"' in parts["[Content_Types].xml"]


def test_the_embedded_face_is_the_font_once_unscrambled():
    parts = _parts(embed_fonts(_docx("Andika")))
    table = parts["word/fontTable.xml"].decode()
    rid, font_key = re.search(r'embedRegular r:id="([^"]+)" w:fontKey="([^"]+)"', table).groups()
    target = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', parts["word/_rels/fontTable.xml.rels"].decode())
    data = _obfuscated(parts["word/" + target.group(1)], font_key)  # XOR undoes itself
    assert TTFont(io.BytesIO(data))["name"].getDebugName(1) == "Andika"


def test_the_file_still_opens_and_a_file_without_our_faces_is_unchanged():
    embedded = embed_fonts(_docx("Andika"))
    assert "Texto en Andika" in Document(io.BytesIO(embedded)).paragraphs[0].text
    plain = _docx("Arial")
    assert embed_fonts(plain) == plain
