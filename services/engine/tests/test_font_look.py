"""Whether a PDF's text was set in a serif face, for its translation's.

Canva marked every font of a slide serif in the PDF, Tuffy and Futura
(sans faces) included, so the translation of words it could not match was
drawn in Times. The fonts are embedded, and each says what it is.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from ct_engine.ingest.font_look import key, own_looks, serif
from ct_engine.ingest.pdf_parser import PdfParser

FACES = Path(__file__).resolve().parents[1] / "ct_engine" / "fonts" / "faces"


def _pdf_with(*fonts: Path) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for k, path in enumerate(fonts):
        name = f"F{k}"
        page.insert_font(fontname=name, fontfile=str(path))
        page.insert_text((72, 100 + 40 * k), "Plants need light", fontname=name, fontsize=18)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sans() -> Path:
    return next((FACES / "noto-sans").glob("*Regular*.ttf"))


@pytest.fixture
def serif_face() -> Path:
    return FACES / "noto-serif" / "NotoSerif-Regular.ttf"


def test_an_embedded_font_says_whether_it_is_serif(sans, serif_face):
    with fitz.open(stream=_pdf_with(sans, serif_face), filetype="pdf") as doc:
        looks = own_looks(doc, doc[0])
    # By key(): the font list's "Noto Sans Regular" is the spans' "NotoSans-Regular".
    assert looks[key("NotoSans-Regular")] is False
    assert looks[key("NotoSerif-Regular")] is True


@pytest.mark.parametrize(
    ("name", "flagged", "own", "flags_lie", "want"),
    [
        # What the font says wins over the PDF's mark.
        ("Tuffy-Regular", True, False, False, False),
        ("Lora-Regular", False, True, False, True),
        # Unsaid: a Google family's listed category (Oranienbaum is a serif
        # display face, in a PDF whose flags were caught lying).
        ("Oranienbaum-Regular", True, None, True, True),
        ("FiraSans-Bold", True, None, False, False),
        # Unsaid: the name, when it is a known face.
        ("TimesNewRomanPSMT", False, None, False, True),
        ("ArialMT", True, None, False, False),
        ("PTSans-Bold", True, None, False, False),
        # Unknown, unsaid: the mark, unless the PDF marked a sans face serif.
        ("Makro-Bold", True, None, False, True),
        ("Makro-Bold", True, None, True, False),
        ("Makro-Bold", False, None, False, False),
    ],
)
def test_which_says_what_the_face_is(name, flagged, own, flags_lie, want):
    assert serif(name, flagged, own, flags_lie) is want


def test_the_parser_believes_the_font_over_the_mark(sans):
    """Noto Sans embedded but marked serif, as Canva marks its fonts."""
    data = _pdf_with(sans)
    with fitz.open(stream=data, filetype="pdf") as doc:
        for xref in range(1, doc.xref_length()):
            if doc.xref_get_key(xref, "Type")[1] == "/FontDescriptor":
                doc.xref_set_key(xref, "Flags", "34")  # Serif | Nonsymbolic
        data = doc.tobytes()
    with fitz.open(stream=data, filetype="pdf") as check:
        flags = {s["flags"] & 4 for b in check[0].get_text("dict")["blocks"]
                 for l in b.get("lines", []) for s in l["spans"]}
    assert flags == {4}, "the test PDF marks its sans font serif"
    with PdfParser().open_bytes(data) as parsed:
        page = parsed.parse_page(0)
    assert page.text_objects and not any(t.style.serif for t in page.text_objects)
