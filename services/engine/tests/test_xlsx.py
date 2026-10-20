"""XLSX ingest and reconstruction.

A spreadsheet fails in its own way: most of it is numbers and formulas that
must come through untouched, and one stored string may be shown by many
cells. The workbook here is written by hand as raw OOXML (openpyxl is not a
dependency), which also means every byte of it is known, so "untouched" can
be checked byte-for-byte.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import zipfile

import pytest

from ct_engine.ingest.registry import parser_for
from ct_engine.ingest.validate import validate_upload
from ct_engine.ingest.xlsx_parser import XlsxParser
from ct_engine.ingest.xlsx_walk import object_id
from ct_engine.model.enums import TranslationStatus
from ct_engine.model.formats import SourceFormat, format_for
from ct_engine.pipeline.translate import translate_page
from ct_engine.reconstruct.xlsx_builder import rebuild_xlsx
from test_translate_pipeline import EchoProvider

XLSX_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# Shared strings, by index:
#   0 directions, 1 "Name" (used twice on sheet 1 and once on sheet 2),
#   2 rich text, 3 "A-12", 4 "3.5", 5 an email, 6 "Yes" (a formula literal),
#   7 a sheet-2 heading.
_SST = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="{NS}" count="10" uniqueCount="8">\
<si><t>Directions: read each question carefully.</t></si>\
<si><t>Name</t></si>\
<si><r><rPr><b/></rPr><t>Bold</t></r><r><t xml:space="preserve"> and plain words</t></r></si>\
<si><t>A-12</t></si>\
<si><t>3.5</t></si>\
<si><t>kim@school.org</t></si>\
<si><t>Yes</t></si>\
<si><t>Vocabulary list</t></si>\
</sst>"""

_FORMULA_CELL = '<c r="C2"><f>SUM(B2:B3)</f><v>84</v></c>'
_NUMBER_CELL = '<c r="B2" s="1"><v>42</v></c>'
_IF_CELL = '<c r="B5"><f>IF(A5="Yes",1,0)</f><v>1</v></c>'
_STR_RESULT = '<c r="C5" t="str"><f>"Total score"</f><v>Total score</v></c>'

_SHEET1 = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS}" xmlns:r="{RNS}">\
<cols><col min="1" max="1" width="40" customWidth="1"/></cols>\
<sheetData>\
<row r="1"><c r="A1" t="s"><v>0</v></c></row>\
<row r="2"><c r="A2" t="s"><v>1</v></c>{_NUMBER_CELL}{_FORMULA_CELL}</row>\
<row r="3"><c r="A3" t="s"><v>1</v></c><c r="B3" t="s"><v>2</v></c></row>\
<row r="4"><c r="A4" t="inlineStr"><is><t>Answer the question below.</t></is></c>\
<c r="B4" t="s"><v>3</v></c><c r="C4" t="s"><v>4</v></c><c r="D4" t="s"><v>5</v></c></row>\
<row r="5"><c r="A5" t="s"><v>6</v></c>{_IF_CELL}{_STR_RESULT}</row>\
</sheetData>\
<mergeCells count="1"><mergeCell ref="A1:D1"/></mergeCells>\
</worksheet>"""

_SHEET2 = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS}" xmlns:r="{RNS}"><sheetData>\
<row r="1"><c r="A1" t="s"><v>7</v></c></row>\
<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>7</v></c></row>\
</sheetData></worksheet>"""

_WORKBOOK = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{NS}" xmlns:r="{RNS}"><sheets>\
<sheet name="Quiz" sheetId="1" r:id="rId1"/>\
<sheet name="Words" sheetId="2" r:id="rId2"/>\
</sheets></workbook>"""

_WB_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>\
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>\
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>\
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\
</Relationships>"""

_STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{NS}"><numFmts count="0"/>\
<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>\
<fills count="1"><fill><patternFill patternType="none"/></fill></fills>\
<borders count="1"><border/></borders>\
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>\
<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>\
<xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>\
</styleSheet>"""

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\
<Default Extension="xml" ContentType="application/xml"/>\
<Default Extension="png" ContentType="image/png"/>\
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\
<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>\
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>\
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\
</Relationships>"""

# Not a real picture; only its bytes surviving matters.
_PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


def _workbook(**overrides: str | bytes) -> bytes:
    parts: dict[str, str | bytes] = {
        "[Content_Types].xml": _CONTENT_TYPES,
        "_rels/.rels": _ROOT_RELS,
        "xl/workbook.xml": _WORKBOOK,
        "xl/_rels/workbook.xml.rels": _WB_RELS,
        "xl/worksheets/sheet1.xml": _SHEET1,
        "xl/worksheets/sheet2.xml": _SHEET2,
        "xl/sharedStrings.xml": _SST,
        "xl/styles.xml": _STYLES,
        "xl/media/image1.png": _PNG,
    }
    parts.update(overrides)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return out.getvalue()


def _pages(data: bytes | None = None):
    with XlsxParser().open_bytes(data or _workbook()) as doc:
        return [doc.parse_page(i) for i in range(doc.page_count)]


def _translate(pages, fn=lambda t: f"<{t}>"):
    for page in pages:
        for obj in page.text_objects:
            obj.translated_text = fn(obj.text)
            obj.translation_status = TranslationStatus.TRANSLATED
    return pages


def _part(data: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(name)


def _cell(xml: str, ref: str) -> str:
    m = re.search(rf'<c r="{ref}"[^>]*?(?:/>|>.*?</c>)', xml)
    assert m, f"cell {ref} not found"
    return m.group(0)


def _shared(data: bytes) -> list[str]:
    """Each <si>'s visible text, in index order."""
    xml = _part(data, "xl/sharedStrings.xml").decode()
    items = re.findall(r"<si>(.*?)</si>", xml)
    return [
        html.unescape("".join(re.findall(r"<t(?: [^>]*)?>([^<]*)</t>", re.sub(r"<rPh.*?</rPh>", "", i))))
        for i in items
    ]


# --- Parsing --------------------------------------------------------------


def test_one_sheet_is_one_page_named_after_its_tab():
    pages = _pages()
    assert len(pages) == 2
    assert [p.diagnostics.extra["xlsx_sheet_name"] for p in pages] == ["Quiz", "Words"]
    assert not any(p.has_fixed_layout for p in pages)


def test_only_words_are_taken():
    texts = [o.text for p in _pages() for o in p.text_objects]
    assert "Directions: read each question carefully." in texts
    assert "Answer the question below." in texts
    assert "Vocabulary list" in texts
    for not_words in ("A-12", "3.5", "kim@school.org"):
        assert not_words not in texts


def test_a_shared_string_is_one_block_on_the_first_sheet_that_shows_it():
    pages = _pages()
    names = [(p.index, o) for p in pages for o in p.text_objects if o.text == "Name"]
    assert len(names) == 1
    page_index, obj = names[0]
    assert page_index == 0
    assert obj.anchor == "xl/sharedStrings.xml#si1"


def test_a_formula_literal_is_not_translated():
    """`=IF(A5="Yes",...)` stops working if A5 says "Sí"."""
    texts = [o.text for p in _pages() for o in p.text_objects]
    assert "Yes" not in texts


def test_formula_results_are_never_taken():
    texts = [o.text for p in _pages() for o in p.text_objects]
    assert "Total score" not in texts


def test_blocks_read_row_by_row_with_stable_ids():
    page = _pages()[0]
    assert [o.text for o in page.text_objects][:3] == [
        "Directions: read each question carefully.",
        "Name",
        "Bold and plain words",
    ]
    ys = [o.bbox.y0 for o in page.text_objects]
    assert ys == sorted(ys)
    for o in page.text_objects:
        assert o.id == object_id(o.anchor)
    # The same in a second parse, which hash() would not promise across runs.
    assert [o.id for o in _pages()[0].text_objects] == [o.id for o in page.text_objects]


def test_rich_text_is_flagged_at_parse_time():
    page = _pages()[0]
    rich = next(o for o in page.text_objects if o.text == "Bold and plain words")
    assert page.diagnostics.extra["xlsx_collapsed_formatting"] == [rich.id]


def test_no_sheet_previews_are_claimed():
    with XlsxParser().open_bytes(_workbook()) as doc:
        assert doc.supports_preview is False
        assert doc.embedded_media_count == 1


# --- Write-back -----------------------------------------------------------


def test_a_shared_string_is_translated_once_and_every_cell_changes():
    src = _workbook()
    result = rebuild_xlsx(src, _translate(_pages()))
    assert result.unresolved == []
    strings = _shared(result.data)
    assert strings[1] == "<Name>"
    # Both cells on sheet 1 and the one on sheet 2 still point at index 1,
    # untouched, so all three now show the translation.
    sheet1 = _part(result.data, "xl/worksheets/sheet1.xml").decode()
    assert _cell(sheet1, "A2") == '<c r="A2" t="s"><v>1</v></c>'
    assert _cell(sheet1, "A3") == '<c r="A3" t="s"><v>1</v></c>'
    assert strings.count("<Name>") == 1


def test_numbers_and_formulas_are_byte_identical():
    result = rebuild_xlsx(_workbook(), _translate(_pages()))
    sheet1 = _part(result.data, "xl/worksheets/sheet1.xml").decode()
    assert _cell(sheet1, "B2") == _NUMBER_CELL
    assert _cell(sheet1, "C2") == _FORMULA_CELL
    assert _cell(sheet1, "B5") == _IF_CELL
    assert _cell(sheet1, "C5") == _STR_RESULT
    assert '<mergeCell ref="A1:D1"/>' in sheet1
    assert '<col min="1" max="1" width="40" customWidth="1"/>' in sheet1


def test_non_words_and_formula_literals_are_left_alone():
    result = rebuild_xlsx(_workbook(), _translate(_pages()))
    strings = _shared(result.data)
    assert strings[3:7] == ["A-12", "3.5", "kim@school.org", "Yes"]


def test_rich_text_collapses_into_the_first_run_and_says_so():
    result = rebuild_xlsx(_workbook(), _translate(_pages()))
    assert _shared(result.data)[2] == "<Bold and plain words>"
    xml = _part(result.data, "xl/sharedStrings.xml").decode()
    rich = re.findall(r"<si>(.*?)</si>", xml)[2]
    assert "<b/>" in rich, "first run lost its formatting"
    codes = [f.code for f in result.stats[0].findings]
    assert codes.count("xlsx_formatting_collapsed") == 1


def test_inline_string_is_translated_in_its_own_cell():
    result = rebuild_xlsx(_workbook(), _translate(_pages()))
    sheet1 = _part(result.data, "xl/worksheets/sheet1.xml").decode()
    assert "&lt;Answer the question below.&gt;" in _cell(sheet1, "A4")


def test_edge_whitespace_is_preserved():
    pages = _translate(_pages(), lambda t: f"  {t}")
    result = rebuild_xlsx(_workbook(), pages)
    # The translator's own edges are replaced by the source's, which has none.
    assert _shared(result.data)[1] == "Name"
    sst = _SST.replace("<t>Name</t>", '<t xml:space="preserve">Name: </t>')
    src = _workbook(**{"xl/sharedStrings.xml": sst})
    pages = _translate(_pages(src), lambda t: "Nombre:")
    result = rebuild_xlsx(src, pages)
    xml = _part(result.data, "xl/sharedStrings.xml").decode()
    assert '<t xml:space="preserve">Nombre: </t>' in xml


def test_untouched_parts_are_byte_identical():
    src = _workbook()
    result = rebuild_xlsx(src, _translate(_pages()))
    with zipfile.ZipFile(io.BytesIO(src)) as a, zipfile.ZipFile(io.BytesIO(result.data)) as b:
        assert a.namelist() == b.namelist()
        changed = {
            n for n in a.namelist()
            if hashlib.sha256(a.read(n)).digest() != hashlib.sha256(b.read(n)).digest()
        }
    # Shared strings for the shared text; sheet 1 only for its inline cell.
    assert changed == {"xl/sharedStrings.xml", "xl/worksheets/sheet1.xml"}


def test_a_workbook_with_only_shared_strings_rewrites_one_part():
    sheet1 = _SHEET1.replace(
        '<c r="A4" t="inlineStr"><is><t>Answer the question below.</t></is></c>', ""
    )
    src = _workbook(**{"xl/worksheets/sheet1.xml": sheet1})
    result = rebuild_xlsx(src, _translate(_pages(src)))
    assert _part(result.data, "xl/worksheets/sheet1.xml") == _part(
        src, "xl/worksheets/sheet1.xml"
    )


def test_output_reopens_with_every_cell():
    src = _workbook()
    result = rebuild_xlsx(src, _translate(_pages()))
    with XlsxParser().open_bytes(src) as before, XlsxParser().open_bytes(result.data) as after:
        assert after.page_count == before.page_count == 2
        assert after.cell_count == before.cell_count


def test_untranslated_blocks_leave_the_source_alone():
    pages = _translate(_pages())
    for p in pages:
        for o in p.text_objects:
            o.translation_status = TranslationStatus.FAILED
    result = rebuild_xlsx(_workbook(), pages)
    assert _part(result.data, "xl/sharedStrings.xml") == _part(
        _workbook(), "xl/sharedStrings.xml"
    )


def test_unresolvable_anchor_is_reported_not_swallowed():
    pages = _translate(_pages())
    pages[0].text_objects[0].anchor = "xl/sharedStrings.xml#si999"
    result = rebuild_xlsx(_workbook(), pages)
    assert result.unresolved == ["xl/sharedStrings.xml#si999"]
    assert any(f.code == "xlsx_anchor_lost" for f in result.stats[0].findings)


def test_through_the_translation_stage():
    """The real translate stage, with a deterministic provider."""
    pages = _pages()
    for page in pages:
        translate_page(page, EchoProvider(str.upper), source_lang="en", target_lang="es")
    result = rebuild_xlsx(_workbook(), pages)
    assert result.unresolved == []
    strings = _shared(result.data)
    assert strings[0] == "DIRECTIONS: READ EACH QUESTION CAREFULLY."
    assert strings[7] == "VOCABULARY LIST"


# --- The door -------------------------------------------------------------


def test_format_detection():
    assert format_for(None, "grades.xlsx") is SourceFormat.XLSX
    assert format_for(XLSX_CT, None) is SourceFormat.XLSX
    fmt = SourceFormat.XLSX
    assert fmt.extension == "xlsx" and fmt.content_type == XLSX_CT
    assert fmt.rebuilds_in_place and not fmt.is_fixed_layout
    assert fmt.page_noun == "sheet"


def test_registry_picks_the_xlsx_parser():
    assert parser_for(None, "a.xlsx").name == "xlsx"
    assert parser_for(XLSX_CT, None).name == "xlsx"
    assert parser_for(None, "a.docx").name != "xlsx"


def test_upload_validation_accepts_xlsx():
    r = validate_upload(_workbook(), "quiz.xlsx")
    assert r.ok, r.message
    assert r.content_type == XLSX_CT
    assert r.page_count == 2
    # And with no extension, by its parts.
    assert validate_upload(_workbook(), None).content_type == XLSX_CT


def test_password_protected_workbook_is_refused_clearly():
    ole = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100
        + "EncryptedPackage".encode("utf-16-le") + b"\x00" * 100
    )
    r = validate_upload(ole, "grades.xlsx")
    assert not r.ok and r.is_encrypted
    assert "password" in r.message


def test_old_xls_is_refused_with_what_to_do():
    r = validate_upload(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 200, "a.xls")
    assert not r.ok and ".xlsx" in r.message


def test_a_workbook_of_only_numbers_is_refused():
    sheet = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS}"><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"""
    src = _workbook(**{"xl/worksheets/sheet1.xml": sheet, "xl/worksheets/sheet2.xml": sheet})
    r = validate_upload(src, "numbers.xlsx")
    assert not r.ok and "no words" in r.message


def test_too_many_text_cells_is_refused(monkeypatch):
    from ct_engine.config import get_settings

    monkeypatch.setattr(get_settings(), "max_xlsx_cells", 2)
    r = validate_upload(_workbook(), "big.xlsx")
    assert not r.ok and "limit" in r.message


def test_upload_route_accepts_xlsx_and_health_lists_it(client):
    import time

    assert "xlsx" in client.get("/api/health").json()["supported_formats"]
    resp = client.post(
        "/api/documents", files={"file": ("quiz.xlsx", _workbook(), XLSX_CT)}
    )
    assert resp.status_code == 200, resp.text
    doc_id = resp.json()["document_id"]
    assert resp.json()["page_count"] == 2

    deadline = time.time() + 60
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("REVIEW_READY", "COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    assert job["stage"] != "FAILED", job
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["pages"][0]["unit_noun"] == "sheet"


@pytest.mark.slow
def test_full_excel_translation_round_trip(client):
    """Upload -> analyze -> translate -> write back -> download, for real."""
    import time

    src = _workbook()
    resp = client.post("/api/documents", files={"file": ("quiz.xlsx", src, XLSX_CT)})
    doc_id = resp.json()["document_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("REVIEW_READY", "COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    started = client.post(
        f"/api/documents/{doc_id}/translate", json={"target_language": "es"}
    )
    assert started.status_code == 200, started.text
    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    assert job["stage"] == "COMPLETE", job

    dl = client.get(f"/api/documents/{doc_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == XLSX_CT
    assert dl.headers["content-disposition"].endswith('.xlsx"')
    with XlsxParser().open_bytes(dl.content) as after:
        assert after.page_count == 2
    sheet1 = _part(dl.content, "xl/worksheets/sheet1.xml").decode()
    assert _cell(sheet1, "C2") == _FORMULA_CELL
    assert _shared(dl.content)[3:7] == ["A-12", "3.5", "kim@school.org", "Yes"]
    assert _part(dl.content, "xl/media/image1.png") == _PNG
