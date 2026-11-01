"""A translated PDF downloaded as Word, easy to edit or looking the same."""
from __future__ import annotations

import io

import pytest
from docx import Document
from test_manuscript import _translated

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _word(client, doc_id, layout=None):
    url = f"/api/documents/{doc_id}/editable-docx" + (f"?layout={layout}" if layout else "")
    r = client.get(url)
    assert r.status_code == 200, r.text
    return r, Document(io.BytesIO(r.content))


@pytest.mark.slow
def test_a_worksheet_comes_out_as_paragraphs_lines_and_its_picture(client):
    doc_id = _translated(client, "04_science_diagram.pdf")
    r, doc = _word(client, doc_id)
    assert "(Spanish).docx" in r.headers["content-disposition"]
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    # In reading order, as words to edit.
    title = next(i for i, t in enumerate(texts) if "celular" in t.lower() and len(t) < 30)
    question = next(i for i, t in enumerate(texts) if t.startswith("1."))
    assert title < question
    # The diagram is one picture, its labels in it rather than loose words.
    assert len(doc.inline_shapes) == 1
    assert not any("Pared" in t for t in texts)
    # Two answer lines under each of the two questions.
    ruled = [p for p in doc.paragraphs if p._p.find(f"{W}pPr/{W}pBdr/{W}between") is not None and not p.text]
    assert len(ruled) >= 4
    # One Word page per PDF page, its number Word's own.
    assert len(doc.sections) == 1
    assert "PAGE" in doc.sections[0].footer._element.xml


@pytest.mark.slow
def test_a_ruled_table_is_a_word_table(client):
    doc_id = _translated(client, "05_tables.pdf")
    _, doc = _word(client, doc_id)
    [table] = doc.tables
    assert (len(table.rows), len(table.columns)) == (6, 5)
    assert table.cell(1, 0).text == "2018"
    assert table.cell(0, 0).text.strip()


@pytest.mark.slow
def test_the_same_look_is_text_boxes_on_the_page(client):
    doc_id = _translated(client, "04_science_diagram.pdf")
    r, doc = _word(client, doc_id, "page")
    assert "same look" in r.headers["content-disposition"]
    body = doc.element.body.xml
    assert "wps:txbx" in body and 'behindDoc="1"' in body
