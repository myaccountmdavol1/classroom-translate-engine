"""The document-wide blocks endpoint behind the manuscript editor.

The editor shows the whole document as one continuous text, so it needs all
of it in one request -- 37 page-by-page fetches before a teacher could type
a word would make the surface unusable.
"""

from __future__ import annotations

import time

import pytest
from conftest import corpus_file


def _translated(client, name: str = "04_science_diagram.pdf") -> str:
    up = client.post(
        "/api/documents", files={"file": (name, corpus_file(name), "application/pdf")}
    )
    doc_id = up.json()["document_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("REVIEW_READY", "COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    client.post(f"/api/documents/{doc_id}/translate", json={"target_language": "es"})
    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("COMPLETE", "FAILED"):
            break
        time.sleep(0.05)
    assert job["stage"] == "COMPLETE", job
    return doc_id


@pytest.mark.slow
def test_the_whole_document_comes_back_in_one_request(client):
    doc_id = _translated(client)
    body = client.get(f"/api/documents/{doc_id}/blocks").json()
    assert len(body["pages"]) == 1
    assert body["total_blocks"] > 0
    assert body["unit_noun"] == "page"


@pytest.mark.slow
def test_locked_blocks_are_shown_not_hidden(client):
    """Filtering to editable text would leave holes where the maths is.

    A teacher reads this surface straight through; a document missing its
    equations and page numbers would read as if content had been lost.
    """
    doc_id = _translated(client, "03_math_worksheet.pdf")
    body = client.get(f"/api/documents/{doc_id}/blocks").json()
    blocks = [b for p in body["pages"] for b in p["blocks"]]
    locked = [b for b in blocks if b["editable"] is False]
    assert locked, "no locked blocks were reported at all"
    assert all(b["locked_reason"] for b in locked)

    # Compare with the per-page editor endpoint, which returns only the
    # correctable ones: the manuscript must be a superset.
    page1 = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()
    editable_ids = {b["id"] for b in page1["blocks"]}
    manuscript_ids = {b["id"] for b in body["pages"][0]["blocks"]}
    assert editable_ids <= manuscript_ids
    assert len(manuscript_ids) > len(editable_ids)


@pytest.mark.slow
def test_a_lock_reason_never_leaks_developer_text(client):
    """`protected_reason` is "role=FORMULA"; a teacher must not see that."""
    doc_id = _translated(client, "03_math_worksheet.pdf")
    body = client.get(f"/api/documents/{doc_id}/blocks").json()
    for page in body["pages"]:
        for block in page["blocks"]:
            reason = block.get("locked_reason") or ""
            assert "role=" not in reason, reason
            assert "_" not in reason.replace(" ", ""), reason
            if reason:
                assert reason[0].isupper() and reason.endswith(".")


@pytest.mark.slow
def test_blocks_holding_only_invisible_characters_are_dropped(client):
    """A PDF emits zero-width spaces constantly; `strip()` keeps them.

    Left in, the manuscript showed rows of apparently empty text.
    """
    doc_id = _translated(client)
    body = client.get(f"/api/documents/{doc_id}/blocks").json()
    invisible = "​‌‍⁠﻿\xa0 \t\n\r"
    for page in body["pages"]:
        for block in page["blocks"]:
            assert block["original"].strip(invisible), repr(block["original"])


@pytest.mark.slow
def test_an_edit_moves_the_page_revision_so_previews_refresh(client):
    """The page image is cache-busted with this number.

    Without it moving, a correction made on the text surface leaves the
    page view showing the render from before the edit.
    """
    doc_id = _translated(client)
    before = client.get(f"/api/documents/{doc_id}/blocks").json()
    page = before["pages"][0]
    target = next(b for b in page["blocks"] if b["editable"])
    start_revision = page["revision"]

    r = client.patch(
        f"/api/documents/{doc_id}/pages/1/blocks/{target['id']}",
        json={"text": "Texto corregido a mano"},
    )
    assert r.status_code == 200, r.text

    after = client.get(f"/api/documents/{doc_id}/blocks").json()
    assert after["pages"][0]["revision"] > start_revision
    edited = next(
        b for b in after["pages"][0]["blocks"] if b["id"] == target["id"]
    )
    assert edited["translation"] == "Texto corregido a mano"
    assert edited["manually_edited"] is True


@pytest.mark.slow
def test_blocks_are_in_reading_order(client):
    """The manuscript is read top to bottom; order is the whole point."""
    doc_id = _translated(client)
    body = client.get(f"/api/documents/{doc_id}/blocks").json()
    for page in body["pages"]:
        tops = [b["bbox"]["y0"] for b in page["blocks"]]
        # Not strictly sorted -- side-by-side columns interleave -- but the
        # first block must not come from the bottom of the page.
        assert tops[0] < page["height"] * 0.5, tops[:3]
