"""Combining boxes: a paragraph the engine split into pieces, made one again.

A slide's centred paragraph often comes back as one box per line. Each
piece is translated alone, as a fragment, and the pieces overlap once the
translation is longer. Combine joins their English, translates it once,
and draws it in one box over the area the pieces covered.
"""

from __future__ import annotations

import fitz
import pytest

from ct_engine.model.enums import TextAlignment
from ct_engine.model.geometry import Rect
from ct_engine.pipeline.edit import (
    EditError,
    add_block,
    combine_blocks,
    current_frame,
    editable_blocks,
    undo,
)
from ct_engine.pipeline.reconstruct_stage import rebuild_export_only
from test_edit import prepared


@pytest.fixture
def translator(monkeypatch):
    """Every English sent, and a translation that marks it."""
    from ct_engine.translate.one_off import OneOffTranslation
    import ct_engine.translate.one_off as one_off

    calls: list[str] = []

    def translate(text, **kwargs):
        calls.append(text)
        return OneOffTranslation(text=f"ES[{text}]", provider="fixture", protected_intact=True)

    monkeypatch.setattr(one_off, "translate_one", translate)
    return calls


def two_blocks(manifest):
    blocks = editable_blocks(manifest.page(0))
    assert len(blocks) >= 2
    return blocks[0], blocks[1]


def lines_of(manifest, store, texts, *, centre=300.0, left=None):
    """Boxes she made, one line each, centred on `centre` or lined up on
    `left`, with their English set."""
    made = []
    y = 200.0
    for text in texts:
        width = 6.0 * len(text)
        x0 = left if left is not None else centre - width / 2
        _r, obj = add_block(manifest, 1, bbox={"x0": x0, "y0": y, "x1": x0 + width, "y1": y + 18},
                            text="x", store=store, target_lang="es")
        obj.approved_source_text = text
        made.append(obj)
        y += 22.0
    return made


def test_the_english_is_joined_and_translated_once(store, translator):
    manifest = prepared(store)
    a, b = two_blocks(manifest)
    english = f"{a.flat_text} {b.flat_text}"

    combine_blocks(manifest, 1, [b.id, a.id], store=store, target_lang="es")

    assert translator == [english], "pieces go to the translator as one sentence, in reading order"
    assert a.flat_text == english
    assert a.translated_text == f"ES[{english}]"
    assert b.suppressed, "the other piece's words still have to come off the page"


def test_a_word_broken_with_a_hyphen_is_mended(store, translator):
    manifest = prepared(store)
    top, low = lines_of(manifest, store, ["Plants use photo-", "synthesis to grow."], left=100.0)

    combine_blocks(manifest, 1, [top.id, low.id], store=store, target_lang="es")

    assert translator == ["Plants use photosynthesis to grow."]


def test_the_box_covers_every_piece_with_the_words_in_the_middle(store, translator):
    manifest = prepared(store)
    a, b = two_blocks(manifest)
    union = Rect.bounding([current_frame(a), current_frame(b)])

    combine_blocks(manifest, 1, [a.id, b.id], store=store, target_lang="es")

    box = a.manual_bbox
    assert box is not None
    for got, want in zip((box.x0, box.y0, box.x1, box.y1), (union.x0, union.y0, union.x1, union.y1)):
        assert got == pytest.approx(want, abs=0.01)
    assert a.manual_vertical_align == "middle"


def test_centred_pieces_stay_centred(store, translator):
    manifest = prepared(store)
    pieces = lines_of(manifest, store, [
        "Magma Math Academy provides various 30-45",
        "minute courses for our teachers, no matter their",
        "level.",
    ])

    combine_blocks(manifest, 1, [p.id for p in pieces], store=store, target_lang="es")

    assert pieces[0].drawn_alignment is TextAlignment.CENTER


def test_pieces_lined_up_on_the_left_stay_left(store, translator):
    manifest = prepared(store)
    pieces = lines_of(manifest, store, [
        "Reflect on what you gleaned",
        "from this LEAP and share your reflection",
        "here.",
    ], left=120.0)

    combine_blocks(manifest, 1, [p.id for p in pieces], store=store, target_lang="es")

    assert pieces[0].drawn_alignment is TextAlignment.LEFT


def test_one_undo_puts_every_piece_back(store, translator):
    manifest = prepared(store)
    a, b = two_blocks(manifest)
    before = (a.flat_text, a.translated_text, a.manual_bbox, b.suppressed)

    combine_blocks(manifest, 1, [a.id, b.id], store=store, target_lang="es")
    undo(manifest, 1, store=store, target_lang="es")

    page = manifest.page(0)
    a2, b2 = page.find_text_object(a.id), page.find_text_object(b.id)
    assert (a2.flat_text, a2.translated_text, a2.manual_bbox, b2.suppressed) == before


def test_boxes_she_made_are_deleted_into_the_first(store, translator):
    manifest = prepared(store)
    pieces = lines_of(manifest, store, ["Take this time", "to plan."])

    combine_blocks(manifest, 1, [p.id for p in pieces], store=store, target_lang="es")

    ids = {o.id for o in manifest.page(0).text_objects}
    assert pieces[0].id in ids and pieces[1].id not in ids


def test_the_pdf_has_the_combined_words_and_neither_original(store, translator):
    manifest = prepared(store)
    a, b = two_blocks(manifest)
    words_a, words_b = a.flat_text.strip()[:16], b.flat_text.strip()[:16]

    combine_blocks(manifest, 1, [a.id, b.id], store=store, target_lang="es")
    export = rebuild_export_only(manifest, store=store, target_lang="es")
    doc = fitz.open(stream=store.get_bytes(export.export_key), filetype="pdf")
    try:
        text = " ".join(doc[0].get_text().split())
    finally:
        doc.close()

    assert "ES[" in text
    # Their words are left only inside the one combined translation.
    assert text.count(words_a) == 1 and text.count(words_b) == 1


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_boxes_is_refused(store, translator, count):
    manifest = prepared(store)
    ids = [b.id for b in editable_blocks(manifest.page(0))[:count]]
    with pytest.raises(EditError):
        combine_blocks(manifest, 1, ids, store=store, target_lang="es")
    assert translator == []


def test_a_shape_is_not_combined(store, translator):
    manifest = prepared(store)
    a, _b = two_blocks(manifest)
    (shape,) = lines_of(manifest, store, ["inside a shape"])
    shape.manual_shape = "rect"
    with pytest.raises(EditError):
        combine_blocks(manifest, 1, [a.id, shape.id], store=store, target_lang="es")


def test_the_route_combines(client, translator):
    from test_api import _translate, _upload, _wait_for_stage, _wait_terminal

    doc_id = _upload(client, "01_clean_worksheet.pdf")["document_id"]
    _wait_for_stage(client, doc_id)
    _translate(client, doc_id)
    _wait_terminal(client, doc_id)
    blocks = client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]
    pair = [b["id"] for b in blocks if b["editable"]][:2]

    r = client.post(f"/api/documents/{doc_id}/pages/1/blocks/combine", json={"object_ids": pair})

    assert r.status_code == 200, r.text
    after = {b["id"]: b for b in client.get(f"/api/documents/{doc_id}/blocks").json()["pages"][0]["blocks"]}
    # Listed but suppressed, as Delete leaves one: the editor hides it.
    assert after[pair[1]]["suppressed"] and not after[pair[0]]["suppressed"]
