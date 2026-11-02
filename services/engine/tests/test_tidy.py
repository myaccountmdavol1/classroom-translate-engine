"""Fix layout: flagged blocks given more room, kept only where it helped."""

from __future__ import annotations

import fitz
import pytest

from ct_engine.analysis.block_split import split_page_blocks
from ct_engine.analysis.classify import classify_page
from ct_engine.analysis.reading_order import apply_reading_order
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.model.document import DocumentManifest
from ct_engine.pipeline.edit import push_history, reset_edit, undo
from ct_engine.pipeline.reconstruct_stage import reconstruct_and_export, verify_page
from ct_engine.pipeline.tidy import (
    LAYOUT_CODES,
    FixLayoutResult,
    fix_layout,
    share_sizes,
    tidy_page,
)
from ct_engine.pipeline.translate import translate_page
from ct_engine.util.hashing import sha256_bytes
from test_translate_pipeline import EchoProvider
from test_copy_paste import _api_doc
from test_edit import first_editable, prepared


def test_tidy_fields_are_undone(store):
    manifest = prepared(store)
    page = manifest.page(0)
    obj = first_editable(manifest)
    push_history(page)
    obj.tidy_room = True
    obj.tidy_font_size = 9.0
    undo(manifest, 1, store=store, target_lang="es")
    obj = page.find_text_object(obj.id)
    assert obj.tidy_room is False
    assert obj.tidy_font_size is None


def test_an_old_snapshot_without_tidy_fields_clears_them(store):
    manifest = prepared(store)
    page = manifest.page(0)
    obj = first_editable(manifest)
    push_history(page)
    for saved in page.edit_history[-1]["blocks"].values():
        saved.pop("tidy_room", None)
        saved.pop("tidy_font_size", None)
    obj.tidy_room = True
    obj.tidy_font_size = 9.0
    undo(manifest, 1, store=store, target_lang="es")
    obj = page.find_text_object(obj.id)
    assert obj.tidy_room is False and obj.tidy_font_size is None


def test_reset_clears_tidy_room(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.tidy_room = True
    obj.tidy_font_size = 9.0
    reset_edit(manifest, 1, obj.id, store=store, target_lang="es")
    obj = manifest.page(0).find_text_object(obj.id)
    assert obj.tidy_room is False and obj.tidy_font_size is None



LONG = "Smoke alarms are installed on every level of the home."


def _form_pdf() -> bytes:
    """One long item with empty space below it and a rule to its right,
    and an unrelated line far down the page."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), LONG, fontsize=11)
    page.insert_text((72, 520), "Fire extinguisher is located in the kitchen.", fontsize=11)
    page.draw_line((430, 80), (430, 560), width=1)
    data = doc.tobytes()
    doc.close()
    return data


def _translated(store, data, transform):
    store.put_bytes("documents/tidy/source.pdf", data)
    manifest = DocumentManifest(
        document_id="tidy", filename="form.pdf", content_type="application/pdf",
        byte_size=len(data), sha256=sha256_bytes(data),
        source_key="documents/tidy/source.pdf",
    )
    with PdfParser().open_bytes(data) as doc:
        for i in range(doc.page_count):
            page = doc.parse_page(i)
            split_page_blocks(page)
            apply_reading_order(page)
            classify_page(page, is_first_page=(i == 0))
            translate_page(page, EchoProvider(transform), source_lang="en", target_lang="es")
            manifest.pages.append(page)
    manifest.analysis.page_count = len(manifest.pages)
    export = reconstruct_and_export(manifest, store=store, target_lang="es")
    # The job runner verifies each page after the export; do the same so the
    # page carries the findings a teacher would see.
    for page in manifest.pages:
        verify_page(page, list(export.findings_by_page.get(page.index, [])))
    return manifest


def _long_text(t):
    return (t + " ") * 6 if t.startswith("Smoke") else t


def _layout_ids(page):
    return {i for f in page.qa.findings if f.code in LAYOUT_CODES for i in f.object_ids}


def _smoke(page):
    return next(o for o in page.text_objects if o.flat_text.startswith("Smoke"))


def test_a_squeezed_item_gets_its_size_back(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    smoke = _smoke(page)
    assert smoke.id in _layout_ids(page)      # the first pass shrank it

    assert tidy_page(manifest, 1, store=store, target_lang="es") is True

    smoke = page.find_text_object(smoke.id)
    assert smoke.tidy_room is True
    assert smoke.id not in _layout_ids(page)
    other = next(o for o in page.text_objects if o.flat_text.startswith("Fire"))
    assert smoke.drawn_bbox.intersection_area(other.bbox) == 0
    assert smoke.drawn_bbox.x1 <= 430


def test_one_undo_puts_the_page_back(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    before = {o.id: o.translated_font_size for o in page.text_objects}
    assert tidy_page(manifest, 1, store=store, target_lang="es") is True
    undo(manifest, 1, store=store, target_lang="es")
    assert {o.id: o.translated_font_size for o in page.text_objects} == before
    assert not any(o.tidy_room for o in page.text_objects)


def test_a_second_run_changes_nothing(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    tidy_page(manifest, 1, store=store, target_lang="es")
    history = len(manifest.page(0).edit_history)
    assert tidy_page(manifest, 1, store=store, target_lang="es") is False
    assert len(manifest.page(0).edit_history) == history


def test_a_page_with_nothing_flagged_is_left_alone(store):
    # Identity transform: the page is checked, nothing runs long.
    manifest = _translated(store, _form_pdf(), lambda t: t)
    page = manifest.page(0)
    assert not _layout_ids(page)
    revision = page.revision
    assert tidy_page(manifest, 1, store=store, target_lang="es") is False
    assert page.revision == revision
    assert page.edit_history == []


def _run_in_pdf() -> bytes:
    """The long item, plus a bold run-in label printed over the start of its
    paragraph: an overlap more room cannot clear."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), LONG, fontsize=11)
    page.draw_line((430, 80), (430, 560), width=1)
    page.insert_text((72, 300), "How to use", fontsize=9, fontname="hebo")
    page.insert_text(
        (72, 300), "            Mark Yes, No or N/A for each item and write a note.",
        fontsize=9,
    )
    page.insert_text((72, 311), "Carry each No to the summary at the end of the form.", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def test_a_block_it_could_not_help_is_left_as_it_was(store):
    manifest = _translated(store, _run_in_pdf(), _long_text)
    page = manifest.page(0)
    overlap = next(f for f in page.qa.findings if f.code == "collision_text")
    boxes = {i: page.find_text_object(i).drawn_bbox for i in overlap.object_ids}

    assert tidy_page(manifest, 1, store=store, target_lang="es") is True

    assert _smoke(page).id not in _layout_ids(page)   # the item it could fix
    for i, box in boxes.items():                      # the overlap it could not
        obj = page.find_text_object(i)
        assert obj.tidy_room is False
        assert obj.drawn_bbox == box


def _seed_history(manifest, page):
    """Give the page one undo step, one redo step and a checked-off item."""
    push_history(page)
    page.redo_history.append(dict(page.edit_history[-1]))
    manifest.reviewed_findings.append(f"text_overflow:{page.number}:x")
    return (
        list(page.edit_history), list(page.redo_history),
        list(manifest.reviewed_findings),
    )


def test_a_try_that_helps_nothing_costs_her_nothing(store, monkeypatch):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    history, redo, reviewed = _seed_history(manifest, page)
    # The builder's room is not used, so the page rebuilds but nothing improves.
    monkeypatch.setattr("ct_engine.pipeline.tidy.share_sizes", lambda *a: False)
    monkeypatch.setattr("ct_engine.pipeline.tidy._rerender", _no_change_rerender)

    assert tidy_page(manifest, 1, store=store, target_lang="es") is False
    assert page.edit_history == history
    assert page.redo_history == redo
    assert manifest.reviewed_findings == reviewed
    assert not any(o.tidy_room for o in page.text_objects)


def _no_change_rerender(manifest, page, obj, *, store, target_lang):
    # Rebuild exactly as before, whatever tidy_room says.
    from ct_engine.pipeline import edit
    saved = {o.id: o.tidy_room for o in page.text_objects}
    for o in page.text_objects:
        o.tidy_room = False
    try:
        return edit._rerender(manifest, page, obj, store=store, target_lang=target_lang)
    finally:
        for o in page.text_objects:
            o.tidy_room = saved[o.id]


def test_a_failed_rebuild_is_put_back_once(store, monkeypatch):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    history, redo, reviewed = _seed_history(manifest, page)
    from ct_engine.pipeline import edit
    from ct_engine.pipeline.edit import EditError

    calls = {"n": 0}

    def failing_first(manifest, page, obj, *, store, target_lang):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EditError("rebuild failed")
        return edit._rerender(manifest, page, obj, store=store, target_lang=target_lang)

    monkeypatch.setattr("ct_engine.pipeline.tidy._rerender", failing_first)
    with pytest.raises(EditError, match="rebuild failed"):
        tidy_page(manifest, 1, store=store, target_lang="es")
    assert page.edit_history == history
    assert page.redo_history == redo
    assert manifest.reviewed_findings == reviewed
    assert not any(o.tidy_room for o in page.text_objects)


def test_siblings_share_the_smallest_size_only_when_it_is_close(store):
    manifest = prepared(store)
    page = manifest.page(0)
    column = [o for o in page.text_objects if o.translated_font_size][:3]
    assert len(column) == 3
    # A size no other block on the page has, so only these three group.
    for o in column:
        o.bbox = o.bbox.model_copy(update={"x0": 72.0})
        o.style.font_size = 31.0
        o.tidy_font_size = None
    column[0].translated_font_size = 28.5   # 92% of 31: close enough
    column[1].translated_font_size = 31.0
    column[2].translated_font_size = 31.0
    assert share_sizes(page, {column[0].id}) is True
    assert column[1].tidy_font_size == 28.5 and column[2].tidy_font_size == 28.5
    assert column[0].tidy_font_size is None

    for o in column:
        o.tidy_font_size = None
    column[0].translated_font_size = 20.0   # 65% of 31: would make new items
    assert share_sizes(page, {column[0].id}) is False
    assert all(o.tidy_font_size is None for o in column)


def _column(store):
    """Three single-line rows at one margin and size, stacked closely, the
    first drawn smaller (flagged)."""
    manifest = prepared(store)
    page = manifest.page(0)
    column = [o for o in page.text_objects if o.translated_font_size][:3]
    assert len(column) == 3
    for n, o in enumerate(column):
        o.lines = o.lines[:1]
        o.bbox = o.bbox.model_copy(
            update={"x0": 72.0, "x1": 300.0, "y0": 100.0 + n * 20, "y1": 114.0 + n * 20}
        )
        o.style.font_size = 31.0
        o.tidy_font_size = None
        o.translated_font_size = 31.0
    column[0].translated_font_size = 28.5
    return page, column


def test_multi_line_paragraphs_are_not_grouped(store):
    page, column = _column(store)
    for o in column:
        o.lines = o.lines * 2
    assert share_sizes(page, {column[0].id}) is False
    assert all(o.tidy_font_size is None for o in column)


def test_rows_far_apart_are_not_grouped(store):
    page, column = _column(store)
    for o in column[1:]:
        o.bbox = o.bbox.model_copy(update={"y0": o.bbox.y0 + 300, "y1": o.bbox.y1 + 300})
    assert share_sizes(page, {column[0].id}) is False
    assert all(o.tidy_font_size is None for o in column)


def test_stacked_one_line_rows_are_grouped(store):
    page, column = _column(store)
    assert share_sizes(page, {column[0].id}) is True
    assert column[1].tidy_font_size == 28.5


def _layout_key(page, finding):
    return f"{finding.code}:{page.number}:" + ",".join(sorted(finding.object_ids))


def test_a_kept_run_keeps_her_other_checked_off_items(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    # A reading item she checked off.
    reading = "source_time_uncertain:1:abc"
    manifest.reviewed_findings.append(reading)

    assert tidy_page(manifest, 1, store=store, target_lang="es") is True
    assert reading in manifest.reviewed_findings


def test_a_kept_run_asks_again_about_blocks_it_changed(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    # The block's own item is unchecked; a second item about the same block
    # was checked off. Fix layout changes the block, so that check goes.
    finding = next(f for f in page.qa.findings if f.code in LAYOUT_CODES)
    stale = f"collision_text:{page.number}:" + ",".join(sorted(finding.object_ids))
    manifest.reviewed_findings.append(stale)
    page.qa.findings.append(finding.model_copy(update={"code": "collision_text"}))

    assert tidy_page(manifest, 1, store=store, target_lang="es") is True
    assert stale not in manifest.reviewed_findings


def test_items_she_checked_off_are_not_tidied(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    for f in page.qa.findings:
        if f.code in LAYOUT_CODES:
            manifest.reviewed_findings.append(_layout_key(page, f))
    assert tidy_page(manifest, 1, store=store, target_lang="es") is False
    assert page.edit_history == []
    assert not any(o.tidy_room for o in page.text_objects)


def test_placing_a_box_clears_fix_layout_marks(store):
    from ct_engine.pipeline.edit import apply_layout
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.tidy_room = True
    obj.tidy_font_size = 9.0
    box = obj.bbox
    apply_layout(
        manifest, 1, obj.id,
        bbox={"x0": box.x0, "y0": box.y0, "x1": box.x1 + 5, "y1": box.y1 + 5},
        store=store, target_lang="es",
    )
    obj = manifest.page(0).find_text_object(obj.id)
    assert obj.tidy_room is False and obj.tidy_font_size is None


def test_an_unexpected_error_fails_only_that_page(store, monkeypatch):
    manifest = _translated(store, _form_pdf(), _long_text)
    page = manifest.page(0)
    history, redo, reviewed = _seed_history(manifest, page)

    def boom(*a, **k):
        raise RuntimeError("surprise")

    monkeypatch.setattr("ct_engine.pipeline.tidy.share_sizes", boom)
    result = fix_layout(manifest, [1], store=store, target_lang="es")
    assert result.failed == [1] and result.changed == []
    assert page.edit_history == history
    assert page.redo_history == redo
    assert manifest.reviewed_findings == reviewed
    assert not any(o.tidy_room for o in page.text_objects)


def test_fix_layout_reports_changed_and_failed_pages(store):
    manifest = _translated(store, _form_pdf(), _long_text)
    result = fix_layout(manifest, [1, 7], store=store, target_lang="es")
    assert [p.number for p in result.changed] == [1]
    assert result.failed == [7]



# --- The route ----------------------------------------------------------------

def test_fix_layout_route(client):
    doc_id, _page = _api_doc(client)
    res = client.post(f"/api/documents/{doc_id}/fix-layout", json={"pages": [1, 99]})
    assert res.status_code == 200
    body = res.json()
    assert body["failed"] == [99]
    assert body["changed"] == []


def test_fix_layout_route_success_path(client, monkeypatch):
    import ct_engine.api.routes as routes

    def stub(manifest, pages, *, store, target_lang):
        page = manifest.page(0)
        push_history(page)
        return FixLayoutResult(changed=[page], failed=[])

    monkeypatch.setattr(routes, "fix_layout", stub)
    doc_id, _page = _api_doc(client)

    res = client.post(f"/api/documents/{doc_id}/fix-layout", json={"pages": [1]})
    assert res.status_code == 200
    body = res.json()
    assert [e["page_number"] for e in body["changed"]] == [1]
    entry = body["changed"][0]
    assert set(entry) == {"page_number", "revision", "can_undo", "can_redo"}
    assert entry["can_undo"] is True
    assert body["failed"] == []

    # The change was saved, not only reported.
    page_blocks = client.get(f"/api/documents/{doc_id}/pages/1/blocks").json()
    assert page_blocks["can_undo"] is True


def test_fix_layout_route_needs_pages(client):
    doc_id, _page = _api_doc(client)
    assert client.post(f"/api/documents/{doc_id}/fix-layout", json={"pages": []}).status_code == 422
