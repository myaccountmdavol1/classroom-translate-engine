"""Teacher recovery: previews do not edit; approval is exact, stale-safe, undoable."""
import json
import fitz
import pytest
from test_edit import prepared, first_editable
from ct_engine.pipeline.recovery import preview_repair, accept_repair, keep_original
from ct_engine.pipeline.edit import EditError, undo
from ct_engine.model.enums import TranslationStatus
from ct_engine.translate.protection import mask_text, restore_text
from ct_engine.translate.value_checks import value_findings
from ct_engine.api.findings import group_findings, finding_keys
from ct_engine.model.document import DocumentManifest


def test_preview_accept_undo_persistence_and_manual_pin(store, monkeypatch):
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.translated_text = 'Nombre revisado: ______________________'
    obj.manually_edited = True
    obj.translation_status = TranslationStatus.MANUAL_EDIT
    monkeypatch.setattr('ct_engine.translate.one_off.translate_one', lambda *a, **kw: pytest.fail('Must reuse translation'))
    state = manifest.model_dump_json()
    assets = {key: store.get_bytes(key) for key in store.list_prefix('documents/edit')}
    preview = preview_repair(manifest, 1, obj.id, store=store, target_lang='es')
    assert manifest.model_dump_json() == state
    assert all(store.get_bytes(key) == data for key, data in assets.items())
    assert preview['translation'] == obj.translated_text
    accept_repair(manifest, 1, obj.id, preview['token'], store=store, target_lang='es')
    assert obj.retry_background and obj.manually_edited
    saved = DocumentManifest.model_validate_json(manifest.model_dump_json())
    assert saved.page(0).find_text_object(obj.id).retry_background
    undo(saved, 1, store=store, target_lang='es')
    assert not saved.page(0).find_text_object(obj.id).retry_background
    assert saved.page(0).find_text_object(obj.id).translated_text == 'Nombre revisado: ______________________'
    assert store.get_bytes(manifest.source_key) == assets[manifest.source_key]


def test_preview_refuses_a_repair_that_would_drop_an_answer_blank(store):
    manifest = prepared(store)
    obj = first_editable(manifest)  # 'Name: ______'
    obj.translated_text = 'Una traducción revisada'
    obj.translation_status = TranslationStatus.MANUAL_EDIT
    with pytest.raises(EditError, match='reliable repair'):
        preview_repair(manifest, 1, obj.id, store=store, target_lang='es')


def test_stale_preview_cannot_overwrite_an_edit(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    preview = preview_repair(manifest, 1, obj.id, store=store, target_lang='es')
    manifest.page(0).revision += 1
    with pytest.raises(EditError, match='changed'):
        accept_repair(manifest, 1, obj.id, preview['token'], store=store, target_lang='es')
    assert not obj.retry_background


def test_keep_original_is_unresolved_even_when_reviewed_and_undoable(store):
    manifest = prepared(store)
    obj = first_editable(manifest)
    keep_original(manifest, 1, obj.id, store=store, target_lang='es')
    groups = group_findings(manifest)
    manifest.reviewed_findings = list(finding_keys(groups))
    assert any(g['code'] == 'original_kept' for g in group_findings(manifest))
    undo(manifest, 1, store=store, target_lang='es')
    assert not obj.keep_original
    assert not any(g['code'] == 'original_kept' for g in group_findings(manifest))


def test_clock_and_numeric_date_protection():
    source = 'Meet at 6:30 p.m. on 09/24/2026 until 8:32 p.m.'
    masked = mask_text(source)
    assert [s.category for s in masked.spans] == ['time', 'date', 'time']
    assert restore_text(masked.masked_text, masked.spans).text == source


def test_time_and_number_checks_do_not_guess(store):
    manifest = prepared(store)
    page = manifest.page(0)
    obj = first_editable(manifest)
    obj.lines[0].spans[0].text = 'Meet at 630 p.m.'
    obj.lines = [obj.lines[0]]
    obj.lines[0].spans = [obj.lines[0].spans[0]]
    obj.translated_text = 'A las 6:30 p.m.'
    codes = {f.code for f in value_findings(page) if obj.id in f.object_ids}
    assert codes == {'source_time_uncertain', 'numeric_content_changed'}
    assert obj.flat_text == 'Meet at 630 p.m.'
    obj.approved_source_text = 'Meet at 6:30 p.m.'
    assert not [f for f in value_findings(page) if obj.id in f.object_ids]


def paper_sample(background=False, rule=False):
    from ct_engine.ingest.pdf_parser import PdfParser
    pdf = fitz.open()
    p = pdf.new_page(width=400, height=250)
    if background:
        p.draw_rect(fitz.Rect(38, 38, 340, 100), fill=(.4, .7, .9))
    p.insert_text((40, 60), 'Our conferences will begin at noon.', fontsize=14, fontname='hebo')
    with PdfParser().open_bytes(pdf.tobytes()) as parsed:
        obj = parsed.parse_page(0).text_objects[0]
    if rule:
        p.draw_line((40, 55), (300, 55), width=1)
    return pdf, obj


def test_paper_retry_accepts_text_but_refuses_colour_and_crossing_rule():
    from ct_engine.reconstruct.builder import estimate_paper_background
    for background, rule, expected in [(False, False, True), (True, False, False), (False, True, False)]:
        pdf, obj = paper_sample(background, rule)
        try:
            assert (estimate_paper_background(pdf[0], obj) is not None) == expected
        finally:
            pdf.close()


def test_failed_repair_leaves_state_and_page_pixels_alone(store, monkeypatch):
    manifest = prepared(store)
    obj = first_editable(manifest)
    from ct_engine.model.enums import TextSource
    obj.source = TextSource.OCR
    state = manifest.model_dump_json()
    original = store.get_bytes(manifest.page(0).renders.translated_preview_key)
    monkeypatch.setattr('ct_engine.reconstruct.builder.estimate_background', lambda *a, **k: (None, 100))
    monkeypatch.setattr('ct_engine.reconstruct.builder.estimate_paper_background', lambda *a, **k: None)
    monkeypatch.setattr('ct_engine.reconstruct.builder.estimate_scanned_background', lambda *a, **k: None)
    with pytest.raises(EditError, match='reliable repair'):
        preview_repair(manifest, 1, obj.id, store=store, target_lang='es')
    assert manifest.model_dump_json() == state
    assert store.get_bytes(manifest.page(0).renders.translated_preview_key) == original


@pytest.mark.parametrize("source,translation,flagged", [
    # "double" written as a numeral: more digits, none lost. Real packet.
    ("What can you double and then add 5 to get 13?", "什么数乘以2后再加5等于13？", False),
    ("Find half of 12.", "求12的一半。", False),
    # A number dropped or changed is still caught.
    ("Add 5 to get 13.", "加5得到31。", True),
    ("Add 5 to get 13.", "加5。", True),
    # Small numbers written as words are a correct translation.
    ("Jose bought 3 CDs.", "José compró tres CD.", False),
    ("Solving 2-Step Equations", "Resolución de ecuaciones en dos pasos", False),
])
def test_numbers_flagged_only_when_the_source_ones_are_lost(store, source, translation, flagged):
    manifest = prepared(store)
    obj = first_editable(manifest)
    obj.approved_source_text = source
    obj.translated_text = translation
    codes = {f.code for f in value_findings(manifest.page(0)) if obj.id in f.object_ids}
    assert ("numeric_content_changed" in codes) is flagged
