"""Preview a section repair without changing canonical page assets or state."""
from __future__ import annotations

import base64
import json
import time
import uuid
import fitz

from .edit import EditError, _editable_object, _rerender, push_history
from ..model.enums import TranslationStatus
from ..reconstruct.builder import rebuild_page
from .reconstruct_stage import verify_page

FIELDS = ('translated_text', 'machine_translated_text', 'translation_status',
          'retry_background', 'keep_original')
BLOCKING = {'scanned_text_not_replaced', 'font_unavailable', 'text_overflow',
            'protected_content_lost', 'untranslated_text'}


def proposal_key(manifest, number):
    # One bounded, private pending proposal per page; never included in exports.
    return f'documents/{manifest.document_id}/repair-proposals/{number}.json'


def preview_repair(manifest, number, object_id, *, store, target_lang):
    original_page = manifest.page(number - 1)
    if original_page is None or not original_page.has_fixed_layout:
        raise EditError('This section does not have a page preview.')
    page = original_page.model_copy(deep=True)
    obj = _editable_object(page, object_id)
    if obj.suppressed:
        raise EditError('Restore this text before repairing it.')
    obj.keep_original = False
    obj.retry_background = True
    if not obj.translated_text or obj.translation_status == TranslationStatus.FAILED:
        from ..translate.one_off import translate_one, OneOffError
        try:
            result = translate_one(obj.flat_text, target_lang=target_lang,
                                   source_lang=manifest.analysis.detected_language)
        except OneOffError as exc:
            raise EditError('We could not translate this section. Check the original wording and try again.') from exc
        if not result.protected_intact:
            raise EditError('A number or formula could not be kept. Check the original wording first.')
        obj.translated_text = obj.machine_translated_text = result.text
        obj.translation_only_text = None
        obj.translation_status = TranslationStatus.TRANSLATED
    with fitz.open(stream=store.get_bytes(manifest.source_key), filetype='pdf') as pdf:
        pdf_page = pdf[number - 1]
        clip = fitz.Rect(*obj.bbox.expanded(12).as_tuple()) & pdf_page.rect
        original = pdf_page.get_pixmap(clip=clip, dpi=144, alpha=False).tobytes('png')
        stats = rebuild_page(pdf_page, page, target_lang=target_lang)
        qa = verify_page(page, stats.findings)
        warnings = [f for f in qa.findings if not f.object_ids or object_id in f.object_ids]
        if any(f.code in BLOCKING or f.severity == 'error' for f in warnings):
            raise EditError('We could not make a reliable repair. Compare the original below and correct any misread words. Pictures and lines have been left untouched.')
        # Show the whole affected area, including any automatic text growth.
        if obj.render_bbox:
            clip |= fitz.Rect(*obj.render_bbox.expanded(12).as_tuple()) & pdf_page.rect
        proposed = pdf_page.get_pixmap(clip=clip, dpi=144, alpha=False).tobytes('png')
    token = uuid.uuid4().hex
    raw = obj.model_dump(mode='json')
    store.put_text(proposal_key(manifest, number), json.dumps({
        'token': token, 'revision': original_page.revision, 'object_id': object_id,
        'target': target_lang, 'expires': time.time() + 1800,
        'fields': {k: raw[k] for k in FIELDS},
    }))
    return {'token': token, 'original': obj.flat_text, 'translation': obj.translated_text,
            'original_image': 'data:image/png;base64,' + base64.b64encode(original).decode(),
            'proposed_image': 'data:image/png;base64,' + base64.b64encode(proposed).decode(),
            'warnings': [f.message for f in warnings]}


def accept_repair(manifest, number, object_id, token, *, store, target_lang):
    key = proposal_key(manifest, number)
    if not store.exists(key):
        raise EditError('That preview has expired. Choose Fix this section again.')
    proposal = json.loads(store.get_text(key))
    page = manifest.page(number - 1)
    if (page is None or proposal['revision'] != page.revision or proposal['token'] != token
        or proposal['object_id'] != object_id or proposal['target'] != target_lang
        or proposal['expires'] < time.time()):
        raise EditError('This page changed since the preview. Choose Fix this section again.')
    obj = _editable_object(page, object_id)
    push_history(page)
    for k, v in proposal['fields'].items():
        setattr(obj, k, TranslationStatus(v) if k == 'translation_status' else v)
    result = _rerender(manifest, page, obj, store=store, target_lang=target_lang)
    store.delete(key)
    return result


def keep_original(manifest, number, object_id, *, store, target_lang):
    page = manifest.page(number - 1)
    if page is None:
        raise EditError('That page could not be found.')
    obj = _editable_object(page, object_id)
    push_history(page)
    obj.keep_original = True
    return _rerender(manifest, page, obj, store=store, target_lang=target_lang)
