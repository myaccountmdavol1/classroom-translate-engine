"""Writing the page editor's work back into the teacher's Word file.

A Word file is shown and edited as pages (see ingest/word_render.py), but
she uploaded a .docx and gets a .docx back. Each Word paragraph is matched
to the page blocks its words became, by aligning the two word streams in
reading order, and receives their translation -- including everything she
did in the editor: English she corrected, text she removed, colours.

When one page block holds the words of two Word paragraphs (the page reader
joined them) the blocks cannot be split back, so that paragraph is
translated on its own. Such translations are kept, so a later download does
not translate them again.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher

from ..fonts import catalog
from ..model.enums import TextSource, TranslationStatus
from ..reconstruct.docx_builder import rebuild_docx
from ..storage import keys as K
from ..util.hashing import sha256_bytes
from ..util.logging import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
# A list number or bullet the page shows but Word draws from its numbering:
# written into the paragraph it would appear twice ("1. 1. Abre...").
_MARKER = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\([a-zA-Z]\)|[a-hA-H][.)]|[•●▪◦‣o\-–])\s+")
# Words of a paragraph that must be found in its blocks for them to stand in
# for it. Below this the match is a coincidence of common words.
MIN_COVERAGE = 0.8


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _block_output(obj) -> str | None:
    """What a page block says now, after everything done in the editor."""
    if obj.suppressed:
        return ""
    if obj.keep_original or obj.translation_status in (
        TranslationStatus.PROTECTED, TranslationStatus.NOT_REQUIRED,
    ):
        return obj.flat_text
    return obj.translated_text


def match_paragraphs(units, blocks):
    """For each Word paragraph, the page blocks that hold exactly its words.

    Returns {unit index: [block, ...]} for paragraphs whose words all sit in
    blocks belonging to them alone; the rest are left out.
    """
    left, left_owner = [], []
    for i, unit in enumerate(units):
        for w in _words(unit.text):
            left.append(w)
            left_owner.append(i)
    right, right_owner = [], []
    for j, block in enumerate(blocks):
        for w in _words(block.text):
            right.append(w)
            right_owner.append(j)

    hits: dict[int, Counter] = {}
    block_units: dict[int, set[int]] = {}
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            u, blk = left_owner[a + k], right_owner[b + k]
            hits.setdefault(u, Counter())[blk] += 1
            block_units.setdefault(blk, set()).add(u)

    unit_words = Counter(left_owner)
    matched: dict[int, list] = {}
    for u, per_block in hits.items():
        shared = any(len(block_units[blk]) > 1 for blk in per_block)
        if shared or sum(per_block.values()) < MIN_COVERAGE * unit_words[u]:
            continue
        matched[u] = [blocks[j] for j in sorted(per_block)]
    return matched


def _touched(block) -> bool:
    """Did she change this block in the editor?"""
    return bool(
        block.approved_source_text is not None
        or block.manually_edited
        or block.suppressed
        or block.keep_original
        or block.manual_color
        or block.manual_underline
        or block.manual_highlight
        or block.manual_list
        or block.manual_font_family
        or block.translation_status is TranslationStatus.MANUAL_EDIT
    )


def _strip_marker(unit_text: str, block, text: str) -> str:
    if _MARKER.match(block.text) and not _MARKER.match(unit_text):
        return _MARKER.sub("", text, count=1)
    return text


def _office_classes(manifest):
    if manifest.office_format == "pptx":
        from ..ingest.pptx_parser import PptxParser as Parser
        from ..reconstruct.pptx_builder import rebuild_pptx as rebuild
    else:
        from ..ingest.docx_parser import DocxParser as Parser
        rebuild = rebuild_docx
    return Parser, rebuild


def _alone_key(unit) -> str:
    return sha256_bytes(unit.text.strip().encode())[:16]


def _paragraphs(manifest, store):
    """The file, its parsed pages, its paragraphs, and which page blocks
    stand in for each."""
    Parser, _ = _office_classes(manifest)
    source = store.get_bytes(manifest.office_source_key)
    with Parser().open_bytes(source) as doc:
        word_pages = [doc.parse_page(i) for i in range(doc.page_count)]
    units = [u for p in word_pages for u in p.text_objects if u.anchor and u.text.strip()]

    # Text in pictures is not a Word paragraph; only the page's own text is.
    # Boxes she made in the editor (added, duplicated or pasted) have no
    # paragraph either, and are left out here on purpose: one drawn inside
    # a picture reaches the file through that picture (word_pictures), and
    # one elsewhere shows on the pages only. The editor tells her so.
    blocks = [
        t
        for p in manifest.pages
        for t in sorted(p.text_objects, key=lambda t: t.reading_index)
        if t.source is TextSource.NATIVE and not t.is_manual_block
    ]
    return source, word_pages, units, match_paragraphs(units, blocks)


def _translate_alone(manifest, *, store, target_lang, units, matched, translate=True):
    """Translate the paragraphs that are translated by themselves, keeping
    each in the cache. Returns the cache and which paragraphs those are.
    With `translate` False no translator is called: only the cache is read."""
    from ..translate.one_off import OneOffError, translate_many

    cache_key = K.doc_prefix(manifest.document_id) + f"/export/word-alone-v2-{target_lang}.json"
    try:
        alone = json.loads(store.get_text(cache_key))
    except Exception:
        alone = {}

    # Pass 1: which paragraphs are translated by themselves.
    wrapped: set[int] = set()   # split over blocks only by the page's wrapping
    unmatched: set[int] = set()  # joined with another paragraph on the page
    for i, unit in enumerate(units):
        found = matched.get(i)
        if not found:
            unmatched.add(i)
        elif (len(found) > 1 or "\n" in unit.text.strip()) and not any(
            _touched(b) for b in found
        ):
            # Split over several page blocks only because the page wrapped
            # it ("...open the drop-down menu and" / "select From a custom
            # URL."): translated whole it reads as one sentence again.
            wrapped.add(i)

    # All of them in one request rather than one per line, one after
    # another, inside the download: on a long file that kept the teacher
    # waiting a minute and spent the allowance twice when she clicked again.
    # A line break of its own survives only translated line by line, so a
    # paragraph is sent as its lines and put back together.
    todo = [
        i for i in sorted(wrapped | unmatched) if _alone_key(units[i]) not in alone
    ]
    lines: list[str] = []
    for i in todo:
        lines.extend(line for line in units[i].text.split("\n") if line.strip())
    translated: list = [None] * len(lines)
    if lines and translate:
        try:
            translated = translate_many(
                lines, target_lang=target_lang,
                source_lang=manifest.analysis.detected_language,
            )
        except OneOffError as exc:
            log.warning("Word export: paragraphs could not be translated: %s", exc.diagnostic)
            translated = [None] * len(lines)
        except Exception as exc:
            # Whatever stopped the translator (in the browser it has run out
            # of memory), the paragraphs keep their words and are counted,
            # so the page and the download say so rather than nothing.
            log.warning("Word export: the translator broke on the paragraphs: %r", exc)
            translated = [None] * len(lines)
    alone_changed = False
    at = 0
    for i in todo:
        out, complete = [], True
        for line in units[i].text.split("\n"):
            if not line.strip():
                out.append(line)
                continue
            result = translated[at]
            at += 1
            if result is None:
                complete = False
            else:
                out.append(result.text)
        # A paragraph with any line missing is not kept: half English, half
        # translated would read as finished.
        if complete:
            alone[_alone_key(units[i])] = "\n".join(out)
            alone_changed = True

    if alone_changed:
        store.put_text(cache_key, json.dumps(alone))
    return alone, wrapped, unmatched


def translate_word_paragraphs(
    manifest, *, store, target_lang: str, translate: bool = True,
) -> int:
    """Translate, ahead of the download, the paragraphs the download
    translates by themselves, and keep them for it.

    Run by the translation job: in the browser the translators are made
    ready for a job, and a download after the page is reloaded is not one,
    so translated only then the whole file kept its English. Returns how
    many paragraphs still have no translation and would keep their
    original words. With `translate` False it only counts, from the cache.
    """
    _, _, units, matched = _paragraphs(manifest, store)
    alone, _, unmatched = _translate_alone(
        manifest, store=store, target_lang=target_lang, units=units, matched=matched,
        translate=translate,
    )
    return sum(1 for i in unmatched if _alone_key(units[i]) not in alone)


def build_office_export(
    manifest, *, store, target_lang: str, translated_pdf: bytes | None = None,
):
    """The teacher's .docx or .pptx with every paragraph translated as on the
    pages. Slides work exactly as Word does: each slide's paragraphs are
    matched to the blocks they became on the slide's page."""
    _, rebuild = _office_classes(manifest)
    source, word_pages, units, matched = _paragraphs(manifest, store)
    # The job has translated these already (translate_word_paragraphs, which
    # counts them on the manifest). When it did, the download reads its
    # translations only and never calls a translator: in the browser,
    # translating outside a job crashed the engine and the download never
    # finished, and the job's count is what the teacher was told. A
    # paragraph it could not translate keeps its words, counted as today.
    # A document from before the job did this (no count) translates here.
    job_did_it = getattr(manifest, "word_paragraphs_kept", None) is not None
    alone, wrapped, unmatched = _translate_alone(
        manifest, store=store, target_lang=target_lang, units=units, matched=matched,
        translate=not job_did_it,
    )

    # Pass 2: write each paragraph's translation.
    kept_original = 0
    for i, unit in enumerate(units):
        found = matched.get(i)
        if i in wrapped or i in unmatched:
            text = alone.get(_alone_key(unit))
            if text is not None:
                unit.translated_text = text
                unit.translation_status = TranslationStatus.TRANSLATED
                continue
            if i in unmatched:
                # Nothing on the pages stands for it, so it keeps its words.
                kept_original += 1
                continue
            # Translated by itself it failed; the page's own blocks still
            # hold its translation, which is what the editor showed.
            log.warning("Word export: paragraph %d uses its page blocks' translation", i)
        parts = [_block_output(b) for b in found]
        if all(p is None for p in parts):
            continue
        kept = [
            _strip_marker(unit.text, b, p) if k == 0 else p
            for k, (b, p) in enumerate(zip(found, parts))
            if p
        ]
        unit.translated_text = " ".join(x.strip() for x in kept)
        unit.translation_status = TranslationStatus.TRANSLATED
        if len(found) == 1:
            b = found[0]
            unit.manual_color = b.manual_color
            unit.manual_font_size = None  # page sizes are not Word sizes
            # Underline and highlight belong to the words, so they carry, on
            # the one run that holds the paragraph's translation. Where the
            # words sit in the box, alignment, justify and line spacing do
            # not: a Word paragraph has no box, and those would need edits
            # to each paragraph's own properties, which alignment never had
            # either. The editor says so for Word and PowerPoint files.
            unit.manual_underline = b.manual_underline
            unit.manual_highlight = b.manual_highlight
            # Her list becomes a real one where the file can hold it, and
            # each line keeps its marker where it cannot (office_lists).
            # Her indents without a list stay on the pages, as alignment
            # does.
            unit.manual_list = b.manual_list
            unit.manual_list_style = b.manual_list_style
            unit.manual_list_start = b.manual_list_start
            unit.manual_indents = b.manual_indents
            # A font from the catalog is named in the file only where it
            # has the language's letters; the older styles are not carried.
            if catalog.covers(b.manual_font_family, target_lang):
                unit.manual_font_family = b.manual_font_family
        unit.suppressed = all(b.suppressed for b in found)

    log.info(
        "Word export: %d of %d paragraphs from the pages, %d translated alone",
        len(matched), len(units), len(units) - len(matched),
    )
    if kept_original:
        log.warning(
            "Word export: %d paragraph(s) could not be translated and keep "
            "their original words", kept_original,
        )
    result = rebuild(source, word_pages)
    # Reported with the file, never only logged: what the editor showed and
    # what the saved file holds must not differ without the teacher knowing.
    # `paragraphs_kept_original` counts paragraphs left in the source
    # language; `pictures_kept_pages` lists, once per picture, the page of
    # each picture whose words are still the original ones.
    result.paragraphs_kept_original = kept_original
    result.pictures_kept_pages = []
    if translated_pdf is not None:
        from .word_pictures import (
            pages_with_labelled_pictures,
            replace_pictures,
            translated_pictures,
        )

        try:
            data, pictures = replace_pictures(
                result.data, translated_pictures(manifest, translated_pdf, store),
            )
        except Exception:
            # The words are the document; a picture that fails stays as it
            # was -- every one of them, and each is reported.
            log.exception("Translated pictures could not be put into the Word file")
            result.pictures_kept_pages = pages_with_labelled_pictures(manifest)
        else:
            result.data = data
            result.pictures_kept_pages = sorted(pictures.kept_pages)
            log.info(
                "Word export: %d picture(s) translated, %d left as they were",
                pictures.translated, pictures.unmatched,
            )
    return result


# The name it had before slides were added.
build_word_export = build_office_export


# -- the finished file, built by the job ---------------------------------------
#
# In the browser, a download after the page was reloaded crashed the engine
# building a long book's file (36 pictures); the same work inside the job
# succeeds. So the job builds the file once and stores it, and the download
# hands it over -- but only while nothing it holds has changed since.

# Part of every stored file's stamp. Bump it whenever what word_export or
# word_pictures writes into the file changes, so a file stored by an older
# engine is built again rather than handed over.
_WORD_EXPORT_VERSION = 1


def office_export_stamp(manifest, *, target_lang: str, export_key: str | None) -> str:
    """What the document looked like, for the Word file, when it was built.

    Every page's fingerprint, the one the export's own rebuild uses to find
    edited pages: every text object, picture and region on the page, all
    but previews, checks and counters. An edit by any route -- a block's
    words, a box drawn on a picture, one removed or moved -- changes it,
    because it compares what is there rather than trusting each route to
    mark it. With it the job's paragraph count (the download writes its
    translations only when the job counted them), the file read, the
    language and the translated pages it was built from.
    """
    from .reconstruct_stage import _page_fingerprint

    raw = json.dumps(
        {
            "version": _WORD_EXPORT_VERSION,
            "target_lang": target_lang,
            "export_key": export_key,
            "office_source_key": manifest.office_source_key,
            "word_paragraphs_kept": getattr(manifest, "word_paragraphs_kept", None),
            "pages": [_page_fingerprint(p) for p in manifest.pages],
        },
        sort_keys=True,
    )
    return sha256_bytes(raw.encode("utf-8"))


def store_office_export(
    manifest, *, store, target_lang: str, export_key: str, job_id: str | None = None
) -> bool:
    """Build the finished Word file and keep it for the download.

    Never raises: a file that could not be built or kept is simply built
    again at download, as before. The old record is replaced first, so a
    failure here can never leave an earlier build to be handed over.

    In the browser a crash while building (WebAssembly running out of
    memory) cannot be caught: the worker dies and the job resumes into this
    same step, to crash again until it fails. So a mark naming the job is
    kept in the record's place while building; a job that finds its own
    mark crashed here before, and leaves the file to the download."""
    doc_id = manifest.document_id
    meta_key = K.word_final_meta_key(doc_id, target_lang)
    if job_id is not None:
        try:
            previous = json.loads(store.get_text(meta_key))
        except Exception:
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("building")
            and previous.get("job_id") == job_id
        ):
            log.warning(
                "The Word file was being built when this job stopped; "
                "it is left to the download."
            )
            try:
                store.delete(meta_key)
            except Exception:
                pass
            return False
    try:
        store.delete(meta_key)
        if job_id is not None:
            store.put_text(
                meta_key,
                json.dumps({"building": True, "job_id": job_id,
                            "target_lang": target_lang}),
                content_type="application/json",
            )
        stamp = office_export_stamp(
            manifest, target_lang=target_lang, export_key=export_key
        )
        translated_pdf = store.get_bytes(export_key)
        result = build_office_export(
            manifest, store=store, target_lang=target_lang,
            translated_pdf=translated_pdf,
        )
        # The translated pages are the largest thing held; let them go
        # before the file is written.
        del translated_pdf
        meta = {
            "target_lang": target_lang,
            "stamp": stamp,
            "pictures_kept_pages": list(result.pictures_kept_pages or []),
            "paragraphs_kept_original": int(result.paragraphs_kept_original or 0),
        }
        store.put_bytes(K.word_final_key(doc_id, target_lang), result.data)
        del result
        # Written last: a record means the file beside it is complete.
        store.put_text(meta_key, json.dumps(meta), content_type="application/json")
    except Exception as exc:
        log.warning("The finished Word file could not be kept: %r", exc)
        try:
            store.delete(meta_key)
        except Exception:
            pass
        return False
    return True


def stored_office_export(manifest, *, store, target_lang: str, export_key: str | None):
    """The file the job built and its record, or None when there is none,
    it is in another language, or anything it holds has changed since."""
    doc_id = manifest.document_id
    try:
        meta = json.loads(store.get_text(K.word_final_meta_key(doc_id, target_lang)))
    except Exception:
        return None
    if meta.get("building") or meta.get("target_lang") != target_lang:
        # A build that never finished is never handed over.
        return None
    try:
        stamp = office_export_stamp(
            manifest, target_lang=target_lang, export_key=export_key
        )
    except Exception as exc:
        log.warning("Could not tell whether the stored Word file is current: %r", exc)
        return None
    if meta.get("stamp") != stamp:
        return None
    try:
        data = store.get_bytes(K.word_final_key(doc_id, target_lang))
    except Exception:
        return None
    return data, meta
