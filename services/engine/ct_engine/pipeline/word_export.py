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


def build_office_export(
    manifest, *, store, target_lang: str, translated_pdf: bytes | None = None,
):
    """The teacher's .docx or .pptx with every paragraph translated as on the
    pages. Slides work exactly as Word does: each slide's paragraphs are
    matched to the blocks they became on the slide's page."""
    from ..translate.one_off import OneOffError, translate_many

    if manifest.office_format == "pptx":
        from ..ingest.pptx_parser import PptxParser as Parser
        from ..reconstruct.pptx_builder import rebuild_pptx as rebuild
    else:
        from ..ingest.docx_parser import DocxParser as Parser
        rebuild = rebuild_docx

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
    matched = match_paragraphs(units, blocks)

    cache_key = K.doc_prefix(manifest.document_id) + f"/export/word-alone-v2-{target_lang}.json"
    try:
        alone = json.loads(store.get_text(cache_key))
    except Exception:
        alone = {}

    def alone_key(unit) -> str:
        return sha256_bytes(unit.text.strip().encode())[:16]

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
        i for i in sorted(wrapped | unmatched) if alone_key(units[i]) not in alone
    ]
    lines: list[str] = []
    for i in todo:
        lines.extend(line for line in units[i].text.split("\n") if line.strip())
    translated: list = []
    if lines:
        try:
            translated = translate_many(
                lines, target_lang=target_lang,
                source_lang=manifest.analysis.detected_language,
            )
        except OneOffError as exc:
            log.warning("Word export: paragraphs could not be translated: %s", exc.diagnostic)
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
            alone[alone_key(units[i])] = "\n".join(out)
            alone_changed = True

    # Pass 2: write each paragraph's translation.
    kept_original = 0
    for i, unit in enumerate(units):
        found = matched.get(i)
        if i in wrapped or i in unmatched:
            text = alone.get(alone_key(unit))
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

    if alone_changed:
        store.put_text(cache_key, json.dumps(alone))
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
