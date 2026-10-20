"""Blurry screenshots: whole lines, and no translation of a misreading.

The installation guide pastes ~70 dpi screenshots into its pages. Read at
150 dpi their captions came back as scraps ("also be", "RL whe") that were
translated alone and painted over the middle of sentences; one heading came
back in two halves ("Add Chrome app or ext" | "sion by ID").
"""

from __future__ import annotations

from conftest import corpus_file
from ct_engine.ingest.pdf_parser import PdfParser
from ct_engine.pipeline.image_text import _join_line_pieces, extract_image_text
from ct_engine.providers.ocr.base import OCRProvider, OCRRegion, OCRResult, OCRWord


def _region(text, x0, y0, confidences, *, height=20.0, char=10.0):
    words, x = [], x0
    for word, conf in zip(text.split(), confidences):
        words.append(OCRWord(text=word, bbox=(x, y0, x + char * len(word), y0 + height), confidence=conf))
        x += char * (len(word) + 1)
    mean = sum(c * len(w) for w, c in zip(text.split(), confidences)) / len(text.replace(" ", ""))
    return OCRRegion(text=text, bbox=(x0, y0, x - char, y0 + height), confidence=round(mean, 4), words=words)


class _Scripted(OCRProvider):
    name = "scripted"
    sends_data_externally = False

    def __init__(self, regions):
        self.regions = regions

    def supports_language(self, lang):
        return True

    def recognize(self, image_png, *, language=None, detect_handwriting=False, sparse=False):
        return OCRResult(regions=list(self.regions), provider=self.name)


def test_a_line_split_at_a_letter_gap_is_one_line_again():
    left = _region("Add Chrome app or ext", 10, 100, [0.95] * 5)
    right = _region("sion by ID", left.bbox[2] + 8, 101, [0.9] * 3)
    far = _region("Requests", 900, 100, [0.95])
    joined = _join_line_pieces([right, far, left])
    texts = sorted(r.text for r in joined)
    assert texts == ["Add Chrome app or ext sion by ID", "Requests"]


def _labels(regions):
    with PdfParser().open_bytes(corpus_file("04_science_diagram.pdf")) as doc:
        page = doc.parse_page(0)
        stats = extract_image_text(doc, page, _Scripted(regions), language="en")
    return page, stats


def test_a_line_with_a_guessed_word_is_left_in_english_and_listed():
    sure = _region("Cell Wall", 20, 20, [0.95, 0.93])
    guessed = _region("Chrome apps and extensions can also be added", 20, 60,
                      [0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9])
    page, stats = _labels([sure, guessed])
    by_text = {t.flat_text: t for t in page.text_objects}
    assert not by_text["Cell Wall"].keep_original
    assert by_text["Chrome apps and extensions can also be added"].keep_original
    flagged = [f for f in stats.findings if f.code == "ocr_low_confidence"]
    assert flagged and flagged[0].object_ids == [by_text["Chrome apps and extensions can also be added"].id]


def test_a_shaky_sentence_is_offered_but_a_shaky_scrap_is_not():
    sentence = _region("Chrome Web Store you must also specify", 20, 20, [0.55] * 7)
    scrap = _region("eel pl", 20, 60, [0.55, 0.55])
    page, _ = _labels([sentence, scrap])
    texts = {t.flat_text: t for t in page.text_objects}
    assert texts["Chrome Web Store you must also specify"].keep_original
    assert "eel pl" not in texts


def test_a_word_the_translator_gave_back_unchanged_is_not_repainted():
    """'COMPIAR' (a misread CONFIRM) came back as itself and was painted
    over the picture's own, better lettering."""
    from ct_engine.model.enums import TranslationStatus
    from ct_engine.reconstruct.builder import rebuild_document

    data = corpus_file("04_science_diagram.pdf")
    with PdfParser().open_bytes(data) as doc:
        page = doc.parse_page(0)
        extract_image_text(doc, page, _Scripted([
            _region("COMPIAR", 20, 20, [0.75]),
            _region("Cell Wall", 20, 60, [0.95, 0.93]),
        ]), language="en")
    for t in page.text_objects:
        t.translated_text = "COMPIAR" if t.flat_text == "COMPIAR" else f"[es] {t.flat_text}"
        t.translation_status = TranslationStatus.TRANSLATED
    _, stats = rebuild_document(data, [page], target_lang="es")
    by_text = {t.flat_text: t for t in page.text_objects}
    assert by_text["COMPIAR"].drawn_layout is None
    # Left alone, and not reported as something that failed to cover.
    assert all(by_text["COMPIAR"].id not in f.object_ids for f in stats[0].findings)


def test_a_checked_line_leaves_the_list():
    """Once she has confirmed or corrected its English, a hard-to-read line
    is no longer something to check."""
    from ct_engine.api.findings import _actionable
    from ct_engine.model.enums import TranslationStatus

    page, stats = _labels([_region("Chrome apps and extensions can also be added", 20, 20,
                                   [0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9])])
    finding = next(f for f in stats.findings if f.code == "ocr_low_confidence")
    obj = page.find_text_object(finding.object_ids[0])
    obj.translation_status = TranslationStatus.TRANSLATED
    assert _actionable(page, finding) is not None
    obj.approved_source_text = "Chrome apps and extensions can also be added"
    assert _actionable(page, finding) is None
