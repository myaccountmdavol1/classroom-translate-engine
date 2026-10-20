"""Source-language detection.

Detection is advisory: it pre-fills the UI and picks OCR language packs. It
never silently overrides a teacher's explicit choice, and it is never guessed
from the filename.
"""

from __future__ import annotations

import re

from ..util.logging import get_logger

log = get_logger(__name__)

# Scripts that are identifiable from codepoints alone, which is far more
# reliable than statistical detection on the short, fragmentary text typical
# of worksheets.
_SCRIPT_RANGES: list[tuple[str, tuple[tuple[int, int], ...]]] = [
    ("zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("ja", ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("ar", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("he", ((0x0590, 0x05FF),)),
    ("ru", ((0x0400, 0x04FF),)),
    ("bn", ((0x0980, 0x09FF),)),
    ("hi", ((0x0900, 0x097F),)),
    ("th", ((0x0E00, 0x0E7F),)),
    ("el", ((0x0370, 0x03FF),)),
]

_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[lang] = counts.get(lang, 0) + 1
                break
    return counts


def detect_language(text: str) -> tuple[str | None, float, int]:
    """Return (ISO-639-1 code | None, confidence 0-1, chars considered).

    Refuses to answer on very short samples rather than returning a confident
    wrong answer -- a misdetected source language degrades both OCR and
    translation, so "unknown" is the more useful result.
    """
    sample = " ".join(text.split())
    if not sample:
        return None, 0.0, 0

    letters = [c for c in sample if c.isalpha()]
    n_letters = len(letters)
    if n_letters < 12:
        return None, 0.0, n_letters

    # Script evidence first: unambiguous and robust on short text.
    counts = _script_counts(sample)
    if counts:
        lang, hits = max(counts.items(), key=lambda kv: kv[1])
        share = hits / n_letters
        if share >= 0.20:
            # Japanese and Chinese share Han characters; the presence of kana
            # is what distinguishes them, and kana already mapped to "ja".
            if lang == "zh" and counts.get("ja", 0) > 0:
                lang = "ja"
            return lang, round(min(0.99, 0.6 + share * 0.4), 3), n_letters

    if n_letters < 25:
        return None, 0.0, n_letters

    try:
        from langdetect import DetectorFactory, detect_langs

        # langdetect is randomized by default; a document must not get a
        # different detected language on re-analysis.
        DetectorFactory.seed = 0
        results = detect_langs(sample[:5000])
        if results:
            top = results[0]
            return str(top.lang), round(float(top.prob), 3), n_letters
    except Exception as exc:
        log.debug("langdetect failed: %r", exc)

    return None, 0.0, n_letters


def sample_document_text(pages_text: list[str], max_chars: int = 6000) -> str:
    """Build a language-detection sample spread across the document.

    Sampling only the first page would misread a document whose cover sheet
    is in one language and whose body is in another.
    """
    if not pages_text:
        return ""
    non_empty = [t for t in pages_text if t and t.strip()]
    if not non_empty:
        return ""
    per_page = max(200, max_chars // max(1, min(len(non_empty), 12)))
    step = max(1, len(non_empty) // 12)
    chunks = [non_empty[i][:per_page] for i in range(0, len(non_empty), step)]
    return " ".join(chunks)[:max_chars]
