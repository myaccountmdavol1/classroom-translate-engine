"""Conservative checks for visible numeric content; never invent a correction."""
import re
import unicodedata
from collections import Counter
from ..model.document import QAFinding

# A missing colon in a scanned English clock is ambiguous: ask about the source.
COMPACT_CLOCK = re.compile(r'(?<!\d)\d{3,4}\s*[ap]\s*\.(?:\s*m\s*\.?)?', re.I)
CLOCK = re.compile(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*[ap]\.?\s*m\.?)?', re.I)
NUMERIC_DATE = re.compile(r'(?<!\d)\d{1,4}[/-]\d{1,2}[/-]\d{1,4}(?!\d)')


def values(text):
    normalized = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return Counter(re.findall(r'\d+', normalized))


def value_findings(page):
    out = []
    for obj in page.text_objects:
        if obj.protected or obj.suppressed or obj.keep_original:
            continue
        if COMPACT_CLOCK.search(obj.flat_text) and obj.approved_source_text is None:
            out.append(QAFinding(severity='warning', code='source_time_uncertain',
                message='A time may have been read incorrectly. Compare the original and confirm its wording.',
                object_ids=[obj.id], bbox=obj.bbox))
        # Only a source number that is MISSING (or appears fewer times) is a
        # problem. Extra digits are the translator writing number words as
        # numerals -- "double" became "乘以2" -- and flagging those buried
        # the real cases on a real packet.
        # Single digits are left out: translators spell small numbers as
        # words ("3 CDs" -> "tres CD", "2-Step" -> "dos pasos"), which is
        # right. A larger number going missing is still caught.
        lost = (
            values(obj.flat_text) - values(obj.translated_text)
            if obj.translated_text else {}
        )
        if any(len(n) > 1 for n in lost):
            # Bilingual output intentionally repeats source values.
            if obj.flat_text.strip() in obj.translated_text and obj.translated_text != obj.flat_text:
                continue
            out.append(QAFinding(severity='warning', code='numeric_content_changed',
                message='Numbers differ from the original. Check dates, times, and quantities before sharing.',
                object_ids=[obj.id], bbox=obj.bbox))
    return out
