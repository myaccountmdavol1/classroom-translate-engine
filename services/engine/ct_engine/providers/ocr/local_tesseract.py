"""Local OCR via Tesseract (spec sec. 16).

Default OCR provider: free, and nothing leaves the machine, which is the
right posture for scanned student work.

Tesseract is an OPTIONAL system dependency (a binary, not a wheel):

    macOS   brew install tesseract tesseract-lang
    Debian  apt-get install tesseract-ocr tesseract-ocr-all

When it is absent this provider says so plainly instead of pretending to
read the page.

Granularity: regions are LINES, not words. A line is the smallest unit that
still carries enough context to translate, and it is the unit the
reconstruction engine needs to cover and redraw. Word boxes are still used
underneath to compute a character-weighted confidence.
"""

from __future__ import annotations

import io
import re
import shutil
from dataclasses import replace
from functools import lru_cache

from ...util.logging import get_logger
from ..registry import register_ocr
from .base import OCRError, OCRProvider, OCRRegion, OCRResult, OCRWord

log = get_logger(__name__)

# Our BCP-47 codes mapped to Tesseract's traineddata names.
_TESSERACT_LANGS: dict[str, str] = {
    "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita",
    "pt": "por", "pt-BR": "por", "pt-PT": "por",
    "zh": "chi_sim", "zh-Hans": "chi_sim", "zh-Hant": "chi_tra",
    "ar": "ara", "ru": "rus", "ko": "kor", "ja": "jpn", "bn": "ben",
    "ur": "urd", "he": "heb", "uk": "ukr", "pl": "pol", "vi": "vie",
    "el": "ell", "hi": "hin", "ht": "fra",  # no Haitian model; French is closest
    "tr": "tur", "nl": "nld", "fa": "fas", "th": "tha",
}

# Tesseract reports -1 for non-text boxes. Its confidence is unreliable at
# the low end: on the corpus it scored a correctly-read "A student" at 9.0
# while neighbouring words scored 96. A high threshold therefore deletes
# correct words. The bar is set just above Tesseract's "not text" sentinel
# and uncertainty is surfaced through the LINE confidence and a review flag
# instead of by discarding text. Note that per-word thresholds do not filter
# noise either: stray dash marks on the corpus scored 53-78.
MIN_WORD_CONFIDENCE = 1.0
REVIEW_CONFIDENCE = 0.72
# A "word" with no letter or digit that Tesseract itself doubted is an icon
# or a border read as text ("[«", "}}" beside a key icon on a real slide).
# Letters are never dropped this way: a correct "A" can score 9.
JUNK_SYMBOL_CONFIDENCE = 0.5

# Tesseract reads a capital I in sans-serif type as a bar: "| will follow".
# A bar before a lowercase word, or carrying an apostrophe ("|'m"), is an I;
# a bar before "Name" is left alone, since that is what a table rule looks
# like. "lf" is not an English word but a misread "If".
_BAR_AS_I = re.compile(r"^\|(?=['’][a-z]+$)")
_BAR_STUCK = re.compile(r"^\|(?=[a-z]{2,}$)")


def _repair_words(words: list[OCRWord]) -> tuple[list[OCRWord], list[OCRWord]]:
    """Fix Tesseract's usual misreads of print; return (kept, junk)."""
    kept: list[OCRWord] = []
    junk: list[OCRWord] = []
    for w in words:
        if not any(c.isalnum() for c in w.text) and w.confidence < JUNK_SYMBOL_CONFIDENCE:
            junk.append(w)
        else:
            kept.append(w)
    # An icon to the left of a sentence leaves a scrap before its first word
    # ("a", "e", "©" before "I will never share..."). Dropped while it is
    # short, doubted and lower-case, and what follows starts a sentence; "A
    # student", with a doubted capital and a lower-case word after it, stays.
    while (
        len(kept) > 1
        and kept[0].confidence < JUNK_SYMBOL_CONFIDENCE
        and sum(c.isalnum() for c in kept[0].text) <= 2
        and not kept[0].text[:1].isupper()
        and (kept[1].text[:1].isupper() or kept[1].text[:1] == "|")
    ):
        junk.append(kept.pop(0))
    # Right to left, so each word sees its neighbour already repaired:
    # "lf |" is only "If" once the bar has become "I".
    for i in range(len(kept) - 1, -1, -1):
        w = kept[i]
        nxt = kept[i + 1].text if i + 1 < len(kept) else ""
        text = w.text
        if text == "|" and nxt[:1].islower():
            text = "I"
        elif _BAR_AS_I.match(text):
            text = "I" + text[1:]
        elif _BAR_STUCK.match(text):
            # The bar and the next word read as one: "|have".
            text = "I " + text[1:]
        elif text == "lf" and nxt[:1].isalpha():
            text = "If"
        if text != w.text:
            kept[i] = replace(w, text=text)
    return kept, junk


@lru_cache(maxsize=1)
def tesseract_binary() -> str | None:
    return shutil.which("tesseract")


@lru_cache(maxsize=1)
def installed_languages() -> frozenset[str]:
    try:
        import pytesseract

        return frozenset(pytesseract.get_languages(config=""))
    except Exception as exc:
        log.debug("Could not list Tesseract languages: %r", exc)
        return frozenset()


@register_ocr("local")
class LocalTesseractOCRProvider(OCRProvider):
    name = "local"
    sends_data_externally = False

    @staticmethod
    def tesseract_code(language: str | None) -> str:
        if not language:
            return "eng"
        if language in _TESSERACT_LANGS:
            return _TESSERACT_LANGS[language]
        return _TESSERACT_LANGS.get(language.split("-")[0], "eng")

    def supports_language(self, lang: str | None) -> bool:
        if tesseract_binary() is None:
            return False
        code = self.tesseract_code(lang)
        available = installed_languages()
        # An empty list means we could not query; assume the default pack.
        return not available or code in available

    def recognize(
        self,
        image_png: bytes,
        *,
        language: str | None = None,
        detect_handwriting: bool = False,
        sparse: bool = False,
    ) -> OCRResult:
        # PSM 3 = automatic page segmentation, for a page of prose.
        # PSM 11 = sparse text, for scattered diagram labels.
        return self._read(image_png, language, psm=11 if sparse else 3)

    def recognize_line(
        self, image_png: bytes, *, language: str | None = None
    ) -> OCRResult | None:
        # PSM 7 = a single text line. On the real slide's straightened
        # "HARBOR" it read 0.96, where sparse reading of the tilted word
        # gave 0.65; a tilted "Photosynthesis Lab" that sparse reading lost
        # past ~8 degrees read at 0.96 at every angle to 40.
        return self._read(image_png, language, psm=7)

    def _read(self, image_png: bytes, language: str | None, *, psm: int) -> OCRResult:
        if tesseract_binary() is None:
            return OCRResult(
                provider=self.name,
                error=(
                    "Text recognition is not installed on this server, so "
                    "scanned pages cannot be read."
                ),
            )

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            return OCRResult(provider=self.name, error=f"OCR unavailable: {exc}")

        code = self.tesseract_code(language)
        available = installed_languages()
        if available and code not in available:
            # Fall back to English rather than failing, but say what happened:
            # a silently wrong language pack degrades accuracy invisibly.
            log.warning(
                "Tesseract language %r is not installed; falling back to eng", code
            )
            code = "eng" if "eng" in available else next(iter(available), "eng")

        try:
            with Image.open(io.BytesIO(image_png)) as image:
                image = image.convert("L")
                data = pytesseract.image_to_data(
                    image,
                    lang=code,
                    config=f"--psm {psm}",
                    output_type=pytesseract.Output.DICT,
                )
        except Exception as exc:
            # Returned rather than raised: one unreadable page must be flagged,
            # not allowed to abort a 100-page document.
            log.warning("Tesseract failed: %r", exc)
            return OCRResult(
                provider=self.name,
                error="This page could not be read by text recognition.",
            )

        regions = self._group_into_lines(data)
        total_chars = sum(len(r.text) for r in regions)
        mean = (
            sum(r.confidence * len(r.text) for r in regions) / total_chars
            if total_chars
            else 0.0
        )
        return OCRResult(
            regions=regions,
            mean_confidence=round(mean, 4),
            detected_language=language,
            provider=self.name,
        )

    @staticmethod
    def _group_into_lines(data: dict) -> list[OCRRegion]:
        """Assemble Tesseract word rows into line regions."""
        n = len(data.get("text", []))
        lines: dict[tuple[int, int, int], dict] = {}
        dropped: dict[tuple[int, int, int], list[tuple[float, float, float, float]]] = {}

        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            key_for_drop = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            if conf < MIN_WORD_CONFIDENCE:
                # Keep the geometry even though the text is unusable, so the
                # reconstruction engine still covers those pixels.
                dropped.setdefault(key_for_drop, []).append(
                    (
                        float(data["left"][i]),
                        float(data["top"][i]),
                        float(data["left"][i]) + float(data["width"][i]),
                        float(data["top"][i]) + float(data["height"][i]),
                    )
                )
                continue

            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            left, top = int(data["left"][i]), int(data["top"][i])
            width, height = int(data["width"][i]), int(data["height"][i])

            entry = lines.setdefault(
                key,
                {
                    "words": [],
                    "x0": left, "y0": top,
                    "x1": left + width, "y1": top + height,
                    "conf_sum": 0.0, "chars": 0,
                },
            )
            entry["words"].append(
                OCRWord(
                    text=text,
                    bbox=(float(left), float(top),
                          float(left + width), float(top + height)),
                    confidence=round(conf / 100.0, 4),
                )
            )
            entry["x0"] = min(entry["x0"], left)
            entry["y0"] = min(entry["y0"], top)
            entry["x1"] = max(entry["x1"], left + width)
            entry["y1"] = max(entry["y1"], top + height)
            entry["conf_sum"] += conf * len(text)
            entry["chars"] += len(text)

        regions: list[OCRRegion] = []
        # Sort by block, then paragraph, then line: Tesseract's own numbering
        # is already close to reading order.
        for idx, key in enumerate(sorted(lines)):
            entry = lines[key]
            words, junk = _repair_words(entry["words"])
            # Junk keeps its place among the boxes to paint over.
            discarded = dropped.get(key, []) + [w.bbox for w in junk]
            if not words:
                continue
            regions.append(
                OCRRegion(
                    text=" ".join(w.text for w in words),
                    bbox=(
                        float(entry["x0"]), float(entry["y0"]),
                        float(entry["x1"]), float(entry["y1"]),
                    ),
                    confidence=round(entry["conf_sum"] / entry["chars"] / 100.0, 4),
                    line_index=idx,
                    words=words,
                    discarded_boxes=discarded,
                    # Tesseract cannot distinguish print from handwriting;
                    # claiming otherwise would be a fabricated signal.
                    handwritten=None,
                )
            )
        return regions
