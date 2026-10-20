"""Text recognition in the browser: Tesseract 5 (tesseract.js-core 7).

The worker loads tesseract.js-core and a language pack before a translation
job that may need them (never at boot: OCR happens in the translation job,
not in analysis), and exposes them as `js.ctTess`. Its recognize() is a
plain synchronous call into the core's WebAssembly, so the engine's OCR
stage calls it mid-job exactly as it would call the tesseract binary.

Everything after the raw read is the native provider's own code: the TSV is
parsed by pytesseract's own file_to_dict, the rows go through
_group_into_lines and _repair_words unchanged, so the browser reads lines
the way the server does. What differs is only where the words come from.
"""

from __future__ import annotations

import io
from typing import Any

from ct_engine.providers.ocr.base import OCRResult
from ct_engine.providers.ocr.local_tesseract import LocalTesseractOCRProvider
from ct_engine.providers.registry import OCR_PROVIDERS
from ct_engine.util.logging import get_logger

log = get_logger(__name__)

# The tesseract binary writes this header as the first line of its TSV and
# pytesseract parses it; the core's GetTSVText() returns the rows only.
TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
)

# What a teacher reads in the page's findings ("Page 3 could not be read.
# <this>"). The engine's own wording speaks of a server, which is wrong here.
NOT_READY = (
    "The text reader did not start on this computer, so scanned pages and "
    "words in pictures were left as they are."
)
READ_FAILED = "This page could not be read by text recognition."


def _bridge() -> Any | None:
    """The worker's js.ctTess, once it has a language loaded; else None.

    Looked up on every call rather than once: the worker loads the reader
    between Python entries, after this provider may already exist. Outside
    Pyodide (the native tests) a fake `js` module stands in.
    """
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return None
    bridge = getattr(js, "ctTess", None)
    if bridge is None or not getattr(bridge, "ready", False):
        return None
    return bridge


def _to_js_bytes(data: bytes) -> Any:
    """PNG bytes as the Uint8Array the core reads, or as-is outside Pyodide."""
    try:
        from pyodide.ffi import to_js  # type: ignore[import-not-found]
    except ImportError:
        return data
    return to_js(data)


def installed_languages() -> frozenset[str]:
    """The Tesseract language packs loaded in this worker right now."""
    bridge = _bridge()
    if bridge is None:
        return frozenset()
    return frozenset(str(code) for code in bridge.langs)


class BrowserTesseractOCRProvider(LocalTesseractOCRProvider):
    """The local provider, reading through the worker's synchronous core."""

    name = "browser"
    sends_data_externally = False

    def supports_language(self, lang: str | None) -> bool:
        return self.tesseract_code(lang) in installed_languages()

    # recognize() and recognize_line() are inherited: they choose psm 3, 11
    # or 7 and call _read, which is the one place that differs.
    def _read(self, image_png: bytes, language: str | None, *, psm: int) -> OCRResult:
        bridge = _bridge()
        if bridge is None:
            return OCRResult(provider=self.name, error=NOT_READY)

        available = installed_languages()
        code = self.tesseract_code(language)
        if code not in available:
            # The same fallback as the server: English, and said in the log.
            # The worker loads the document's own pack when one exists, so
            # this happens only for a language with no pack at all.
            log.warning("Text reader has no %r pack; reading as eng", code)
            code = "eng" if "eng" in available else next(iter(sorted(available)), "eng")

        try:
            from PIL import Image
            from pytesseract.pytesseract import file_to_dict

            # Exactly what pytesseract hands the tesseract binary: the
            # picture made grey and saved as a PNG.
            with Image.open(io.BytesIO(image_png)) as image:
                grey = image.convert("L")
                buf = io.BytesIO()
                grey.save(buf, format="PNG")
            # psm always explicit: the core keeps the last value it was given,
            # and a page read at 11 must not make the next one sparse too.
            tsv = bridge.recognize(_to_js_bytes(buf.getvalue()), code, int(psm))
            data = file_to_dict(TSV_HEADER + str(tsv), "\t", -1)
        except Exception as exc:
            # Returned, not raised: one unreadable page is flagged, and the
            # rest of the document still translates.
            log.warning("Text reader failed (psm %s): %r", psm, exc)
            return OCRResult(provider=self.name, error=READ_FAILED)

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


def install() -> None:
    """Register the provider as 'browser' (CT_OCR_PROVIDER=browser)."""
    OCR_PROVIDERS["browser"] = BrowserTesseractOCRProvider


def uninstall() -> None:
    """Take it out again (for tests)."""
    if OCR_PROVIDERS.get("browser") is BrowserTesseractOCRProvider:
        del OCR_PROVIDERS["browser"]
