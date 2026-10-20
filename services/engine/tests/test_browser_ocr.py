"""Text recognition in the browser (ct_browser.ocr), checked natively.

In the browser, `js.ctTess.recognize(png, lang, psm)` is tesseract.js-core
called synchronously in the worker, and returns Tesseract's TSV without its
header line. Here a fake `js` module plays that part. The provider must turn
that TSV into exactly what the server's provider makes of pytesseract's
image_to_data dict, so the rest of the engine cannot tell them apart.
"""

from __future__ import annotations

import io
import shutil
import sys
import types

import pytest
from PIL import Image, ImageDraw

from ct_browser import ocr as browser_ocr
from ct_browser.ocr import NOT_READY, READ_FAILED, TSV_HEADER, BrowserTesseractOCRProvider
from ct_engine.providers.ocr.local_tesseract import LocalTesseractOCRProvider
from ct_engine.providers.registry import OCR_PROVIDERS, get_ocr_provider

# Rows as tesseract.js-core's GetTSVText() returns them: no header. Two
# lines in one paragraph, a "|" read for a capital I (repaired by the
# engine), a word with a float confidence, and a -1 non-text row.
CORE_TSV = (
    "1\t1\t0\t0\t0\t0\t0\t0\t800\t300\t-1\t\n"
    "2\t1\t1\t0\t0\t0\t20\t20\t700\t120\t-1\t\n"
    "3\t1\t1\t1\t0\t0\t20\t20\t700\t120\t-1\t\n"
    "4\t1\t1\t1\t1\t0\t20\t20\t500\t40\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t20\t20\t30\t40\t91.5\t|\n"
    "5\t1\t1\t1\t1\t2\t60\t20\t120\t40\t95.2\twill\n"
    "5\t1\t1\t1\t1\t3\t190\t20\t150\t40\t88\tfollow\n"
    "4\t1\t1\t1\t2\t0\t20\t100\t400\t40\t-1\t\n"
    "5\t1\t1\t1\t2\t1\t20\t100\t180\t40\t96\tdirections\n"
    "5\t1\t1\t1\t2\t2\t210\t100\t60\t40\t0.4\t~\n"
)


class FakeCore:
    """The worker's js.ctTess, recording every call."""

    def __init__(self, tsv: str = CORE_TSV, langs=("eng",)) -> None:
        self.ready = True
        self.langs = list(langs)
        self.tsv = tsv
        self.calls: list[tuple[str, int, tuple[int, int], str]] = []

    def recognize(self, png, lang, psm):
        with Image.open(io.BytesIO(bytes(png))) as im:
            self.calls.append((lang, psm, im.size, im.mode))
        if isinstance(self.tsv, Exception):
            raise self.tsv
        return self.tsv


@pytest.fixture
def core(monkeypatch):
    fake = FakeCore()
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(ctTess=fake))
    return fake


def _png(size=(800, 300), mode="RGB") -> bytes:
    im = Image.new(mode, size, "white")
    ImageDraw.Draw(im).text((20, 20), "I will follow directions", fill="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _as_tuples(regions):
    return [
        (r.text, r.bbox, r.confidence, r.line_index,
         [(w.text, w.bbox, w.confidence) for w in r.words], sorted(r.discarded_boxes))
        for r in regions
    ]


def test_registered_as_browser_only_when_installed():
    assert "browser" not in OCR_PROVIDERS
    browser_ocr.install()
    try:
        assert OCR_PROVIDERS["browser"] is BrowserTesseractOCRProvider
        assert isinstance(get_ocr_provider("browser"), BrowserTesseractOCRProvider)
    finally:
        browser_ocr.uninstall()
    assert "browser" not in OCR_PROVIDERS


def test_lines_match_what_the_server_provider_makes_of_the_same_tsv(core):
    from pytesseract.pytesseract import file_to_dict

    result = BrowserTesseractOCRProvider().recognize(_png())
    expected = LocalTesseractOCRProvider._group_into_lines(
        file_to_dict(TSV_HEADER + CORE_TSV, "\t", -1)
    )
    assert result.error is None
    assert _as_tuples(result.regions) == _as_tuples(expected)
    assert [r.text for r in result.regions] == ["I will follow", "directions"]
    assert result.regions[1].discarded_boxes  # the doubted "~" is still covered
    assert result.provider == "browser"


@pytest.mark.parametrize(
    "call,psm",
    [
        (lambda p, png: p.recognize(png), 3),
        (lambda p, png: p.recognize(png, sparse=True), 11),
        (lambda p, png: p.recognize_line(png), 7),
    ],
)
def test_page_segmentation_mode_is_always_set(core, call, psm):
    call(BrowserTesseractOCRProvider(), _png())
    assert core.calls[-1][1] == psm


def test_the_core_gets_a_grey_png_like_the_binary_does(core):
    BrowserTesseractOCRProvider().recognize(_png((640, 200), "RGBA"))
    lang, psm, size, mode = core.calls[-1]
    assert (lang, size, mode) == ("eng", (640, 200), "L")


def test_languages_come_from_the_loaded_packs(core):
    core.langs = ["eng", "spa"]
    p = BrowserTesseractOCRProvider()
    assert p.supports_language("es") and p.supports_language("en")
    assert not p.supports_language("ar")
    p.recognize(_png(), language="es")
    assert core.calls[-1][0] == "spa"


def test_a_language_with_no_pack_is_read_as_english(core):
    BrowserTesseractOCRProvider().recognize(_png(), language="ar")
    assert core.calls[-1][0] == "eng"


def test_no_reader_is_an_honest_error_without_server_wording(monkeypatch):
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace())
    result = BrowserTesseractOCRProvider().recognize(_png())
    assert result.regions == [] and result.error == NOT_READY
    assert "server" not in NOT_READY.lower() and "reload" not in NOT_READY.lower()
    assert not BrowserTesseractOCRProvider().supports_language("en")


def test_a_reader_still_loading_counts_as_no_reader(core):
    core.ready = False
    assert BrowserTesseractOCRProvider().recognize(_png()).error == NOT_READY
    assert core.calls == []


def test_a_failed_read_is_reported_not_raised(core):
    core.tsv = RuntimeError("bad image")
    result = BrowserTesseractOCRProvider().recognize(_png())
    assert result.error == READ_FAILED and result.regions == []


def test_an_empty_page_reads_as_no_lines(core):
    core.tsv = "1\t1\t0\t0\t0\t0\t0\t0\t800\t300\t-1\t\n"
    result = BrowserTesseractOCRProvider().recognize(_png())
    assert result.error is None and result.regions == [] and result.mean_confidence == 0.0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract binary")
@pytest.mark.parametrize("psm", [3, 11, 7])
def test_real_tesseract_tsv_gives_the_server_providers_lines(monkeypatch, psm):
    """TSV captured from a real Tesseract, fed through the browser provider,
    gives exactly the lines the server provider reads from the same image."""
    import pytesseract

    from ct_engine.providers.ocr import local_tesseract

    if not local_tesseract.installed_languages() or "eng" not in local_tesseract.installed_languages():
        pytest.skip("no eng pack")
    im = Image.new("L", (900, 260), "white")
    d = ImageDraw.Draw(im)
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=40)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    d.text((20, 30), "Photosynthesis Lab", fill="black", font=font)
    if psm != 7:
        d.text((20, 140), "Write your answer below.", fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    png = buf.getvalue()

    raw = pytesseract.image_to_data(im, lang="eng", config=f"--psm {psm}")
    header, _, rows = raw.partition("\n")
    assert header.startswith("level\tpage_num")
    fake = FakeCore(tsv=rows)
    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(ctTess=fake))

    browser = BrowserTesseractOCRProvider()
    local = LocalTesseractOCRProvider()
    if psm == 7:
        got, want = browser.recognize_line(png), local.recognize_line(png)
    else:
        got = browser.recognize(png, sparse=psm == 11)
        want = local.recognize(png, sparse=psm == 11)
    assert want.error is None and want.regions
    assert _as_tuples(got.regions) == _as_tuples(want.regions)
    assert got.mean_confidence == want.mean_confidence
