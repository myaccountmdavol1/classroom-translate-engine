"""Upload validation: reject bad input at the door with a usable message."""

from __future__ import annotations

from conftest import corpus_file
from ct_engine.ingest.validate import validate_docx, validate_pdf, validate_upload


def test_accepts_real_pdf():
    r = validate_pdf(corpus_file("01_clean_worksheet.pdf"), "worksheet.pdf")
    assert r.ok and r.page_count == 1 and r.content_type == "application/pdf"


def test_reports_true_page_count_for_long_document():
    r = validate_pdf(corpus_file("12_long_100p.pdf"), "packet.pdf")
    assert r.ok and r.page_count == 100


def test_rejects_empty_file():
    r = validate_pdf(b"", "x.pdf")
    assert not r.ok and "empty" in r.message.lower()


def test_rejects_non_pdf_content_with_pdf_name():
    """A renamed .docx must not reach the parser."""
    r = validate_pdf(b"PK\x03\x04" + b"\x00" * 600, "actually_a_docx.pdf")
    assert not r.ok
    assert "does not look like a PDF" in r.message


def test_pdf_bytes_named_docx_are_rejected():
    """Dispatch is by extension, so this reaches the DOCX validator.

    It must fail there on the bytes, not be waved through because the name
    looked plausible.
    """
    r = validate_upload(corpus_file("01_clean_worksheet.pdf"), "worksheet.docx")
    assert not r.ok
    assert "does not look like a Word document" in r.message


def test_rejects_unsupported_extension():
    r = validate_upload(corpus_file("01_clean_worksheet.pdf"), "worksheet.pages")
    assert not r.ok
    for supported in ("PDF", "Word", "PowerPoint"):
        assert supported in r.message


def test_rejects_truncated_pdf():
    data = corpus_file("01_clean_worksheet.pdf")[:120]
    r = validate_pdf(data, "broken.pdf")
    assert not r.ok
    assert r.diagnostic  # developer detail retained


def test_error_messages_are_teacher_facing():
    """No stack traces or library names in the user-visible message."""
    for data, name in [(b"", "a.pdf"), (b"PK\x03\x04", "b.pdf")]:
        r = validate_pdf(data, name)
        assert not r.ok
        assert "Traceback" not in r.message
        assert "pypdf" not in r.message.lower()
        assert r.message.endswith(".")
