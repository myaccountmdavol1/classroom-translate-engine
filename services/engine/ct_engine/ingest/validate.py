"""Upload validation.

Runs before anything expensive. Deliberately uses pypdf here rather than
PyMuPDF: PyMuPDF is lenient and will happily open files it then fails on
mid-parse, so a second, stricter opinion at the door catches corrupt uploads
while the user is still watching.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..util.logging import get_logger

log = get_logger(__name__)

PDF_MAGIC = b"%PDF-"


@dataclass
class ValidationResult:
    ok: bool
    # Teacher-facing reason when not ok.
    message: str | None = None
    diagnostic: str | None = None
    content_type: str = "application/pdf"
    page_count: int = 0
    pdf_version: str | None = None
    is_encrypted: bool = False
    warnings: list[str] = field(default_factory=list)


def _detect_pdf_version(head: bytes) -> str | None:
    if not head.startswith(PDF_MAGIC):
        return None
    try:
        return head[5:8].decode("ascii", errors="ignore").strip()
    except Exception:
        return None


def _read_head(path: Path, n: int = 1024) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def validate_pdf(data: bytes | Path, filename: str | None = None) -> ValidationResult:
    """Validate a PDF given as bytes or as a file on disk.

    Given a path, only the first kilobyte is read for sniffing and the strict
    reader opens the file itself, so a 100 MB upload is never held in memory
    just to be checked.
    """
    settings = get_settings()
    warnings: list[str] = []

    if isinstance(data, Path):
        try:
            size = data.stat().st_size
            head = _read_head(data) if size else b""
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))
    else:
        size = len(data)
        head = data[:1024]

    if not size:
        return ValidationResult(False, "That file is empty.", "zero-length upload")

    if size > settings.max_upload_bytes:
        mb = settings.max_upload_bytes // (1024 * 1024)
        return ValidationResult(
            False,
            f"That file is larger than the {mb} MB limit.",
            f"{size} bytes",
        )

    # Content sniffing, not just the extension: a renamed .docx must not get
    # as far as the parser.
    if PDF_MAGIC not in head[:512]:
        return ValidationResult(
            False,
            "That file does not look like a PDF. It may be damaged, or it may "
            "be a different kind of file with a .pdf name.",
            f"magic bytes: {head[:8]!r}",
        )
    if not head.startswith(PDF_MAGIC):
        warnings.append("PDF header is not at the start of the file.")

    version = _detect_pdf_version(head[head.find(PDF_MAGIC):])

    # Structural check with a strict reader.
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency is required
        return ValidationResult(False, "Server is misconfigured.", repr(exc))

    # A path is opened as a file handle, not passed as a path: pypdf reads a
    # path into memory whole, but reads a handle lazily, seeking as it goes.
    with contextlib.ExitStack() as stack:
        try:
            source = (
                stack.enter_context(open(data, "rb"))
                if isinstance(data, Path)
                else io.BytesIO(data)
            )
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))
        return _check_pdf_structure(
            PdfReader, PdfReadError, source, version, warnings, settings
        )


def _check_pdf_structure(
    PdfReader, PdfReadError, source, version, warnings, settings
) -> ValidationResult:
    encrypted = False
    try:
        reader = PdfReader(source, strict=False)
        encrypted = bool(reader.is_encrypted)
        if encrypted:
            # Many school PDFs carry an owner password with empty user
            # password; those decrypt with "" and are perfectly usable.
            try:
                if reader.decrypt("") == 0:
                    return ValidationResult(
                        False,
                        "That PDF is password-protected. Please remove the "
                        "password and upload it again.",
                        "decrypt('') returned 0",
                        is_encrypted=True,
                    )
                warnings.append(
                    "This PDF had usage restrictions, which were opened with an "
                    "empty password."
                )
            except Exception as exc:
                return ValidationResult(
                    False,
                    "That PDF is password-protected and could not be opened.",
                    repr(exc),
                    is_encrypted=True,
                )

        page_count = len(reader.pages)
    except PdfReadError as exc:
        return ValidationResult(
            False,
            "That PDF appears to be damaged and could not be opened. Try "
            "re-saving or re-exporting it, then upload again.",
            repr(exc),
        )
    except Exception as exc:
        return ValidationResult(
            False,
            "That PDF could not be opened.",
            repr(exc),
        )

    if page_count == 0:
        return ValidationResult(False, "That PDF has no pages.", "page_count=0")

    if page_count > settings.max_page_count:
        return ValidationResult(
            False,
            f"That document has {page_count} pages, which is over the current "
            f"limit of {settings.max_page_count} pages.",
            f"page_count={page_count}",
        )

    return ValidationResult(
        ok=True,
        content_type="application/pdf",
        page_count=page_count,
        pdf_version=version,
        is_encrypted=encrypted,
        warnings=warnings,
    )


# --- DOCX -----------------------------------------------------------------

ZIP_MAGIC = b"PK\x03\x04"
DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
# An OLE2 compound file: the pre-2007 binary .doc, and also a .docx that was
# actually saved as .doc. Worth naming exactly, because "re-save as .docx" is
# something a teacher can act on and "unsupported file" is not.
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def validate_docx(data: bytes | Path, filename: str | None = None) -> ValidationResult:
    """Structural validation of a .docx upload.

    Mirrors `validate_pdf`: sniff the bytes, then have a stricter reader open
    the file before anything expensive starts. The page limit has no meaning
    here -- Word decides pagination when it opens the document -- so the
    equivalent size bound is on paragraphs.
    """
    settings = get_settings()
    if isinstance(data, Path):
        try:
            data = data.read_bytes()
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))
    warnings: list[str] = []

    if not data:
        return ValidationResult(False, "That file is empty.", "zero-length upload")

    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes // (1024 * 1024)
        return ValidationResult(
            False,
            f"That file is larger than the {mb} MB limit.",
            f"{len(data)} bytes",
        )

    if data[:8] == OLE2_MAGIC:
        return ValidationResult(
            False,
            "That is an older Word document (.doc). Open it in Word and use "
            "Save As to save it as a .docx file, then upload it again.",
            "OLE2 compound file",
        )

    if not data.startswith(ZIP_MAGIC):
        return ValidationResult(
            False,
            "That file does not look like a Word document. It may be damaged, "
            "or it may be a different kind of file with a .docx name.",
            f"magic bytes: {data[:8]!r}",
        )

    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "word/document.xml" not in names:
                # A .pptx and an .xlsx are both valid ZIPs with the same
                # magic bytes, so this is the check that actually tells the
                # Office formats apart.
                kind = (
                    "PowerPoint presentation"
                    if "ppt/presentation.xml" in names
                    else "Excel workbook"
                    if "xl/workbook.xml" in names
                    else None
                )
                if kind:
                    return ValidationResult(
                        False,
                        f"That looks like a {kind}, not a Word document.",
                        f"parts={sorted(names)[:5]}",
                    )
                return ValidationResult(
                    False,
                    "That file is a ZIP archive, but not a Word document.",
                    "word/document.xml missing",
                )
            bad = z.testzip()
            if bad is not None:
                return ValidationResult(
                    False,
                    "That Word document appears to be damaged. Try re-saving "
                    "it from Word, then upload it again.",
                    f"corrupt entry: {bad}",
                )
    except Exception as exc:
        return ValidationResult(
            False,
            "That Word document could not be opened. It may be damaged.",
            repr(exc),
        )

    # Open it the way the pipeline will, so a file that passes the door also
    # parses. This is the same reasoning as using pypdf above: a lenient
    # check that defers failure to mid-pipeline helps nobody.
    try:
        from .docx_parser import DocxParser

        with DocxParser().open_bytes(data) as doc:
            paragraphs = doc.paragraph_count
            sections = doc.page_count
    except Exception as exc:
        return ValidationResult(
            False,
            "That Word document could not be read.",
            repr(exc),
        )

    if paragraphs == 0:
        return ValidationResult(
            False, "That Word document has no text in it.", "paragraph_count=0"
        )

    if paragraphs > settings.max_docx_paragraphs:
        return ValidationResult(
            False,
            f"That document has {paragraphs} paragraphs, which is over the "
            f"current limit of {settings.max_docx_paragraphs}.",
            f"paragraph_count={paragraphs}",
        )

    return ValidationResult(
        ok=True,
        content_type=DOCX_CONTENT_TYPE,
        page_count=sections,
        warnings=warnings,
    )


PPTX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)


def validate_pptx(data: bytes | Path, filename: str | None = None) -> ValidationResult:
    """Structural validation of a .pptx upload.

    Mirrors `validate_docx`. A slide count IS meaningful here -- one slide is
    one page, always -- so the ordinary page limit applies rather than a
    paragraph one.
    """
    settings = get_settings()
    if isinstance(data, Path):
        try:
            data = data.read_bytes()
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))

    if not data:
        return ValidationResult(False, "That file is empty.", "zero-length upload")

    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes // (1024 * 1024)
        return ValidationResult(
            False,
            f"That file is larger than the {mb} MB limit.",
            f"{len(data)} bytes",
        )

    if data[:8] == OLE2_MAGIC:
        return ValidationResult(
            False,
            "That is an older PowerPoint file (.ppt). Open it in PowerPoint "
            "and use Save As to save it as a .pptx file, then upload it again.",
            "OLE2 compound file",
        )

    if not data.startswith(ZIP_MAGIC):
        return ValidationResult(
            False,
            "That file does not look like a PowerPoint presentation. It may be "
            "damaged, or it may be a different kind of file with a .pptx name.",
            f"magic bytes: {data[:8]!r}",
        )

    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "ppt/presentation.xml" not in names:
                kind = (
                    "Word document"
                    if "word/document.xml" in names
                    else "Excel workbook"
                    if "xl/workbook.xml" in names
                    else None
                )
                if kind:
                    return ValidationResult(
                        False,
                        f"That looks like a {kind}, not a PowerPoint presentation.",
                        f"parts={sorted(names)[:5]}",
                    )
                return ValidationResult(
                    False,
                    "That file is a ZIP archive, but not a PowerPoint presentation.",
                    "ppt/presentation.xml missing",
                )
            bad = z.testzip()
            if bad is not None:
                return ValidationResult(
                    False,
                    "That presentation appears to be damaged. Try re-saving it "
                    "from PowerPoint, then upload it again.",
                    f"corrupt entry: {bad}",
                )
    except Exception as exc:
        return ValidationResult(
            False,
            "That presentation could not be opened. It may be damaged.",
            repr(exc),
        )

    try:
        from .pptx_parser import PptxParser

        with PptxParser().open_bytes(data) as doc:
            slides = doc.page_count
    except Exception as exc:
        return ValidationResult(
            False, "That presentation could not be read.", repr(exc)
        )

    if slides == 0:
        return ValidationResult(
            False, "That presentation has no slides.", "slide_count=0"
        )
    if slides > settings.max_page_count:
        return ValidationResult(
            False,
            f"That presentation has {slides} slides, which is over the current "
            f"limit of {settings.max_page_count}.",
            f"slide_count={slides}",
        )

    return ValidationResult(
        ok=True, content_type=PPTX_CONTENT_TYPE, page_count=slides
    )


XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
# The name of the stream Office uses for an encrypted package, as it appears
# (UTF-16LE) in the OLE2 directory. A password-protected .xlsx is not a ZIP
# at all: Excel wraps the whole encrypted package in a compound file, so
# without this check it would be reported as an old .xls, which is wrong and
# sends the teacher to fix the wrong thing.
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")


def validate_xlsx(data: bytes | Path, filename: str | None = None) -> ValidationResult:
    """Structural validation of a .xlsx upload.

    Mirrors `validate_pptx`. One sheet is one page, so the page limit applies
    to sheets; the real size bound is on text cells, as it is on paragraphs
    for Word.
    """
    settings = get_settings()
    if isinstance(data, Path):
        try:
            data = data.read_bytes()
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))

    if not data:
        return ValidationResult(False, "That file is empty.", "zero-length upload")

    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes // (1024 * 1024)
        return ValidationResult(
            False,
            f"That file is larger than the {mb} MB limit.",
            f"{len(data)} bytes",
        )

    if data[:8] == OLE2_MAGIC:
        if _ENCRYPTED_PACKAGE in data:
            return ValidationResult(
                False,
                "That Excel file is password-protected. Open it in Excel, "
                "remove the password (File > Info > Protect Workbook), save "
                "it, then upload it again.",
                "OLE2 EncryptedPackage",
                is_encrypted=True,
            )
        return ValidationResult(
            False,
            "That is an older Excel file (.xls). Open it in Excel and use "
            "Save As to save it as a .xlsx file, then upload it again.",
            "OLE2 compound file",
        )

    if not data.startswith(ZIP_MAGIC):
        return ValidationResult(
            False,
            "That file does not look like an Excel workbook. It may be "
            "damaged, or it may be a different kind of file with a .xlsx name.",
            f"magic bytes: {data[:8]!r}",
        )

    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "xl/workbook.xml" not in names:
                kind = (
                    "Word document"
                    if "word/document.xml" in names
                    else "PowerPoint presentation"
                    if "ppt/presentation.xml" in names
                    else None
                )
                if kind:
                    return ValidationResult(
                        False,
                        f"That looks like a {kind}, not an Excel workbook.",
                        f"parts={sorted(names)[:5]}",
                    )
                return ValidationResult(
                    False,
                    "That file is a ZIP archive, but not an Excel workbook.",
                    "xl/workbook.xml missing",
                )
            bad = z.testzip()
            if bad is not None:
                return ValidationResult(
                    False,
                    "That workbook appears to be damaged. Try re-saving it "
                    "from Excel, then upload it again.",
                    f"corrupt entry: {bad}",
                )
    except Exception as exc:
        return ValidationResult(
            False,
            "That workbook could not be opened. It may be damaged.",
            repr(exc),
        )

    try:
        from .xlsx_parser import XlsxParser

        with XlsxParser().open_bytes(data) as doc:
            sheets = doc.page_count
            cells = doc.translatable_count
    except Exception as exc:
        return ValidationResult(False, "That workbook could not be read.", repr(exc))

    if sheets == 0:
        return ValidationResult(
            False, "That workbook has no worksheets.", "sheet_count=0"
        )
    if cells == 0:
        return ValidationResult(
            False,
            "That workbook has no words in it to translate -- only numbers, "
            "formulas or codes.",
            "translatable_cells=0",
        )
    if sheets > settings.max_page_count:
        return ValidationResult(
            False,
            f"That workbook has {sheets} sheets, which is over the current "
            f"limit of {settings.max_page_count}.",
            f"sheet_count={sheets}",
        )
    if cells > settings.max_xlsx_cells:
        return ValidationResult(
            False,
            f"That workbook has {cells} cells of text, which is over the "
            f"current limit of {settings.max_xlsx_cells}.",
            f"translatable_cells={cells}",
        )

    return ValidationResult(
        ok=True, content_type=XLSX_CONTENT_TYPE, page_count=sheets
    )


def validate_upload(
    data: bytes | Path, filename: str | None = None
) -> ValidationResult:
    """Validate an upload of any supported format.

    Dispatches on the filename extension, because that is what the teacher
    chose and what the error messages need to talk about. The per-format
    validators then sniff the bytes, so a mislabelled file is still caught --
    a .docx renamed to .pdf fails the PDF magic-byte check, and vice versa.

    `data` may be a path to the file on disk. A PDF is then checked without
    reading it into memory; the Office formats are read, since their checks
    open the whole package anyway.
    """
    ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
    if ext == "docx":
        return validate_docx(data, filename)
    if ext == "pptx":
        return validate_pptx(data, filename)
    if ext == "xlsx":
        return validate_xlsx(data, filename)
    if ext == "xls":
        return ValidationResult(
            False,
            "That is an older Excel file (.xls). Open it in Excel and use "
            "Save As to save it as a .xlsx file, then upload it again.",
            "extension=xls",
        )
    if ext == "pdf":
        return validate_pdf(data, filename)
    if isinstance(data, Path):
        if not ext:
            try:
                head = _read_head(data, 8)
            except OSError as exc:
                return ValidationResult(False, "That file could not be read.", repr(exc))
            if head[:5] == PDF_MAGIC:
                return validate_pdf(data, filename)
        try:
            data = data.read_bytes()
        except OSError as exc:
            return ValidationResult(False, "That file could not be read.", repr(exc))
    if ext:
        return ValidationResult(
            False,
            f"Classroom Translate supports PDF, Word (.docx), PowerPoint "
            f"(.pptx) and Excel (.xlsx) files. This file appears to be a "
            f".{ext} file.",
            f"extension={ext}",
        )
    # No extension: fall back to the bytes.
    if data[:5] == PDF_MAGIC:
        return validate_pdf(data, filename)
    if data[:4] == ZIP_MAGIC:
        # All the Office formats are ZIPs, so the part list is what tells
        # them apart. Each validator reports the mismatch clearly if it is
        # wrong.
        try:
            import zipfile

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                parts = z.namelist()
                if "ppt/presentation.xml" in parts:
                    return validate_pptx(data, filename)
                if "xl/workbook.xml" in parts:
                    return validate_xlsx(data, filename)
        except Exception:
            pass
        return validate_docx(data, filename)
    return ValidationResult(
        False,
        "Classroom Translate could not tell what kind of file that is. "
        "Please upload a PDF, Word (.docx), PowerPoint (.pptx) or Excel "
        "(.xlsx) file.",
        f"magic bytes: {data[:8]!r}",
    )
