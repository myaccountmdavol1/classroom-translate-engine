"""What each supported source format implies for the pipeline.

One place to ask "is this a PDF or a Word file?", so the answer cannot drift
between the upload route, the reconstruction stage and the download route --
which is how a teacher ends up downloading a .docx named .pdf.
"""

from __future__ import annotations

from enum import Enum

PDF_CONTENT_TYPE = "application/pdf"
DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PPTX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


class SourceFormat(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    PPTX = "PPTX"
    XLSX = "XLSX"

    @property
    def extension(self) -> str:
        return {"PDF": "pdf", "DOCX": "docx", "PPTX": "pptx", "XLSX": "xlsx"}[
            self.value
        ]

    @property
    def content_type(self) -> str:
        return {
            "PDF": PDF_CONTENT_TYPE,
            "DOCX": DOCX_CONTENT_TYPE,
            "PPTX": PPTX_CONTENT_TYPE,
            "XLSX": XLSX_CONTENT_TYPE,
        }[self.value]

    @property
    def is_fixed_layout(self) -> bool:
        """True when the source states where its text sits.

        A PDF states the position of every glyph, and a PPTX states the box
        of every shape; both are honoured exactly, so their coordinates are
        real and the review overlay can be trusted. A DOCX states nothing --
        Word lays the document out when it opens the file -- and neither does
        an XLSX in any useful sense: a cell has a grid position, not a box on
        a printed page, and Excel decides what is visible when it opens.

        This does NOT say how the output is built. Only a PDF is
        reconstructed by redrawing; the Office formats are rebuilt by editing
        the original package, which is what `rebuilds_in_place` below says.
        """
        return self not in (SourceFormat.DOCX, SourceFormat.XLSX)

    @property
    def rebuilds_in_place(self) -> bool:
        """True when the output is the source file with its text replaced."""
        return self in (SourceFormat.DOCX, SourceFormat.PPTX, SourceFormat.XLSX)

    @property
    def page_noun(self) -> str:
        """What one unit of this format is called in teacher-facing text."""
        return {
            "PDF": "page",
            "DOCX": "section",
            "PPTX": "slide",
            "XLSX": "sheet",
        }[self.value]


def format_for(content_type: str | None, filename: str | None = None) -> SourceFormat:
    name = (filename or "").lower()
    if name.endswith(".docx"):
        return SourceFormat.DOCX
    if name.endswith(".pptx"):
        return SourceFormat.PPTX
    if name.endswith(".xlsx"):
        return SourceFormat.XLSX
    if name.endswith(".pdf"):
        return SourceFormat.PDF
    ct = (content_type or "").lower()
    if ct == DOCX_CONTENT_TYPE:
        return SourceFormat.DOCX
    if ct == PPTX_CONTENT_TYPE:
        return SourceFormat.PPTX
    if ct == XLSX_CONTENT_TYPE:
        return SourceFormat.XLSX
    return SourceFormat.PDF


def format_of(manifest) -> SourceFormat:
    """The format the pipeline works on.

    A Word or PowerPoint file shown as pages is worked on as the PDF made
    from it: every page tool (editor, picture text, previews) applies
    unchanged.
    """
    if getattr(manifest, "office_source_key", None):
        return SourceFormat.PDF
    return format_for(manifest.content_type, manifest.filename)


def output_format_of(manifest) -> SourceFormat:
    """The format the teacher downloads: what she uploaded."""
    if getattr(manifest, "office_source_key", None):
        return SourceFormat.PPTX if manifest.office_format == "pptx" else SourceFormat.DOCX
    return format_of(manifest)
