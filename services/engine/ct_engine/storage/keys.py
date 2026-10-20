"""Canonical storage key construction.

Centralised so that the layout can change in one place, and so that no module
hand-builds a key string with a subtly different shape.
"""

from __future__ import annotations


def doc_prefix(document_id: str) -> str:
    return f"documents/{document_id}"


def source_key(document_id: str, ext: str = "pdf") -> str:
    return f"{doc_prefix(document_id)}/source.{ext}"


def manifest_key(document_id: str) -> str:
    return f"{doc_prefix(document_id)}/manifest.json"


def job_key(document_id: str, job_id: str) -> str:
    return f"{doc_prefix(document_id)}/jobs/{job_id}.json"


def jobs_prefix(document_id: str) -> str:
    return f"{doc_prefix(document_id)}/jobs"


def page_prefix(document_id: str, index: int) -> str:
    return f"{doc_prefix(document_id)}/pages/{index:04d}"


def page_source_render_key(document_id: str, index: int) -> str:
    return f"{page_prefix(document_id, index)}/source.png"


def page_source_thumb_key(document_id: str, index: int) -> str:
    return f"{page_prefix(document_id, index)}/source.thumb.png"


def page_translated_render_key(document_id: str, index: int) -> str:
    return f"{page_prefix(document_id, index)}/translated.png"


def page_plate_key(document_id: str, index: int) -> str:
    """The page with its translations removed and nothing drawn back."""
    return f"{page_prefix(document_id, index)}/plate.png"


def page_translated_thumb_key(document_id: str, index: int) -> str:
    return f"{page_prefix(document_id, index)}/translated.thumb.png"


def asset_key(document_id: str, content_hash: str, ext: str) -> str:
    """Key for an extracted image asset, addressed by content hash.

    Content addressing means a logo repeated on 100 pages is stored once and
    OCR'd once.
    """
    safe_ext = (ext or "bin").lstrip(".").lower()[:8] or "bin"
    return f"{doc_prefix(document_id)}/assets/{content_hash}.{safe_ext}"


def export_key(document_id: str, filename: str = "translated.pdf") -> str:
    return f"{doc_prefix(document_id)}/export/{filename}"


def report_key(document_id: str) -> str:
    return f"{doc_prefix(document_id)}/export/report.json"


def export_state_key(document_id: str) -> str:
    """What each page looked like when the stored export was built.

    Lets a download after an edit rebuild only the pages that changed.
    """
    return f"{doc_prefix(document_id)}/export/pages.json"


def upload_notes_key(document_id: str) -> str:
    """Warnings found while the upload was checked.

    Kept apart from the manifest's analysis, which the analysis stage builds
    afresh and would otherwise drop them from.
    """
    return f"{doc_prefix(document_id)}/upload-notes.json"
