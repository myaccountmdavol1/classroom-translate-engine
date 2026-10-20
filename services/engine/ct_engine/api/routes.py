"""HTTP API.

Two rules shape this module:

  * Teachers see plain language; developers see diagnostics. Technical detail
    is attached only when the service runs in development (spec sec. 48).
  * Documents are addressed by an unguessable id and served through the API,
    never from a public static path (spec sec. 43).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from ..config import get_settings
from ..ingest.pdf_parser import PdfParser
from ..api.access import access_required
from ..ingest.registry import SUPPORTED_EXTENSIONS
from ..ingest.validate import validate_upload
from ..model.formats import SourceFormat, format_for, format_of, output_format_of
from ..languages import LANGUAGES, get_language, source_language_name
from ..model.document import DocumentManifest
from ..model.enums import OutputLanguageMode, ProcessingMode
from ..model.enums import JobStage, TextSource
from ..model.job import JobSettings
from ..pipeline.edit import (
    EditError,
    add_block,
    add_blocks,
    apply_bulk,
    apply_edit,
    apply_layout,
    can_undo,
    editable_blocks,
    lock_reason,
    remove_block,
    remove_blocks,
    reset_edit,
    restore_block,
    translate_specs,
    undo,
    redo,
)
from ..pipeline.manifest_store import document_lock
from ..providers import ProviderNotAvailable, get_translation_provider
from ..providers.translation.chain import SERVICE_UNREACHABLE, untranslated_reason
from ..render.debug_overlay import ALL_LAYERS, render_overlay
from ..fonts.registry import REGISTRY
from ..reconstruct import lists
from ..reconstruct.fitting import DEFAULT_LEADING, MIN_LEADING, justifiable
from ..reconstruct.builder import _source_look, turned_frame_of
from ..reconstruct.sprite import render_sprite, sprite_area
from ..storage import keys as K
from ..storage.base import InvalidKeyError, ObjectNotFoundError
from ..util.hashing import sha256_bytes
from ..util.logging import get_logger
from .deps import get_job_runner, get_manifest_store, get_object_store
from .findings import group_findings, finding_keys
from pydantic import BaseModel, Field

from .schemas import DocumentOut, JobOut, UploadResponse

log = get_logger(__name__)
router = APIRouter()

PNG_CACHE = "private, max-age=3600"


def _dev_detail(diagnostic: str | None) -> str | None:
    return diagnostic if get_settings().environment == "development" else None


def _fail(status: int, message: str, diagnostic: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": message, "diagnostic": _dev_detail(diagnostic)},
    )


# Findings that name blocks left in the original language. The page-wide
# "untranslated_text" lists every such block on its page, and the per-block
# ones name some of the same blocks again, so blocks are counted, not findings.
_UNTRANSLATED_CODES = ("untranslated_text", "segment_untranslated", "translation_unavailable")


def _blocks_left_untranslated(pages) -> int:
    total = 0
    for p in pages:
        ids: set[str] = set()
        unnamed = 0
        for f in p.qa.findings:
            if f.code not in _UNTRANSLATED_CODES:
                continue
            if f.object_ids:
                ids.update(f.object_ids)
            else:
                # An older finding that names no block still stands for one.
                unnamed += 1
        total += len(ids) + unnamed
    return total


def _load_manifest(document_id: str) -> DocumentManifest:
    manifests = get_manifest_store()
    try:
        manifest = manifests.load_manifest(document_id)
    except Exception as exc:
        raise _fail(500, "That document could not be read.", repr(exc)) from exc
    if manifest is None:
        raise _fail(404, _missing_document_message())
    return manifest


def _missing_document_message() -> str:
    """Why a document is gone, as far as the server can tell.

    On the free host the disk does not survive a restart, so the usual cause
    is a restart, not the retention window -- and "it may have expired" sent
    a teacher back from lunch looking for the wrong reason. The store knows
    when it last started out empty; a restart inside the retention window is
    stated as the fact it is.
    """
    settings = get_settings()
    hours = settings.retention_hours
    keep = (
        " Please upload it again. To keep your corrections next time, use "
        "\u201cSave project to reopen later\u201d before you leave."
    )
    started = get_object_store().created_at()
    if started is not None:
        age = _dt.datetime.now(_dt.timezone.utc) - started
        if age < _dt.timedelta(hours=hours):
            when = _ago(age)
            return (
                "That document could not be found. The translation server "
                f"restarted {when}, which clears documents uploaded before "
                "then." + keep
            )
    return (
        "That document could not be found. Documents are deleted "
        f"{hours} hours after they are uploaded." + keep
    )


def _ago(age: _dt.timedelta) -> str:
    minutes = int(age.total_seconds() // 60)
    if minutes < 2:
        return "a moment ago"
    if minutes < 90:
        return f"{minutes} minutes ago"
    return f"about {round(minutes / 60)} hours ago"


def _latest_job(document_id: str):
    jobs = get_manifest_store().list_jobs(document_id)
    return jobs[-1] if jobs else None


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@router.get("/health")
async def health() -> dict:
    # async: answered on the event loop itself, never queued behind busy
    # worker threads, so the host's 5-second health check always gets a reply.
    from ..util.memory import container_memory_mb

    s = get_settings()
    return {
        "status": "ok",
        "environment": s.environment,
        "storage_backend": s.storage_backend,
        "translation_provider": s.translation_provider,
        "ocr_provider": s.ocr_provider,
        "retention_hours": s.retention_hours,
        "max_page_count": s.max_page_count,
        # Whole megabytes, so the upload page can say "Up to 100 MB" and
        # refuse a bigger file before sending it.
        "max_upload_mb": s.max_upload_bytes // (1024 * 1024),
        # Whether the engine is behind the shared access code. Reported so a
        # deployment that forgot to set one is visible rather than quietly
        # open to the internet, and so the web app knows to ask for it.
        "access_required": access_required(),
        "supported_formats": sorted(SUPPORTED_EXTENSIONS | {"png", "jpg", "jpeg"}),
        # How close the server is to its memory limit; see util/memory.py.
        "memory": container_memory_mb(),
    }


@router.get("/languages")
def languages() -> dict:
    return {"languages": [lang.model_dump() for lang in LANGUAGES]}


@router.get("/output-modes")
def output_modes() -> dict:
    """Output language modes, with the example text shown in the UI."""
    return {
        "modes": [
            {
                "value": OutputLanguageMode.TRANSLATED_ONLY.value,
                "label": "Translated only",
                "description": "Only the translated text appears.",
                "example": "Escribe tu respuesta.",
                "default": True,
            },
            {
                "value": OutputLanguageMode.BILINGUAL_TRANSLATION_FIRST.value,
                "label": "Bilingual — translation first",
                "description": "Translation, followed by the English.",
                "example": "Escribe tu respuesta. (Write your answer.)",
                "default": False,
            },
            {
                "value": OutputLanguageMode.BILINGUAL_ENGLISH_FIRST.value,
                "label": "Bilingual — English first",
                "description": "English, followed by the translation.",
                "example": "Write your answer. (Escribe tu respuesta.)",
                "default": False,
            },
        ],
        "processing_modes": [
            {
                "value": ProcessingMode.STANDARD.value,
                "label": "Standard",
                "description": "Faster, for straightforward documents.",
            },
            {
                "value": ProcessingMode.HIGH_FIDELITY.value,
                "label": "High fidelity",
                "description": (
                    "Recommended for worksheets, exams, and diagrams. Takes "
                    "longer and checks the layout more carefully."
                ),
                "default": True,
            },
        ],
    }


# --------------------------------------------------------------------------
# Upload + analysis
# --------------------------------------------------------------------------
UPLOAD_CHUNK = 1024 * 1024


def _too_large(limit: int) -> HTTPException:
    return _fail(
        413,
        f"That file is larger than the {limit // (1024 * 1024)} MB limit. Try "
        "splitting it into smaller files, or saving it with smaller pictures.",
    )


def _spool_upload(src, scratch: Path, limit: int) -> tuple[Path, int, str]:
    """Copy an upload to a file of our own a chunk at a time, hashing it.

    The size is checked on every chunk, so a file over the limit is refused
    after one chunk too many -- it never reaches memory whole, and never
    reaches the store at all.
    """
    fd, tmp = tempfile.mkstemp(dir=str(scratch), suffix=".upload")
    path = Path(tmp)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            src.seek(0)
            while chunk := src.read(UPLOAD_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise _too_large(limit)
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, size, digest.hexdigest()


_OFFICE_MEDIA = {SourceFormat.DOCX: "word/media/", SourceFormat.PPTX: "ppt/media/"}


def _has_pictures(data: bytes, fmt: SourceFormat) -> bool:
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return any(n.startswith(_OFFICE_MEDIA[fmt]) for n in z.namelist())
    except Exception:
        # Validation already opened it; if it cannot be listed now, say
        # nothing rather than guess either way.
        return False


def _upload_notes(document_id: str) -> dict:
    """The upload-time warnings kept for this document (see keys.py)."""
    try:
        return json.loads(get_object_store().get_text(K.upload_notes_key(document_id)))
    except ObjectNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Upload notes for %s are unreadable: %r", document_id, exc)
        return {}


@router.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    store = get_object_store()
    manifests = get_manifest_store()

    limit = settings.max_upload_bytes
    # The size the request declared, when it declared one: refused before a
    # byte is copied. The copy below enforces it again as it goes, so a
    # request that understates its size gets no further.
    if file.size is not None and file.size > limit:
        raise _too_large(limit)
    # Copied and hashed off the event loop: on it, a large file's disk write
    # and SHA-256 held /api/health past the host's 5-second check.
    upload_path, upload_size, upload_sha = await run_in_threadpool(
        _spool_upload, file.file, store.scratch_dir(), limit
    )
    try:
        return await _accept_upload(
            file.filename, upload_path, upload_size, upload_sha,
            store=store, manifests=manifests,
        )
    finally:
        # Already gone when the store took it over; this is for a refusal.
        upload_path.unlink(missing_ok=True)


async def _accept_upload(
    filename: str | None,
    upload_path: Path,
    upload_size: int,
    upload_sha: str,
    *,
    store,
    manifests,
) -> UploadResponse:
    # A PDF (or a spreadsheet) is checked straight from the spooled file and
    # never read into memory here. A picture is converted from memory, and a
    # Word or PowerPoint file is read once below for laying out as pages --
    # held once, never written back out: the spooled file is moved into the
    # store as-is.
    data: bytes | None = None
    original_mime = None
    is_image = (filename or "").lower().endswith((".png", ".jpg", ".jpeg"))
    if is_image:
        from ..ingest.image_import import image_to_pdf
        raw = await run_in_threadpool(upload_path.read_bytes)
        try:
            data, original_mime = await run_in_threadpool(image_to_pdf, raw)
        except ValueError as exc:
            raise _fail(400, str(exc)) from exc
        del raw
    # Everything slow below runs off the event loop. Run on it, a slide deck
    # laid out by LibreOffice held the loop for minutes, /api/health went
    # unanswered past the host's 5-second check, and it restarted a busy
    # but healthy service -- taking every document with it.
    result = await run_in_threadpool(
        validate_upload,
        data if data is not None else upload_path,
        "page.pdf" if is_image else filename,
    )
    if not result.ok:
        raise _fail(400, result.message or "That file could not be used.",
                    result.diagnostic)

    fmt = format_for(result.content_type, filename)
    notes: list[str] = list(result.warnings)

    # A Word or PowerPoint file is shown as pages, like a PDF, when
    # LibreOffice is here to lay it out; the original is kept, and is what
    # she downloads -- a Google Slides deck goes back to Drive as Slides.
    paged = False
    office_fmt = fmt
    pictures_not_read: str | None = None
    if fmt in (SourceFormat.DOCX, SourceFormat.PPTX):
        from ..ingest.word_render import WordRenderError, can_render_word, office_to_pdf
        reason: str | None = None
        data = await run_in_threadpool(upload_path.read_bytes)
        if not can_render_word():
            reason = "This server cannot lay out this kind of file as pages."
        else:
            try:
                pdf = await run_in_threadpool(office_to_pdf, data, fmt.extension)
            except WordRenderError as exc:
                log.warning("%s shown without pages instead: %s", fmt.value, exc)
                reason = str(exc)
            else:
                checked = await run_in_threadpool(validate_upload, pdf, "document.pdf")
                if checked.ok:
                    data, result, paged = pdf, checked, True
                    fmt = SourceFormat.PDF
                else:
                    log.warning("%s pages did not validate: %s", fmt.value,
                                checked.diagnostic)
                    reason = "The file could not be laid out as pages."
        # Without pages, nothing reads the pictures: the text-only parsers
        # never open them. Said here, where the reason is known, rather than
        # leaving her to find the diagrams still in English.
        if reason and await run_in_threadpool(_has_pictures, data, fmt):
            pictures_not_read = reason
            notes.append(
                f"{reason} It opens as a list of its text instead, so words "
                "inside its pictures will not be translated; the pictures "
                "stay exactly as they are. To translate them, download it as "
                "a PDF and upload that, or split it into smaller files."
            )

    # uuid4 hex: unguessable, so a document URL cannot be enumerated.
    document_id = uuid.uuid4().hex
    source_key = K.source_key(document_id, fmt.extension)
    # The upload file itself is the source unless it was converted (a
    # picture, or Office laid out as pages); then it is kept as the original.
    upload_is_source = not is_image and not paged
    if upload_is_source:
        await run_in_threadpool(
            store.put_file, source_key, upload_path, content_type=result.content_type
        )
        source_size, source_sha = upload_size, upload_sha
    else:
        assert data is not None  # converted, so held in memory above
        await run_in_threadpool(
            store.put_bytes, source_key, data, content_type=result.content_type
        )
        source_size = len(data)
        source_sha = await run_in_threadpool(sha256_bytes, data)
    del data

    manifest = DocumentManifest(
        document_id=document_id,
        filename=filename or "document.pdf",
        content_type=result.content_type,
        byte_size=source_size,
        sha256=source_sha,
        source_key=source_key,
    )
    if paged:
        word_key = K.source_key(document_id, office_fmt.extension)
        await run_in_threadpool(
            store.put_file, word_key, upload_path, content_type=office_fmt.content_type
        )
        manifest.office_source_key = word_key
        manifest.office_format = office_fmt.extension
        manifest.original_upload_key = word_key
        manifest.original_sha256 = upload_sha
        manifest.original_content_type = office_fmt.content_type
    if is_image:
        original_key = K.doc_prefix(document_id) + "/original-upload"
        await run_in_threadpool(
            store.put_file, original_key, upload_path, content_type=original_mime
        )
        manifest.original_upload_key = original_key
        manifest.original_sha256 = upload_sha
        manifest.original_content_type = original_mime
    manifest.analysis.page_count = result.page_count
    manifest.analysis.warnings.extend(notes)
    if notes:
        # The analysis stage rebuilds manifest.analysis from the pages, which
        # would drop these; kept beside it, they are added back when the
        # document is read (get_document, the report).
        store.put_text(
            K.upload_notes_key(document_id),
            json.dumps({"warnings": notes, "pictures_not_read": pictures_not_read}),
            content_type="application/json",
        )
    manifests.save_manifest(manifest)

    job = get_job_runner().submit_analysis(document_id)
    log.info(
        "Accepted upload (%.1f KB, %d pages) as %s",
        source_size / 1024,
        result.page_count,
        document_id,
    )
    return UploadResponse(
        document_id=document_id,
        job_id=job.job_id,
        filename=manifest.filename,
        byte_size=source_size,
        page_count=result.page_count,
        warnings=notes,
    )


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str) -> DocumentOut:
    manifest = _load_manifest(document_id)
    return DocumentOut.of(
        manifest,
        _latest_job(document_id),
        upload_warnings=_upload_notes(document_id).get("warnings", []),
    )


@router.get("/documents/{document_id}/job", response_model=JobOut)
def get_job(document_id: str) -> JobOut:
    """Poll target for the progress display. Cheap: job state only."""
    job = _latest_job(document_id)
    if job is None:
        raise _fail(404, "No processing job was found for that document.")
    return JobOut.of(job)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    removed = get_manifest_store().delete_document(document_id)
    return {"deleted": removed > 0, "objects_removed": removed}


@router.post("/documents/{document_id}/cancel")
def cancel_processing(document_id: str) -> dict:
    """Stop the document's processing: "Start over" while it is still running.

    A queued job is dropped; a running one stops at its next page or stage
    and ends as Cancelled. `cancelled` is False when nothing was running.
    """
    _load_manifest(document_id)  # 404 for a document that does not exist
    return {"cancelled": bool(get_job_runner().cancel(document_id))}


@router.post("/documents/{document_id}/retry")
def retry_pages(document_id: str, pages: list[int]) -> dict:
    job = _latest_job(document_id)
    if job is None:
        raise _fail(404, "No processing job was found for that document.")
    get_job_runner().retry_pages(document_id, job.job_id, pages)
    return {"retrying": pages, "job_id": job.job_id}


# --------------------------------------------------------------------------
# Page detail
# --------------------------------------------------------------------------
@router.get("/documents/{document_id}/pages/{number}")
def get_page(document_id: str, number: int) -> dict:
    """Full object model for one page, for the viewer and debug panel."""
    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    if page is None:
        raise _fail(404, f"Page {number} does not exist in this document.")
    return page.model_dump(mode="json")


def _serve_png(
    key: str, filename: str, missing: str = "That page image is not available yet."
) -> Response:
    store = get_object_store()
    try:
        data = store.get_bytes(key)
    except (ObjectNotFoundError, InvalidKeyError) as exc:
        # InvalidKeyError: an id no document could have.
        raise _fail(404, missing, str(exc)) from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": PNG_CACHE,
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )


# The page-image routes build the storage key from the page number rather
# than reading it from the manifest: every one of these keys is fixed by
# storage/keys.py, and parsing a 100-page manifest (about 2 MB) once per
# thumbnail made the page strip fill in slowly on the free server. A page that
# does not exist, or is not drawn yet, has no file, so it is still a 404.
def _page_png(
    document_id: str, number: int, key_for, filename: str, missing: str
) -> Response:
    if number < 1:
        raise _fail(404, missing)
    return _serve_png(key_for(document_id, number - 1), filename, missing)


@router.get("/documents/{document_id}/pages/{number}/preview")
def page_preview(document_id: str, number: int) -> Response:
    return _page_png(
        document_id, number, K.page_source_render_key, f"page-{number}.png",
        f"No preview is available for page {number}.",
    )


@router.get("/documents/{document_id}/pages/{number}/thumb")
def page_thumb(document_id: str, number: int) -> Response:
    return _page_png(
        document_id, number, K.page_source_thumb_key, f"page-{number}-thumb.png",
        f"No thumbnail is available for page {number}.",
    )


@router.get("/documents/{document_id}/pages/{number}/overlay")
def page_overlay(
    document_id: str,
    number: int,
    layers: str = Query(
        default=",".join(ALL_LAYERS),
        description="Comma-separated overlay layers to draw.",
    ),
    labels: bool = Query(default=True),
) -> Response:
    """Debug overlay: the object model drawn on the page (spec sec. 37)."""
    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    if page is None:
        raise _fail(404, f"Page {number} does not exist in this document.")
    if not page.renders.source_preview_key:
        raise _fail(404, f"No preview is available for page {number}.")

    requested = tuple(
        layer.strip() for layer in layers.split(",") if layer.strip() in ALL_LAYERS
    )
    if not requested:
        requested = ALL_LAYERS

    store = get_object_store()
    try:
        png = store.get_bytes(page.renders.source_preview_key)
    except ObjectNotFoundError as exc:
        raise _fail(404, "That page image is not available yet.", str(exc)) from exc

    try:
        composed = render_overlay(png, page, layers=requested, labels=labels)
    except Exception as exc:
        raise _fail(500, "The debug overlay could not be drawn.", repr(exc)) from exc

    return Response(
        content=composed,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/documents/{document_id}/source")
def get_source(document_id: str) -> Response:
    """Serve the original upload. Never a public static path."""
    manifest = _load_manifest(document_id)
    store = get_object_store()
    try:
        data = store.get_bytes(manifest.original_upload_key or manifest.source_key)
    except ObjectNotFoundError as exc:
        raise _fail(404, "The original file is no longer available.", str(exc)) from exc
    return Response(
        content=data,
        media_type=manifest.original_content_type or manifest.content_type,
        headers={
            "Content-Disposition": _content_disposition("inline", manifest.filename),
            "Cache-Control": "private, no-store",
        },
    )


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    target_language: str
    source_language: str | None = None
    output_mode: OutputLanguageMode = OutputLanguageMode.TRANSLATED_ONLY
    processing_mode: ProcessingMode = ProcessingMode.HIGH_FIDELITY
    translate_image_text: bool = True


@router.post("/documents/{document_id}/translate", response_model=JobOut)
def start_translation(document_id: str, body: TranslateRequest) -> JobOut:
    manifest = _load_manifest(document_id)

    language = get_language(body.target_language)
    if language is None:
        raise _fail(400, "That language is not supported.")

    if manifest.processed_page_count == 0:
        latest = _latest_job(document_id)
        if latest is not None and latest.stage == JobStage.FAILED:
            # Reading it already ended in failure: waiting will not help.
            raise _fail(
                409,
                "This file could not be read, so it can't be translated. "
                "Please upload a different copy.",
                f"job={latest.job_id} stage=FAILED",
            )
        if latest is not None and latest.stage == JobStage.CANCELLED:
            # Reading it was stopped: it will never finish on its own.
            raise _fail(
                409,
                "Reading this file was stopped, so it can't be translated. "
                "Please upload it again.",
                f"job={latest.job_id} stage=CANCELLED",
            )
        raise _fail(
            409,
            "This document has not finished being analyzed yet. Please wait a "
            "moment and try again.",
        )

    # Fail before queuing rather than halfway through a 100-page document.
    try:
        provider = get_translation_provider()
    except Exception as exc:
        # Not 503: the web client reads 503 as "the server is waking up" and
        # retries for minutes, and no amount of waiting fixes a missing setup.
        raise _fail(
            500,
            getattr(exc, "message", "Translation is not configured on this server."),
            getattr(exc, "diagnostic", None) or repr(exc),
        ) from exc

    if not provider.supports(body.source_language, body.target_language):
        raise _fail(
            400,
            f"{language.english_name} is not available with the translation "
            "engine configured on this server.",
            f"provider={provider.name}",
        )

    # Every engine that offers this language has used up its allowance:
    # refuse now, saying when it comes back, rather than queue a job that
    # would leave every page in English.
    source_lang = body.source_language or manifest.analysis.detected_language
    # (untranslated_reason names the allowance only when every one is out.)
    reason = untranslated_reason(provider, source_lang, body.target_language)
    if reason != SERVICE_UNREACHABLE:
        # 409, not 503: clients retry 503 as a server that is still waking,
        # which would hide this message behind minutes of waiting.
        raise _fail(409, reason, f"provider={provider.name} exhausted")

    settings = JobSettings(
        target_language=body.target_language,
        source_language=body.source_language or manifest.analysis.detected_language,
        output_mode=body.output_mode,
        processing_mode=body.processing_mode,
        translate_image_text=body.translate_image_text,
    )
    job = get_job_runner().submit_translation(document_id, settings)
    return JobOut.of(job)


@router.get("/documents/{document_id}/pages/{number}/translated")
def translated_preview(document_id: str, number: int) -> Response:
    return _page_png(
        document_id, number, K.page_translated_render_key,
        f"page-{number}-translated.png",
        f"No translated page {number} is available yet.",
    )


@router.get("/documents/{document_id}/pages/{number}/translated-thumb")
def translated_thumb(document_id: str, number: int) -> Response:
    return _page_png(
        document_id, number, K.page_translated_thumb_key,
        f"page-{number}-translated-thumb.png",
        f"No translated page {number} is available yet.",
    )


@router.get("/documents/{document_id}/pages/{number}/plate")
def page_plate(document_id: str, number: int) -> Response:
    """The page with its translations removed: the live editor's background."""
    return _page_png(
        document_id, number, K.page_plate_key, f"page-{number}-plate.png",
        f"No editing background for page {number} yet.",
    )


@router.get("/documents/{document_id}/pages/{number}/blocks/{block_id}/sprite")
def block_sprite(
    document_id: str,
    number: int,
    block_id: str,
    scale: float = Query(2.0, ge=0.5, le=6.0),
) -> Response:
    """One translated block drawn alone, transparent, for the live editor.

    Drawn on request from the block's recorded layout by the same renderer
    as the PDF. Cheap -- a few lines of text -- so it is not cached on the
    server; the browser caches it by the page revision in the URL.
    """
    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    obj = page.find_text_object(block_id) if page else None
    if obj is None or obj.drawn_layout is None or obj.drawn_bbox is None:
        raise _fail(404, "That text has not been drawn on the page.")
    try:
        png = render_sprite(obj, scale=scale)
    except ValueError as exc:
        raise _fail(404, str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": PNG_CACHE},
    )


def _content_disposition(disposition: str, filename: str) -> str:
    """A Content-Disposition header that survives any file name (RFC 6266).

    The real name goes in `filename*`, UTF-8 encoded per RFC 5987, so an
    Arabic or Vietnamese file name arrives intact; `filename` is an ASCII
    stand-in for anything too old to read the first. A raw non-ASCII name
    in the header could not be encoded at all, and the request failed.
    """
    fallback = "".join(
        c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in filename
    )
    return (
        f"{disposition}; filename*=UTF-8''{quote(filename, safe='')}; "
        f'filename="{fallback}"'
    )


def _download_name(
    manifest: DocumentManifest, job, extension: str, note: str = ""
) -> str:
    """The original name with the language added, e.g. "Fractions (Spanish).pdf".

    Several language copies of one worksheet stay told apart in Downloads and
    in Drive.
    """
    stem = manifest.filename.rsplit(".", 1)[0].strip() or "document"
    lang = get_language(job.settings.target_language)
    name = lang.english_name if lang else job.settings.target_language
    return f"{stem} ({name}{note}).{extension}"


def _fresh_export(document_id: str):
    """The manifest and job, with the export rebuilt if edits are pending.

    Only once the completion gate has passed.
    """
    manifest = _load_manifest(document_id)
    job = _latest_job(document_id)
    if job is None or not job.export_key:
        raise _fail(404, "This document has not been translated yet.")
    if job.stage.value != "COMPLETE":
        raise _fail(
            409,
            "This translation is not finished yet, so it cannot be downloaded.",
            f"stage={job.stage.value}",
        )

    # Rebuild once here if edits have landed since the last build, rather
    # than on every edit (spec sec. 36). Only the pages that changed are
    # redrawn; see rebuild_export_only.
    if job.export_stale:
        from ..pipeline.reconstruct_stage import rebuild_export_only

        with document_lock(document_id):
            manifest = _load_manifest(document_id)
            job = _latest_job(document_id) or job
            if job.export_stale:
                try:
                    export = rebuild_export_only(
                        manifest,
                        store=get_object_store(),
                        target_lang=job.settings.target_language,
                    )
                except Exception as exc:
                    raise _fail(
                        500,
                        "The translated file could not be prepared with your "
                        "latest changes.",
                        repr(exc),
                    ) from exc
                if export.output_page_count != manifest.source_page_count:
                    raise _fail(
                        500,
                        f"The rebuilt document has {export.output_page_count} pages "
                        f"but the original has {manifest.source_page_count}.",
                    )
                job.export_key = export.export_key
                job.export_stale = False
                get_manifest_store().save_job(job)
    return manifest, job


_DOWNLOAD_CHUNK = 256 * 1024


def _read_and_close(fh):
    """The open file in pieces, closed when done or when the client leaves."""
    try:
        while chunk := fh.read(_DOWNLOAD_CHUNK):
            yield chunk
    finally:
        fh.close()


# Read by the web app after an Office download; listed in the CORS
# expose_headers in main.py, or the browser would hide them from the page.
PICTURES_KEPT_HEADER = "X-CT-Pictures-Kept"
PARAGRAPHS_KEPT_HEADER = "X-CT-Paragraphs-Kept"
PICTURE_NOT_REPLACED = "office_picture_not_replaced"


def _record_pictures_kept(document_id: str, kept_pages: list[int]) -> None:
    """Note on each page a picture that kept its original words in the file.

    The old notes are cleared first, so a repeat download neither repeats
    them nor keeps one for a picture that has since come through. Nothing is
    saved when nothing changed. A failure here never fails the download.
    """
    from ..model.document import QAFinding

    counts: dict[int, int] = {}
    for n in kept_pages:
        counts[n] = counts.get(n, 0) + 1
    try:
        with document_lock(document_id):
            manifest = _load_manifest(document_id)
            changed = False
            for page in manifest.pages:
                before = [f for f in page.qa.findings if f.code == PICTURE_NOT_REPLACED]
                want = counts.get(page.number, 0)
                if len(before) == want:
                    continue
                page.qa.findings = [
                    f for f in page.qa.findings if f.code != PICTURE_NOT_REPLACED
                ]
                page.qa.findings.extend(
                    QAFinding(
                        severity="warning",
                        code=PICTURE_NOT_REPLACED,
                        message=(
                            f"A picture on page {page.number} kept its original "
                            "words in the saved file."
                        ),
                    )
                    for _ in range(want)
                )
                changed = True
            if changed:
                get_manifest_store().save_manifest(manifest)
    except Exception as exc:
        log.warning("Could not note kept pictures for %s: %r", document_id, exc)


@router.get("/documents/{document_id}/download")
def download_translated(document_id: str) -> Response:
    """Serve the finished file, but only once the completion gate has passed."""
    manifest, job = _fresh_export(document_id)
    store = get_object_store()
    # A translated Word document comes back as a Word document, so the
    # teacher can keep editing it. Serving it as application/pdf would make
    # the browser save a file Word refuses to open.
    fmt = output_format_of(manifest)
    headers = {
        "Content-Disposition": _content_disposition(
            "attachment", _download_name(manifest, job, fmt.extension)
        ),
        "Cache-Control": "private, no-store",
    }

    if manifest.office_source_key:
        # Shown as pages, handed back as the Word or PowerPoint file she
        # uploaded, with the pages' translations and her edits written in.
        from ..pipeline.word_export import build_office_export
        try:
            export = build_office_export(
                manifest, store=store, target_lang=job.settings.target_language,
                translated_pdf=store.get_bytes(job.export_key),
            )
        except Exception as exc:
            raise _fail(
                500, "The translated file could not be prepared.", repr(exc),
            ) from exc
        kept_pages = list(getattr(export, "pictures_kept_pages", None) or [])
        _record_pictures_kept(document_id, kept_pages)
        if kept_pages:
            # What the editor showed and what the file holds differ; the page
            # says so. "<count>;pages=<page,page>", one count per picture.
            unique = ",".join(str(n) for n in sorted(set(kept_pages)))
            headers[PICTURES_KEPT_HEADER] = f"{len(kept_pages)};pages={unique}"
        kept_paragraphs = int(getattr(export, "paragraphs_kept_original", 0) or 0)
        if kept_paragraphs > 0:
            headers[PARAGRAPHS_KEPT_HEADER] = str(kept_paragraphs)
        return Response(
            content=export.data, media_type=fmt.content_type, headers=headers
        )

    path = store.local_path(job.export_key)
    if path is not None:
        # Sent from disk in pieces: a large PDF is never read into memory
        # just to be handed over. The file is opened here, not when the body
        # starts streaming: a download after an edit can replace the export
        # in place, and the length sent must be that of the file whose bytes
        # follow. An open handle keeps the file it was opened on.
        try:
            fh = open(path, "rb")
        except FileNotFoundError as exc:
            raise _fail(
                404, "The translated file is no longer available.", str(exc),
            ) from exc
        headers["Content-Length"] = str(os.fstat(fh.fileno()).st_size)
        return StreamingResponse(
            _read_and_close(fh), media_type=fmt.content_type, headers=headers,
            # Also closed if the body never started (the client went away).
            background=BackgroundTask(fh.close),
        )
    try:
        data = store.get_bytes(job.export_key)
    except ObjectNotFoundError as exc:
        raise _fail(404, "The translated file is no longer available.", str(exc)) from exc
    return Response(content=data, media_type=fmt.content_type, headers=headers)


@router.get("/documents/{document_id}/report")
def translation_report(document_id: str) -> dict:
    """The end-of-run report (spec sec. 47). States problems plainly."""
    manifest = _load_manifest(document_id)
    job = _latest_job(document_id)
    if job is None:
        raise _fail(404, "This document has not been translated yet.")

    lang = get_language(job.settings.target_language)
    fmt = format_of(manifest)
    out_fmt = output_format_of(manifest)
    pages = manifest.pages
    review = [p.number for p in pages if p.qa.status.value == "REVIEW_SUGGESTED"]
    required = [p.number for p in pages if p.qa.status.value == "REVIEW_REQUIRED"]

    compromises: list[str] = []
    for p in pages:
        for f in p.qa.findings:
            if f.code in ("font_reduced", "text_overflow", "collision_response_area",
                          "image_aspect_changed", "shaping_unavailable"):
                compromises.append(f"Page {p.number}: {f.message}")

    # OCR facts. A teacher needs to know which pages were *read* rather than
    # extracted, because those carry recognition risk.
    scanned_pages = [p.number for p in pages if p.is_scanned]
    ocr_objects = [
        t for p in pages for t in p.text_objects if t.source.value == "OCR"
    ]
    ocr_confidences = [t.confidence for t in ocr_objects if t.confidence is not None]
    mean_ocr = (
        round(sum(ocr_confidences) / len(ocr_confidences), 3)
        if ocr_confidences
        else None
    )
    low_confidence_pages = sorted(
        {
            p.number
            for p in pages
            for f in p.qa.findings
            if f.code in ("ocr_low_confidence", "ocr_discarded", "ocr_failed")
        }
    )

    # Text inside pictures. "Partially" is the honest answer whenever any
    # label was too unclear to read (spec sec. 47: never claim perfection).
    image_labels = [
        t for p in pages for t in p.text_objects if t.source.value == "IMAGE_OCR"
    ]
    image_labels_translated = sum(
        1
        for t in image_labels
        if t.translation_status.value in ("TRANSLATED", "MANUAL_EDIT")
    )
    images_with_unreadable = sorted(
        {
            p.number
            for p in pages
            for f in p.qa.findings
            if f.code in (
                "image_label_unreadable", "image_text_unreadable", "image_text_not_read"
            )
        }
    )
    examined_images = [i for p in pages for i in p.image_objects]
    embedded_media = manifest.analysis.embedded_media_count
    if fmt.rebuilds_in_place and embedded_media:
        # Pictures in an Office file are never opened, so nothing can be said
        # about text inside them. Saying "no pictures" would be false, and
        # saying "processed" would be a claim the engine has not earned.
        image_text_status = "Not examined; pictures are left exactly as they are"
        reason = _upload_notes(document_id).get("pictures_not_read")
        if reason:
            # Why they were not read, when the cause was the file not being
            # laid out as pages -- that one she can do something about.
            image_text_status += (
                f". {reason} To translate words in pictures, download it as "
                "a PDF and upload that."
            )
    elif not examined_images and not embedded_media:
        image_text_status = "No pictures in this document"
    elif not job.settings.translate_image_text:
        # She chose not to have them read; "no text" would be a claim.
        image_text_status = "Not examined (turned off)"
    elif not examined_images:
        image_text_status = "Not examined"
    elif images_with_unreadable:
        # Ahead of "no readable text": a picture that could not be read at
        # all is not a picture with no words in it.
        image_text_status = "Partially processed"
    elif not image_labels:
        image_text_status = "No readable text found inside pictures"
    else:
        image_text_status = "Processed"

    # Counted from the rebuilt file, both sides measured the same way, so
    # this line says what a teacher reads it as saying: how much of the
    # artwork came through. It used to count how many images the engine had
    # managed to cache a copy of, which reported "40 / 42 -- with
    # exceptions" on a document whose pictures were pixel-identical.
    counted = [
        (p.diagnostics.extra.get("source_image_count"),
         p.diagnostics.extra.get("output_image_count"))
        for p in pages
    ]
    measured = [(a, b) for a, b in counted if isinstance(a, int) and isinstance(b, int)]
    if measured:
        images_total = sum(a for a, _ in measured)
        images_kept = sum(min(a, b) for a, b in measured)
    else:
        # Nothing rebuilt yet (or a format that does not report counts).
        images_total = sum(len(p.image_objects) for p in pages)
        images_kept = images_total
    if fmt.rebuilds_in_place:
        # Reconstruction rewrites text inside runs and copies every other
        # part of the package through, so the picture bytes are the original
        # bytes by construction. This is stated as a fact, not a measurement.
        images_total = images_kept = embedded_media
    untranslated = _blocks_left_untranslated(pages)

    return {
        "source_language": source_language_name(
            job.settings.source_language or manifest.analysis.detected_language
        ),
        "target_language": lang.english_name if lang else job.settings.target_language,
        # For lang= on translated text in the page.
        "target_language_code": lang.code if lang else job.settings.target_language,
        "translation_mode": job.settings.output_mode.value,
        "processing_mode": job.settings.processing_mode.value,
        "translation_engine": get_settings().translation_provider,
        # A Word document's "pages" are sections, and Word repaginates when
        # it opens the file. The report says so rather than quoting a page
        # count that will not match what the teacher prints.
        "output_format": out_fmt.name,
        "output_extension": out_fmt.extension,
        "unit_noun": fmt.page_noun,
        "original_pages": manifest.source_page_count,
        "translated_pages": manifest.translated_page_count,
        "rendered_pages": manifest.rendered_page_count,
        "verified_pages": manifest.verified_page_count,
        "page_correspondence": (
            f"{manifest.rendered_page_count} / {manifest.source_page_count}"
            f" {fmt.page_noun}s"
        ),
        "page_counts_match": manifest.rendered_page_count == manifest.source_page_count,
        "original_images_preserved": (
            "Yes"
            if images_total and images_kept == images_total
            else ("Not applicable" if not images_total else "With exceptions")
        ),
        "images_preserved_count": f"{images_kept} / {images_total}",
        # Stated plainly: image text is a later phase and we do not imply
        # otherwise (spec sec. 47 -- never claim perfection).
        "scanned_pages": scanned_pages,
        "scanned_pages_read": len([p for p in pages if p.ocr_done and p.is_scanned]),
        "text_recognition_confidence": mean_ocr,
        "pages_hard_to_read": low_confidence_pages,
        "embedded_image_text": image_text_status,
        "image_labels_translated": image_labels_translated,
        "images_with_unreadable_text": images_with_unreadable,
        "automated_content_review": (
            f"{manifest.verified_page_count} / {manifest.source_page_count} "
            f"{fmt.page_noun}s"
        ),
        "pages_review_suggested": review,
        "pages_review_required": required,
        "pages_failed": manifest.failed_pages,
        "blocks_left_untranslated": untranslated,
        "layout_compromises": compromises[:40],
        "status": job.stage.value,
        "downloadable": job.stage.value == "COMPLETE",
    }


# --------------------------------------------------------------------------
# Manual correction (spec sec. 36)
# --------------------------------------------------------------------------
class BoxIn(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class EditRequest(BaseModel):
    """Any combination of text and layout changes, applied in one call."""

    text: str | None = Field(default=None, max_length=8000)
    bbox: BoxIn | None = None
    color: str | None = Field(default=None, max_length=7)
    font_size: float | None = None
    alignment: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    # "sans", "serif", "mono", a catalog font ("andika", see
    # fonts/catalog.py), or "auto" to go back to the automatic choice.
    font_family: str | None = Field(default=None, max_length=16)
    # Background colour "#rrggbb", or "none" to remove it.
    fill: str | None = Field(default=None, max_length=7)
    # True: size the words to fill the box (and refit when it changes).
    fit: bool | None = None
    # "front" or "back": stack this box above or below everything else.
    layer: str | None = Field(default=None, max_length=5)
    # Tilted picture words: "straight" draws the translation level,
    # "follow" turns it with the original words again.
    rotation: str | None = Field(default=None, max_length=8)
    # "top", "middle" or "bottom": where the words sit in the box.
    vertical_align: str | None = Field(default=None, max_length=6)
    # 1.0, 1.15, 1.5 or 2.0 times the size; 0 goes back to automatic.
    line_spacing: float | None = None
    underline: bool | None = None
    # Colour behind the words only, "#rrggbb", or "none" to remove it.
    highlight: str | None = Field(default=None, max_length=7)
    # A line around the box: "#rrggbb" or "none"; a weight in points (one
    # of fitting.BORDER_WEIGHTS); "solid", "dashed" or "dotted".
    border_color: str | None = Field(default=None, max_length=7)
    border_width: float | None = None
    border_dash: str | None = Field(default=None, max_length=6)
    # Space inside the box in points: 0, 2, 4 or 8.
    padding: float | None = None
    # Degrees clockwise to turn the box about its centre; 0 is straight.
    # With a bbox, the bbox is the unturned frame.
    angle: float | None = None
    # Lists (reconstruct/lists.py): "bullet", "number", or "none" to turn
    # the list off (her indents are kept).
    # (Named list_kind here: a field called `list` would hide the type
    # from the annotations below it.)
    list_kind: str | None = Field(default=None, max_length=6, alias="list")
    # The style of the list's first level: "disc", "circle", "square",
    # "dash", "check", or "decimal", "lower-alpha", "upper-alpha",
    # "lower-roman" (and "-paren" forms). Alone, it turns that kind on.
    list_style: str | None = Field(default=None, max_length=20)
    # The first number, 1 to 999.
    list_start: int | None = None
    # +1 or -1: every line of the box one level in or out (0 to 3).
    indent: int | None = None
    # Each line's level, 0 to 3, by line; with `text`, saved with it.
    list_levels: list[int] | None = Field(default=None, max_length=200)
    # A box she made drawn as a shape (reconstruct/shapes.SHAPES): "rect",
    # "roundRect", "ellipse". A plain string, checked in pipeline/edit.
    shape: str | None = Field(default=None, max_length=32)


class AddBlockRequest(BaseModel):
    """A new text block, for content the engine did not find."""

    # The text to draw, or empty when `source_text` is given instead.
    text: str = Field(default="", max_length=8000)
    # English (source-language) text to translate and draw, for a teacher
    # who cannot write the target language.
    source_text: str | None = Field(default=None, max_length=8000)
    # A background colour to cover what is beneath the box.
    fill: str | None = Field(default=None, max_length=7)
    bbox: BoxIn
    color: str | None = Field(default=None, max_length=7)
    font_size: float | None = None
    alignment: str | None = None


class BoxSpecIn(BaseModel):
    """One box to make: a duplicate, a pasted copy, or text pasted from
    outside the app. Every value is read by the same checks as an edit
    (pipeline/edit._spec_object)."""

    # The editor's own id for the box it is already showing; sent back
    # with the box the engine made, so it can swap one for the other.
    client_id: str = Field(max_length=64)
    # The FRAME in page points: the unturned box when `angle` is set.
    bbox: BoxIn
    # Degrees clockwise; 0 is straight.
    angle: float = 0.0
    # The translation to draw.
    text: str = Field(default="", max_length=8000)
    # The English it says, kept as the box's source so it can be changed in
    # English later. Null for a box typed straight into the translation.
    source_text: str | None = Field(default=None, max_length=8000)
    # True: translate `source_text` now and draw that (`text` is ignored).
    translate: bool = False
    # The machine translation of the box copied, so Reset on the copy
    # brings back what Reset on the original would.
    machine_text: str | None = Field(default=None, max_length=8000)
    # A size locks it; Fit fills the box. Not both.
    font_size: float | None = None
    fit: bool = False
    # The face the engine picks when the type style is automatic
    # (`font_look` in a block), so a copy is drawn in the same one.
    font_look: str = Field(default="sans", max_length=5)
    # "auto", an older style ("sans", "serif", "mono") or a catalog font.
    font_family: str = Field(default="auto", max_length=16)
    bold: bool | None = None
    italic: bool | None = None
    color: str | None = Field(default=None, max_length=7)
    alignment: str | None = Field(default=None, max_length=8)
    vertical_align: str = Field(default="top", max_length=6)
    line_spacing: float = 0.0
    underline: bool = False
    highlight: str = Field(default="none", max_length=7)
    fill: str = Field(default="none", max_length=7)
    border_color: str = Field(default="none", max_length=7)
    border_width: float | None = None
    border_dash: str = Field(default="solid", max_length=6)
    padding: float = 0.0
    # Text pasted from outside: after translating, make the box as tall as
    # the rows it is drawn in.
    fit_height: bool = False
    # Its list, as an edit sets it: "none", "bullet" or "number", the
    # first level's style, the first number and each line's level.
    list_kind: str = Field(default="none", max_length=6, alias="list")
    list_style: str | None = Field(default=None, max_length=20)
    list_start: int | None = None
    list_levels: list[int] | None = Field(default=None, max_length=200)
    # Drawn as a shape (reconstruct/shapes.SHAPES); null is a plain box.
    shape: str | None = Field(default=None, max_length=32)


class BatchAddRequest(BaseModel):
    blocks: list[BoxSpecIn] = Field(min_length=1, max_length=200)


class RemoveRequest(BaseModel):
    object_ids: list[str] = Field(min_length=1, max_length=400)


class BulkRequest(BaseModel):
    object_ids: list[str] = Field(min_length=1, max_length=400)
    color: str | None = Field(default=None, max_length=7)
    font_size: float | None = None
    alignment: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    font_family: str | None = Field(default=None, max_length=16)
    fill: str | None = Field(default=None, max_length=7)
    fit: bool | None = None
    layer: str | None = Field(default=None, max_length=5)
    vertical_align: str | None = Field(default=None, max_length=6)
    line_spacing: float | None = None
    underline: bool | None = None
    highlight: str | None = Field(default=None, max_length=7)
    border_color: str | None = Field(default=None, max_length=7)
    border_width: float | None = None
    border_dash: str | None = Field(default=None, max_length=6)
    padding: float | None = None
    # Each box turns about its own centre.
    angle: float | None = None
    # Lists (reconstruct/lists.py): "bullet", "number", or "none" to turn
    # the list off (her indents are kept).
    # (Named list_kind here: a field called `list` would hide the type
    # from the annotations below it.)
    list_kind: str | None = Field(default=None, max_length=6, alias="list")
    # The style of the list's first level: "disc", "circle", "square",
    # "dash", "check", or "decimal", "lower-alpha", "upper-alpha",
    # "lower-roman" (and "-paren" forms). Alone, it turns that kind on.
    list_style: str | None = Field(default=None, max_length=20)
    # The first number, 1 to 999.
    list_start: int | None = None
    # +1 or -1: every line of the box one level in or out (0 to 3).
    indent: int | None = None
    # Boxes she made drawn as this shape; others in the selection keep theirs.
    shape: str | None = Field(default=None, max_length=32)


def _line_heights(obj) -> tuple[float, float]:
    """(line_height, auto_line_height) as multiples of the size.

    The first is the spacing the picture really used, including the
    source's own, so the editor's stand-in text matches it. The second is
    where automatic spacing starts with no choice of hers, for the moment
    she picks Automatic, before the engine has drawn it.
    """
    auto = DEFAULT_LEADING
    # Divided by the size lay_out starts from (its base_size), so this is
    # the ratio fit_text really tries first.
    size = obj.manual_font_size or obj.style.font_size
    if not obj.manual_fit and not obj.manual_size_locked and obj.line_spacing and size:
        ratio = obj.line_spacing / size
        if MIN_LEADING <= ratio <= 2.2:
            auto = round(ratio, 4)
    layout = obj.drawn_layout
    leading = layout.leading if layout is not None else None
    if layout is not None and not leading:
        # Drawn before the leading was saved (a project from before line
        # spacing): read it off the baselines, the closest two, since a
        # blank row leaves a double gap.
        baselines = sorted({round(b, 2) for _, b, _ in layout.lines})
        gaps = [y1 - y0 for y0, y1 in zip(baselines, baselines[1:]) if y1 - y0 > 0.01]
        leading = min(gaps) if gaps else None
    if layout is not None and leading and layout.size:
        drawn = round(leading / layout.size, 4)
    else:
        drawn = obj.manual_line_spacing or auto
    return drawn, auto


def _printed_rows(obj) -> list[str]:
    rows: list[list] = []
    for line in sorted(obj.lines, key=lambda ln: (ln.bbox.y0, ln.bbox.x0)):
        text = line.text.replace("\u200b", "").strip()
        if not text:
            continue
        mid = (line.bbox.y0 + line.bbox.y1) / 2
        if rows and rows[-1][0].bbox.y0 <= mid <= rows[-1][0].bbox.y1:
            rows[-1].append(line)
        else:
            rows.append([line])
    texts = [
        "   ".join(
            ln.text.replace("\u200b", "").strip()
            for ln in sorted(row, key=lambda ln: ln.bbox.x0)
            if ln.text.replace("\u200b", "").strip()
        )
        for row in rows
    ]
    # A row that only wraps the one above ("...to the" / "other side using
    # inverse operations") joins it: translated alone, half a sentence came
    # back garbled. A wrap starts in lower case after a row that had not
    # finished its sentence.
    out: list[str] = []
    for text in texts:
        if out and text[:1].islower() and not out[-1].rstrip().endswith((".", ":", "?", "!")):
            out[-1] = f"{out[-1]} {text}"
        else:
            out.append(text)
    return out


def _auto_base_size(obj) -> float | None:
    """Where the engine starts sizing a block whose size it chooses (see
    `auto_base_size` in _block_out), capped as builder.lay_out caps a
    turned picture label at its measured letters' height."""
    if obj.manual_fit or (obj.manual_size_locked and obj.manual_font_size):
        return None
    base = obj.manual_font_size or obj.style.font_size or 10.0
    ob = obj.oriented
    if (
        obj.source is TextSource.IMAGE_OCR and ob is not None
        and obj.render_angle and not obj.manual_font_size
    ):
        base = min(base, round(max(4.0, ob.height * 0.92), 1))
    return base


def _list_out(obj) -> dict:
    """A block's list as the editor reads it (reconstruct/lists.py)."""
    text = obj.translated_text or ""
    font = None
    layout = obj.drawn_layout
    if layout is not None and (layout.font_builtin or layout.font_path):
        # The face the words were drawn in decides which face a marker
        # takes, so the plan sent is the one drawn.
        font = REGISTRY._load(
            builtin=layout.font_builtin,
            path=Path(layout.font_path) if layout.font_path else None,
        )
    paras = lists.plan_paragraphs(text, obj, font)
    return {
        "list": obj.manual_list,
        # The first level's style as drawn (the default when unset), the
        # first number, and each line's level.
        "list_style": (
            (obj.manual_list_style or lists.DEFAULT_STYLE[obj.manual_list])
            if obj.manual_list else None
        ),
        "list_start": obj.manual_list_start or 1,
        "list_levels": lists.effective_levels(
            obj.manual_indents, len(lists.paragraphs_of(text))
        ),
        # Numbers or bullets typed in the text itself, and whether a list
        # of each kind may be turned on over them; `message` says why not.
        "own_markers": lists.own_markers(text, lists.source_hang(obj)),
        # What the engine draws for each line: its marker (typed in the
        # text, or the list's own), the marker's face, its level, and the
        # source's own marker-to-text distance in points.
        "paragraphs": lists.paragraphs_out(paras),
    }


def _block_out(obj, page_width: float = 612.0, page_height: float = 792.0) -> dict:
    # A hand-placed box is exactly where she put it. Otherwise frame where
    # the rebuild really drew the words, which widening or reflow can move
    # well outside the source box.
    frame = obj.render_bbox if obj.manual_bbox else (obj.drawn_bbox or obj.render_bbox)
    layout = obj.drawn_layout
    if layout is not None and layout.quad:
        # A turned block's box is its UNTURNED frame; the editor turns it
        # about its centre by `angle` (see builder.turned_frame_of). Her
        # own angle's frame is exactly her box.
        frame = (
            obj.manual_bbox
            if obj.manual_angle is not None and obj.manual_bbox is not None
            else turned_frame_of(layout)[0]
        )
    line_height, auto_line_height = _line_heights(obj)
    return {
        "id": obj.id,
        "role": obj.role.value,
        "source": obj.source.value,
        "original": obj.flat_text,
        # The block's text row by row as printed, so the editor can offer
        # the original's line breaks to a teacher who cannot see them in
        # the translation.
        "original_lines": _printed_rows(obj),
        "translation": obj.translated_text,
        "machine_translation": obj.machine_translated_text,
        "manually_edited": obj.manually_edited,
        "status": obj.translation_status.value,
        "confidence": obj.confidence,
        "norm_bbox": obj.norm_bbox.model_dump(),
        "bbox": obj.bbox.model_dump(),
        # Where the translation is actually drawn, which is what the editor
        # must show and drag.
        # Where the block's sprite goes, in page points: the drawn frame grown
        # by the sprite's padding (sprite.sprite_area). For a turned block
        # it is in the unturned frame and turns with the box. None when
        # nothing is drawn for it.
        "sprite_bbox": (
            sprite_area(obj).model_dump()
            if obj.drawn_layout is not None and obj.drawn_bbox is not None
            else None
        ),
        # Changes only when this block's picture would: the browser keeps
        # every other block's sprite cached across an edit. The salt marks
        # pictures drawn since turned blocks' sprites became unturned, so a
        # browser never keeps an old pre-turned one.
        "sprite_rev": (
            sha256_bytes(
                b"s2" + (
                    obj.drawn_layout.model_dump_json() + obj.drawn_bbox.model_dump_json()
                ).encode()
            )[:12]
            if obj.drawn_layout is not None and obj.drawn_bbox is not None
            else None
        ),
        "render_bbox": frame.model_dump(),
        "render_norm_bbox": frame.normalized(page_width, page_height).model_dump(),
        "manual_bbox": obj.manual_bbox.model_dump() if obj.manual_bbox else None,
        "color": obj.manual_color or obj.style.color,
        "background_color": obj.background_color,
        "manual_color": obj.manual_color,
        "bold": obj.render_bold,
        "italic": obj.render_italic,
        "font_family": obj.manual_font_family,
        "fill": obj.manual_fill,
        "manual_bold": obj.manual_bold,
        "manual_italic": obj.manual_italic,
        # What is drawn, when the size follows the box.
        "font_size": (obj.drawn_layout.size if obj.manual_fit and obj.drawn_layout else None)
        or obj.manual_font_size
        or obj.translated_font_size
        or obj.style.font_size,
        "fit": obj.manual_fit,
        # Where the engine starts sizing a block whose size it chooses (not
        # Fit, not locked): it steps down from here until the words fit the
        # box (fitting.fit_text), so the editor's stand-in text can do the
        # same while a resize or a new spacing waits for the picture. Null
        # when the size is fixed.
        "auto_base_size": _auto_base_size(obj),
        # How far right (page x) the engine would widen the words into
        # empty space when they no longer fit (null when it never would).
        "widen_x1": None if obj.manual_bbox else obj.widen_x1,
        # Stacking order: higher is drawn on top.
        "z": obj.z_order,
        "manual_font_size": obj.manual_font_size,
        "has_manual_layout": obj.has_manual_layout,
        # As drawn: a justified source the parser detected reads LEFT,
        # because that is how it is drawn (see drawn_alignment).
        "alignment": obj.drawn_alignment.value,
        "manual_alignment": (
            obj.manual_alignment.value if obj.manual_alignment else None
        ),
        "vertical_align": obj.manual_vertical_align or "top",
        "manual_vertical_align": obj.manual_vertical_align,
        # Her line spacing; null is automatic.
        "line_spacing": obj.manual_line_spacing,
        "line_height": line_height,
        "auto_line_height": auto_line_height,
        "underline": obj.manual_underline,
        "highlight": obj.manual_highlight,
        # False when justified text in this script is drawn from its start
        # (fitting.justifiable), so the editor can say so and do the same.
        "justifiable": justifiable(obj.translated_text or ""),
        # The picture's own line breaks, so the editor's stand-in text
        # breaks where the picture does. Empty before it is drawn.
        # Each row also carries its column and marker (see DrawnRow):
        # `indent` and `hang` in points, the marker text and its face.
        "rows": (
            [
                {
                    "text": r.text, "end": r.end, "spread": r.spread,
                    "indent": r.indent, "hang": r.hang,
                    "marker": r.marker, "marker_face": r.marker_face,
                }
                for r in obj.drawn_layout.rows
            ]
            if obj.drawn_layout is not None
            else []
        ),
        **_list_out(obj),
        # The catalog font the words were drawn in (null when automatic or
        # an older style), and True when her font had no letters for this
        # language and the standard one for it was used.
        "font_face": layout.face if layout is not None else None,
        "font_fallback": bool(layout is not None and layout.face_fallback),
        "suppressed": obj.suppressed,
        # Tilted words in a picture: the slant the translation is drawn at
        # (degrees clockwise; 0 when level or set straight), the slant the
        # original words were measured at, and the unturned frame's size.
        # render_bbox is then the unturned frame and sprite_bbox the
        # unturned picture's area: the editor turns both by `angle` about
        # the frame's centre. Her free angle counts here too.
        "angle": (
            obj.drawn_layout.angle if obj.drawn_layout is not None else obj.render_angle
        ),
        "source_angle": obj.oriented.angle if obj.oriented is not None else None,
        "frame": (
            {"width": round(obj.oriented.width, 2), "height": round(obj.oriented.height, 2)}
            if obj.oriented is not None
            else None
        ),
        "manual_rotation": obj.manual_rotation,
        # Her own angle (null when straight or following a tilt), and
        # whether a turned picture label was centred along its frame
        # rather than drawn with its alignment.
        "manual_angle": obj.manual_angle,
        "centred": bool(layout is not None and layout.centred),
        # The box's border: colour (null when there is none), weight in
        # points and dash ("solid", "dashed" or "dotted"). The weight and
        # dash are sent while the border is off too: they are kept for next
        # time, so turning it back on draws the weight it had.
        "border_color": obj.manual_border_color,
        "border_width": obj.border_weight,
        "border_dash": obj.manual_border_dash or "solid",
        # A box she made drawn as a shape ("rect", "roundRect", "ellipse"),
        # or null for a plain box. The name is passed on as stored: an
        # editor that does not know it draws a rectangle, as the engine does.
        "shape": obj.manual_shape,
        # Space inside the box in points; 0 when none. The words lay out in
        # the box shrunk by it (fitting.inset), clamped the same way there.
        "padding": obj.manual_padding or 0,
        # Left in the original language on purpose: read too uncertainly to
        # translate, or kept by her choice.
        "keep_original": obj.keep_original,
        "is_manual_block": obj.is_manual_block,
        # The face the engine picks while the type style is automatic
        # ("sans", "serif" or "mono"), so a copy can be drawn in it.
        "font_look": _source_look(obj),
        # The English a copy of this block carries: what she typed for a
        # box she made (null when she typed the translation itself), the
        # source text for one the engine found. Not `original`, which for a
        # box typed in the translation is that translation.
        "copy_source": (
            obj.approved_source_text if obj.is_manual_block else obj.flat_text
        ),
    }


# Characters that occupy no space. A PDF producer emits them constantly as
# kerning and bidi hints, and `str.strip()` does not remove them -- so a
# block containing nothing but a zero-width space looked like real text and
# would have shown up in the manuscript as a run of empty rows.
_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\xa0"


def _has_visible_text(text: str | None) -> bool:
    return bool((text or "").strip(_INVISIBLE + " \t\n\r"))


def _target_language(document_id: str) -> str:
    job = _latest_job(document_id)
    if job is None:
        raise _fail(409, "This document has not been translated yet.")
    return job.settings.target_language


@router.get("/documents/{document_id}/findings")
def document_findings(document_id: str) -> dict:
    """Everything worth checking, grouped, for the editor's problem rail."""
    manifest = _load_manifest(document_id)
    groups = group_findings(manifest)
    return {"groups": groups, "total": sum(len(g["items"]) for g in groups),
        "reviewed": sorted(set(manifest.reviewed_findings) & finding_keys(groups))}


@router.get("/documents/{document_id}/blocks")
def document_blocks(document_id: str) -> dict:
    """Every correctable block in the document, in reading order.

    The manuscript editor shows the whole document as one continuous text,
    so it needs all of it at once. Fetching page by page would be 37 round
    trips on the test packet before a teacher could type a single word, and
    would make scrolling stutter as each page arrived.

    This is text only -- no page images -- so it stays small: the same
    packet is about 60 KB here.
    """
    manifest = _load_manifest(document_id)
    pages: list[dict] = []
    for page in manifest.pages:
        # EVERY text block, not only the correctable ones. The manuscript is
        # something a teacher reads straight through, and filtering to what
        # is editable would leave holes exactly where the equations, question
        # numbers and page numbers are -- the document would read as if it
        # were missing its maths. Protected blocks are shown and marked
        # locked instead, so their absence from editing is visible and
        # explained rather than silent.
        editable_ids = {o.id for o in editable_blocks(page)}
        ordered = sorted(
            page.text_objects,
            key=lambda o: (o.reading_index if o.reading_index >= 0 else 1_000_000),
        )
        blocks = []
        for obj in ordered:
            # A box she made with no words: a cover (fill or border) or a
            # shape, which is the shape she drew whatever its colours.
            is_cover = obj.is_manual_block and (
                obj.manual_fill is not None or obj.manual_border_color is not None
                or obj.manual_shape is not None
            )
            if not _has_visible_text(obj.flat_text) and not is_cover:
                continue
            out = _block_out(obj, page.width, page.height)
            out["editable"] = obj.id in editable_ids
            out["locked_reason"] = None if obj.id in editable_ids else lock_reason(obj)
            blocks.append(out)
        pages.append(
            {
                "page_number": page.number,
                "revision": page.revision,
                # Whether the live editor can show this page as background
                # plus movable sprites. Older documents, and Office pages,
                # fall back to the rendered page picture.
                "has_plate": page.renders.plate_key is not None,
                "width": page.width,
                "height": page.height,
                "status": page.qa.status.value,
                "can_undo": can_undo(page),
        "can_redo": bool(page.redo_history),
                # Why this page is flagged, so the editor can say so beside
                # the text rather than making a teacher open the report.
                "notes": [
                    f.message
                    for f in page.qa.findings
                    if f.severity in ("warning", "error")
                ][:4],
                "blocks": blocks,
            }
        )
    return {
        "document_id": document_id,
        "unit_noun": format_of(manifest).page_noun,
        "pages": pages,
        "total_blocks": sum(len(p["blocks"]) for p in pages),
    }


@router.get("/documents/{document_id}/pages/{number}/blocks")
def page_blocks(document_id: str, number: int) -> dict:
    """The correctable text on one page, for the review editor."""
    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    if page is None:
        raise _fail(404, f"Page {number} does not exist in this document.")
    return {
        "page_number": number,
        "revision": page.revision,
        "width": page.width,
        "height": page.height,
        "can_undo": can_undo(page),
        "can_redo": bool(page.redo_history),
        "blocks": [
            _block_out(o, page.width, page.height) for o in editable_blocks(page)
        ],
    }


def _run_edit(
    document_id: str,
    number: int,
    object_id: str,
    text: str | None,
    *,
    layout: EditRequest | None = None,
    list_levels: list[int] | None = None,
):
    manifests = get_manifest_store()
    store = get_object_store()

    # Serialize per document: an edit is read-modify-write on the manifest,
    # and two concurrent saves would otherwise lose one of them.
    with document_lock(document_id):
        manifest = _load_manifest(document_id)
        target = _target_language(document_id)
        try:
            if layout is not None:
                result = apply_layout(
                    manifest,
                    number,
                    object_id,
                    bbox=layout.bbox.model_dump() if layout.bbox else None,
                    color=layout.color,
                    font_size=layout.font_size,
                    alignment=layout.alignment,
                    bold=layout.bold,
                    italic=layout.italic,
                    font_family=layout.font_family,
                    fill=layout.fill,
                    fit=layout.fit,
                    layer=layout.layer,
                    rotation=getattr(layout, "rotation", None),
                    vertical_align=layout.vertical_align,
                    line_spacing=layout.line_spacing,
                    underline=layout.underline,
                    highlight=layout.highlight,
                    border_color=layout.border_color,
                    border_width=layout.border_width,
                    border_dash=layout.border_dash,
                    padding=layout.padding,
                    angle=layout.angle,
                    list=layout.list_kind,
                    list_style=layout.list_style,
                    list_start=layout.list_start,
                    indent=layout.indent,
                    list_levels=layout.list_levels,
                    shape=layout.shape,
                    store=store,
                    target_lang=target,
                )
            elif text is None:
                result = reset_edit(
                    manifest, number, object_id, store=store, target_lang=target
                )
            else:
                result = apply_edit(
                    manifest, number, object_id, text, list_levels=list_levels,
                    store=store, target_lang=target,
                )
        except EditError as exc:
            raise _fail(400, exc.message, exc.diagnostic) from exc
        manifests.save_manifest(manifest)

        # The rendered page is already correct; only the single downloadable
        # file is now behind, and it is rebuilt on demand.
        job = _latest_job(document_id)
        if job is not None:
            job.export_stale = True
            manifests.save_job(job)

    page = result.page
    obj = page.find_text_object(object_id)
    return {
        "page_number": number,
        "revision": page.revision,
        "status": result.status.value,
        "layout_confidence": result.layout_confidence,
        "lost_protected": result.lost_protected,
        "warnings": result.warnings,
        "block": _block_out(obj, page.width, page.height) if obj else None,
    }


def _mark_export_stale(document_id: str, manifests) -> None:
    job = _latest_job(document_id)
    if job is not None:
        job.export_stale = True
        manifests.save_job(job)


def _mutate(document_id: str, number: int, operation, *, object_id: str | None = None):
    """Run a page mutation under the document lock and persist it."""
    manifests = get_manifest_store()
    store = get_object_store()
    with document_lock(document_id):
        manifest = _load_manifest(document_id)
        target = _target_language(document_id)
        try:
            result = operation(manifest, target, store)
        except EditError as exc:
            raise _fail(400, exc.message, exc.diagnostic) from exc
        manifests.save_manifest(manifest)
        _mark_export_stale(document_id, manifests)

    page = result.page
    block = page.find_text_object(object_id) if object_id else None
    return {
        "page_number": number,
        "revision": page.revision,
        "status": result.status.value,
        "layout_confidence": result.layout_confidence,
        "lost_protected": result.lost_protected,
        "warnings": result.warnings,
        "can_undo": can_undo(page),
        "can_redo": bool(page.redo_history),
        "block": _block_out(block, page.width, page.height) if block else None,
    }


@router.patch("/documents/{document_id}/pages/{number}/blocks/{object_id}")
def edit_block(
    document_id: str, number: int, object_id: str, body: EditRequest
) -> dict:
    """Change a block's text, its placement, its colour, its size, or several
    at once."""
    has_layout = any(
        v is not None
        for v in (
            body.bbox,
            body.color,
            body.font_size,
            body.alignment,
            body.bold,
            body.italic,
            body.font_family,
            body.fill,
            body.fit,
            body.layer,
            body.rotation,
            body.vertical_align,
            body.line_spacing,
            body.underline,
            body.highlight,
            body.border_color,
            body.border_width,
            body.border_dash,
            body.padding,
            body.angle,
            body.list_kind,
            body.list_style,
            body.list_start,
            body.indent,
            body.shape,
        )
    )
    if body.text is None and not has_layout and body.list_levels is None:
        raise _fail(400, "Nothing to change was provided.")

    # Text first, so a combined call does not lose one of the two. Line
    # levels sent with the text are saved with it, in the same step.
    if body.text is not None:
        result = _run_edit(
            document_id, number, object_id, body.text, list_levels=body.list_levels
        )
        if not has_layout:
            return result
        body = body.model_copy(update={"list_levels": None})
    return _run_edit(
        document_id, number, object_id, None, layout=body
    )


@router.post("/documents/{document_id}/pages/{number}/blocks/{object_id}/reset")
def reset_block(document_id: str, number: int, object_id: str) -> dict:
    return _run_edit(document_id, number, object_id, None)


@router.post("/documents/{document_id}/pages/{number}/blocks")
def create_block(document_id: str, number: int, body: AddBlockRequest) -> dict:
    """Add a text block where the engine found nothing.

    OCR misses stylised text -- a heading in white on a dark band is often
    not detected at all -- and without this such a document cannot be
    finished.
    """
    manifests = get_manifest_store()
    store = get_object_store()
    with document_lock(document_id):
        manifest = _load_manifest(document_id)
        target = _target_language(document_id)
        try:
            result, created = add_block(
                manifest,
                number,
                bbox=body.bbox.model_dump(),
                text=body.text,
                source_text=body.source_text,
                fill=body.fill,
                color=body.color,
                font_size=body.font_size,
                alignment=body.alignment,
                store=store,
                target_lang=target,
            )
        except EditError as exc:
            raise _fail(400, exc.message, exc.diagnostic) from exc
        manifests.save_manifest(manifest)
        _mark_export_stale(document_id, manifests)

    page = result.page
    return {
        "page_number": number,
        "revision": page.revision,
        "status": result.status.value,
        "warnings": result.warnings,
        "block": _block_out(created, page.width, page.height),
    }


@router.delete("/documents/{document_id}/pages/{number}/blocks/{object_id}")
def delete_block(document_id: str, number: int, object_id: str) -> dict:
    """Remove a block.

    A block a person added is deleted. A block the engine found is
    suppressed: its original glyphs are still taken off the page, but nothing
    is drawn back.
    """
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: remove_block(
            manifest, number, object_id, store=store, target_lang=target
        ),
        object_id=object_id,
    )


@router.post("/documents/{document_id}/pages/{number}/blocks/{object_id}/restore")
def restore_suppressed_block(
    document_id: str, number: int, object_id: str
) -> dict:
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: restore_block(
            manifest, number, object_id, store=store, target_lang=target
        ),
        object_id=object_id,
    )


class MoveIn(BaseModel):
    id: str = Field(max_length=200)
    bbox: BoxIn


class PositionsRequest(BaseModel):
    moves: list[MoveIn] = Field(min_length=1, max_length=400)
    # Lining boxes up by an edge lines up their words too: this alignment
    # goes to `align_ids` in the same undo step as the moves.
    alignment: str | None = Field(default=None, max_length=8)
    align_ids: list[str] = Field(default_factory=list, max_length=400)


@router.post("/documents/{document_id}/pages/{number}/moves")
def move_blocks(document_id: str, number: int, body: PositionsRequest) -> dict:
    """Move several blocks at once, as one undo step: a group drag, or
    lining up and spreading out a selection."""
    from ..pipeline.edit import apply_positions
    moves = [{"id": m.id, "bbox": m.bbox.model_dump()} for m in body.moves]
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: apply_positions(
            manifest, number, moves, alignment=body.alignment,
            align_ids=body.align_ids, store=store, target_lang=target,
        ),
    )


@router.post("/documents/{document_id}/pages/{number}/blocks/batch")
def create_blocks_batch(document_id: str, number: int, body: BatchAddRequest) -> dict:
    """Make several boxes in one step: Duplicate, Paste, and text pasted
    from outside the app (translated from English).

    All or nothing, and one undo step. Each box made comes back with the
    editor's own id for it, in the order sent.
    """
    specs = [spec.model_dump(by_alias=True) for spec in body.blocks]
    if any(spec["translate"] for spec in specs):
        # Text pasted from outside is translated first, outside the
        # document lock, so a slow translator holds up no other save. Only
        # the source language is read from this copy of the document.
        try:
            translate_specs(
                _load_manifest(document_id), specs,
                target_lang=_target_language(document_id),
            )
        except EditError as exc:
            raise _fail(400, exc.message, exc.diagnostic) from exc
    made: list = []

    def operation(manifest, target, store):
        result, created = add_blocks(
            manifest, number, specs, store=store, target_lang=target
        )
        made.extend((obj, result.page) for obj in created)
        return result

    out = _mutate(document_id, number, operation)
    out["created"] = [
        {
            "client_id": spec.client_id,
            # As the document's block list has it, so the editor can put
            # it straight in place of the box it was showing.
            "block": _block_out(obj, page.width, page.height)
            | {"editable": True, "locked_reason": None},
        }
        for spec, (obj, page) in zip(body.blocks, made)
    ]
    return out


@router.post("/documents/{document_id}/pages/{number}/blocks/remove")
def remove_blocks_route(document_id: str, number: int, body: RemoveRequest) -> dict:
    """Remove several blocks as one undo step: Cut, or Delete on a
    selection. Boxes she made are deleted; blocks the engine found are
    suppressed, as Delete does one."""
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: remove_blocks(
            manifest, number, body.object_ids, store=store, target_lang=target
        ),
    )


@router.patch("/documents/{document_id}/pages/{number}/blocks")
def bulk_edit_blocks(document_id: str, number: int, body: BulkRequest) -> dict:
    """Apply one colour, size or alignment to many blocks in a single pass."""
    if all(
        v is None
        for v in (
            body.color, body.font_size, body.alignment, body.bold, body.italic,
            body.font_family, body.fill, body.fit, body.layer,
            body.vertical_align, body.line_spacing, body.underline, body.highlight,
            body.border_color, body.border_width, body.border_dash, body.padding,
            body.angle, body.list_kind, body.list_style, body.list_start, body.indent,
            body.shape,
        )
    ):
        raise _fail(400, "Nothing to change was provided.")
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: apply_bulk(
            manifest,
            number,
            body.object_ids,
            color=body.color,
            font_size=body.font_size,
            alignment=body.alignment,
            bold=body.bold,
            italic=body.italic,
            font_family=body.font_family,
            fill=body.fill,
            fit=body.fit,
            layer=body.layer,
            vertical_align=body.vertical_align,
            line_spacing=body.line_spacing,
            underline=body.underline,
            highlight=body.highlight,
            border_color=body.border_color,
            border_width=body.border_width,
            border_dash=body.border_dash,
            padding=body.padding,
            angle=body.angle,
            list=body.list_kind,
            list_style=body.list_style,
            list_start=body.list_start,
            indent=body.indent,
            shape=body.shape,
            store=store,
            target_lang=target,
        ),
    )


@router.post("/documents/{document_id}/pages/{number}/undo")
def undo_last_change(document_id: str, number: int) -> dict:
    """Step back one change on this page."""
    return _mutate(
        document_id,
        number,
        lambda manifest, target, store: undo(
            manifest, number, store=store, target_lang=target
        ),
    )


@router.post("/documents/{document_id}/pages/{number}/redo")
def redo_last_change(document_id: str, number: int) -> dict:
    return _mutate(document_id, number, lambda manifest, target, store:
        redo(manifest, number, store=store, target_lang=target))


@router.get("/documents/{document_id}/project")
def download_project(document_id: str) -> Response:
    from ..pipeline.project_package import export_project
    _load_manifest(document_id)
    with document_lock(document_id):
        job = _latest_job(document_id)
        if job is None or not job.is_terminal:
            raise _fail(409, "Wait for processing to finish before saving a project.")
        try:
            data = export_project(document_id, get_object_store())
        except ValueError as exc:
            raise _fail(400, str(exc)) from exc
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="classroom.ctproject"',
        "Cache-Control": "private, no-store",
    })


@router.post("/projects", response_model=UploadResponse)
async def open_project(file: UploadFile = File(...)) -> UploadResponse:
    from ..pipeline.project_package import import_project, MAX_PACKAGE_BYTES
    data = await file.read(MAX_PACKAGE_BYTES + 1)
    try:
        manifest = import_project(data, get_object_store())
    except Exception as exc:
        raise _fail(400, "This project could not be opened. Use an intact Classroom Translate project file.") from exc
    job = _latest_job(manifest.document_id)
    if job is None:
        get_manifest_store().delete_document(manifest.document_id)
        raise _fail(400, "This project is missing its processing record. Try the original document instead.")
    return UploadResponse(document_id=manifest.document_id, job_id=job.job_id if job else "",
        filename=manifest.filename, byte_size=manifest.byte_size,
        page_count=manifest.source_page_count, warnings=[])


@router.get("/documents/{document_id}/editable-pptx")
def download_editable_pptx(document_id: str) -> Response:
    from ..reconstruct.layered_pptx import export_layered_pptx, styled_pages
    manifest = _load_manifest(document_id)
    if format_of(manifest).value != "PDF":
        raise _fail(400, "Use the main download for editable Office documents.")
    job = _latest_job(document_id)
    if job and job.settings.target_language.split("-")[0] in ("ar", "he", "ur"):
        raise _fail(400, "Editable right-to-left export is not ready. Download the PDF to keep the text correct.")
    with document_lock(document_id):
        manifest, job = _fresh_export(document_id)
        try:
            pdf = get_object_store().get_bytes(job.export_key)
        except ObjectNotFoundError as exc:
            raise _fail(404, "The translated file is no longer available.", str(exc)) from exc
        try:
            data = export_layered_pptx(pdf, style_notes=styled_pages(manifest.pages))
        except ValueError as exc:
            raise _fail(400, str(exc)) from exc
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", headers={
        "Content-Disposition": _content_disposition(
            "attachment", _download_name(manifest, job, "pptx", ", editable")
        ),
        "Cache-Control": "private, no-store",
    })


class TextsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=600)
    target_language: str
    source_language: str | None = None


@router.post("/translate-texts")
def translate_texts(body: TextsRequest) -> dict:
    """Translate a list of short texts that are not a document.

    A Google Form is translated in the teacher's browser: it sends the
    form's questions and choices here and writes the answers into a copy of
    the form. Each item comes back with `ok`; when false the original text
    is returned, because a damaged translation of a quiz question is worse
    than an untranslated one.
    """
    from ..translate.one_off import OneOffError, translate_many

    if get_language(body.target_language) is None:
        raise _fail(400, "That language is not supported.")
    if sum(len(t) for t in body.texts) > 300_000:
        raise _fail(413, "That is too much text to translate in one go.")
    try:
        results = translate_many(
            body.texts, target_lang=body.target_language,
            source_lang=body.source_language,
        )
    except OneOffError as exc:
        raise _fail(503, exc.message if hasattr(exc, "message") else str(exc),
                    getattr(exc, "diagnostic", None)) from exc
    return {
        "translations": [
            {"text": r.text if r else original, "ok": r is not None}
            for r, original in zip(results, body.texts)
        ]
    }


class SourceCorrection(BaseModel):
    # Empty only takes the words out of a shape (translate-english); every
    # other use refuses it in pipeline/edit, with its own message.
    text: str = Field(max_length=8000)
    # translate-english only: each line's level, 0 to 3, kept as the
    # box's indents with the new English.
    levels: list[int] | None = Field(default=None, max_length=200)


@router.patch("/documents/{document_id}/pages/{number}/blocks/{object_id}/source")
def correct_original_text(document_id: str, number: int, object_id: str, body: SourceCorrection) -> dict:
    from ..pipeline.edit import correct_source
    return _mutate(document_id, number,
        lambda manifest, target, store: correct_source(manifest, number, object_id, body.text,
            store=store, target_lang=target), object_id=object_id)


@router.post("/documents/{document_id}/pages/{number}/blocks/{object_id}/translate-english")
def translate_block_from_english(
    document_id: str, number: int, object_id: str, body: SourceCorrection
) -> dict:
    """Set what a block should say, in English, and translate it again."""
    from ..pipeline.edit import translate_from_english
    return _mutate(document_id, number,
        lambda manifest, target, store: translate_from_english(
            manifest, number, object_id, body.text, store=store, target_lang=target,
            levels=body.levels),
        object_id=object_id)


@router.get("/documents/{document_id}/pages/{number}/blocks/{object_id}/back-translation")
def back_translate_block(document_id: str, number: int, object_id: str) -> dict:
    """What a block's translation says, translated back into the source language.

    For a teacher who cannot read the translation: she checks its meaning in
    English. Read-only -- nothing on the page changes.
    """
    from ..translate.one_off import OneOffError, translate_one

    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    obj = page.find_text_object(object_id) if page else None
    if obj is None or not (obj.translated_text or "").strip():
        raise _fail(404, "That text has no translation to check.")
    source = (manifest.analysis.detected_language or "en").split("-")[0]
    if get_language(source) is None:  # "und" on a scan that was not detected
        source = "en"
    try:
        result = translate_one(obj.translated_text, target_lang=source)
    except OneOffError as exc:
        raise _fail(502, str(exc)) from exc
    return {"text": result.text, "language": source}


class ReviewDecision(BaseModel):
    key: str = Field(max_length=10000)
    reviewed: bool


@router.patch("/documents/{document_id}/review")
def review_finding(document_id: str, body: ReviewDecision) -> dict:
    with document_lock(document_id):
        manifest = _load_manifest(document_id)
        if body.key not in finding_keys(group_findings(manifest)):
            raise _fail(400, "That item has changed. Reload the page to check the latest version.")
        reviewed = set(manifest.reviewed_findings)
        if body.reviewed:
            reviewed.add(body.key)
        else:
            reviewed.discard(body.key)
        manifest.reviewed_findings = sorted(reviewed)
        get_manifest_store().save_manifest(manifest)
    return document_findings(document_id)


@router.post('/documents/{document_id}/pages/{number}/blocks/{object_id}/repair-preview')
def repair_preview(document_id: str, number: int, object_id: str):
    from ..pipeline.recovery import preview_repair
    with document_lock(document_id):
        try:
            return preview_repair(_load_manifest(document_id), number, object_id,
                store=get_object_store(), target_lang=_target_language(document_id))
        except EditError as exc:
            raise _fail(400, exc.message) from exc


class RepairAcceptance(BaseModel):
    token: str = Field(pattern=r'^[a-f0-9]{32}$')


@router.post('/documents/{document_id}/pages/{number}/blocks/{object_id}/repair')
def repair_accept(document_id: str, number: int, object_id: str, body: RepairAcceptance):
    from ..pipeline.recovery import accept_repair
    return _mutate(document_id, number,
        lambda manifest, target, store: accept_repair(manifest, number, object_id, body.token,
            store=store, target_lang=target), object_id=object_id)


@router.post('/documents/{document_id}/pages/{number}/blocks/{object_id}/keep-original')
def keep_section_original(document_id: str, number: int, object_id: str):
    from ..pipeline.recovery import keep_original
    return _mutate(document_id, number,
        lambda manifest, target, store: keep_original(manifest, number, object_id,
            store=store, target_lang=target), object_id=object_id)


@router.get('/documents/{document_id}/pages/{number}/blocks/{object_id}/original-crop')
def original_crop(document_id: str, number: int, object_id: str):
    import fitz
    manifest = _load_manifest(document_id)
    page = manifest.page(number - 1)
    obj = page.find_text_object(object_id) if page else None
    if obj is None or not page.has_fixed_layout:
        raise _fail(404, 'That original section could not be found.')
    with fitz.open(stream=get_object_store().get_bytes(manifest.source_key), filetype='pdf') as pdf:
        source = pdf[number - 1]
        clip = fitz.Rect(*obj.bbox.expanded(12).as_tuple()) & source.rect
        png = source.get_pixmap(clip=clip, dpi=144, alpha=False).tobytes('png')
    return Response(png, media_type='image/png', headers={'Cache-Control': 'private, no-store'})
