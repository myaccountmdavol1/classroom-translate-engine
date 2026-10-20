"""Classroom Translate document engine (FastAPI).

Run:  uvicorn ct_engine.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.access import HEADER as ACCESS_HEADER, access_required, check_access
from .api.deps import get_job_runner, get_manifest_store
from .api.routes import router
# Importing these registers the providers in the registry.
from .providers.translation import local_argos as _local_argos  # noqa: F401
from .providers.translation import remote as _remote  # noqa: F401
from .providers.ocr import local_tesseract as _local_ocr  # noqa: F401
from .config import get_settings
from .jobs.retention import RetentionSweeper
from .storage import get_store
from .util.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    manifests = get_manifest_store()
    runner = get_job_runner()
    sweeper = RetentionSweeper(store, manifests)
    sweeper.start()

    # Pick up anything that was mid-flight when the process last stopped: a
    # deploy, a crash, or a host suspending an idle machine would otherwise
    # abandon the job silently.
    try:
        resumed = runner.resume_interrupted()
        if resumed:
            log.info("Resumed %d interrupted job(s) on startup", len(resumed))
    except Exception:
        log.exception("Could not resume interrupted jobs")
    log.info(
        "Classroom Translate engine ready (env=%s, storage=%s)",
        settings.environment,
        settings.storage_path,
    )
    try:
        yield
    finally:
        sweeper.stop()
        runner.shutdown(wait=False)
        # Drop the cached singletons so a subsequent lifespan in the same
        # process (tests, embedded runs, hot reload) builds a fresh runner
        # instead of reusing this shut-down pool.
        get_job_runner.cache_clear()
        get_manifest_store.cache_clear()
        log.info("Engine shut down")


app = FastAPI(
    title="Classroom Translate Engine",
    description=(
        "Document translation and reconstruction engine. Decomposes source "
        "documents into a coordinate-faithful object model, translates the "
        "eligible content, and rebuilds the pages."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def require_access(request: Request, call_next):
    """Gate every request on the shared passphrase, when one is configured.

    A middleware rather than a dependency on each route: a route added later
    is protected by default, which is the direction the mistake should go in.

    REGISTERED BEFORE CORS, which matters. Starlette applies the most
    recently added middleware outermost, so registering this after CORS put
    it in front, and a 401 returned from here never passed through
    CORSMiddleware -- it reached the browser with no `Access-Control-Allow-
    Origin` header, was blocked, and surfaced as "could not reach the
    service" instead of "wrong code". Every test passed, because the
    TestClient does not enforce CORS. Registering it first makes CORS the
    outer layer, so refusals are readable by the page that caused them.

    Unexpected errors are answered HERE for the same reason. Starlette runs
    an `exception_handler(Exception)` in its outermost layer, outside CORS,
    so every 500 reached the browser without the header and was shown as
    "could not reach the translation service" -- a service she has no way
    to check. Caught inside CORS, she reads what actually happened.
    """
    try:
        check_access(request)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            detail = {**detail, "detail": detail["error"]}
        return JSONResponse(status_code=exc.status_code, content=detail)
    try:
        return await call_next(request)
    except Exception as exc:
        return unhandled_exception_response(request, exc)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    # PATCH is required by the manual-correction endpoint. Omitting it made
    # the browser preflight fail while the TestClient (which bypasses CORS)
    # still passed -- so the method list is asserted in the test suite now.
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    # Not readable cross-origin unless named here: without it the page could
    # not see the file name, and every download was saved as "translated".
    # The X-CT-* pair tells the page what an Office download kept in English.
    expose_headers=["Content-Disposition", "X-CT-Pictures-Kept", "X-CT-Paragraphs-Kept"],
)


app.include_router(router, prefix="/api")


def error_body(message: str, diagnostic: str | None = None) -> dict:
    """The one error shape: {error, detail, diagnostic}.

    `detail` repeats `error` so a client reading either key gets the same
    plain sentence.
    """
    return {"error": message, "detail": message, "diagnostic": diagnostic}


# Starlette's own refusals (an unknown address, the wrong method) carry a
# bare phrase like "Not Found". These are the sentences shown instead.
_PLAIN_STATUS = {
    404: "That address was not found on the translation server.",
    405: "That action is not available here.",
}


# Registered for Starlette's HTTPException, which FastAPI's subclasses, so
# routing 404s and 405s come back in the same shape as the routes' own errors.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Normalize errors to {error, detail, diagnostic} so the UI has one shape."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = {**detail, "detail": detail["error"]}
    elif not isinstance(exc, HTTPException) and exc.status_code in _PLAIN_STATUS:
        content = error_body(_PLAIN_STATUS[exc.status_code], str(detail))
    else:
        content = error_body(str(detail))
    return JSONResponse(
        status_code=exc.status_code, content=content, headers=exc.headers
    )


def _validation_message(exc: RequestValidationError) -> str:
    """Say what to change, for the checks a person can actually trip."""
    errors = exc.errors()
    kind = str(errors[0].get("type", "")) if errors else ""
    ctx = (errors[0].get("ctx") or {}) if errors else {}
    if kind == "string_too_long":
        limit = ctx.get("max_length")
        if isinstance(limit, int):
            return f"That text is too long. Please keep it under {limit:,} characters."
        return "That text is too long. Please shorten it and try again."
    if kind == "string_too_short":
        return "Please type some text first."
    return "That request could not be understood. Please reload the page and try again."


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    diagnostic = (
        str(exc.errors()[:3]) if settings.environment == "development" else None
    )
    return JSONResponse(
        status_code=422, content=error_body(_validation_message(exc), diagnostic)
    )


def unhandled_exception_response(request: Request, exc: Exception) -> JSONResponse:
    # Never leak an internal error to a teacher; log the detail instead.
    log.exception(
        "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    diagnostic = (
        f"{type(exc).__name__}: {exc}"
        if settings.environment == "development"
        else None
    )
    return JSONResponse(
        status_code=500,
        content=error_body(
            "Something went wrong on our side. Please try again.", diagnostic
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Only reached by an error the access middleware did not catch -- in a
    # middleware outside it, say. Such a reply has no CORS header.
    return unhandled_exception_response(request, exc)


@app.get("/")
def root() -> dict:
    return {"service": "classroom-translate-engine", "docs": "/docs"}
