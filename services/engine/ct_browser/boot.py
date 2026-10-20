"""Start the unchanged engine inside Pyodide, and the entry points the worker calls.

The worker (apps/web/src/engine/engine.worker.js) sets os.environ, then calls
start(). After that every HTTP request the page makes is one call to handle(),
and every job is one call to run_next_job(), each its own "entry" on the
worker's one-at-a-time JS queue. Nothing in Python ever runs two things at
once, and nothing runs between entries.

Only the places where a browser differs from a server are changed, all from
outside ct_engine (see each shim below for why):

  * the job pool records work instead of starting a thread (executor.py);
  * Starlette's threadpool runs sync routes in place;
  * httpx's Pyodide transport stays off JSPI (a synchronous XMLHttpRequest
    from the worker, which needs no JSPI and works on every Chromebook);
  * PyMuPDF looks for fonts under /fonts, Emscripten's "system" folder;
  * a job may be picked up again more often (reloads are expected here);
  * a running job's checkpoints also ask the page whether she pressed Stop
    (cancel.py), since the request itself cannot reach Python mid-job;
  * each job and manifest the engine saves is also posted to the host page
    as the exact JSON the /job and /documents/{id} routes would return, so
    the page can be answered while Python is busy with a job;
  * every write to the store is told to the page, in order (journal.py), so
    the page can keep the lasting copy in the browser's own disk (OPFS);
  * a job that ran out of memory is left to be picked up again by a fresh
    worker, rather than ending FAILED (see _memory_failure).

Deliberately NOT done: the FastAPI lifespan. Its retention sweeper starts a
thread (retention runs in the page, over every copy the browser holds), and
the jobs it would resume are resumed by resume() below, once the page has
put back the documents they belong to.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .executor import DeferredJobExecutor
from .journal import StoreJournal

# Reloads, a tab handing the engine to another, and recycling the worker all
# interrupt jobs in the browser in the normal course of things; three
# attempts (the server's limit for crash loops) is too few here.
BROWSER_MAX_RESUME_ATTEMPTS = 8
# PyMuPDF's "system" font folder in Pyodide (platform.system() is
# "Emscripten"). Big script fonts are written here when a document needs them.
EMSCRIPTEN_FONT_DIR = Path("/fonts")
# Snapshots of a running job are posted at most this often (a stage change
# or the end of the job always goes at once; the last one is always sent at
# the end of the entry).
JOB_POST_INTERVAL_S = 0.25
DOCUMENT_POST_INTERVAL_S = 1.0

BASE_URL = "http://engine.local"
# Not passed on to the app: httpx sets its own for the in-process request.
_DROPPED_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}

Post = Callable[[str], Any]
Journal = Callable[[int, str, str, "str | None"], Any]

# What a job that ran out of memory says while it waits for a fresh worker,
# and when it has run out too often to be tried again.
MEMORY_PAUSED = (
    "This computer ran short of memory, so the work paused. It picks up "
    "again where it left off."
)
MEMORY_GAVE_UP = (
    "This computer ran short of memory several times while working on this "
    "document. Close other tabs you are not using and try again, or split "
    "the file into smaller parts."
)


async def _run_sync_inline(
    func: Callable[..., Any],
    *args: Any,
    abandon_on_cancel: bool = False,
    cancellable: bool | None = None,
    limiter: Any = None,
    **kwargs: Any,
) -> Any:
    """anyio.to_thread.run_sync without the thread: Pyodide cannot start one.

    Starlette runs every sync route through this. Running it in place is
    safe because the worker never runs two Python entries at once.
    """
    return func(*args, **kwargs)


def _relay_module():
    """ct_browser.relay, or None in a build that leaves it out.

    Without it, CT_TRANSLATION_PROVIDER names the engine's own providers,
    which need keys the browser never has, and translating says it is not
    set up rather than failing later.
    """
    try:
        from . import relay
    except ImportError:
        return None
    return relay


def _as_bytes(body: Any) -> bytes:
    """A request body from JS (Uint8Array/ArrayBuffer proxy) or Python."""
    # JS null arrives as pyodide's JsNull, not None.
    if body is None or type(body).__name__ == "JsNull":
        return b""
    if isinstance(body, (bytes, bytearray, memoryview)):
        return bytes(body)
    to_bytes = getattr(body, "to_bytes", None)
    if to_bytes is not None:
        return bytes(to_bytes())
    to_py = getattr(body, "to_py", None)
    if to_py is not None:
        return bytes(to_py())
    raise TypeError(f"cannot read a request body of type {type(body).__name__}")


class BrowserEngine:
    """The engine as the worker sees it: start once, then requests and jobs."""

    def __init__(
        self, post: Post | None = None, cancel_checker=None, journal: Journal | None = None
    ) -> None:
        self._post_fn: Post = post or (lambda _message: None)
        # Where each store write is told (journal.py); None tells no one.
        self._journal_sink = journal
        self.journal: StoreJournal | None = None
        # The last stage each job reached before it ended, and the jobs that
        # ran out of memory in this worker (see _memory_failure).
        self._last_stage: dict[str, Any] = {}
        self._memory_failed: set[str] = set()
        # Whether the page asked to stop a running job (cancel.py); None
        # asks the worker's ctCancelRequested. Tests pass their own.
        self._cancel_checker = cancel_checker
        self._undo: list[Callable[[], None]] = []
        self._job_posted: dict[str, tuple[float, str]] = {}  # doc -> (time, stage)
        self._job_pending: dict[str, str] = {}  # doc -> message not yet posted
        self._doc_posted: dict[str, float] = {}
        self._doc_pending: dict[str, Any] = {}  # doc -> manifest not yet posted
        self.client = None
        self.runner = None
        self.manifests = None
        self.relay = None

    # -- start ----------------------------------------------------------------
    def start(self) -> dict:
        t0 = time.perf_counter()
        self._install_shims()

        # The 'browser' OCR provider and the relay translators, in place
        # before any job asks for them (each is applied by an explicit call).
        from . import ocr

        ocr.install()
        self._undo.append(ocr.uninstall)
        # Stop and "Start over" reach a running job (see cancel.py).
        from . import cancel

        cancel.install(self._cancel_checker)
        self._undo.append(cancel.uninstall)
        self.relay = _relay_module()
        if self.relay is not None:
            self.relay.install()
            self._undo.append(self.relay.uninstall)

        import httpx

        from ct_engine.api.deps import get_job_runner, get_manifest_store
        from ct_engine.main import app

        # Replaced before anything can be submitted: nothing has been
        # uploaded yet, and the lifespan (which resumes jobs) never runs.
        runner = get_job_runner()
        original_pool = runner._pool
        runner._pool = DeferredJobExecutor()
        original_pool.shutdown(wait=False)
        self.runner = runner
        self.manifests = get_manifest_store()
        if self.manifests is not runner.manifests:
            raise RuntimeError("the job runner and the routes use different manifest stores")
        if self.manifests.store is not runner.store:
            raise RuntimeError("the job runner and the routes use different stores")
        # Before anything can write: every write from here on is told.
        if self._journal_sink is not None:
            self.journal = StoreJournal(runner.store, self._journal_sink)
            self.journal.install()
            self._undo.append(self.journal.uninstall)
        self._hook_snapshots(self.manifests)

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url=BASE_URL,
            timeout=None,
        )
        import fitz

        return {
            "python": sys.version.split()[0],
            "pymupdf": getattr(fitz, "VersionBind", None),
            "relay": self.relay is not None,
            "boot_seconds": round(time.perf_counter() - t0, 3),
        }

    def _install_shims(self) -> None:
        import anyio.to_thread

        original_run_sync = anyio.to_thread.run_sync
        anyio.to_thread.run_sync = _run_sync_inline
        self._undo.append(lambda: setattr(anyio.to_thread, "run_sync", original_run_sync))

        # Only exists inside Pyodide. With JSPI available httpx would suspend
        # the WebAssembly stack for each request, which the worker's
        # synchronous Python (and older Chromebooks) cannot use; with
        # can_run_sync() false it makes a synchronous XMLHttpRequest instead.
        try:
            import httpx._transports.jsfetch as jsfetch  # type: ignore[import-not-found]
        except ImportError:
            jsfetch = None
        if jsfetch is not None and hasattr(jsfetch, "can_run_sync"):
            original_can_run_sync = jsfetch.can_run_sync
            jsfetch.can_run_sync = lambda: False
            self._undo.append(lambda: setattr(jsfetch, "can_run_sync", original_can_run_sync))

        # Before the first font lookup: the registry reads this map whenever
        # it lists its search folders, keyed by platform.system().
        from ct_engine.fonts import registry

        had = "Emscripten" in registry._SYSTEM_DIRS
        previous = registry._SYSTEM_DIRS.get("Emscripten")
        registry._SYSTEM_DIRS["Emscripten"] = [EMSCRIPTEN_FONT_DIR]

        def restore_fonts() -> None:
            if had:
                registry._SYSTEM_DIRS["Emscripten"] = previous
            else:
                registry._SYSTEM_DIRS.pop("Emscripten", None)

        self._undo.append(restore_fonts)

        # Read when a job is resumed (runner.resume_interrupted), not at import.
        from ct_engine.jobs import runner as runner_mod

        original_attempts = runner_mod.MAX_RESUME_ATTEMPTS
        runner_mod.MAX_RESUME_ATTEMPTS = max(original_attempts, BROWSER_MAX_RESUME_ATTEMPTS)
        self._undo.append(
            lambda: setattr(runner_mod, "MAX_RESUME_ATTEMPTS", original_attempts)
        )

    # -- snapshots ------------------------------------------------------------
    def _hook_snapshots(self, manifests) -> None:
        """Post each saved job and manifest, as the routes would return them.

        Wrapped on the one ManifestStore the routes and the runner share
        (an instance attribute, so nothing else in the process changes).
        """
        save_job = manifests.save_job
        save_manifest = manifests.save_manifest

        def save_job_and_post(job) -> None:
            self._memory_failure(job)
            save_job(job)
            self._job_saved(job)

        def save_manifest_and_post(manifest) -> None:
            save_manifest(manifest)
            self._manifest_saved(manifest)

        manifests.save_job = save_job_and_post
        manifests.save_manifest = save_manifest_and_post

        def unhook() -> None:
            for name in ("save_job", "save_manifest"):
                manifests.__dict__.pop(name, None)

        self._undo.append(unhook)

    def _post(self, message: dict) -> None:
        try:
            self._post_fn(json.dumps(message))
        except Exception:  # a page that cannot hear must never stop a job
            pass

    def _memory_failure(self, job) -> None:
        """A job that failed because Python ran out of memory is not a failed
        document: the wasm heap only grows, and a fresh worker has room. So
        instead of saving FAILED, the job is put back at the stage it had
        reached, with a line in its log, and the worker is told to start
        afresh after this job (run_next_job's "recycle"); the engine's own
        resume then carries on from the last page it saved. Past the
        engine's resume limit it fails, with words that say why.
        """
        from ct_engine.jobs import runner as runner_mod
        from ct_engine.model.enums import JobStage

        if not job.is_terminal:
            self._last_stage[job.job_id] = job.stage
            return
        diagnostic = job.error_diagnostic or ""
        if job.stage is not JobStage.FAILED or not diagnostic.startswith("MemoryError"):
            self._last_stage.pop(job.job_id, None)
            return
        self._memory_failed.add(job.job_id)
        if job.resume_count >= runner_mod.MAX_RESUME_ATTEMPTS:
            job.error = MEMORY_GAVE_UP
            self._last_stage.pop(job.job_id, None)
            return
        translation = (job.settings.target_language or "und") != "und"
        job.stage = self._last_stage.pop(
            job.job_id, JobStage.TRANSLATING if translation else JobStage.ANALYZING
        )
        job.error = None
        job.finished_at = None
        job.log("warning", MEMORY_PAUSED)

    def _job_saved(self, job) -> None:
        from ct_engine.api.schemas import JobOut

        doc = job.document_id
        message = {
            "type": "job",
            "document_id": doc,
            "job": JobOut.of(job).model_dump(mode="json"),
            # Not part of the route's answer: lets the page say the job was
            # paused and picked up again.
            "resume_count": job.resume_count,
        }
        now = time.monotonic()
        last_time, last_stage = self._job_posted.get(doc, (0.0, ""))
        stage = job.stage.value
        if job.is_terminal or stage != last_stage or now - last_time >= JOB_POST_INTERVAL_S:
            self._job_posted[doc] = (now, stage)
            self._job_pending.pop(doc, None)
            self._post(message)
        else:
            self._job_pending[doc] = json.dumps(message)

    def _manifest_saved(self, manifest) -> None:
        doc = manifest.document_id
        now = time.monotonic()
        if now - self._doc_posted.get(doc, 0.0) >= DOCUMENT_POST_INTERVAL_S:
            self._doc_posted[doc] = now
            self._doc_pending.pop(doc, None)
            self._post(self._document_message(manifest))
        else:
            # Kept as the object: built into JSON once, at the end of the
            # entry, rather than for every page.
            self._doc_pending[doc] = manifest

    @staticmethod
    def _document_message(manifest) -> dict:
        from ct_engine.api.routes import _upload_notes
        from ct_engine.api.schemas import DocumentOut

        out = DocumentOut.of(
            manifest,
            None,  # the host merges in the newest job snapshot it holds
            upload_warnings=_upload_notes(manifest.document_id).get("warnings", []),
        )
        return {
            "type": "document",
            "document_id": manifest.document_id,
            "document": out.model_dump(mode="json"),
        }

    def flush(self) -> None:
        """Send the snapshots held back by the throttle: the newest of each."""
        pending_jobs, self._job_pending = self._job_pending, {}
        for doc, raw in pending_jobs.items():
            self._job_posted[doc] = (time.monotonic(), json.loads(raw)["job"]["stage"])
            try:
                self._post_fn(raw)
            except Exception:
                pass
        pending_docs, self._doc_pending = self._doc_pending, {}
        for doc, manifest in pending_docs.items():
            self._doc_posted[doc] = time.monotonic()
            try:
                self._post(self._document_message(manifest))
            except Exception:
                pass

    # -- requests -------------------------------------------------------------
    async def handle(
        self, method: str, path: str, headers: dict | str | None, body: Any = None
    ) -> tuple[int, list[list[str]], bytes]:
        """One HTTP request through the real FastAPI app, in this process.

        `path` is the engine's own ("/api/..."), with its query string.
        The class code travels in the X-CT-Access header like any request;
        it is copied into relay.ACCESS_CODE for the relay translators.
        """
        if isinstance(headers, str):
            headers = json.loads(headers) if headers else {}
        sent = {
            k: v for k, v in (headers or {}).items() if k.lower() not in _DROPPED_HEADERS
        }
        code = next((v for k, v in sent.items() if k.lower() == "x-ct-access"), None) or None
        token = self._set_access_code(code)
        try:
            response = await self.client.request(
                method.upper(), path, headers=sent, content=_as_bytes(body) or None
            )
        finally:
            self._reset_access_code(token)
            self.flush()
        return (
            response.status_code,
            [[k, v] for k, v in response.headers.multi_items()],
            response.content,
        )

    def _set_access_code(self, code: str | None):
        # The relay translators read it; a job queued by this request takes
        # it along (the pool captures the context at submit).
        if self.relay is None:
            return None
        return self.relay.ACCESS_CODE.set(code)

    def _reset_access_code(self, token) -> None:
        if token is not None:
            self.relay.ACCESS_CODE.reset(token)

    # -- jobs -----------------------------------------------------------------
    @property
    def pool(self) -> DeferredJobExecutor:
        return self.runner._pool

    def has_pending_job(self) -> bool:
        return self.pool.pending() > 0

    def next_job(self) -> dict | None:
        """What the next job is, so the worker can prepare before it runs.

        `ocr_language` is the Tesseract pack the job may need: set only for a
        translation of a document with scanned pages, or with pictures whose
        words are to be translated (OCR runs in the translation job, never
        in analysis).
        """
        args = self.pool.peek()
        if not args or len(args) < 2:
            return None
        document_id, job_id = str(args[0]), str(args[1])
        info: dict = {
            "document_id": document_id,
            "job_id": job_id,
            "translation": False,
            "ocr_language": None,
        }
        try:
            job = self.manifests.load_job(document_id, job_id)
            if job is None:
                return info
            info["translation"] = (job.settings.target_language or "und") != "und"
            if not info["translation"]:
                return info
            manifest = self.manifests.load_manifest(document_id)
            if manifest is None:
                return info
            from ct_engine.pipeline.ocr_stage import pages_needing_ocr
            from ct_engine.providers.ocr.local_tesseract import LocalTesseractOCRProvider

            reads = bool(pages_needing_ocr(manifest.pages)) or (
                job.settings.translate_image_text
                and any(p.image_objects for p in manifest.pages)
            )
            if reads:
                info["ocr_language"] = LocalTesseractOCRProvider.tesseract_code(
                    job.settings.source_language or manifest.analysis.detected_language
                )
        except Exception as exc:  # the job itself will report what is wrong
            info["error"] = repr(exc)[:300]
        return info

    def run_next_job(self) -> dict:
        """Run the next waiting job to its end, as one Python entry."""
        args = self.pool.peek()
        started = time.perf_counter()
        try:
            future = self.pool.run_next()
        finally:
            self.flush()
        out: dict = {"ran": future is not None, "seconds": round(time.perf_counter() - started, 3)}
        if args and len(args) >= 2:
            out["document_id"], out["job_id"] = str(args[0]), str(args[1])
            if out["job_id"] in self._memory_failed:
                self._memory_failed.discard(out["job_id"])
                # Start a fresh worker before anything else runs; the job is
                # picked up again there (resume()).
                out["recycle"] = True
        if future is not None and future.exception() is not None:
            import traceback

            exc = future.exception()
            out["error"] = "".join(traceback.format_exception(exc))[-1500:]
        return out

    # -- picking jobs up again --------------------------------------------------
    def resume(
        self, document_ids: list[str] | None = None, stops=(), code: str | None = None
    ) -> list[str]:
        """The engine's own resume after a restart, for the documents the
        page has just put back in this worker's store.

        `document_ids` None means every document in the store (at boot, the
        page puts back exactly the ones with unfinished jobs). A list limits
        it to those, so a document opened later, while another document's job
        is queued or running, never has that other job queued twice.

        `stops` are (document_id, job_id) pairs she stopped before the tab
        that was running them closed: written down as the route's cancel
        would have, so resume ends them rather than running them again.

        `code` is the class code saved in this browser. A job picked up again
        has no request of its own to carry it, and the relay translators
        need it; the pool keeps it with each job it queues here.
        """
        token = self._set_access_code(code or None)
        try:
            return self._resume(document_ids, stops)
        finally:
            self._reset_access_code(token)

    def _resume(self, document_ids, stops) -> list[str]:
        runner = self.runner
        for document_id, job_id in stops or ():
            try:
                job = self.manifests.load_job(str(document_id), str(job_id))
            except Exception:
                job = None
            if job is not None and not job.is_terminal:
                runner._write_cancel_marker(str(document_id), str(job_id))
        try:
            if document_ids is None:
                return runner.resume_interrupted()
            wanted = {str(d) for d in document_ids} - set(runner._active)
            if not wanted:
                return []
            store = runner.store
            own = vars(store)
            had = "list_prefix" in own
            previous = own.get("list_prefix")
            every = store.list_prefix

            def only_these(prefix: str):
                for key in every(prefix):
                    parts = key.split("/")
                    if len(parts) > 1 and parts[1] in wanted:
                        yield key

            store.list_prefix = only_these
            try:
                return runner.resume_interrupted()
            finally:
                if had:
                    own["list_prefix"] = previous
                else:
                    own.pop("list_prefix", None)
        finally:
            self.flush()

    def stats(self) -> dict:
        return {
            "journal": self.journal.stats() if self.journal else None,
            "pending_jobs": self.pool.pending() if self.runner else 0,
        }

    # -- tests and restarts ---------------------------------------------------------
    def stop(self) -> None:
        """Undo every shim (the native tests share one process with others)."""
        if self.runner is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        for undo in reversed(self._undo):
            try:
                undo()
            except Exception:
                pass
        self._undo.clear()
        from ct_engine.api.deps import get_job_runner, get_manifest_store

        get_job_runner.cache_clear()
        get_manifest_store.cache_clear()


# ---- Module-level entry points for the worker ------------------------------------
_ENGINE: BrowserEngine | None = None


def start(
    env: dict | str | None = None, post: Post | None = None, journal: Journal | None = None
) -> str:
    """Boot the engine once. `env` is set before ct_engine is first imported
    (its settings are read once); `journal` hears every store write. Returns
    a JSON summary for the worker."""
    global _ENGINE
    if _ENGINE is not None:
        raise RuntimeError("the engine is already started")
    if isinstance(env, str):
        env = json.loads(env) if env else {}
    os.environ.update({str(k): str(v) for k, v in (env or {}).items()})
    root = os.environ.get("CT_STORAGE_ROOT")
    if root:
        os.makedirs(root, exist_ok=True)
    if sys.platform == "emscripten":
        EMSCRIPTEN_FONT_DIR.mkdir(parents=True, exist_ok=True)
    engine = BrowserEngine(post, journal=journal)
    summary = engine.start()
    _ENGINE = engine
    return json.dumps(summary)


def _engine() -> BrowserEngine:
    if _ENGINE is None:
        raise RuntimeError("the engine has not been started")
    return _ENGINE


async def handle(method: str, path: str, headers_json: str | None, body: Any = None):
    """One request; returns [status, headers JSON, body] for the worker."""
    status, headers, content = await _engine().handle(method, path, headers_json, body)
    try:
        from pyodide.ffi import to_js  # type: ignore[import-not-found]
    except ImportError:
        return [status, json.dumps(headers), content]
    return to_js([status, json.dumps(headers), to_js(content)])


def has_pending_job() -> bool:
    return _engine().has_pending_job()


def next_job() -> str:
    info = _engine().next_job()
    return json.dumps(info) if info else ""


def run_next_job() -> str:
    return json.dumps(_engine().run_next_job())


def resume(payload: str | None = None) -> str:
    """{"ids": null | [document ids], "stops": [[document id, job id], ...],
    "code": class code | null} -> {"resumed": [job ids]}. See
    BrowserEngine.resume."""
    data = json.loads(payload) if payload else {}
    resumed = _engine().resume(data.get("ids"), data.get("stops") or (), data.get("code"))
    return json.dumps({"resumed": resumed})


def stats() -> str:
    return json.dumps(_engine().stats())
