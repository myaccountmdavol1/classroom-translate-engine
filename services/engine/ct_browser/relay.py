"""DeepL and Google through the web app's translation relay.

When the engine runs in a teacher's browser it cannot hold the DeepL or
Google key: anything sent to a browser can be copied. Each batch goes to the
web app instead (``CT_RELAY_URL`` + ``/deepl`` or ``/google``, served by
apps/web/src/app/api/mt), which checks the class code, adds the key and
forwards the batch unchanged. The relay passes DeepL's and Google's answers
back verbatim, so these providers subclass the engine's own and inherit,
untouched:

* 456 sets DeepL aside until next month; Google's 403 with a daily marker
  sets Google aside until tomorrow; a 403 without one is not a quota;
* 429 and 5xx are tried again after their Retry-After (capped);
* the shared cache, placeholder masking and the deepl,google fallback chain.

Only three things differ, each because of the relay:

* **401 from the relay** means the class code is wrong. Asking again, or
  asking Google instead, cannot help (the same relay checks the same code),
  so inside a translation job the job FAILS at once with words the teacher
  can act on. A document left in English and marked complete would look
  finished to someone who cannot read the target language.
* **504 from the relay** means it gave up waiting on DeepL or Google after
  55 s. The work may already have been done and billed, so the batch is
  never sent again: the service is rested exactly as after a timeout, and
  the rest of the document goes to the next one.
* **This side's timeout is 50 s**, under the relay's 55 s, so normally the
  engine gives up first, which (a ReadTimeout) is also never retried.

The class code travels per request: whoever handles a request copies its
X-CT-Access header into ``ACCESS_CODE``. A job runs later, elsewhere, so the
executor that runs jobs must run each one in the context it was submitted
from (``contextvars.copy_context()`` at submit). The browser worker's job
executor does; scripts/run_packet_via_relay.py shows the native equivalent.
"""

from __future__ import annotations

import contextvars
import functools
import os
from typing import Callable, ClassVar

import httpx

from ct_engine.jobs.runner import JobCancelled, JobRunner, _utcnow
from ct_engine.model.job import JobStage
from ct_engine.providers.registry import TRANSLATION_PROVIDERS
from ct_engine.providers.translation.base import (
    TranslationError,
    TranslationResult,
    TranslationSegment,
)
from ct_engine.providers.translation.remote import (
    DeepLTranslationProvider,
    GoogleTranslationProvider,
    _from_tags,
    _to_tags,
)
from ct_engine.util.logging import get_logger

log = get_logger(__name__)

ACCESS_HEADER = "X-CT-Access"

# The class code of the request (or job) being handled. boot.handle copies
# each request's X-CT-Access header here. None sends no header, which a
# relay that needs a code refuses with 401.
ACCESS_CODE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ct_access_code", default=None
)

# Below the relay's 55 s (UPSTREAM_TIMEOUT_MS in mtRelay.ts). See the module
# docstring for why the order matters.
RELAY_TIMEOUT = httpx.Timeout(50.0, connect=10.0)

# Mirrors MAX_TOTAL_CHARS and MAX_ITEM_CHARS in
# apps/web/src/lib/server/mtRelay.ts; tests/test_browser_relay.py fails if
# they drift apart. A page with more text than this is sent as several
# requests rather than refused.
MAX_TOTAL_CHARS = 5_000
MAX_ITEM_CHARS = 5_000

# What the teacher reads when the class code is refused mid-job. The same
# words the access-code screen uses (AccessGate.tsx), so she knows which
# code is meant.
CLASS_CODE_REFUSED = (
    "The class access code was not accepted. Your school may have changed "
    "it. Enter the current one, then translate again."
)

# Tests replace this with an httpx.MockTransport.
_transport: httpx.BaseTransport | None = None


class ClassCodeRefused(JobCancelled):
    """The relay refused the class code during a translation job.

    A JobCancelled, so it is a BaseException: it passes through the
    engine's per-page and per-batch ``except Exception`` guards (which would
    otherwise turn it into one untranslated page after another) straight to
    the job's own handler, which ends the job (see install()).
    """


class _JobAccess:
    """One translation job's view of the class code."""

    # The running job's, set for its length by install()'s wrapper, so a
    # refusal is remembered for that job only: both providers stop asking,
    # and the next job (perhaps with a corrected code) starts fresh.
    #
    current: ClassVar[contextvars.ContextVar[_JobAccess | None]] = (
        contextvars.ContextVar("ct_job_access", default=None)
    )

    def __init__(self) -> None:
        self.refused = False


def _job_access() -> _JobAccess | None:
    return _JobAccess.current.get()


def _relay_endpoint(path: str) -> str:
    base = os.environ.get("CT_RELAY_URL", "").strip().rstrip("/")
    if not base:
        raise TranslationError(
            "Translation is not set up yet.",
            diagnostic="CT_RELAY_URL is not set",
        )
    return f"{base}/{path}"


def _says_access_required(resp: httpx.Response) -> bool:
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("code") == "access_required"


def _groups(
    segments: list[TranslationSegment], size: Callable[[TranslationSegment], int]
) -> list[list[TranslationSegment]]:
    """Consecutive runs of segments, each within the relay's character cap.

    One segment over the cap still goes alone: the relay refuses it with
    413 and that batch is reported untranslated, which is the honest result.
    """
    groups: list[list[TranslationSegment]] = []
    current: list[TranslationSegment] = []
    total = 0
    for seg in segments:
        n = size(seg)
        if current and total + n > MAX_TOTAL_CHARS:
            groups.append(current)
            current, total = [], 0
        current.append(seg)
        total += n
    if current:
        groups.append(current)
    return groups


class _RelayMixin:
    """What both relay providers share. Listed first, so it wins the MRO."""

    name: str
    _relay_path: str
    _refused_here = False  # a refusal outside any job, for this instance

    def _relay_init(self) -> None:
        self._endpoint = _relay_endpoint(self._relay_path)

    def _payload_chars(self, seg: TranslationSegment) -> int:
        raise NotImplementedError

    def _check_access(self) -> None:
        job = _job_access()
        if (job is not None and job.refused) or self._refused_here:
            # No request: the relay already said no to this code.
            raise TranslationError(CLASS_CODE_REFUSED)

    def _translate_with_retry(self, segments, *, source_lang, target_lang):
        self._check_access()
        out: list[TranslationResult] = []
        for group in _groups(segments, self._payload_chars):
            out.extend(
                super()._translate_with_retry(  # type: ignore[misc]
                    group, source_lang=source_lang, target_lang=target_lang
                )
            )
        return out

    def _post(self, payload: dict) -> dict:
        self._check_access()
        headers = {}
        code = ACCESS_CODE.get()
        if code:
            headers[ACCESS_HEADER] = code
        # Only the native harness sets this: in a browser, Origin is the
        # browser's to send (it always does on a POST) and cannot be set.
        origin = os.environ.get("CT_RELAY_ORIGIN", "").strip()
        if origin:
            headers["Origin"] = origin
        with httpx.Client(timeout=RELAY_TIMEOUT, transport=_transport) as client:
            resp = client.post(self._endpoint, headers=headers, json=payload)

        if resp.status_code == 401 and _says_access_required(resp):
            job = _job_access()
            log.warning("%s: the relay refused the class code", self.name)
            if job is not None:
                job.refused = True
                raise ClassCodeRefused()
            # Outside a job (a single block re-translated, pages retried),
            # the request fails like any other, and neither provider here
            # asks again.
            self._refused_here = True
            raise TranslationError(CLASS_CODE_REFUSED)
        if resp.status_code == 504:
            # A timeout, as far as the engine's retry rules go: never sent
            # again, and the service is rested (remote._REST_AFTER_TIMEOUT).
            raise httpx.ReadTimeout(
                "The translation service did not answer in time.", request=resp.request
            )
        resp.raise_for_status()
        return resp.json()


class RelayDeepL(_RelayMixin, DeepLTranslationProvider):
    """DeepL, reached through /api/mt/deepl. No key on this side."""

    _relay_path = "deepl"

    def __init__(self) -> None:  # noqa: D107 - the parent's needs a key
        self._relay_init()

    def _payload_chars(self, seg: TranslationSegment) -> int:
        return len(_to_tags(seg.text.strip()))

    # The body and the reading of DeepL's reply are the engine's own
    # (DeepLTranslationProvider._translate_batch); only the transport
    # differs. test_browser_relay.py checks the two send the same body and
    # read the same reply, so a change there that is not made here fails.
    def _translate_batch(self, segments, *, source_lang, target_lang):
        edges = []
        for seg in segments:
            core = seg.text.strip()
            start = seg.text.find(core) if core else len(seg.text)
            edges.append((seg.text[:start], seg.text[start + len(core):], core))
        payload = {
            "text": [_to_tags(core) for _, _, core in edges],
            "target_lang": self._code(target_lang),
            "tag_handling": "xml",
            "preserve_formatting": True,
        }
        if source_lang:
            payload["source_lang"] = source_lang.split("-")[0].upper()
        data = self._post(payload)
        out = data.get("translations", [])
        if len(out) != len(segments):
            raise TranslationError(
                "The translation service returned an unexpected response.",
                diagnostic=f"expected {len(segments)}, got {len(out)}",
            )
        return [
            TranslationResult(
                segment_id=seg.segment_id,
                text=lead
                + self._clean(_from_tags(t.get("text", ""), core), core, target_lang).strip()
                + trail,
                provider=self.name,
            )
            for seg, t, (lead, trail, core) in zip(segments, out, edges)
        ]


class RelayGoogle(_RelayMixin, GoogleTranslationProvider):
    """Google, reached through /api/mt/google. No key on this side, so none
    ever appears in a URL either."""

    _relay_path = "google"

    def __init__(self) -> None:  # noqa: D107 - the parent's needs a key
        self._relay_init()

    def _payload_chars(self, seg: TranslationSegment) -> int:
        return len(seg.text)

    # As for DeepL: GoogleTranslationProvider._translate_batch with the
    # transport swapped, checked by test_browser_relay.py.
    def _translate_batch(self, segments, *, source_lang, target_lang):
        payload = {
            "q": [s.text for s in segments],
            "target": self._code(target_lang),
            "format": "text",
        }
        if source_lang:
            payload["source"] = source_lang.split("-")[0]
        data = self._post(payload)
        translations = data.get("data", {}).get("translations", [])
        if len(translations) != len(segments):
            raise TranslationError(
                "The translation service returned an unexpected response.",
                diagnostic=f"expected {len(segments)} results, got {len(translations)}",
            )
        return [
            TranslationResult(
                segment_id=seg.segment_id,
                text=t.get("translatedText", ""),
                provider=self.name,
            )
            for seg, t in zip(segments, translations)
        ]


# --- Installing -------------------------------------------------------------
_saved: dict | None = None


def install() -> None:
    """Use the relay for 'deepl' and 'google' in this process.

    Replaces the registry entries (so CT_TRANSLATION_PROVIDER=deepl,google
    builds the relay providers) and wraps two JobRunner methods so a refused
    class code fails the translation job with CLASS_CODE_REFUSED. Safe to
    call more than once.
    """
    global _saved
    if _saved is not None:
        return
    _saved = {
        "deepl": TRANSLATION_PROVIDERS.get("deepl"),
        "google": TRANSLATION_PROVIDERS.get("google"),
        "_run_translation": JobRunner._run_translation,
        "_mark_cancelled": JobRunner._mark_cancelled,
    }
    TRANSLATION_PROVIDERS["deepl"] = RelayDeepL
    TRANSLATION_PROVIDERS["google"] = RelayGoogle

    run_translation = _saved["_run_translation"]
    mark_cancelled = _saved["_mark_cancelled"]

    @functools.wraps(run_translation)
    def _run_translation(self, document_id: str, job_id: str) -> None:
        token = _JobAccess.current.set(_JobAccess())
        try:
            run_translation(self, document_id, job_id)
        finally:
            _JobAccess.current.reset(token)

    @functools.wraps(mark_cancelled)
    def _mark_cancelled(self, job) -> None:
        # The job's own handler for JobCancelled lands here. A
        # ClassCodeRefused is not a stop the teacher asked for: the job
        # failed, and she is told why.
        access = _job_access()
        if access is None or not access.refused:
            mark_cancelled(self, job)
            return
        job.stage = JobStage.FAILED
        job.error = CLASS_CODE_REFUSED
        job.error_diagnostic = "translation relay answered 401 access_required"
        job.finished_at = _utcnow()
        job.log("error", CLASS_CODE_REFUSED)
        self.manifests.save_job(job)

    JobRunner._run_translation = _run_translation
    JobRunner._mark_cancelled = _mark_cancelled


def uninstall() -> None:
    """Put the engine's own providers and methods back (for tests)."""
    global _saved
    if _saved is None:
        return
    for name in ("deepl", "google"):
        if _saved[name] is None:
            TRANSLATION_PROVIDERS.pop(name, None)
        else:
            TRANSLATION_PROVIDERS[name] = _saved[name]
    JobRunner._run_translation = _saved["_run_translation"]
    JobRunner._mark_cancelled = _saved["_mark_cancelled"]
    _saved = None
