"""Stop a running job from the page, in a browser where Python has one thread.

On a server, Stop is a request like any other: the route adds the job to
JobRunner._cancel_requested from another thread, and the job sees it at its
next checkpoint (_check_cancel, between pages and between stages) and ends
CANCELLED. In the browser the request cannot reach Python while the job
holds the worker's only thread, so a translation she abandoned ("Start
over", Stop) would carry on to the end, sending every page to DeepL and
Google and spending the school's allowance.

So the checkpoint also asks the page. The engine host (apps/web/src/engine/
host.ts) answers Stop at once in JS and remembers the job; the worker's
``ctCancelRequested(job_id)`` asks it with a synchronous request, which the
Service Worker hands to the host (whose thread is free). When the answer is
yes, the job is marked exactly as the route would mark it and the engine's
own checkpoint stops it: the same JobCancelled, the same CANCELLED job, the
same tidy-up.

The question costs a millisecond or two, and is asked at most every
CHECK_INTERVAL_S per job: analysis checks on every use of the open document,
far more often than that. Between pages it is asked every time (a page takes
longer than the interval), so a stopped translation sends no further page.
"""

from __future__ import annotations

import functools
import time
from typing import Callable

from ct_engine.jobs.runner import JobRunner

# Short enough that a page boundary is never skipped (a page, with its
# DeepL round trip, takes longer); long enough that analysis, which checks
# on every use of the document, asks at most 20 times a second.
CHECK_INTERVAL_S = 0.05

Checker = Callable[[str], bool]


def _ask_the_page(job_id: str) -> bool:
    """The worker's ctCancelRequested; False wherever there is none."""
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return False
    ask = getattr(js, "ctCancelRequested", None)
    if ask is None:
        return False
    return bool(ask(job_id))


_saved: dict | None = None


def install(checker: Checker | None = None) -> None:
    """Make JobRunner's checkpoints also ask `checker` (default: the page).

    A checker that fails, or a page that cannot answer, never stops a job:
    the job carries on as it would have without a Stop.
    """
    global _saved
    if _saved is not None:
        return
    ask = checker or _ask_the_page
    original = JobRunner._check_cancel
    last_asked: dict[str, float] = {}

    @functools.wraps(original)
    def _check_cancel(self, job) -> None:
        job_id = job.job_id
        if job_id not in self._cancel_requested:
            now = time.monotonic()
            if now - last_asked.get(job_id, float("-inf")) >= CHECK_INTERVAL_S:
                last_asked[job_id] = now
                try:
                    stop = ask(job_id)
                except Exception:
                    stop = False
                if stop:
                    # What the route's cancel() does for a running job. Not
                    # under the runner's guard: there is no other thread, and
                    # a checkpoint may already be inside a guarded section.
                    self._cancel_requested.add(job_id)
                    last_asked.pop(job_id, None)
        original(self, job)

    _saved = {"_check_cancel": original}
    JobRunner._check_cancel = _check_cancel


def uninstall() -> None:
    """Put the engine's own checkpoint back (for tests)."""
    global _saved
    if _saved is None:
        return
    JobRunner._check_cancel = _saved["_check_cancel"]
    _saved = None
