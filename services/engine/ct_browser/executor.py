"""The job pool for an engine with no threads.

On a server, JobRunner hands each job to a ThreadPoolExecutor and returns.
Pyodide cannot start threads, and running the job inside submit() is not an
option: JobRunner._enqueue calls submit() while it holds its _guard lock, and
the job's own _job_ended takes that same lock again, so an inline run
deadlocks (seen in the first spike). So submit() here only RECORDS the work.

The worker's JS queue then runs one job at a time as its own Python entry
(boot.run_next_job -> run_next), after the request that queued it has
answered. Requests waiting in the JS queue go first, so an edit or a Stop
sent while a job is still waiting is answered before the job starts, and a
job cancelled while it waits never runs at all.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import contextvars
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Work:
    future: cf.Future
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    # The request's context (its class code, above all), so the job runs as
    # the request that queued it would have.
    context: contextvars.Context


class DeferredJobExecutor:
    """A stand-in for JobRunner's ThreadPoolExecutor: submit records, run_next runs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._queue: collections.deque[_Work] = collections.deque()
        self._shutdown = False
        self._running: _Work | None = None

    # -- the ThreadPoolExecutor surface JobRunner uses ---------------------
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> cf.Future:
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        future: cf.Future = cf.Future()
        self._queue.append(
            _Work(future, fn, args, kwargs, contextvars.copy_context())
        )
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Refuse new work; with cancel_futures, drop what is still waiting.

        `wait` is accepted for the interface and has nothing to wait for:
        nothing runs except inside run_next(), which the caller drives.
        """
        self._shutdown = True
        if cancel_futures:
            while self._queue:
                self._queue.popleft().future.cancel()

    def map(self, *args: Any, **kwargs: Any):  # pragma: no cover - never used for jobs
        raise NotImplementedError("the job pool runs one submitted job at a time")

    def __enter__(self) -> "DeferredJobExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.shutdown(wait=True)
        return False

    # -- driven by the worker's queue ---------------------------------------
    def _drop_cancelled(self) -> None:
        while self._queue and self._queue[0].future.cancelled():
            self._queue.popleft()

    def pending(self) -> int:
        """Jobs waiting to run (cancelled ones no longer count)."""
        return sum(1 for w in self._queue if not w.future.cancelled())

    def peek(self) -> tuple | None:
        """The positional arguments of the next job to run, or None.

        JobRunner submits work(document_id, job_id), so the worker can say
        which document is about to start -- and load the text reader first
        when that document needs it -- before the job holds the thread.
        """
        self._drop_cancelled()
        return self._queue[0].args if self._queue else None

    @property
    def running(self) -> tuple | None:
        """The arguments of the job running right now, if one is."""
        return self._running.args if self._running else None

    def run_next(self) -> cf.Future | None:
        """Run the next job to the end. None when nothing is waiting.

        set_running_or_notify_cancel() comes first, exactly as in a real
        pool: a job cancelled while it waited is skipped, never started.
        """
        while self._queue:
            work = self._queue.popleft()
            if not work.future.set_running_or_notify_cancel():
                continue
            self._running = work
            try:
                result = work.context.run(work.fn, *work.args, **work.kwargs)
            except BaseException as exc:  # noqa: BLE001 - a pool reports, never raises
                work.future.set_exception(exc)
            else:
                work.future.set_result(result)
            finally:
                self._running = None
            return work.future
        return None
