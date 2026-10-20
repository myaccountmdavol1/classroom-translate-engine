"""The store journal: every write the engine makes, handed to the page in order.

In the browser the engine's store lives in Pyodide's memory (MEMFS), which
goes away with the tab. The page keeps the lasting copy in the browser's
private disk (OPFS), and Python cannot write that itself: OPFS is reached
only through promises, and Python here never waits on a promise in the
middle of a job (see boot.py). So each write the engine makes to its store is
also told to the page, in the order it was made, the moment it is made; the
engine host (apps/web/src/engine/opfs-store.ts) applies them strictly first
in, first out, each key written under a temporary name and then moved over
the old one. Whatever the host has applied when a tab dies is therefore a
PREFIX of the engine's own writes -- exactly what a server's disk holds after
the process is killed -- and the engine's own resume (runner.
resume_interrupted) picks up from it as it does after a server restart.

Wrapped on the one cached store instance the routes and the job runner
share (instance attributes, so nothing else in the process changes):
put_bytes, put_file, put_text, delete and delete_prefix. A write that calls
another of them (the base put_text calls put_bytes) is told once. A write
that raises is not told: it did not happen. Nothing here ever yields.

What is told is (seq, op, key, path): op "put" with the file's path in the
store (the worker reads the bytes straight from MEMFS, so the bytes are
copied once, into a buffer it hands over to the page without copying
again), "delete", or "delete_prefix"; or "lost" (no path) for a write whose
file cannot be found, so every write has its number and the page can tell
a gap. Telling never stops a write: a page that cannot hear is counted, and
the engine carries on.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

WRITE_METHODS = ("put_bytes", "put_file", "put_text", "delete", "delete_prefix")

Sink = Callable[[int, str, str, "str | None"], Any]


class StoreJournal:
    def __init__(self, store, sink: Sink) -> None:
        self.store = store
        self.sink = sink
        self.seq = 0
        self.failures = 0
        self.last_error: str | None = None
        self._depth = 0
        self._previous: dict[str, Any] | None = None

    # -- wiring -------------------------------------------------------------
    def install(self) -> None:
        if self._previous is not None:
            return
        own = vars(self.store)
        self._previous = {name: own.get(name) for name in WRITE_METHODS}
        for name in WRITE_METHODS:
            setattr(self.store, name, self._wrap(name, getattr(self.store, name)))

    def uninstall(self) -> None:
        if self._previous is None:
            return
        own = vars(self.store)
        for name, previous in self._previous.items():
            if previous is None:
                own.pop(name, None)
            else:
                own[name] = previous
        self._previous = None

    def _wrap(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def write(*args: Any, **kwargs: Any) -> Any:
            key = args[0] if args else kwargs.get("key", kwargs.get("prefix"))
            self._depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                self._depth -= 1
            if self._depth == 0:
                self._tell(name, str(key))
            return result

        return write

    # -- telling ---------------------------------------------------------------
    def _tell(self, method: str, key: str) -> None:
        path: str | None = None
        if method == "delete":
            op = "delete"
        elif method == "delete_prefix":
            op = "delete_prefix"
            key = key.rstrip("/")
        else:
            op = "put"
            try:
                found = self.store.local_path(key)
            except Exception:
                found = None
            if found is None:
                # Numbered and told all the same, as "lost": the page then
                # knows its copy of this document is missing a write, and
                # stops trusting it, rather than carrying on past a gap.
                self.failures += 1
                self.last_error = f"no file for {key}"
                op = "lost"
            else:
                path = str(found)
        self.seq += 1
        try:
            self.sink(self.seq, op, key, path)
        except Exception as exc:  # a page that cannot hear never stops a write
            self.failures += 1
            self.last_error = repr(exc)[:300]

    def stats(self) -> dict:
        return {"seq": self.seq, "failures": self.failures, "last_error": self.last_error}
