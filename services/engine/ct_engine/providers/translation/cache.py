"""Translation cache keyed by content hash (spec sec. 50).

Educational documents repeat themselves heavily: "Show your work.", answer
choices, running headers. Caching by content means a 100-page packet
translates each distinct string once.

The key includes the provider, so switching engines never serves output from
the previous one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ...util.hashing import translation_cache_key


class TranslationCache:
    """Process-local LRU cache. Thread-safe.

    Deliberately in-memory: cached student text must not outlive the job on
    disk, and the win is within a single document anyway. The cap is sized
    for one large packet (100 pages of distinct strings fit easily) and keeps
    the idle footprint to a few MB on a 512 MB host, where that headroom is
    what LibreOffice needs.
    """

    def __init__(self, max_entries: int = 5_000) -> None:
        # Insertion order is recency order: a hit moves to the end, and the
        # front is the least recently used, so eviction is O(1).
        self._data: OrderedDict[str, str] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(
        self, text: str, source: str | None, target: str, provider: str
    ) -> str | None:
        key = translation_cache_key(text, source, target, provider)
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
                self._data.move_to_end(key)
            return value

    def put(
        self, text: str, source: str | None, target: str, provider: str, result: str
    ) -> None:
        key = translation_cache_key(text, source, target, provider)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = result
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return {"entries": len(self._data), "hits": self.hits, "misses": self.misses}


# One shared cache per process.
GLOBAL_CACHE = TranslationCache()
