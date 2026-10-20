"""Scheduled deletion of expired documents (spec sec. 43).

School documents contain student work. The default posture is that nothing is
kept: uploads and everything derived from them are removed once the retention
window passes, without anyone having to remember to delete them.
"""

from __future__ import annotations

import datetime as _dt
import threading

from ..config import get_settings
from ..pipeline.manifest_store import ManifestStore
from ..storage.base import ObjectStore
from ..util.logging import get_logger

log = get_logger(__name__)


def expired_document_ids(
    store: ObjectStore, manifests: ManifestStore, *, now: _dt.datetime | None = None
) -> list[str]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(hours=get_settings().retention_hours)
    expired: list[str] = []
    seen: set[str] = set()
    for key in store.list_prefix("documents"):
        parts = key.split("/")
        if len(parts) < 2:
            continue
        doc_id = parts[1]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        try:
            manifest = manifests.load_manifest(doc_id)
        except Exception:
            continue
        if manifest is None:
            continue
        created = manifest.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=_dt.timezone.utc)
        if created < cutoff:
            expired.append(doc_id)
    return expired


def sweep_once(store: ObjectStore, manifests: ManifestStore) -> int:
    removed = 0
    for doc_id in expired_document_ids(store, manifests):
        try:
            manifests.delete_document(doc_id)
            removed += 1
            log.info("Retention: deleted expired document %s", doc_id)
        except Exception as exc:
            log.warning("Retention: could not delete %s: %r", doc_id, exc)
    return removed


class RetentionSweeper:
    """Periodic background sweep. Stops cleanly on shutdown."""

    def __init__(self, store: ObjectStore, manifests: ManifestStore) -> None:
        self.store = store
        self.manifests = manifests
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        settings = get_settings()
        interval = max(60, settings.retention_sweep_minutes * 60)

        def loop() -> None:
            # Sweep once at startup so a restart cleans up anything that
            # expired while the service was down.
            while not self._stop.is_set():
                try:
                    sweep_once(self.store, self.manifests)
                except Exception:
                    log.exception("Retention sweep failed")
                self._stop.wait(interval)

        self._thread = threading.Thread(target=loop, name="ct-retention", daemon=True)
        self._thread.start()
        log.info(
            "Retention sweeper started (window=%sh, interval=%smin)",
            settings.retention_hours,
            settings.retention_sweep_minutes,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
