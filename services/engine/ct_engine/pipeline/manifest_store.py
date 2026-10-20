"""Persistence for document manifests and jobs.

Every stage writes through here, which is what makes processing resumable:
a crash loses at most the page in flight (spec sec. 40).
"""

from __future__ import annotations

import json
import threading

from ..model.document import DocumentManifest
from ..model.job import DocumentJob
from ..storage import keys as K
from ..storage.base import ObjectNotFoundError, ObjectStore
from ..util.logging import get_logger

log = get_logger(__name__)

# Manifest writes are read-modify-write, so concurrent page workers on the
# same document must serialize. Per-document locks keep unrelated documents
# fully parallel.
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def document_lock(document_id: str) -> threading.RLock:
    with _locks_guard:
        lock = _locks.get(document_id)
        if lock is None:
            lock = threading.RLock()
            _locks[document_id] = lock
        return lock


class ManifestStore:
    def __init__(self, store: ObjectStore) -> None:
        self.store = store

    # -- manifests ---------------------------------------------------------
    def save_manifest(self, manifest: DocumentManifest) -> None:
        manifest.touch()
        payload = manifest.model_dump_json(indent=None)
        self.store.put_text(
            K.manifest_key(manifest.document_id),
            payload,
            content_type="application/json",
        )

    def load_manifest(self, document_id: str) -> DocumentManifest | None:
        try:
            raw = self.store.get_text(K.manifest_key(document_id))
        except ObjectNotFoundError:
            return None
        try:
            return DocumentManifest.model_validate_json(raw)
        except Exception as exc:
            # A manifest that will not parse is a bug or a schema change, not
            # a missing document; surfacing it as "not found" would hide it.
            log.error("Manifest for %s is unreadable: %r", document_id, exc)
            raise

    def manifest_exists(self, document_id: str) -> bool:
        return self.store.exists(K.manifest_key(document_id))

    # -- jobs --------------------------------------------------------------
    def save_job(self, job: DocumentJob) -> None:
        job.touch()
        self.store.put_text(
            K.job_key(job.document_id, job.job_id),
            job.model_dump_json(),
            content_type="application/json",
        )

    def load_job(self, document_id: str, job_id: str) -> DocumentJob | None:
        try:
            raw = self.store.get_text(K.job_key(document_id, job_id))
        except ObjectNotFoundError:
            return None
        return DocumentJob.model_validate_json(raw)

    def list_jobs(self, document_id: str) -> list[DocumentJob]:
        jobs: list[DocumentJob] = []
        for key in self.store.list_prefix(K.jobs_prefix(document_id)):
            if not key.endswith(".json"):
                continue
            try:
                jobs.append(DocumentJob.model_validate_json(self.store.get_text(key)))
            except Exception as exc:
                log.warning("Skipping unreadable job %s: %r", key, exc)
        return sorted(jobs, key=lambda j: j.created_at)

    def delete_document(self, document_id: str) -> int:
        return self.store.delete_prefix(K.doc_prefix(document_id))
