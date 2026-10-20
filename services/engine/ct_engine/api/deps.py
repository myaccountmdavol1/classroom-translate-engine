"""Shared API dependencies (singletons wired once at import)."""

from __future__ import annotations

from functools import lru_cache

from ..jobs.runner import JobRunner
from ..pipeline.manifest_store import ManifestStore
from ..storage import get_store
from ..storage.base import ObjectStore


@lru_cache(maxsize=1)
def get_manifest_store() -> ManifestStore:
    return ManifestStore(get_store())


@lru_cache(maxsize=1)
def get_job_runner() -> JobRunner:
    return JobRunner(get_store(), get_manifest_store())


def get_object_store() -> ObjectStore:
    return get_store()
