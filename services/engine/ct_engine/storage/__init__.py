"""Storage layer."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import (
    InvalidKeyError,
    ObjectNotFoundError,
    ObjectStore,
    StorageError,
    validate_key,
)
from .local_fs import LocalFileSystemStore


@lru_cache(maxsize=1)
def get_store() -> ObjectStore:
    settings = get_settings()
    backend = settings.storage_backend.lower()
    if backend == "local":
        return LocalFileSystemStore(settings.storage_path)
    # Fail loudly rather than silently falling back to local storage: a
    # production deployment that thinks it is writing to S3 but is actually
    # writing to a container filesystem would lose every document on restart.
    raise StorageError(
        f"Unsupported CT_STORAGE_BACKEND={settings.storage_backend!r}. "
        "Supported: 'local'."
    )


__all__ = [
    "ObjectStore", "LocalFileSystemStore", "StorageError", "ObjectNotFoundError",
    "InvalidKeyError", "validate_key", "get_store",
]
