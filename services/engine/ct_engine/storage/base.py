"""Storage abstraction.

Local development uses the filesystem; production can later swap in object
storage without touching the pipeline (spec sec. 42). Everything the pipeline
persists -- source uploads, extracted assets, page renders, manifests,
exports -- goes through this interface and is addressed by an opaque string
key.

Key convention (hierarchical, but callers must treat keys as opaque):

    documents/{document_id}/source.pdf
    documents/{document_id}/manifest.json
    documents/{document_id}/jobs/{job_id}.json
    documents/{document_id}/pages/{index:04d}/source.png
    documents/{document_id}/pages/{index:04d}/source.thumb.png
    documents/{document_id}/assets/{sha256}.{ext}
    documents/{document_id}/export/translated.pdf
"""

from __future__ import annotations

import datetime as _dt
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

# Keys are restricted to a conservative character set. This is the single
# chokepoint that makes path traversal impossible for a filesystem backend,
# so it is enforced here rather than in each implementation.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class StorageError(RuntimeError):
    pass


class InvalidKeyError(StorageError):
    pass


class ObjectNotFoundError(StorageError):
    pass


def validate_key(key: str) -> str:
    if not key or not _KEY_RE.match(key):
        raise InvalidKeyError(f"Invalid storage key: {key!r}")
    # Reject traversal and absolute-looking keys even though the charset
    # already makes them hard to express.
    if ".." in key or key.startswith("/") or "//" in key:
        raise InvalidKeyError(f"Invalid storage key: {key!r}")
    return key


class ObjectStore(ABC):
    """Content-addressed-ish blob storage with an opaque string key."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Write `data` at `key`, overwriting. Returns the key."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read the object. Raises ObjectNotFoundError if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete one object. Returns True if something was removed."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under `prefix`. Returns the count removed."""

    @abstractmethod
    def list_prefix(self, prefix: str) -> Iterator[str]: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    def put_file(self, key: str, path: Path, *, content_type: str | None = None) -> str:
        """Move the file at `path` into the store at `key`. Returns the key.

        The file is taken over: it no longer exists at `path` afterwards.
        This is how a large upload or export reaches the store without the
        whole thing ever being held in memory as one `bytes` object. Create
        the file under `scratch_dir()` so a filesystem backend can rename it
        into place instead of copying it.
        """
        path = Path(path)
        try:
            return self.put_bytes(key, path.read_bytes(), content_type=content_type)
        finally:
            path.unlink(missing_ok=True)

    def scratch_dir(self) -> Path:
        """Where to create a temporary file that will be given to `put_file`."""
        return Path(tempfile.gettempdir())

    def created_at(self) -> _dt.datetime | None:
        """When this store started out empty, if the backend knows.

        On a host whose disk does not survive a restart, this is when the
        server last started from nothing -- which is what tells a teacher
        "the server restarted" apart from "the document expired".
        """
        return None

    def put_text(self, key: str, text: str, *, content_type: str = "text/plain") -> str:
        return self.put_bytes(key, text.encode("utf-8"), content_type=content_type)

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")

    @abstractmethod
    def local_path(self, key: str) -> Path | None:
        """Filesystem path for the object, when one exists.

        Returns None for backends with no local file (e.g. S3). Callers that
        need a real path -- PyMuPDF opening a large PDF without buffering it
        in memory -- must handle None by falling back to bytes.
        """
