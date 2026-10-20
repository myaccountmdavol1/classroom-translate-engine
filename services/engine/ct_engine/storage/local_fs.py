"""Filesystem-backed object store for local development and single-node runs."""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

from ..util.logging import get_logger
from .base import ObjectNotFoundError, ObjectStore, validate_key

log = get_logger(__name__)

# Both live beside the keys, never among them: a key must start with a letter
# or digit, so neither name can be addressed, listed or swept as a document.
_SCRATCH = ".scratch"
_CREATED_MARKER = ".created"
# A scratch file this old belongs to a process that was killed mid-upload.
_SCRATCH_MAX_AGE_S = 6 * 3600


class LocalFileSystemStore(ObjectStore):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / _CREATED_MARKER
        # Dated only when the store really is starting out empty. A store
        # that already holds documents but no date has an unknown start, and
        # dating it now would claim a restart that never happened.
        if not marker.exists() and not any(self.root.iterdir()):
            marker.write_text(_dt.datetime.now(_dt.timezone.utc).isoformat())
        self._scratch = self.root / _SCRATCH
        self._scratch.mkdir(exist_ok=True)
        self._clear_old_scratch()
        log.info("Local object store rooted at %s", self.root)

    def _clear_old_scratch(self) -> None:
        # Only old files: another process on the same root (a reload, a
        # test) may be writing a fresh one right now.
        cutoff = time.time() - _SCRATCH_MAX_AGE_S
        for p in self._scratch.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = (self.root / key).resolve()
        # Defence in depth: even with a validated key, confirm the resolved
        # path is inside the root before touching the filesystem.
        if not str(path).startswith(str(self.root) + os.sep) and path != self.root:
            raise ObjectNotFoundError(f"Key escapes storage root: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then rename. An
        # interrupted write must never leave a truncated manifest or a
        # half-written PDF that a resumed job would happily read.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return key

    def put_file(self, key: str, path: Path, *, content_type: str | None = None) -> str:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = Path(path)
        try:
            with open(path, "rb+") as fh:
                os.fsync(fh.fileno())
            # A rename when the file was made in scratch_dir() (same disk),
            # so a 100 MB upload is never copied, and a reader never sees a
            # half-written object.
            os.replace(path, dest)
        except OSError:
            # Another filesystem: copy beside the destination, then rename.
            fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
            os.close(fd)
            try:
                shutil.copyfile(path, tmp)
                with open(tmp, "rb+") as fh:
                    os.fsync(fh.fileno())
                os.replace(tmp, dest)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            finally:
                path.unlink(missing_ok=True)
        return key

    def scratch_dir(self) -> Path:
        self._scratch.mkdir(exist_ok=True)
        return self._scratch

    def created_at(self) -> _dt.datetime | None:
        try:
            value = _dt.datetime.fromisoformat(
                (self.root / _CREATED_MARKER).read_text().strip()
            )
        except (OSError, ValueError):
            return None
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"No object at key {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.is_file():
            path.unlink()
            return True
        return False

    def delete_prefix(self, prefix: str) -> int:
        base = self._path(prefix.rstrip("/"))
        if base.is_dir():
            count = sum(1 for _ in base.rglob("*") if _.is_file())
            shutil.rmtree(base, ignore_errors=True)
            return count
        if base.is_file():
            base.unlink()
            return 1
        return 0

    def list_prefix(self, prefix: str) -> Iterator[str]:
        base = self._path(prefix.rstrip("/"))
        if not base.is_dir():
            return
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                yield str(p.relative_to(self.root))

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"No object at key {key!r}")
        return path.stat().st_size

    def local_path(self, key: str) -> Path | None:
        path = self._path(key)
        return path if path.is_file() else None
