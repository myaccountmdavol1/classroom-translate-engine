"""Giving back memory between pages, so a big document fits a small server.

MuPDF keeps every picture it has decoded in a cache that only grows: on a
20-slide deck of photographs the engine climbed ~6 MB a slide and went past
the free server's 512 MB while translating, and the host restarted it,
losing every document. Emptied after each page it stayed flat at ~80 MB.
"""

from __future__ import annotations


def release_page_memory() -> None:
    """Drop MuPDF's decoded-picture cache; call after finishing a page."""
    try:
        import fitz

        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass


def container_memory_mb() -> dict | None:
    """Memory in use on this server, for /api/health.

    The free host restarts the whole service past 512 MB, so this is the
    number to watch while a big document is translated. Read from the
    container's own cgroup (what the host enforces) and, failing that, the
    sum of every process's resident memory. None where neither exists
    (macOS during development).
    """
    from pathlib import Path

    for current, limit in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            used = int(Path(current).read_text().strip())
            raw = Path(limit).read_text().strip()
            cap = None if raw == "max" else int(raw)
            return {
                "used_mb": round(used / 2**20),
                "limit_mb": round(cap / 2**20) if cap and cap < 2**50 else None,
            }
        except (OSError, ValueError):
            continue
    total = 0
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            continue
    return {"used_mb": round(total / 1024), "limit_mb": None} if total else None


def give_back_memory() -> None:
    """Return freed memory to the system after a job or before LibreOffice.

    Python and MuPDF free memory but glibc keeps it: on the live server an
    idle engine sat at ~395 MB after a few documents (~200 MB fresh), so
    LibreOffice laying out the next slide deck tipped it over 512 MB and the
    host restarted everything. malloc_trim hands the freed pages back.
    """
    import gc

    gc.collect()
    release_page_memory()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc (macOS during development)
