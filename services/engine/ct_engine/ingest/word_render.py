"""Showing a Word file as pages.

Word lays a document out when it opens it; the .docx says nothing about
where a line falls. LibreOffice does the same layout and writes it as a PDF,
which every page tool here already understands: the editor, the reading of
pictures, the previews. The teacher still downloads a Word file -- the PDF
is only how the pages are shown and edited (see pipeline/word_export.py).

Measured on the installation guide (5 pages, 7 pictures): about 1 second
and 240 MB at peak, in a separate process that exits when it is done.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ..config import get_settings
from ..util.logging import get_logger

log = get_logger(__name__)

_MAC_APP = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
# Pictures kept exactly, not re-saved as JPEG at a lower resolution: the
# words inside a screenshot are read from these pixels, and on the
# installation guide the JPEG copies lost whole lines ("Locally applied").
_LOSSLESS = (
    '{"UseLosslessCompression":{"type":"boolean","value":"true"},'
    '"ReduceImageResolution":{"type":"boolean","value":"false"}}'
)
# Word files go through Writer's exporter, slides through Impress's.
_PDF_EXPORT = {
    "docx": f"pdf:writer_pdf_Export:{_LOSSLESS}",
    "pptx": f"pdf:impress_pdf_Export:{_LOSSLESS}",
}


# One conversion at a time: LibreOffice peaks near 240 MB, and two at once
# on a 512 MB server would push the engine itself out of memory.
_ONE_AT_A_TIME = threading.Lock()
_OOM_ADJ = "/proc/self/oom_score_adj"


class WordRenderError(RuntimeError):
    """The file could not be laid out as pages."""


def soffice_binary() -> str | None:
    configured = get_settings().soffice_path
    if configured:
        return configured if Path(configured).exists() else None
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    return _MAC_APP if Path(_MAC_APP).exists() else None


def can_render_word() -> bool:
    return soffice_binary() is not None


def docx_to_pdf(data: bytes) -> bytes:
    """Lay a .docx out as pages, as LibreOffice does, and return the PDF."""
    return office_to_pdf(data, "docx")


def office_to_pdf(data: bytes, ext: str) -> bytes:
    """Lay a .docx or .pptx out as pages (one per slide) and return the PDF."""
    binary = soffice_binary()
    if binary is None:
        raise WordRenderError("LibreOffice is not installed on this server.")
    limit = get_settings().office_pages_max_mb
    if len(data) > limit * 2**20:
        raise WordRenderError(
            f"Files over {limit:g} MB are not laid out as pages on this server."
        )
    from ..util.memory import give_back_memory

    # Room for LibreOffice: whatever the last document left behind goes first.
    give_back_memory()
    with _ONE_AT_A_TIME, tempfile.TemporaryDirectory(prefix="ct-word-") as tmp:
        work = Path(tmp)
        source = work / f"document.{ext}"
        source.write_bytes(data)
        # A profile of its own: two conversions at once otherwise fight over
        # the shared one, and a first run would write into the home folder.
        profile = (work / "profile").as_uri()
        try:
            command = [
                binary, "--headless", "--norestore", "--nolockcheck",
                f"-env:UserInstallation={profile}",
                "--convert-to", _PDF_EXPORT[ext], "--outdir", str(work), str(source),
            ]
            if Path(_OOM_ADJ).exists():
                # First to go if memory runs out. A big slide deck full of
                # pictures took the free server past 512 MB and the kernel
                # killed the ENGINE, losing every document on it; killed
                # instead, LibreOffice fails this one file, which then opens
                # as a text list. Set in a shell that execs LibreOffice, so
                # it and its helper processes inherit it.
                command = [
                    "sh", "-c",
                    f'echo 1000 > {_OOM_ADJ} 2>/dev/null; exec "$0" "$@"',
                    *command,
                ]
            done = subprocess.run(
                command,
                capture_output=True,
                timeout=get_settings().word_render_timeout_s,
                env={**os.environ, "HOME": str(work)},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WordRenderError("Laying out the file as pages took too long.") from exc
        out = work / "document.pdf"
        if done.returncode != 0 or not out.exists():
            log.warning(
                "LibreOffice failed (%s): %s", done.returncode,
                done.stderr.decode(errors="replace")[-500:],
            )
            raise WordRenderError("The file could not be laid out as pages.")
        return out.read_bytes()
