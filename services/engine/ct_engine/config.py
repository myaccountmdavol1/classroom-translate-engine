"""Environment-driven configuration.

Every provider choice and every limit is settable from the environment, and
no credential is ever hard-coded (spec sec. 49). Defaults are the local,
zero-cost, privacy-preserving options: nothing leaves the machine unless an
operator explicitly configures a remote provider.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CT_",
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service ---
    environment: str = "development"
    log_level: str = "INFO"
    # Browser origins allowed to call the engine. The Next.js dev server by
    # default; production deployments must set this explicitly.
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    # Regex alternative to the list above, for origins whose hostname is not
    # fixed. Vercel mints a new URL per deployment, so a deployed frontend
    # talking to a locally-run engine cannot be allow-listed by exact value.
    # Empty by default: this widens who may call the engine, so it is opt-in.
    cors_origin_regex: str | None = None

    # --- Storage ---
    storage_backend: str = "local"  # local | s3 (s3 not implemented yet)
    storage_root: Path = Path("./data")

    # --- Upload limits ---
    # 100 MB. The free server has 512 MB of memory, and a translated PDF is
    # rebuilt with the original and the new file both open; much past this,
    # a big scanned packet ran it out of memory at "Rebuilding pages", and
    # the restart lost every teacher's documents. /api/health reports it so
    # the upload page states the real limit.
    max_upload_bytes: int = 100 * 1024 * 1024
    max_page_count: int = 250
    # A Word document has no page count until Word opens it, so the size
    # bound is on paragraphs instead. 4000 is roughly a 150-page document,
    # matching the spirit of the page limit above.
    max_docx_paragraphs: int = 4000
    # A spreadsheet's size is its text cells, not its sheets: one sheet can
    # hold a whole gradebook. 20000 cells of text is far beyond any
    # worksheet and still a bounded amount of translation.
    max_xlsx_cells: int = 20000
    allowed_content_types: list[str] = Field(
        default_factory=lambda: [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]
    )

    # --- Rendering ---
    # Preview DPI for the review viewer. 144 = 2x CSS pixels at 72pt, which is
    # crisp on retina displays without exploding storage on a 100-page doc.
    preview_dpi: int = 144
    thumbnail_width_px: int = 220
    # DPI used when rasterizing a page for OCR. 300 is the floor for reliable
    # small-text recognition on scanned worksheets.
    ocr_dpi: int = 300
    # LibreOffice, for showing a Word file as pages. Found on PATH or in the
    # macOS app when unset; without it a Word file opens as a text list.
    soffice_path: str | None = None
    word_render_timeout_s: int = 180
    # Larger Word or PowerPoint files are not laid out as pages here: on the
    # free 512 MB server LibreOffice plus the engine went over the limit and
    # the host restarted the service. They open as a text list instead.
    office_pages_max_mb: float = 12.0

    # --- Analysis thresholds ---
    # A page with fewer than this many extractable characters is treated as
    # having no usable text layer.
    scanned_text_char_threshold: int = 24
    # ...unless it also has no images, in which case it is simply blank.
    text_coverage_scanned_threshold: float = 0.005

    # --- Providers ---
    translation_provider: str = "local"
    ocr_provider: str = "local"

    # --- Access ---
    # One shared passphrase that every request must carry. Unset means the
    # engine is open, which is correct on localhost and wrong in public --
    # /api/health reports which it is so a deployment cannot get this wrong
    # quietly.
    access_passphrase: str | None = None

    google_translate_api_key: str | None = None
    deepl_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_vision_credentials: str | None = None
    azure_vision_endpoint: str | None = None
    azure_vision_key: str | None = None

    # --- Concurrency ---
    # Page-level worker count. Kept modest by default: PyMuPDF releases the
    # GIL for rendering, but memory scales with concurrent page pixmaps.
    page_concurrency: int = 4
    max_job_concurrency: int = 2

    # --- Retention / privacy ---
    # Uploaded student work is deleted after this many hours (spec sec. 43).
    retention_hours: int = 24
    retention_sweep_minutes: int = 30
    # When false, document text never appears in logs at any level.
    log_document_text: bool = False

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_root).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
