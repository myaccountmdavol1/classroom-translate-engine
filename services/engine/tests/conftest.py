"""Shared fixtures.

Storage is redirected to a temp directory before any engine module is
imported, so a test run can never touch real uploaded documents.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parent.parent
CORPUS = ENGINE_ROOT / "corpus" / "out"
sys.path.insert(0, str(ENGINE_ROOT))

_TMP_STORAGE = tempfile.mkdtemp(prefix="ct-test-storage-")
os.environ["CT_STORAGE_ROOT"] = _TMP_STORAGE
os.environ["CT_ENVIRONMENT"] = "development"
os.environ["CT_LOG_LEVEL"] = "WARNING"
# Deterministic page ordering in tests; concurrency is exercised separately.
os.environ["CT_PAGE_CONCURRENCY"] = "2"
# Pin anything a developer's local .env might otherwise change, so a test run
# never depends on whose machine it is on. A stray CT_CORS_ORIGIN_REGEX in
# .env silently broke an assertion about the default.
os.environ.setdefault("CT_CORS_ORIGINS", '["http://localhost:3000"]')
os.environ["CT_CORS_ORIGIN_REGEX"] = ""
os.environ["CT_TRANSLATION_PROVIDER"] = "local"
os.environ["CT_OCR_PROVIDER"] = "local"
# An unset variable still falls through to the developer's .env. Explicit
# empty overrides keep offline tests from sending corpus text to paid APIs.
for credential in (
    "GOOGLE_TRANSLATE_API_KEY", "DEEPL_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "GOOGLE_VISION_CREDENTIALS", "AZURE_VISION_KEY",
):
    os.environ[f"CT_{credential}"] = ""



def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(_TMP_STORAGE, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def corpus() -> Path:
    """Build the test corpus once per session if it is missing."""
    needed = ["01_clean_worksheet.pdf", "13_rotated_pages.pdf", "12_long_100p.pdf"]
    if not all((CORPUS / n).exists() for n in needed):
        subprocess.run(
            [sys.executable, str(ENGINE_ROOT / "corpus" / "build_corpus.py")],
            check=True,
            cwd=str(ENGINE_ROOT),
            capture_output=True,
        )
    if not all(
        (CORPUS / n).exists() for n in ["20_worksheet.docx", "21_two_sections.docx"]
    ):
        subprocess.run(
            [sys.executable, str(ENGINE_ROOT / "corpus" / "build_docx_corpus.py")],
            check=True,
            cwd=str(ENGINE_ROOT),
            capture_output=True,
        )
    if not (CORPUS / "22_lesson.pptx").exists():
        subprocess.run(
            [sys.executable, str(ENGINE_ROOT / "corpus" / "build_pptx_corpus.py")],
            check=True,
            cwd=str(ENGINE_ROOT),
            capture_output=True,
        )
    return CORPUS


def corpus_file(name: str) -> bytes:
    return (CORPUS / name).read_bytes()


@pytest.fixture
def store(tmp_path):
    from ct_engine.storage.local_fs import LocalFileSystemStore

    return LocalFileSystemStore(tmp_path / "store")


@pytest.fixture
def parser():
    from ct_engine.ingest.pdf_parser import PdfParser

    return PdfParser()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from ct_engine.main import app

    with TestClient(app) as c:
        yield c
