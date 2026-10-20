"""Logging setup with a privacy-aware helper.

Student document contents must not be written to logs unless an operator
explicitly opts in (spec sec. 43), so all call sites that want to log text go
through `safe_text`.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # These are chatty at DEBUG and say nothing we need.
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def safe_text(text: str | None, limit: int = 60) -> str:
    """Render text for a log line, redacting unless explicitly enabled."""
    from ..config import get_settings

    if text is None:
        return "<none>"
    if not get_settings().log_document_text:
        return f"<{len(text)} chars redacted>"
    snippet = text.replace("\n", " ")[:limit]
    return f"{snippet!r}{'...' if len(text) > limit else ''}"
