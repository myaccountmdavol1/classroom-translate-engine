"""Content hashing, used for asset dedupe and translation caching."""

from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(data: bytes | str, length: int = 16) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def translation_cache_key(
    text: str, source_lang: str | None, target_lang: str, provider: str
) -> str:
    """Stable key for a translation result.

    Includes the provider so switching engines does not serve stale output
    from the previous one (spec sec. 50).
    """
    payload = f"{provider}\x1f{source_lang or 'auto'}\x1f{target_lang}\x1f{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
