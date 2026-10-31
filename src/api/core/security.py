"""Security utilities for the Brain MRI AI Detection System.

Provides:
- Simple API-key authentication (v1)
- Input sanitisation helpers
- Rate-limiting placeholder
"""

import re
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.core.config import settings

# ── API-Key Authentication ────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """Validate the ``X-API-Key`` header against the configured key.

    Returns the validated key on success; raises *401* otherwise.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key


# ── Input Sanitisation ────────────────────────────────────────────

_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]")


def sanitise_filename(filename: str) -> str:
    """Strip or replace characters that could cause path-traversal or injection.

    Parameters
    ----------
    filename:
        Raw filename from the client upload.

    Returns
    -------
    str
        Sanitised filename safe for filesystem use.
    """
    # Remove directory components
    filename = filename.replace("\\", "/").split("/")[-1]
    # Replace unsafe chars with underscores
    filename = _SAFE_FILENAME_RE.sub("_", filename)
    # Limit length
    return filename[:255] if filename else "unnamed"


def sanitise_string(value: str, max_length: int = 1024) -> str:
    """Basic string sanitisation to prevent log injection.

    Strips control characters and limits length.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value)
    return cleaned[:max_length]


# ── Rate-Limiting Placeholder ─────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate-limiter placeholder.

    A production deployment should use Redis-backed token-bucket or
    sliding-window counters.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # In-memory store: ip -> list[timestamp]
        self._store: dict[str, list[float]] = {}

    async def check(self, client_ip: str) -> bool:
        """Return ``True`` if the request is allowed, ``False`` if rate-limited.

        .. note:: This is a placeholder — does **not** enforce limits yet.
        """
        # TODO: implement sliding-window counter
        return True
