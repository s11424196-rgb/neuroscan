"""
API client for communicating with the NeuroScan AI backend.

Handles file uploads, prediction requests, heatmap retrieval, and report
downloads.  Falls back to demo mode automatically when the backend is
unreachable.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional

import requests
from PIL import Image

# Backend base URL — configurable via environment variable
BASE_URL: str = os.getenv("NEUROSCAN_API_URL", "http://localhost:8000")

# Timeout for all requests (seconds)
REQUEST_TIMEOUT: int = int(os.getenv("NEUROSCAN_API_TIMEOUT", "60"))
MAX_RETRIES: int = 2


def _url(path: str) -> str:
    """Build a full URL from a relative API path."""
    return f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Execute an HTTP request with simple retry logic.

    Args:
        method: HTTP method (``GET``, ``POST``, etc.).
        url: Full request URL.
        **kwargs: Forwarded to ``requests.request``.

    Returns:
        ``requests.Response`` object.

    Raises:
        requests.exceptions.ConnectionError: If all retries fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url, timeout=REQUEST_TIMEOUT, **kwargs
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
    # Should not reach here, but satisfy the type checker
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_backend_available() -> bool:
    """Check whether the backend API server is reachable.

    Returns:
        ``True`` if the backend responds to a health-check within 3 s.
    """
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def upload_scan(file_bytes: bytes, filename: str) -> str:
    """Upload an MRI scan file to the backend.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used for MIME detection server-side).

    Returns:
        Server-assigned ``scan_id`` string.

    Raises:
        requests.exceptions.HTTPError: On server-side validation errors.
        requests.exceptions.ConnectionError: If backend unreachable.
    """
    files = {"file": (filename, io.BytesIO(file_bytes))}
    resp = _request_with_retry("POST", _url("/api/v1/upload"), files=files)
    data: Dict[str, Any] = resp.json()
    return data["scan_id"]


def run_prediction(scan_id: str, analysis_type: str) -> Dict[str, Any]:
    """Request a prediction for a previously uploaded scan.

    Args:
        scan_id: Identifier returned by :func:`upload_scan`.
        analysis_type: ``"tumour"``, ``"alzheimer"``, or ``"both"``.

    Returns:
        Prediction result dictionary from the backend.
    """
    resp = _request_with_retry(
        "POST",
        _url(f"/api/v1/predict/{scan_id}"),
        json={"analysis_type": analysis_type},
    )
    return resp.json()


def get_heatmap(scan_id: str) -> Image.Image:
    """Retrieve the Grad-CAM heatmap image for a scan.

    Args:
        scan_id: Identifier returned by :func:`upload_scan`.

    Returns:
        Heatmap as a PIL Image.
    """
    resp = _request_with_retry("GET", _url(f"/api/v1/heatmap/{scan_id}"))
    return Image.open(io.BytesIO(resp.content))


def download_report(scan_id: str, fmt: str = "pdf") -> bytes:
    """Download a generated report from the backend.

    Args:
        scan_id: Identifier returned by :func:`upload_scan`.
        fmt: Report format — ``"pdf"`` or ``"csv"``.

    Returns:
        Raw report bytes.
    """
    resp = _request_with_retry(
        "GET",
        _url(f"/api/v1/report/{scan_id}"),
        params={"format": fmt},
    )
    return resp.content
