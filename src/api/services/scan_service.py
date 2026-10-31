"""Scan management service.

Handles file storage, validation, status tracking, and cleanup for
uploaded MRI scans.
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from src.api.core.config import settings
from src.api.core.security import sanitise_filename
from src.api.models.schemas import ScanInfoResponse, ScanUploadResponse

logger = logging.getLogger(__name__)

# Allowed extensions mapped to a canonical type label
_ALLOWED_EXTENSIONS: Dict[str, str] = {
    ".dcm": "dicom",
    ".dicom": "dicom",
    ".nii": "nifti",
    ".nii.gz": "nifti",
    ".gz": "nifti",  # commonly .nii.gz
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
}


class ScanService:
    """Manage uploaded scan files on disk with in-memory status tracking."""

    def __init__(self) -> None:
        self._upload_dir: Path = Path(settings.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        # In-memory registry (swap for a DB table in production)
        self._registry: Dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────

    async def save_upload(
        self,
        filename: str,
        content: bytes,
    ) -> ScanUploadResponse:
        """Persist an uploaded file and register it for tracking.

        Parameters
        ----------
        filename:
            Original filename from the client (will be sanitised).
        content:
            Raw file bytes.

        Returns
        -------
        ScanUploadResponse
            Metadata about the stored scan.

        Raises
        ------
        ValueError
            If the file type is not supported or the file exceeds the size limit.
        """
        safe_name = sanitise_filename(filename)
        file_type = self._detect_type(safe_name)

        if file_type is None:
            raise ValueError(
                f"Unsupported file type for '{safe_name}'. "
                f"Allowed: {', '.join(sorted(set(_ALLOWED_EXTENSIONS.values())))}"
            )

        if len(content) > settings.max_file_size_bytes:
            raise ValueError(
                f"File exceeds maximum allowed size of {settings.max_file_size_mb} MB."
            )

        scan_id = uuid.uuid4().hex
        scan_dir = self._upload_dir / scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)
        dest = scan_dir / safe_name
        dest.write_bytes(content)

        now = datetime.utcnow()
        record = {
            "scan_id": scan_id,
            "filename": safe_name,
            "file_type": file_type,
            "file_size_bytes": len(content),
            "upload_time": now,
            "status": "uploaded",
            "path": str(dest),
            "prediction_available": False,
        }
        self._registry[scan_id] = record

        logger.info(
            "Scan saved: id=%s type=%s size=%d",
            scan_id,
            file_type,
            len(content),
        )

        return ScanUploadResponse(
            scan_id=scan_id,
            filename=safe_name,
            file_type=file_type,
            file_size_bytes=len(content),
            upload_time=now,
            status="uploaded",
        )

    def get_scan_info(self, scan_id: str) -> Optional[ScanInfoResponse]:
        """Return metadata for a registered scan, or ``None``."""
        rec = self._registry.get(scan_id)
        if rec is None:
            return None
        return ScanInfoResponse(
            scan_id=rec["scan_id"],
            filename=rec["filename"],
            file_type=rec["file_type"],
            file_size_bytes=rec["file_size_bytes"],
            upload_time=rec["upload_time"],
            status=rec["status"],
            prediction_available=rec.get("prediction_available", False),
        )

    def get_scan_path(self, scan_id: str) -> Optional[Path]:
        """Return the on-disk path for the scan, or ``None``."""
        rec = self._registry.get(scan_id)
        if rec is None:
            return None
        return Path(rec["path"])

    def update_status(self, scan_id: str, status: str) -> None:
        """Update the processing status for a scan."""
        if scan_id in self._registry:
            self._registry[scan_id]["status"] = status
            if status == "completed":
                self._registry[scan_id]["prediction_available"] = True

    def mark_prediction_available(self, scan_id: str) -> None:
        """Flag that prediction results are available for this scan."""
        if scan_id in self._registry:
            self._registry[scan_id]["prediction_available"] = True

    def get_all_scans_status(self) -> Dict[str, int]:
        """Return counts of scans grouped by status."""
        counts: Dict[str, int] = {}
        for rec in self._registry.values():
            s = rec["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts

    async def cleanup_old_scans(self) -> int:
        """Delete scans older than the retention window.

        Returns the number of scans removed.
        """
        cutoff = datetime.utcnow() - timedelta(days=settings.retention_days)
        to_remove = [
            sid
            for sid, rec in self._registry.items()
            if rec["upload_time"] < cutoff
        ]
        for sid in to_remove:
            rec = self._registry.pop(sid)
            scan_dir = Path(rec["path"]).parent
            if scan_dir.exists():
                import shutil
                shutil.rmtree(scan_dir, ignore_errors=True)
            logger.info("Cleaned up old scan: %s", sid)
        return len(to_remove)

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _detect_type(filename: str) -> Optional[str]:
        lower = filename.lower()
        # Check compound extension first (.nii.gz)
        for ext, label in _ALLOWED_EXTENSIONS.items():
            if lower.endswith(ext):
                return label
        return None
