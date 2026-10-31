"""FastAPI dependency injection providers.

Centralises service instantiation so that routers receive pre-built
singletons via ``Depends()``.
"""

from typing import Optional

from src.api.services.audit_service import AuditService
from src.api.services.inference_service import InferenceService
from src.api.services.report_service import ReportService
from src.api.services.scan_service import ScanService

# ── Singletons (created once, reused across requests) ────────────

_scan_service: Optional[ScanService] = None
_inference_service: Optional[InferenceService] = None
_audit_service: Optional[AuditService] = None
_report_service: Optional[ReportService] = None


def get_scan_service() -> ScanService:
    """Return the shared :class:`ScanService` instance."""
    global _scan_service
    if _scan_service is None:
        _scan_service = ScanService()
    return _scan_service


def get_inference_service() -> InferenceService:
    """Return the shared :class:`InferenceService` instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


def get_audit_service() -> AuditService:
    """Return the shared :class:`AuditService` instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


def get_report_service() -> ReportService:
    """Return the shared :class:`ReportService` instance."""
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service


async def get_current_user() -> str:
    """Placeholder for user identification.

    In v1 this returns a static string. Replace with JWT / OAuth
    token extraction when auth is implemented.
    """
    return "anonymous"
