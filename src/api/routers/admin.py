"""Admin router — model info, audit log, and system statistics.

These endpoints are intended for operators / dashboard views and may
be secured with stricter access controls in production.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import (
    get_audit_service,
    get_inference_service,
    get_scan_service,
)
from src.api.models.schemas import (
    AuditLogResponse,
    ModelInfo,
    SystemStats,
)
from src.api.services.audit_service import AuditService
from src.api.services.inference_service import InferenceService
from src.api.services.scan_service import ScanService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get(
    "/model-info",
    response_model=List[ModelInfo],
    summary="Current model metadata",
)
async def model_info(
    inf_svc: InferenceService = Depends(get_inference_service),
) -> List[ModelInfo]:
    """Return version, architecture, and training info for every loaded model."""
    raw = inf_svc.get_model_info()
    return [ModelInfo(**m) for m in raw]


@router.get(
    "/audit-log",
    response_model=AuditLogResponse,
    summary="Paginated audit log",
)
async def audit_log(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(50, ge=1, le=500, description="Entries per page"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    scan_id: Optional[str] = Query(None, description="Filter by scan ID"),
    user: Optional[str] = Query(None, description="Filter by user"),
    start_date: Optional[str] = Query(
        None, description="ISO date lower bound (inclusive)"
    ),
    end_date: Optional[str] = Query(
        None, description="ISO date upper bound (inclusive)"
    ),
    audit_svc: AuditService = Depends(get_audit_service),
) -> AuditLogResponse:
    """Query the audit log with optional filtering and pagination."""
    entries, total = await audit_svc.query(
        page=page,
        page_size=page_size,
        action=action,
        scan_id=scan_id,
        user=user,
        start_date=start_date,
        end_date=end_date,
    )
    return AuditLogResponse(
        entries=entries,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/stats",
    response_model=SystemStats,
    summary="System usage statistics",
)
async def system_stats(
    audit_svc: AuditService = Depends(get_audit_service),
    scan_svc: ScanService = Depends(get_scan_service),
) -> SystemStats:
    """Return aggregate statistics for the admin dashboard."""
    stats = await audit_svc.get_stats()
    scans_by_status = scan_svc.get_all_scans_status()

    return SystemStats(
        total_scans=stats.get("total_scans", 0),
        total_predictions=stats.get("total_predictions", 0),
        predictions_by_type=stats.get("predictions_by_type", {}),
        scans_by_status=scans_by_status,
    )
