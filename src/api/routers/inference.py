"""Inference router — endpoints for running and retrieving predictions.

Handles prediction requests, result retrieval, and Grad-CAM heatmap
delivery for uploaded MRI scans.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from src.api.dependencies import (
    get_audit_service,
    get_current_user,
    get_inference_service,
    get_scan_service,
)
from src.api.models.schemas import (
    ErrorResponse,
    HeatmapResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.audit_service import AuditService
from src.api.services.inference_service import InferenceService
from src.api.services.scan_service import ScanService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post(
    "/predict/{scan_id}",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Run prediction on an uploaded scan",
)
async def run_prediction(
    scan_id: str,
    body: PredictionRequest,
    scan_svc: ScanService = Depends(get_scan_service),
    inf_svc: InferenceService = Depends(get_inference_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> PredictionResponse:
    """Preprocess the scan, run model inference, and generate Grad-CAM heatmaps.

    In mock mode synthetic predictions and Gaussian-blob heatmaps are returned.
    """
    scan_path = scan_svc.get_scan_path(scan_id)
    if scan_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan '{scan_id}' not found.",
        )

    scan_svc.update_status(scan_id, "processing")

    try:
        response = await inf_svc.predict(
            scan_id=scan_id,
            scan_path=str(scan_path),
            analysis_type=body.analysis_type,
        )
    except Exception as exc:
        scan_svc.update_status(scan_id, "failed")
        logger.exception("Prediction failed for scan %s", scan_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    scan_svc.update_status(scan_id, "completed")

    await audit_svc.log(
        action="predict",
        scan_id=scan_id,
        user=user,
        details=body.analysis_type,
    )

    return response


@router.get(
    "/results/{scan_id}",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Retrieve prediction results",
)
async def get_results(
    scan_id: str,
    inf_svc: InferenceService = Depends(get_inference_service),
    scan_svc: ScanService = Depends(get_scan_service),
) -> PredictionResponse:
    """Return the full prediction results including probabilities and heatmap data."""
    cached = inf_svc.get_cached_results(scan_id)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for scan '{scan_id}'. Run /predict first.",
        )

    from datetime import datetime

    return PredictionResponse(
        scan_id=scan_id,
        analysis_type="both" if len(cached) > 1 else list(cached.keys())[0],
        results=list(cached.values()),
        model_version="0.1.0-mock",
        processing_time_seconds=0.0,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/results/{scan_id}/heatmap",
    response_model=HeatmapResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get Grad-CAM heatmap image",
)
async def get_heatmap(
    scan_id: str,
    task: Optional[str] = "tumour",
    inf_svc: InferenceService = Depends(get_inference_service),
) -> HeatmapResponse:
    """Return the Grad-CAM heatmap overlay as base64-encoded PNG."""
    heatmap = inf_svc.get_heatmap(scan_id, task=task or "tumour")
    if heatmap is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No heatmap found for scan '{scan_id}', task '{task}'.",
        )
    return heatmap
