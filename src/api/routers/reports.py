"""Reports router — endpoints for downloading PDF and CSV reports.

Generates professional reports that include scan images, prediction
results, Grad-CAM heatmaps, and a prominent research disclaimer.
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from src.api.dependencies import (
    get_audit_service,
    get_current_user,
    get_inference_service,
    get_report_service,
    get_scan_service,
)
from src.api.models.schemas import (
    ErrorResponse,
    PredictionResult,
    ReportRequest,
)
from src.api.services.audit_service import AuditService
from src.api.services.inference_service import InferenceService
from src.api.services.report_service import ReportService
from src.api.services.scan_service import ScanService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get(
    "/{scan_id}/pdf",
    responses={
        200: {"content": {"application/pdf": {}}, "description": "PDF report"},
        404: {"model": ErrorResponse},
    },
    summary="Download PDF report",
)
async def download_pdf(
    scan_id: str,
    scan_svc: ScanService = Depends(get_scan_service),
    inf_svc: InferenceService = Depends(get_inference_service),
    report_svc: ReportService = Depends(get_report_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> Response:
    """Generate and return a PDF report for the given scan."""
    scan_info = scan_svc.get_scan_info(scan_id)
    if scan_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan '{scan_id}' not found.",
        )

    results = inf_svc.get_cached_results(scan_id)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction results for scan '{scan_id}'. Run /predict first.",
        )

    # Grab first available heatmap overlay
    overlay_b64 = None
    for task in results:
        heatmap = inf_svc.get_heatmap(scan_id, task)
        if heatmap:
            overlay_b64 = heatmap.overlay_base64
            break

    pdf_bytes = await report_svc.generate_pdf(
        scan_id=scan_id,
        filename=scan_info.filename,
        results=results,
        heatmap_overlay_b64=overlay_b64,
        model_version="0.1.0-mock",
    )

    await audit_svc.log(
        action="download_report",
        scan_id=scan_id,
        user=user,
        details="pdf",
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="report_{scan_id}.pdf"'
        },
    )


@router.get(
    "/{scan_id}/csv",
    responses={
        200: {"content": {"text/csv": {}}, "description": "CSV results"},
        404: {"model": ErrorResponse},
    },
    summary="Download CSV results",
)
async def download_csv(
    scan_id: str,
    inf_svc: InferenceService = Depends(get_inference_service),
    report_svc: ReportService = Depends(get_report_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> Response:
    """Return prediction results for a single scan as CSV."""
    results = inf_svc.get_cached_results(scan_id)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction results for scan '{scan_id}'.",
        )

    rows = _results_to_rows(scan_id, results)
    csv_text = await report_svc.generate_csv(rows)

    await audit_svc.log(
        action="download_report",
        scan_id=scan_id,
        user=user,
        details="csv",
    )

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="results_{scan_id}.csv"'
        },
    )


@router.post(
    "/batch",
    responses={
        200: {"content": {"text/csv": {}}, "description": "Batch CSV report"},
    },
    summary="Generate batch CSV report",
)
async def batch_report(
    body: ReportRequest,
    inf_svc: InferenceService = Depends(get_inference_service),
    report_svc: ReportService = Depends(get_report_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> Response:
    """Combine prediction results for multiple scans into a single CSV."""
    all_rows: List[Dict] = []
    for sid in body.scan_ids:
        results = inf_svc.get_cached_results(sid)
        if results:
            all_rows.extend(_results_to_rows(sid, results))

    if not all_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found for any of the provided scan IDs.",
        )

    csv_text = await report_svc.generate_csv(all_rows)

    await audit_svc.log(
        action="download_report",
        scan_id=",".join(body.scan_ids[:5]),
        user=user,
        details=f"batch_csv ({len(body.scan_ids)} scans)",
    )

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="batch_results.csv"'
        },
    )


# ── Helpers ───────────────────────────────────────────────────────


def _results_to_rows(
    scan_id: str,
    results: Dict[str, PredictionResult],
) -> List[Dict]:
    """Convert cached PredictionResult objects to flat dicts for CSV output."""
    rows = []
    for task, res in results.items():
        rows.append(
            {
                "scan_id": scan_id,
                "task": task,
                "predicted_class": res.predicted_class,
                "confidence": res.confidence,
                "uncertainty_score": res.uncertainty_score,
                "is_low_confidence": res.is_low_confidence,
                **{f"prob_{cls}": prob for cls, prob in res.probabilities.items()},
            }
        )
    return rows
