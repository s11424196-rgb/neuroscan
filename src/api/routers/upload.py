"""Upload router — endpoints for ingesting MRI scan files.

Supports single and batch uploads with file-type validation, size
checks, and optional DICOM anonymisation.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.dependencies import (
    get_audit_service,
    get_current_user,
    get_scan_service,
)
from src.api.models.schemas import (
    BatchUploadResponse,
    ErrorResponse,
    ScanInfoResponse,
    ScanUploadResponse,
)
from src.api.services.audit_service import AuditService
from src.api.services.scan_service import ScanService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["uploads"])


@router.post(
    "/upload",
    response_model=ScanUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
    summary="Upload a single MRI scan",
)
async def upload_scan(
    file: UploadFile = File(..., description="DICOM, NIfTI, JPG, or PNG file"),
    scan_svc: ScanService = Depends(get_scan_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> ScanUploadResponse:
    """Accept a single MRI scan file, validate it, and store it for later analysis."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await scan_svc.save_upload(file.filename, content)
    except ValueError as exc:
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "size" in str(exc).lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=str(exc))

    # Audit (no PHI in the log — only scan_id and file type)
    await audit_svc.log(
        action="upload",
        scan_id=result.scan_id,
        user=user,
        details=f"type={result.file_type}",
    )

    return result


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload multiple MRI scans",
)
async def upload_batch(
    files: List[UploadFile] = File(
        ..., description="List of DICOM / NIfTI / image files"
    ),
    scan_svc: ScanService = Depends(get_scan_service),
    audit_svc: AuditService = Depends(get_audit_service),
    user: str = Depends(get_current_user),
) -> BatchUploadResponse:
    """Accept multiple scan files in a single request."""
    results: List[ScanUploadResponse] = []
    for f in files:
        if not f.filename:
            continue
        content = await f.read()
        if not content:
            continue
        try:
            res = await scan_svc.save_upload(f.filename, content)
            results.append(res)
            await audit_svc.log(
                action="upload",
                scan_id=res.scan_id,
                user=user,
                details=f"type={res.file_type}",
            )
        except ValueError as exc:
            logger.warning("Skipped file %s: %s", f.filename, exc)

    return BatchUploadResponse(scans=results, total=len(results))


@router.get(
    "/scans/{scan_id}",
    response_model=ScanInfoResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get scan info and status",
)
async def get_scan(
    scan_id: str,
    scan_svc: ScanService = Depends(get_scan_service),
) -> ScanInfoResponse:
    """Return metadata and processing status for a previously uploaded scan."""
    info = scan_svc.get_scan_info(scan_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan '{scan_id}' not found.",
        )
    return info
