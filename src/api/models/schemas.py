"""Pydantic schemas for all API request and response bodies.

Every model is strict, immutable, and fully documented so that the
auto-generated OpenAPI spec is self-explanatory.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Scan / Upload ─────────────────────────────────────────────────

class ScanUploadResponse(BaseModel):
    """Returned after a successful scan upload."""

    scan_id: str = Field(..., description="Unique identifier for the uploaded scan.")
    filename: str = Field(..., description="Sanitised original filename.")
    file_type: str = Field(..., description="Detected file type (e.g. 'dicom', 'nifti', 'png').")
    file_size_bytes: int = Field(..., description="Size of the uploaded file in bytes.")
    upload_time: datetime = Field(..., description="Server timestamp of the upload.")
    status: str = Field(default="uploaded", description="Current scan status.")


class BatchUploadResponse(BaseModel):
    """Returned after a batch upload."""

    scans: List[ScanUploadResponse]
    total: int


class ScanInfoResponse(BaseModel):
    """Full metadata for a stored scan."""

    scan_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    upload_time: datetime
    status: str = Field(
        description="One of: uploaded, processing, completed, failed."
    )
    prediction_available: bool = False


# ── Prediction / Inference ────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Body sent to trigger inference on a scan."""

    analysis_type: Literal["tumour", "alzheimer", "both"] = Field(
        ...,
        description="Type of analysis to run.",
    )


class PredictionResult(BaseModel):
    """Prediction output for a single analysis task."""

    task: str = Field(..., description="'tumour' or 'alzheimer'.")
    predicted_class: str = Field(..., description="Most likely class label.")
    probabilities: Dict[str, float] = Field(
        ..., description="Class → probability mapping."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the top prediction."
    )
    uncertainty_score: float = Field(
        ..., ge=0.0, description="Model uncertainty estimate (lower is better)."
    )
    is_low_confidence: bool = Field(
        ..., description="True when confidence falls below the reliability threshold."
    )


class PredictionResponse(BaseModel):
    """Full response returned after running inference."""

    scan_id: str
    analysis_type: str
    results: List[PredictionResult]
    model_version: str
    processing_time_seconds: float
    timestamp: datetime


# ── Heatmap ───────────────────────────────────────────────────────

class HeatmapResponse(BaseModel):
    """Grad-CAM heatmap data."""

    scan_id: str
    task: str
    heatmap_base64: str = Field(..., description="Raw heatmap as base64-encoded PNG.")
    overlay_base64: str = Field(..., description="Heatmap overlaid on the original scan.")


# ── Reports ───────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    """Request body for batch report generation."""

    scan_ids: List[str] = Field(..., min_length=1, description="Scan IDs to include.")
    format: Literal["pdf", "csv"] = Field(default="csv")


# ── Admin ─────────────────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    """Single row in the audit log."""

    id: int
    timestamp: datetime
    action: str
    scan_id: Optional[str] = None
    user: Optional[str] = None
    details: Optional[str] = None


class AuditLogResponse(BaseModel):
    """Paginated audit-log listing."""

    entries: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


class ModelInfo(BaseModel):
    """Metadata about a loaded model."""

    task: str
    architecture: str
    version: str
    training_date: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class SystemStats(BaseModel):
    """High-level usage statistics."""

    total_scans: int
    total_predictions: int
    predictions_by_type: Dict[str, int]
    scans_by_status: Dict[str, int]


# ── Health / Meta ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "ok"
    version: str
    models_loaded: bool
    mock_mode: bool
    gpu_available: bool


# ── Errors ────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None
