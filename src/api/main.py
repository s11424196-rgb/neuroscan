"""FastAPI application entry point for the Brain MRI AI Detection System.

Creates the ASGI app with CORS middleware, exception handlers, startup /
shutdown hooks, and includes all versioned routers.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.core.config import settings
from src.api.dependencies import (
    get_audit_service,
    get_inference_service,
    get_scan_service,
)
from src.api.models.schemas import ErrorResponse, HealthResponse
from src.api.routers import admin, inference, reports, upload

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared services on startup; clean up on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Eagerly initialise singletons so first request isn't slow
    get_scan_service()
    get_inference_service()
    get_audit_service()

    logger.info("Mock inference mode: %s", settings.mock_inference)
    logger.info(
        "GPU available: %s",
        torch.cuda.is_available() if _torch_available() else False,
    )

    yield  # ← application is running

    logger.info("Shutting down %s", settings.app_name)


# ── App Factory ───────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "REST API for brain-tumour and Alzheimer's disease detection "
        "from MRI scans.  Supports DICOM, NIfTI, JPEG, and PNG uploads "
        "with Grad-CAM explainability overlays."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────

app.include_router(upload.router)
app.include_router(inference.router)
app.include_router(reports.router)
app.include_router(admin.router)

# ── Request-level middleware (timing) ─────────────────────────────


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Inject an ``X-Process-Time`` response header."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


# ── Global Exception Handler ─────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that returns a structured error JSON.

    Avoids leaking stack traces to clients while still logging the
    full traceback server-side.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An internal error occurred. Please try again later.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ── Core Endpoints ────────────────────────────────────────────────


@app.get("/", tags=["meta"], summary="API information")
async def root():
    """Return basic API metadata."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe for orchestrators and monitoring."""
    inf_svc = get_inference_service()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_loaded=inf_svc.models_loaded,
        mock_mode=inf_svc.is_mock,
        gpu_available=_torch_available() and torch.cuda.is_available(),
    )


# ── Helpers ───────────────────────────────────────────────────────


def _torch_available() -> bool:
    """Check whether PyTorch is importable without crashing."""
    try:
        import torch  # noqa: F811
        return True
    except ImportError:
        return False
