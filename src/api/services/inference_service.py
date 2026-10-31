"""Inference service with mock-mode support.

Provides a singleton-style service that either delegates to real ML
models (``src.ml`` pipeline) or generates realistic synthetic
predictions with Gaussian-blob heatmaps when ``MOCK_INFERENCE=true``.
"""

import base64
import io
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from src.api.core.config import settings
from src.api.models.schemas import (
    HeatmapResponse,
    PredictionResponse,
    PredictionResult,
)

logger = logging.getLogger(__name__)

# ── Class labels used by the models ──────────────────────────────

TUMOUR_CLASSES: List[str] = ["glioma", "meningioma", "pituitary", "no_tumour"]
ALZHEIMER_CLASSES: List[str] = [
    "non_demented",
    "very_mild_demented",
    "mild_demented",
    "moderate_demented",
]

CONFIDENCE_THRESHOLD = 0.60  # Below this → low-confidence flag

# ── Model metadata (shared between mock and real) ────────────────

MODEL_INFO: Dict[str, dict] = {
    "tumour": {
        "task": "tumour",
        "architecture": "EfficientNet-B0 (fine-tuned)",
        "version": "0.1.0-mock",
        "training_date": None,
        "metrics": {"accuracy": 0.0, "f1_macro": 0.0},
    },
    "alzheimer": {
        "task": "alzheimer",
        "architecture": "ResNet-50 (fine-tuned)",
        "version": "0.1.0-mock",
        "training_date": None,
        "metrics": {"accuracy": 0.0, "f1_macro": 0.0},
    },
}


class InferenceService:
    """Run (or mock) inference on uploaded MRI scans."""

    def __init__(self) -> None:
        self._mock = settings.mock_inference
        self._models_loaded = False
        # Cache: scan_id → {task → PredictionResult}
        self._result_cache: Dict[str, Dict[str, PredictionResult]] = {}
        # Cache: scan_id → {task → (heatmap_b64, overlay_b64)}
        self._heatmap_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}

        if self._mock:
            logger.info("InferenceService running in MOCK mode.")
            self._models_loaded = True
        else:
            logger.info("InferenceService will load real models on first call.")

    # ── Properties ────────────────────────────────────────────────

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    @property
    def is_mock(self) -> bool:
        return self._mock

    # ── Public API ────────────────────────────────────────────────

    async def predict(
        self,
        scan_id: str,
        scan_path: str,
        analysis_type: str,
    ) -> PredictionResponse:
        """Run inference and return structured results.

        Parameters
        ----------
        scan_id:
            Unique scan identifier.
        scan_path:
            Filesystem path to the scan file.
        analysis_type:
            One of ``tumour``, ``alzheimer``, ``both``.

        Returns
        -------
        PredictionResponse
        """
        t0 = time.monotonic()

        tasks: List[str] = (
            ["tumour", "alzheimer"] if analysis_type == "both" else [analysis_type]
        )

        results: List[PredictionResult] = []
        for task in tasks:
            result = await self._run_single(scan_id, scan_path, task)
            results.append(result)

        elapsed = round(time.monotonic() - t0, 4)

        response = PredictionResponse(
            scan_id=scan_id,
            analysis_type=analysis_type,
            results=results,
            model_version=MODEL_INFO[tasks[0]]["version"],
            processing_time_seconds=elapsed,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "Prediction complete: scan=%s type=%s time=%.3fs",
            scan_id,
            analysis_type,
            elapsed,
        )
        return response

    def get_cached_results(self, scan_id: str) -> Optional[Dict[str, PredictionResult]]:
        """Return cached prediction results for a scan, if available."""
        return self._result_cache.get(scan_id)

    def get_heatmap(self, scan_id: str, task: str = "tumour") -> Optional[HeatmapResponse]:
        """Return a cached heatmap for the given scan and task."""
        heatmaps = self._heatmap_cache.get(scan_id)
        if heatmaps is None or task not in heatmaps:
            return None
        hmap_b64, overlay_b64 = heatmaps[task]
        return HeatmapResponse(
            scan_id=scan_id,
            task=task,
            heatmap_base64=hmap_b64,
            overlay_base64=overlay_b64,
        )

    def get_model_info(self) -> List[dict]:
        """Return metadata for all registered models."""
        return list(MODEL_INFO.values())

    # ── Internal ──────────────────────────────────────────────────

    async def _run_single(
        self,
        scan_id: str,
        scan_path: str,
        task: str,
    ) -> PredictionResult:
        """Run (or mock) a single-task prediction."""
        if self._mock:
            return self._mock_predict(scan_id, task)
        else:
            return await self._real_predict(scan_id, scan_path, task)

    # ── Mock inference ────────────────────────────────────────────

    def _mock_predict(self, scan_id: str, task: str) -> PredictionResult:
        """Generate a realistic but synthetic prediction."""
        classes = TUMOUR_CLASSES if task == "tumour" else ALZHEIMER_CLASSES

        # Generate plausible probability distribution
        raw = [random.random() for _ in classes]
        total = sum(raw)
        probs = {c: round(v / total, 4) for c, v in zip(classes, raw)}

        predicted_class = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence = probs[predicted_class]
        uncertainty = round(1.0 - confidence + random.uniform(0, 0.1), 4)
        is_low = confidence < CONFIDENCE_THRESHOLD

        result = PredictionResult(
            task=task,
            predicted_class=predicted_class,
            probabilities=probs,
            confidence=confidence,
            uncertainty_score=uncertainty,
            is_low_confidence=is_low,
        )

        # Cache result
        self._result_cache.setdefault(scan_id, {})[task] = result

        # Generate & cache mock heatmap
        hmap_b64, overlay_b64 = self._generate_mock_heatmap()
        self._heatmap_cache.setdefault(scan_id, {})[task] = (hmap_b64, overlay_b64)

        return result

    @staticmethod
    def _generate_mock_heatmap(size: int = 224) -> Tuple[str, str]:
        """Create a synthetic Gaussian-blob heatmap and an overlay.

        Returns (heatmap_base64, overlay_base64).
        """
        # Create gaussian blob
        y, x = np.ogrid[-1:1:complex(size), -1:1:complex(size)]
        cx, cy = random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)
        sigma = random.uniform(0.2, 0.5)
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        blob = (blob / blob.max() * 255).astype(np.uint8)

        # Heatmap (red channel)
        heatmap_arr = np.zeros((size, size, 3), dtype=np.uint8)
        heatmap_arr[:, :, 0] = blob  # Red
        heatmap_arr[:, :, 1] = (blob * 0.3).astype(np.uint8)  # slight green
        heatmap_img = Image.fromarray(heatmap_arr).filter(
            ImageFilter.GaussianBlur(radius=5)
        )

        # Fake grayscale "brain scan" background
        bg = np.random.randint(20, 80, (size, size), dtype=np.uint8)
        bg_img = Image.fromarray(bg, mode="L").convert("RGB")

        # Overlay: blend heatmap onto background
        overlay_img = Image.blend(bg_img, heatmap_img, alpha=0.45)

        def _to_b64(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return _to_b64(heatmap_img), _to_b64(overlay_img)

    # ── Real inference (placeholder) ──────────────────────────────

    async def _real_predict(
        self,
        scan_id: str,
        scan_path: str,
        task: str,
    ) -> PredictionResult:
        """Invoke the real ML pipeline.

        .. note::
            This is a placeholder.  Once models are trained, hook into
            ``src.ml.pipeline`` here.
        """
        raise NotImplementedError(
            "Real inference is not yet implemented. "
            "Set MOCK_INFERENCE=true to use synthetic predictions."
        )
