"""
Mock inference module for NeuroScan AI demo mode.

Generates synthetic but plausible prediction results and Grad-CAM heatmaps
so the Streamlit frontend can be demonstrated and tested without a running
backend server.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

TUMOUR_CLASSES = [
    "Glioma",
    "Meningioma",
    "Pituitary Tumour",
    "No Tumour",
]

ALZHEIMER_CLASSES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non-Demented",
    "Very Mild Dementia",
]


# ---------------------------------------------------------------------------
# Synthetic heatmap generator
# ---------------------------------------------------------------------------

def _generate_synthetic_heatmap(height: int, width: int) -> np.ndarray:
    """Create a Gaussian-blob heatmap mimicking Grad-CAM output.

    Places 1-3 Gaussian blobs in the central brain-like region of the image
    to simulate areas of model focus.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        2-D numpy array of shape ``(height, width)`` with values 0-255.
    """
    heatmap = np.zeros((height, width), dtype=np.float64)
    rng = np.random.default_rng()
    n_blobs = rng.integers(1, 4)

    for _ in range(n_blobs):
        # Centre blobs roughly in the middle 60 % of the image
        cy = rng.integers(int(height * 0.2), int(height * 0.8))
        cx = rng.integers(int(width * 0.2), int(width * 0.8))
        sigma_y = rng.uniform(height * 0.06, height * 0.18)
        sigma_x = rng.uniform(width * 0.06, width * 0.18)
        amplitude = rng.uniform(0.5, 1.0)

        yy, xx = np.ogrid[:height, :width]
        blob = amplitude * np.exp(
            -(((yy - cy) ** 2) / (2 * sigma_y ** 2)
              + ((xx - cx) ** 2) / (2 * sigma_x ** 2))
        )
        heatmap += blob

    # Normalise to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    return heatmap.astype(np.uint8)


# ---------------------------------------------------------------------------
# Synthetic probability generator
# ---------------------------------------------------------------------------

def _generate_probabilities(
    class_names: list[str],
    dominant_index: int | None = None,
) -> Dict[str, float]:
    """Generate plausible class probabilities that sum to ~1.

    One class is chosen to dominate, with the others receiving smaller
    random scores.

    Args:
        class_names: List of class label strings.
        dominant_index: Index of the dominant class.  Random if ``None``.

    Returns:
        Dict mapping class names to probability floats.
    """
    rng = np.random.default_rng()
    n = len(class_names)
    if dominant_index is None:
        dominant_index = rng.integers(0, n)

    raw = rng.dirichlet(np.ones(n) * 0.3)
    # Boost the dominant class
    raw[dominant_index] += rng.uniform(1.5, 4.0)
    probs = raw / raw.sum()
    return {name: float(probs[i]) for i, name in enumerate(class_names)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mock_predict(
    image: Image.Image,
    analysis_type: str,
) -> Dict[str, Any]:
    """Run a mock prediction on an MRI image.

    Simulates backend inference by generating synthetic predictions and
    heatmaps.  A small artificial delay is introduced to mimic network +
    GPU latency.

    Args:
        image: PIL Image of the uploaded MRI scan.
        analysis_type: One of ``"tumour"``, ``"alzheimer"``, or ``"both"``.

    Returns:
        Dictionary with keys:
            - ``predicted_class`` (str)
            - ``confidence`` (float)
            - ``probabilities`` (dict[str, float])
            - ``heatmap`` (np.ndarray)
            - ``model_info`` (dict)
            - ``processing_time`` (float, seconds)
            - ``analysis_type`` (str)
    """
    start = time.time()

    # Simulate processing latency
    time.sleep(np.random.uniform(0.6, 1.8))

    img_arr = np.array(image.convert("L"))
    h, w = img_arr.shape[:2]

    results: list[Dict[str, Any]] = []

    types_to_run: list[str] = []
    if analysis_type in ("tumour", "both"):
        types_to_run.append("tumour")
    if analysis_type in ("alzheimer", "both"):
        types_to_run.append("alzheimer")

    for atype in types_to_run:
        if atype == "tumour":
            classes = TUMOUR_CLASSES
            arch = "EfficientNet-B0"
            version = "1.2.0-demo"
        else:
            classes = ALZHEIMER_CLASSES
            arch = "ResNet-50"
            version = "1.1.0-demo"

        probs = _generate_probabilities(classes)
        top_class = max(probs, key=probs.get)  # type: ignore[arg-type]
        heatmap = _generate_synthetic_heatmap(h, w)

        results.append(
            {
                "predicted_class": top_class,
                "confidence": probs[top_class],
                "probabilities": probs,
                "heatmap": heatmap,
                "analysis_type": atype,
                "model_info": {
                    "architecture": arch,
                    "version": version,
                    "mode": "demo",
                },
            }
        )

    elapsed = time.time() - start

    # If only one analysis type, return flat dict; otherwise list
    if len(results) == 1:
        results[0]["processing_time"] = elapsed
        return results[0]

    return {
        "results": results,
        "processing_time": elapsed,
        "analysis_type": "both",
    }
