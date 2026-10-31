"""
Explainability utilities – Grad-CAM and Grad-CAM++ implementations.

Generate class-discriminative heatmaps and overlay visualisations for
model interpretability in clinical review workflows.
"""

from src.ml.explainability.gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    overlay_heatmap,
)

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "overlay_heatmap",
]
