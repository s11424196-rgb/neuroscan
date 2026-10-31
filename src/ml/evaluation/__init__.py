"""Evaluation metrics, diagnostic reporting, and inference pipeline."""

from src.ml.evaluation.inference import InferencePipeline, PredictionResult
from src.ml.evaluation.metrics import (
    ModelCard,
    bootstrap_confidence_intervals,
    compute_ece,
    compute_metrics,
    plot_confusion_matrix,
)

__all__ = [
    "compute_metrics",
    "compute_ece",
    "bootstrap_confidence_intervals",
    "plot_confusion_matrix",
    "ModelCard",
    "InferencePipeline",
    "PredictionResult",
]
