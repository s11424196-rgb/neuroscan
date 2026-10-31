"""
Model architectures for brain MRI classification.

Exposes pretrained-backbone classifiers for tumour detection and
Alzheimer's staging, plus a ``ModelFactory`` for convenient construction.
"""

from src.ml.models.classifier import (
    BaseClassifier,
    BrainTumourClassifier,
    AlzheimerClassifier,
    ModelFactory,
)

__all__ = [
    "BaseClassifier",
    "BrainTumourClassifier",
    "AlzheimerClassifier",
    "ModelFactory",
]
