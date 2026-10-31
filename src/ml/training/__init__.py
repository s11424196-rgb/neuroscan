"""
Training loop, dataset abstraction, and patient-aware data splitting.

Provides a :class:`Trainer` driven by a :class:`TrainingConfig` dataclass,
a :class:`BrainMRIDataset`, a :class:`PatientAwareSplitter`, and a
synthetic-data generator for end-to-end pipeline testing.
"""

from src.ml.training.trainer import Trainer, TrainingConfig
from src.ml.training.dataset import (
    BrainMRIDataset,
    PatientAwareSplitter,
    generate_synthetic_dataset,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "BrainMRIDataset",
    "PatientAwareSplitter",
    "generate_synthetic_dataset",
]
