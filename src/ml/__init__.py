"""
Brain MRI AI – Machine Learning package.

Provides preprocessing, model architectures, training utilities,
explainability (Grad-CAM), and evaluation / inference pipelines for
brain-tumour and Alzheimer's classification from MRI scans.
"""

try:
    from src.ml.models.classifier import (
        BaseClassifier,
        BrainTumourClassifier,
        AlzheimerClassifier,
        ModelFactory,
    )
    from src.ml.preprocessing.transforms import (
        get_train_transforms,
        get_val_transforms,
    )
    from src.ml.explainability.gradcam import GradCAM, GradCAMPlusPlus
    from src.ml.training.trainer import Trainer, TrainingConfig
    from src.ml.evaluation.inference import InferencePipeline, PredictionResult
except ImportError:
    # PyTorch or related deep learning deps not yet installed in local environment
    BaseClassifier = None  # type: ignore
    BrainTumourClassifier = None  # type: ignore
    AlzheimerClassifier = None  # type: ignore
    ModelFactory = None  # type: ignore
    get_train_transforms = None  # type: ignore
    get_val_transforms = None  # type: ignore
    GradCAM = None  # type: ignore
    GradCAMPlusPlus = None  # type: ignore
    Trainer = None  # type: ignore
    TrainingConfig = None  # type: ignore
    InferencePipeline = None  # type: ignore
    PredictionResult = None  # type: ignore

from src.ml.training.dataset import (
    BrainMRIDataset,
    PatientAwareSplitter,
    generate_synthetic_dataset,
)
try:
    from src.ml.evaluation.metrics import (
        compute_metrics,
        generate_evaluation_report,
        ModelCard,
    )
except ImportError:
    compute_metrics = None  # type: ignore
    generate_evaluation_report = None  # type: ignore
    ModelCard = None  # type: ignore

__all__ = [
    "BaseClassifier",
    "BrainTumourClassifier",
    "AlzheimerClassifier",
    "ModelFactory",
    "get_train_transforms",
    "get_val_transforms",
    "GradCAM",
    "GradCAMPlusPlus",
    "Trainer",
    "TrainingConfig",
    "BrainMRIDataset",
    "PatientAwareSplitter",
    "generate_synthetic_dataset",
    "compute_metrics",
    "generate_evaluation_report",
    "ModelCard",
    "InferencePipeline",
    "PredictionResult",
]
