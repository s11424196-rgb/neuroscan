"""Common constants and enumerations for NeuroScan AI.

Defines the class labels, supported file types, and other constants
used across the application.
"""

from enum import Enum


class AnalysisType(str, Enum):
    """Type of analysis to perform on a brain MRI scan."""

    TUMOUR = "tumour"
    ALZHEIMER = "alzheimer"
    BOTH = "both"


class TumourClass(str, Enum):
    """Brain tumour classification categories."""

    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    PITUITARY = "pituitary"
    NO_TUMOUR = "no_tumour"


class AlzheimerClass(str, Enum):
    """Alzheimer's disease staging categories."""

    NON_DEMENTED = "non_demented"
    VERY_MILD = "very_mild"
    MILD = "mild"
    MODERATE = "moderate"


class ScanStatus(str, Enum):
    """Status of a scan in the processing pipeline."""

    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    PREDICTING = "predicting"
    COMPLETED = "completed"
    FAILED = "failed"


class BackboneType(str, Enum):
    """Supported model backbone architectures."""

    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    DENSENET121 = "densenet121"


# Display-friendly class labels
TUMOUR_CLASS_LABELS: dict[TumourClass, str] = {
    TumourClass.GLIOMA: "Glioma",
    TumourClass.MENINGIOMA: "Meningioma",
    TumourClass.PITUITARY: "Pituitary Tumour",
    TumourClass.NO_TUMOUR: "No Tumour Detected",
}

ALZHEIMER_CLASS_LABELS: dict[AlzheimerClass, str] = {
    AlzheimerClass.NON_DEMENTED: "Non-Demented",
    AlzheimerClass.VERY_MILD: "Very Mild Dementia",
    AlzheimerClass.MILD: "Mild Dementia",
    AlzheimerClass.MODERATE: "Moderate Dementia",
}

# Supported file extensions
SUPPORTED_EXTENSIONS: set[str] = {
    ".dcm",     # DICOM
    ".nii",     # NIfTI
    ".nii.gz",  # Compressed NIfTI
    ".jpg",
    ".jpeg",
    ".png",
}

# Image dimensions
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_NUM_CHANNELS: int = 3

# Confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
LOW_CONFIDENCE_MESSAGE: str = (
    "⚠️ Low confidence prediction — Human review is strongly recommended. "
    "This result should not be used for clinical decision-making."
)

# Medical disclaimer
DISCLAIMER_TEXT: str = (
    "This tool is for research and educational use only and is not a substitute "
    "for professional medical diagnosis. All results should be reviewed by a "
    "qualified healthcare professional. Do not make medical decisions based "
    "solely on the output of this tool."
)

DISCLAIMER_SHORT: str = (
    "⚠️ Research & Educational Use Only — Not a medical diagnosis tool."
)

# Model defaults
DEFAULT_MC_DROPOUT_ITERATIONS: int = 10
DEFAULT_TEMPERATURE: float = 1.0
MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB

