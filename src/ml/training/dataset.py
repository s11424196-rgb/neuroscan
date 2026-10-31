"""
Dataset classes, patient-aware splitting, and synthetic data generation.

:class:`BrainMRIDataset` loads images from a class-per-directory layout.
:class:`PatientAwareSplitter` guarantees that all images from the same
patient end up in the same split (stratified by class).
:func:`generate_synthetic_dataset` creates fake brain-MRI-like images
for end-to-end pipeline testing without real data.

Directory layout assumed::

    root/
        class_a/
            patient01_001.jpg
            patient01_002.jpg
            patient02_001.jpg
        class_b/
            ...
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:  # type: ignore
        """Fallback Dataset base class when PyTorch is not yet installed."""
        pass

from PIL import Image

logger = logging.getLogger(__name__)

# Supported image extensions.
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────


class BrainMRIDataset(Dataset):
    """PyTorch dataset that loads images from a ``class_name/image.*``
    directory structure.

    Parameters
    ----------
    root_dir:
        Top-level data directory.
    transform:
        Callable applied to each PIL image before returning.
    class_names:
        Explicit list of class sub-directories.  If *None*, all
        sub-directories are discovered automatically (sorted).
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        if class_names is None:
            class_names = sorted(
                d.name
                for d in self.root_dir.iterdir()
                if d.is_dir()
            )
        self.class_names: list[str] = list(class_names)
        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }

        # Collect all samples.
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                logger.warning("Class directory not found: %s", cls_dir)
                continue
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.suffix.lower() in _IMG_EXTENSIONS:
                    self.samples.append((fpath, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            logger.warning(
                "No images found in %s with classes %s",
                self.root_dir,
                self.class_names,
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_path(self, idx: int) -> Path:
        """Return the file path for sample *idx*."""
        return self.samples[idx][0]

    def get_patient_id(self, idx: int) -> str:
        """Extract a patient ID from the filename.

        Heuristic: split the stem on ``_`` and take the first token.
        Fall back to hashing the filename if the split yields nothing
        useful.
        """
        return extract_patient_id(self.samples[idx][0].stem)

    @property
    def targets(self) -> list[int]:
        """All labels as a flat list (for stratification helpers)."""
        return [label for _, label in self.samples]


# ──────────────────────────────────────────────────────────────────────
# Patient ID extraction
# ──────────────────────────────────────────────────────────────────────


def extract_patient_id(stem: str) -> str:
    """Extract a patient identifier from a filename stem.

    Tries, in order:

    1. Match ``patient<NUM>`` or ``pat<NUM>``.
    2. Take everything before the first ``_`` (if the result is not
       purely numeric with < 3 digits – which looks like a slice index).
    3. Fall back to a hash of the full stem.
    """
    # 1. Explicit patient pattern.
    m = re.match(r"(pat(?:ient)?\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 2. First token before underscore.
    parts = stem.split("_")
    candidate = parts[0]
    if len(candidate) >= 3:
        return candidate.lower()

    # 3. Hash fallback – treat every image as its own patient.
    return hashlib.md5(stem.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────────────────────────────
# Patient-aware splitter
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SplitResult:
    """Result of a patient-aware train/val/test split."""

    train_indices: list[int] = field(default_factory=list)
    val_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train_indices),
            "val": len(self.val_indices),
            "test": len(self.test_indices),
        }


class PatientAwareSplitter:
    """Split a dataset so that all images from one patient belong to
    exactly one of train / val / test.

    Stratification by class is approximated: patients are grouped by
    their majority class, and each group is split independently.

    Parameters
    ----------
    train_frac:
        Fraction of patients for training.
    val_frac:
        Fraction for validation.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        if not (0 < train_frac < 1 and 0 < val_frac < 1):
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1")
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac
        self.seed = seed

    def split(self, dataset: BrainMRIDataset) -> SplitResult:
        """Perform a patient-aware, class-stratified split.

        Parameters
        ----------
        dataset:
            A :class:`BrainMRIDataset` instance.

        Returns
        -------
        SplitResult
            Index lists for train / val / test.
        """
        rng = np.random.RandomState(self.seed)

        # Map patient → list of (sample_index, label).
        patient_map: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx in range(len(dataset)):
            pid = dataset.get_patient_id(idx)
            label = dataset.samples[idx][1]
            patient_map[pid].append((idx, label))

        # Determine majority class per patient.
        patient_class: dict[str, int] = {}
        for pid, items in patient_map.items():
            labels = [lb for _, lb in items]
            patient_class[pid] = int(np.argmax(np.bincount(labels)))

        # Group patients by majority class.
        class_patients: dict[int, list[str]] = defaultdict(list)
        for pid, cls in patient_class.items():
            class_patients[cls].append(pid)

        result = SplitResult()

        for cls in sorted(class_patients.keys()):
            pids = class_patients[cls]
            rng.shuffle(pids)
            n = len(pids)
            n_train = max(1, int(round(n * self.train_frac)))
            n_val = max(1, int(round(n * self.val_frac)))
            # Ensure at least 1 for test when possible.
            if n_train + n_val >= n and n >= 3:
                n_train = n - 2
                n_val = 1

            train_pids = pids[:n_train]
            val_pids = pids[n_train : n_train + n_val]
            test_pids = pids[n_train + n_val :]

            for pid in train_pids:
                result.train_indices.extend(i for i, _ in patient_map[pid])
            for pid in val_pids:
                result.val_indices.extend(i for i, _ in patient_map[pid])
            for pid in test_pids:
                result.test_indices.extend(i for i, _ in patient_map[pid])

        logger.info(
            "Patient-aware split: %s",
            result.summary,
        )
        return result


# ──────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────

TUMOUR_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumour"]
ALZHEIMER_CLASSES = ["non_demented", "very_mild", "mild", "moderate"]


def _make_brain_base(
    rng: np.random.RandomState,
    size: int = 224,
) -> np.ndarray:
    """Generate a synthetic brain-shaped grayscale image.

    Creates an elliptical mask with Perlin-noise–like intensity
    variation (approximated with blurred Gaussian noise).
    """
    img = np.zeros((size, size), dtype=np.float64)

    # Elliptical skull mask.
    cy, cx = size // 2, size // 2
    ry, rx = int(size * 0.40), int(size * 0.35)
    yy, xx = np.ogrid[:size, :size]
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0

    # Background intensity inside skull.
    base_intensity = rng.uniform(80, 160)
    noise = rng.randn(size, size) * 15
    img[mask] = base_intensity + noise[mask]

    # Smooth it.
    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

    img = gaussian_filter(img, sigma=3)
    img = np.clip(img, 0, 255)
    return img


def _add_tumour_blob(
    img: np.ndarray,
    rng: np.random.RandomState,
    tumour_type: str,
) -> np.ndarray:
    """Add a synthetic tumour blob to the image.

    Different tumour types get different appearance characteristics.
    """
    size = img.shape[0]
    from scipy.ndimage import gaussian_filter

    # Pick tumour position (within the brain area, offset from centre).
    angle = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(size * 0.05, size * 0.20)
    ty = int(size // 2 + radius * np.sin(angle))
    tx = int(size // 2 + radius * np.cos(angle))

    params = {
        "glioma": {"r": (15, 35), "intensity": (180, 230), "irregular": True},
        "meningioma": {"r": (12, 28), "intensity": (190, 240), "irregular": False},
        "pituitary": {"r": (8, 18), "intensity": (170, 220), "irregular": False},
    }
    cfg = params.get(tumour_type, params["glioma"])

    r = rng.randint(*cfg["r"])
    intensity = rng.uniform(*cfg["intensity"])

    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - ty) ** 2 + (xx - tx) ** 2)
    blob = np.zeros_like(img)
    blob[dist < r] = intensity

    if cfg["irregular"]:
        noise = rng.randn(size, size) * 20
        blob += noise * (dist < r * 1.2)

    blob = gaussian_filter(blob, sigma=r * 0.3)
    img = np.maximum(img, blob)
    return np.clip(img, 0, 255)


def _make_alzheimer_image(
    rng: np.random.RandomState,
    stage: str,
    size: int = 224,
) -> np.ndarray:
    """Generate a synthetic brain image for an Alzheimer's stage.

    Simulates cortical atrophy by darkening the outer ring and adding
    ventricle enlargement (brighter central region) proportional to
    severity.
    """
    img = _make_brain_base(rng, size)
    from scipy.ndimage import gaussian_filter

    severity = {
        "non_demented": 0.0,
        "very_mild": 0.3,
        "mild": 0.6,
        "moderate": 1.0,
    }.get(stage, 0.0)

    if severity > 0:
        cy, cx = size // 2, size // 2
        yy, xx = np.ogrid[:size, :size]

        # Cortical thinning – darken outer ring.
        dist_from_centre = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        outer = dist_from_centre > size * (0.30 - 0.05 * severity)
        img[outer] *= (1.0 - 0.3 * severity)

        # Ventricular enlargement – bright central ellipse.
        vent_r = int(size * 0.05 * (1 + severity * 2))
        vent_mask = dist_from_centre < vent_r
        img[vent_mask] = np.clip(
            img[vent_mask] + 40 * severity, 0, 255
        )

        img = gaussian_filter(img, sigma=2)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(
    output_dir: str | Path,
    task: str = "tumour",
    num_per_class: int = 50,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """Generate a synthetic dataset for pipeline testing.

    Parameters
    ----------
    output_dir:
        Root output directory.  Sub-directories for each class are
        created underneath.
    task:
        ``"tumour"`` or ``"alzheimer"``.
    num_per_class:
        Number of images per class.
    image_size:
        Spatial size of generated images.
    seed:
        Random seed.

    Returns
    -------
    Path
        The *output_dir* as a resolved :class:`~pathlib.Path`.
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)

    if task == "tumour":
        classes = TUMOUR_CLASSES
    elif task == "alzheimer":
        classes = ALZHEIMER_CLASSES
    else:
        raise ValueError(f"Unknown task '{task}'.  Choose 'tumour' or 'alzheimer'.")

    metadata_rows: list[dict[str, str]] = []

    for cls_name in classes:
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_per_class):
            patient_id = f"patient{(i // 3):04d}"  # ~3 images per patient
            fname = f"{patient_id}_{i:04d}.png"

            if task == "tumour":
                img = _make_brain_base(rng, image_size)
                if cls_name != "no_tumour":
                    img = _add_tumour_blob(img, rng, cls_name)
                img = img.astype(np.uint8)
            else:
                img = _make_alzheimer_image(rng, cls_name, image_size)

            Image.fromarray(img, mode="L").save(cls_dir / fname)
            metadata_rows.append(
                {
                    "filename": f"{cls_name}/{fname}",
                    "class": cls_name,
                    "patient_id": patient_id,
                }
            )

    # Write metadata CSV.
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "patient_id"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(
        "Generated %d synthetic %s images in %s",
        len(metadata_rows),
        task,
        output_dir,
    )
    return output_dir.resolve()
