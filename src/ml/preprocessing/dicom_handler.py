"""
DICOM, NIfTI, and standard image loading utilities.

Handles ingestion of medical imaging formats, DICOM PHI anonymisation,
and basic heuristic validation that an image looks like a brain MRI.

Typical usage::

    arr = load_medical_image("scan.dcm")   # → np.ndarray (H, W), uint8
    anonymize_dicom("scan.dcm", "anon.dcm")
    is_ok, reason = validate_brain_mri(arr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# PHI DICOM tags to strip during anonymisation (tag keyword → tag tuple).
_PHI_TAGS: list[Tuple[int, int]] = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1070),  # OperatorsName
    (0x0020, 0x0010),  # StudyID
    (0x0008, 0x0050),  # AccessionNumber
]


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────


def load_dicom(path: str | Path) -> np.ndarray:
    """Load a DICOM file and return the pixel data as a uint8 NumPy array.

    Parameters
    ----------
    path:
        Path to a ``.dcm`` file.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(H, W)`` with dtype ``uint8``.

    Raises
    ------
    ImportError
        If *pydicom* is not installed.
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the DICOM file has no pixel data.
    """
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM files.  "
            "Install it with: pip install pydicom"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path))
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {path}")

    pixel_array: np.ndarray = ds.pixel_array.astype(np.float64)

    # Handle 3-D volumes – extract the middle axial slice.
    if pixel_array.ndim == 3:
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]
        logger.info("Extracted middle slice %d from 3-D DICOM volume.", mid)

    # Normalise to [0, 255] uint8.
    mn, mx = pixel_array.min(), pixel_array.max()
    if mx - mn > 0:
        pixel_array = ((pixel_array - mn) / (mx - mn) * 255.0)
    else:
        pixel_array = np.zeros_like(pixel_array)

    return pixel_array.astype(np.uint8)


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI file and return the middle axial slice as uint8.

    Parameters
    ----------
    path:
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    np.ndarray
        2-D array ``(H, W)`` with dtype ``uint8``.

    Raises
    ------
    ImportError
        If *nibabel* is not installed.
    """
    try:
        import nibabel as nib  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to load NIfTI files.  "
            "Install it with: pip install nibabel"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    data: np.ndarray = np.asarray(img.dataobj, dtype=np.float64)

    # Extract middle slice along the last spatial axis (axial).
    if data.ndim == 3:
        mid = data.shape[2] // 2
        data = data[:, :, mid]
        logger.info("Extracted axial slice %d from NIfTI volume.", mid)
    elif data.ndim == 4:
        # 4-D (e.g. time-series): take first volume, middle slice.
        mid = data.shape[2] // 2
        data = data[:, :, mid, 0]
        logger.info("Extracted axial slice %d from 4-D NIfTI.", mid)

    mn, mx = data.min(), data.max()
    if mx - mn > 0:
        data = ((data - mn) / (mx - mn) * 255.0)
    else:
        data = np.zeros_like(data)

    return data.astype(np.uint8)


def load_standard_image(path: str | Path) -> np.ndarray:
    """Load a standard image file (JPG, PNG, BMP, TIFF) as a uint8 array.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    np.ndarray
        2-D grayscale array ``(H, W)`` with dtype ``uint8``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    # Convert to grayscale for consistency with medical conventions.
    img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def load_medical_image(path: str | Path) -> np.ndarray:
    """Auto-detect the file type and load it as a 2-D uint8 array.

    Supports ``.dcm``, ``.nii``, ``.nii.gz``, and standard image
    formats (JPG, PNG, BMP, TIFF).

    Parameters
    ----------
    path:
        Path to the image.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 array.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Handle .nii.gz (double suffix)
    if path.name.lower().endswith(".nii.gz"):
        return load_nifti(path)

    dispatch = {
        ".dcm": load_dicom,
        ".nii": load_nifti,
        ".jpg": load_standard_image,
        ".jpeg": load_standard_image,
        ".png": load_standard_image,
        ".bmp": load_standard_image,
        ".tif": load_standard_image,
        ".tiff": load_standard_image,
    }

    loader = dispatch.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file format '{suffix}'.  "
            f"Supported: {', '.join(sorted(dispatch.keys()))}"
        )
    return loader(path)


# ──────────────────────────────────────────────────────────────────────
# DICOM anonymisation
# ──────────────────────────────────────────────────────────────────────


def anonymize_dicom(
    input_path: str | Path,
    output_path: str | Path,
    *,
    extra_tags: list[Tuple[int, int]] | None = None,
) -> Path:
    """Strip PHI tags from a DICOM file and write the anonymised copy.

    Parameters
    ----------
    input_path:
        Source DICOM file.
    output_path:
        Destination for the anonymised file.
    extra_tags:
        Additional ``(group, element)`` tag tuples to remove.

    Returns
    -------
    Path
        The *output_path* as a resolved :class:`~pathlib.Path`.
    """
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for anonymisation.  "
            "Install it with: pip install pydicom"
        ) from exc

    input_path = Path(input_path)
    output_path = Path(output_path)

    ds = pydicom.dcmread(str(input_path))

    tags_to_strip = list(_PHI_TAGS)
    if extra_tags:
        tags_to_strip.extend(extra_tags)

    for tag in tags_to_strip:
        if tag in ds:
            del ds[tag]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))
    logger.info("Anonymised DICOM saved to %s", output_path)
    return output_path.resolve()


# ──────────────────────────────────────────────────────────────────────
# Validation heuristic
# ──────────────────────────────────────────────────────────────────────


def validate_brain_mri(
    image: np.ndarray,
    *,
    min_size: int = 64,
    max_aspect_ratio: float = 2.0,
    min_mean_intensity: float = 10.0,
    max_black_fraction: float = 0.90,
) -> Tuple[bool, str]:
    """Run basic heuristic checks on a 2-D image array.

    This is **not** a definitive classifier – just a set of sanity checks
    to flag obviously wrong or corrupt inputs.

    Parameters
    ----------
    image:
        ``(H, W)`` uint8 array.
    min_size:
        Minimum acceptable height and width.
    max_aspect_ratio:
        Maximum allowed ``max(H,W)/min(H,W)``.
    min_mean_intensity:
        Images with a mean pixel value below this are considered too dark.
    max_black_fraction:
        Fraction of near-zero (< 5) pixels allowed.

    Returns
    -------
    (bool, str)
        ``(True, "ok")`` if the image passes, else ``(False, reason)``.
    """
    if image.ndim != 2:
        return False, f"Expected 2-D array, got {image.ndim}-D"

    h, w = image.shape
    if h < min_size or w < min_size:
        return False, f"Image too small: {h}×{w} (min {min_size})"

    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > max_aspect_ratio:
        return False, f"Aspect ratio {aspect:.2f} exceeds {max_aspect_ratio}"

    mean_val = float(image.mean())
    if mean_val < min_mean_intensity:
        return False, f"Mean intensity {mean_val:.1f} below {min_mean_intensity}"

    black_frac = float((image < 5).sum()) / image.size
    if black_frac > max_black_fraction:
        return False, (
            f"{black_frac:.0%} of pixels are near-black "
            f"(threshold {max_black_fraction:.0%})"
        )

    return True, "ok"
