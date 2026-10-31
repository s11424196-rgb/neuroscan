"""
Image preprocessing and augmentation transforms for brain MRI data.

Provides configurable training and validation transform pipelines built on
``torchvision.transforms.v2``.  Both PIL images and NumPy arrays are
accepted as input; the pipeline converts them to tensors internally.

Typical usage::

    train_tfm = get_train_transforms(image_size=224, norm="z-score")
    val_tfm   = get_val_transforms(image_size=224, norm="z-score")
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────

NormMode = Literal["min-max", "z-score"]

# ImageNet statistics – used for z-score normalisation when the backbone
# was pretrained on ImageNet.
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


class _EnsurePIL(v2.Transform):
    """Convert a NumPy array to a PIL Image if necessary."""

    def forward(self, inpt: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
        if isinstance(inpt, np.ndarray):
            # Handle single-channel arrays → L-mode PIL
            if inpt.ndim == 2:
                return Image.fromarray(inpt.astype(np.uint8), mode="L")
            if inpt.ndim == 3 and inpt.shape[2] == 1:
                return Image.fromarray(
                    inpt.squeeze(-1).astype(np.uint8), mode="L"
                )
            return Image.fromarray(inpt.astype(np.uint8))
        if isinstance(inpt, torch.Tensor):
            arr = inpt.numpy()
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                return Image.fromarray(arr.squeeze().astype(np.uint8), mode="L")
            return Image.fromarray(arr.astype(np.uint8))
        return inpt  # already PIL


class _EnsureRGB(v2.Transform):
    """Convert a single-channel PIL Image to 3-channel RGB."""

    def forward(self, inpt: Image.Image) -> Image.Image:
        if isinstance(inpt, Image.Image) and inpt.mode != "RGB":
            return inpt.convert("RGB")
        return inpt


class _MinMaxNormalize(v2.Transform):
    """Scale tensor values to [0, 1] per-image (channel-wise)."""

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        if not isinstance(inpt, torch.Tensor):
            return inpt
        # inpt shape: (C, H, W)
        for c in range(inpt.shape[0]):
            ch = inpt[c]
            mn, mx = ch.min(), ch.max()
            if mx - mn > 0:
                inpt[c] = (ch - mn) / (mx - mn)
            else:
                inpt[c] = torch.zeros_like(ch)
        return inpt


class ElasticDeformation(v2.Transform):
    """Lightweight elastic deformation via ``v2.ElasticTransform``.

    Wraps ``torchvision.transforms.v2.ElasticTransform`` with
    sensible defaults for brain MRI augmentation.
    """

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.3) -> None:
        super().__init__()
        self._inner = v2.ElasticTransform(alpha=alpha, sigma=sigma)
        self.p = p

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return self._inner(inpt)
        return inpt


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def _build_normalize(norm: NormMode) -> v2.Transform:
    """Return the appropriate normalisation transform."""
    if norm == "z-score":
        return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    # min-max normalisation is applied *before* ToTensor
    return _MinMaxNormalize()


def get_train_transforms(
    image_size: int = 224,
    norm: NormMode = "z-score",
    rotation_degrees: float = 15.0,
    horizontal_flip_p: float = 0.5,
    color_jitter_brightness: float = 0.15,
    color_jitter_contrast: float = 0.15,
    elastic_alpha: float = 50.0,
    elastic_sigma: float = 5.0,
    elastic_p: float = 0.3,
) -> v2.Compose:
    """Build a training-time transform pipeline.

    Parameters
    ----------
    image_size:
        Target spatial size (square crop).
    norm:
        ``"min-max"`` for per-image [0,1] scaling, or ``"z-score"`` for
        ImageNet-statistic normalisation.
    rotation_degrees:
        Maximum rotation in degrees (applied ±).
    horizontal_flip_p:
        Probability of random horizontal flip.
    color_jitter_brightness / color_jitter_contrast:
        Intensity jitter ranges.
    elastic_alpha / elastic_sigma / elastic_p:
        Elastic deformation parameters.

    Returns
    -------
    v2.Compose
        A composable transform that accepts PIL images, NumPy arrays,
        or tensors and returns a ``(3, image_size, image_size)`` float tensor.
    """
    tfms: list[v2.Transform] = [
        _EnsurePIL(),
        _EnsureRGB(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomRotation(degrees=rotation_degrees),
        v2.RandomHorizontalFlip(p=horizontal_flip_p),
        v2.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ElasticDeformation(alpha=elastic_alpha, sigma=elastic_sigma, p=elastic_p),
        _build_normalize(norm),
    ]
    return v2.Compose(tfms)


def get_val_transforms(
    image_size: int = 224,
    norm: NormMode = "z-score",
) -> v2.Compose:
    """Build a validation / test transform pipeline (no augmentation).

    Parameters
    ----------
    image_size:
        Target spatial size (square crop).
    norm:
        ``"min-max"`` or ``"z-score"``.

    Returns
    -------
    v2.Compose
        A composable transform that accepts PIL images, NumPy arrays,
        or tensors and returns a ``(3, image_size, image_size)`` float tensor.
    """
    tfms: list[v2.Transform] = [
        _EnsurePIL(),
        _EnsureRGB(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        _build_normalize(norm),
    ]
    return v2.Compose(tfms)
