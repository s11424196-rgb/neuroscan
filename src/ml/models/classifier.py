"""
Classification model architectures for brain tumour and Alzheimer's staging.

All classifiers inherit from :class:`BaseClassifier`, which exposes a
``get_gradcam_target_layer()`` method so that Grad-CAM can automatically
find the last convolutional layer.

:class:`ModelFactory` provides a single entry-point to create any
supported ``(task, backbone)`` combination.

Supported backbones: ``resnet50``, ``efficientnet_b0``, ``densenet121``.

Example::

    model = ModelFactory.create(
        task="tumour",
        backbone="resnet50",
        pretrained=True,
    )
"""

from __future__ import annotations

import abc
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

TUMOUR_CLASSES: list[str] = ["glioma", "meningioma", "pituitary", "no_tumour"]
ALZHEIMER_CLASSES: list[str] = ["non_demented", "very_mild", "mild", "moderate"]

TaskName = Literal["tumour", "alzheimer"]
BackboneName = Literal["resnet50", "efficientnet_b0", "densenet121"]

# ──────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────


class BaseClassifier(nn.Module, abc.ABC):
    """Abstract base for all brain-MRI classifiers.

    Subclasses must implement :meth:`get_gradcam_target_layer` so that
    the explainability module can hook into the correct convolutional
    layer.
    """

    backbone_name: str
    num_classes: int
    class_names: list[str]

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the convolutional layer to use for Grad-CAM."""

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (feature-extraction mode)."""
        for name, param in self.named_parameters():
            if "classifier" not in name and "fc" not in name and "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (fine-tuning mode)."""
        for param in self.parameters():
            param.requires_grad = True

    def enable_mc_dropout(self) -> None:
        """Set all Dropout layers to training mode for MC-dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ──────────────────────────────────────────────────────────────────────
# Backbone builder helpers
# ──────────────────────────────────────────────────────────────────────


def _build_resnet50(
    num_classes: int,
    pretrained: bool,
    freeze: bool,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Return ``(full_model, classifier_head, gradcam_layer)``."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)
    gradcam_layer = backbone.layer4  # last conv block
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        for name, p in backbone.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    return backbone, backbone.fc, gradcam_layer


def _build_efficientnet_b0(
    num_classes: int,
    pretrained: bool,
    freeze: bool,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)
    gradcam_layer = backbone.features[-1]  # last conv block
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        for name, p in backbone.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
    return backbone, backbone.classifier, gradcam_layer


def _build_densenet121(
    num_classes: int,
    pretrained: bool,
    freeze: bool,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    backbone = models.densenet121(weights=weights)
    gradcam_layer = backbone.features.denseblock4  # last dense block
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        for name, p in backbone.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
    return backbone, backbone.classifier, gradcam_layer


_BUILDERS = {
    "resnet50": _build_resnet50,
    "efficientnet_b0": _build_efficientnet_b0,
    "densenet121": _build_densenet121,
}


# ──────────────────────────────────────────────────────────────────────
# Concrete classifiers
# ──────────────────────────────────────────────────────────────────────


class _BackboneClassifier(BaseClassifier):
    """Internal mixin that wraps any supported backbone."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        class_names: list[str],
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if backbone_name not in _BUILDERS:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'.  "
                f"Choose from: {list(_BUILDERS.keys())}"
            )
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.class_names = class_names

        builder = _BUILDERS[backbone_name]
        self.backbone, self._head, self._gradcam_layer = builder(
            num_classes, pretrained, freeze
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, num_classes)``.
        """
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self._gradcam_layer


class BrainTumourClassifier(_BackboneClassifier):
    """Classifier for brain tumour detection.

    Classes: ``glioma``, ``meningioma``, ``pituitary``, ``no_tumour``.

    Parameters
    ----------
    backbone:
        ``"resnet50"`` | ``"efficientnet_b0"`` | ``"densenet121"``.
    pretrained:
        Whether to start from ImageNet-pretrained weights.
    freeze:
        If *True*, freeze backbone for feature-extraction mode.
    num_classes:
        Override the default 4-class output.
    """

    def __init__(
        self,
        backbone: BackboneName = "resnet50",
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 4,
    ) -> None:
        class_names = TUMOUR_CLASSES[:num_classes] if num_classes <= 4 else TUMOUR_CLASSES
        super().__init__(
            backbone_name=backbone,
            num_classes=num_classes,
            class_names=class_names,
            pretrained=pretrained,
            freeze=freeze,
        )


class AlzheimerClassifier(_BackboneClassifier):
    """Classifier for Alzheimer's disease staging.

    Classes: ``non_demented``, ``very_mild``, ``mild``, ``moderate``.

    Parameters
    ----------
    backbone:
        ``"resnet50"`` | ``"efficientnet_b0"`` | ``"densenet121"``.
    pretrained:
        Whether to start from ImageNet-pretrained weights.
    freeze:
        If *True*, freeze backbone for feature-extraction mode.
    num_classes:
        Override the default 4-class output.
    """

    def __init__(
        self,
        backbone: BackboneName = "resnet50",
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 4,
    ) -> None:
        class_names = ALZHEIMER_CLASSES[:num_classes] if num_classes <= 4 else ALZHEIMER_CLASSES
        super().__init__(
            backbone_name=backbone,
            num_classes=num_classes,
            class_names=class_names,
            pretrained=pretrained,
            freeze=freeze,
        )


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────


class ModelFactory:
    """Create classifier instances by task and backbone name.

    Example::

        model = ModelFactory.create(
            task="tumour",
            backbone="efficientnet_b0",
            pretrained=True,
        )
    """

    _TASK_MAP: dict[str, type[BaseClassifier]] = {
        "tumour": BrainTumourClassifier,
        "alzheimer": AlzheimerClassifier,
    }

    @classmethod
    def create(
        cls,
        task: TaskName,
        backbone: BackboneName = "resnet50",
        pretrained: bool = True,
        num_classes: int | None = None,
        freeze: bool = False,
    ) -> BaseClassifier:
        """Instantiate a classifier.

        Parameters
        ----------
        task:
            ``"tumour"`` or ``"alzheimer"``.
        backbone:
            Backbone architecture name.
        pretrained:
            Use ImageNet-pretrained weights.
        num_classes:
            Override default class count (4 for both tasks).
        freeze:
            Freeze backbone parameters.

        Returns
        -------
        BaseClassifier
        """
        if task not in cls._TASK_MAP:
            raise ValueError(
                f"Unknown task '{task}'.  "
                f"Choose from: {list(cls._TASK_MAP.keys())}"
            )
        klass = cls._TASK_MAP[task]
        kwargs: dict = dict(backbone=backbone, pretrained=pretrained, freeze=freeze)
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        return klass(**kwargs)
