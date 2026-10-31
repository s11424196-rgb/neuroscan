"""End-to-end inference pipeline for brain MRI classification and explainability.

Integrates preprocessing, forward inference, Monte Carlo Dropout uncertainty
estimation, and Grad-CAM heatmap generation into a single cohesive interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.ml.explainability.gradcam import GradCAM, overlay_heatmap
from src.ml.models.classifier import BaseClassifier
from src.ml.preprocessing.dicom_handler import load_medical_image
from src.ml.preprocessing.transforms import get_inference_transforms


@dataclass
class PredictionResult:
    """Output container for a single scan prediction."""

    task: str
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    top_2: List[Tuple[str, float]]
    uncertainty_score: float
    is_low_confidence: bool
    heatmap: Optional[np.ndarray] = None  # (H, W) in [0, 1]
    overlay: Optional[np.ndarray] = None  # (H, W, 3) uint8
    original_image: Optional[np.ndarray] = None
    processing_time_seconds: float = 0.0
    model_version: str = "0.1.0"
    disclaimer: str = (
        "Research & Educational Use Only. Not a substitute for professional medical diagnosis."
    )


class InferencePipeline:
    """End-to-end inference pipeline for Brain Tumour and Alzheimer's models."""

    def __init__(
        self,
        model: BaseClassifier,
        task: str,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.65,
        mc_dropout_samples: int = 10,
        model_version: str = "0.1.0",
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.task = task
        self.confidence_threshold = confidence_threshold
        self.mc_dropout_samples = mc_dropout_samples
        self.model_version = model_version

        # Transforms & explainability
        self.transform = get_inference_transforms()
        self.cam: Optional[GradCAM] = None
        try:
            target_layer = self.model.get_gradcam_target_layer()
            self.cam = GradCAM(self.model, target_layer)
        except Exception:
            self.cam = None

    def _estimate_uncertainty(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Perform Monte Carlo Dropout forward passes to estimate predictive uncertainty."""
        if self.mc_dropout_samples <= 1:
            with torch.no_grad():
                self.model.eval()
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                return probs, 0.0

        # Enable dropout during inference for MC sampling
        self.model.train()
        probs_samples = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                probs_samples.append(probs.cpu().numpy())

        self.model.eval()
        stacked = np.stack(probs_samples, axis=0)  # (S, B, C)
        mean_probs = np.mean(stacked, axis=0)       # (B, C)
        variance = np.var(stacked, axis=0)         # (B, C)
        # Average predictive variance across classes as the uncertainty metric
        uncertainty = float(np.mean(variance))

        return torch.from_numpy(mean_probs).to(self.device), round(uncertainty, 4)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        generate_gradcam: bool = True,
    ) -> PredictionResult:
        """Run prediction and explainability on a single scan.

        Parameters
        ----------
        image_input:
            File path (DICOM, NIfTI, JPG, PNG) or PIL Image / NumPy array.
        generate_gradcam:
            Whether to compute Grad-CAM heatmap.

        Returns
        -------
        PredictionResult
        """
        start_time = time.perf_counter()

        # 1. Load input into standard RGB numpy array
        if isinstance(image_input, (str, Path)):
            pil_image = load_medical_image(image_input)
            img_np = np.array(pil_image.convert("RGB"))
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            img_np = np.array(pil_image)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_np = np.stack([image_input] * 3, axis=-1)
            else:
                img_np = image_input
            if img_np.dtype != np.uint8:
                img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        # 2. Preprocessing
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # 3. Inference with uncertainty
        probs_tensor, uncertainty = self._estimate_uncertainty(tensor)
        probs_np = probs_tensor.cpu().numpy()[0]
        class_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[class_idx])

        # 4. Class mappings
        class_names = getattr(self.model, "class_names", [f"Class_{i}" for i in range(len(probs_np))])
        predicted_class = class_names[class_idx]

        probabilities = {
            c_name: float(round(probs_np[i], 4)) for i, c_name in enumerate(class_names)
        }

        # Top 2 predictions
        sorted_pairs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_2 = sorted_pairs[:2]

        is_low_conf = bool(confidence < self.confidence_threshold)

        # 5. Grad-CAM
        heatmap = None
        overlay = None
        if generate_gradcam and self.cam is not None:
            try:
                # self.cam operates on single input or batch
                heatmaps = self.cam.generate_heatmap(tensor, target_class=class_idx)
                heatmap = heatmaps[0]  # shape (H, W) in [0, 1]
                overlay = overlay_heatmap(img_np, heatmap, alpha=0.45)
            except Exception:
                heatmap = None
                overlay = None

        elapsed = time.perf_counter() - start_time

        return PredictionResult(
            task=self.task,
            predicted_class=predicted_class,
            class_index=class_idx,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            top_2=top_2,
            uncertainty_score=uncertainty,
            is_low_confidence=is_low_conf,
            heatmap=heatmap,
            overlay=overlay,
            original_image=img_np,
            processing_time_seconds=round(elapsed, 4),
            model_version=self.model_version,
        )

