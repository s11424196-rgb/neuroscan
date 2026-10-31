"""
Grad-CAM and Grad-CAM++ implementations for model explainability.

Generates class-discriminative saliency heatmaps that highlight the
image regions most influential to a model's prediction.

Typical usage::

    cam = GradCAM(model, model.get_gradcam_target_layer())
    heatmap = cam.generate_heatmap(input_tensor, target_class=0)
    overlay = overlay_heatmap(original_pil, heatmap, alpha=0.4)

Both :class:`GradCAM` and :class:`GradCAMPlusPlus` support batched inputs
and return per-sample heatmaps.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# Overlay helper (module-level function)
# ──────────────────────────────────────────────────────────────────────


def overlay_heatmap(
    original_image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap on top of the original image.

    Parameters
    ----------
    original_image:
        RGB image as PIL or ``(H, W, 3)`` uint8 NumPy array.
    heatmap:
        ``(H_cam, W_cam)`` float32 array in ``[0, 1]``.
    alpha:
        Blending weight for the heatmap.
    colormap:
        OpenCV colormap constant.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 blended image.
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.convert("RGB"))

    h, w = original_image.shape[:2]

    # Resize heatmap to match original.
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = np.float32(heatmap_color) * alpha + np.float32(original_image) * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Base Grad-CAM
# ──────────────────────────────────────────────────────────────────────


class GradCAM:
    """Standard Grad-CAM (Selvaraju *et al.*, 2017).

    Registers forward and backward hooks on the *target_layer* to
    capture activations and gradients, then combines them into a
    class-discriminative heatmap.

    Parameters
    ----------
    model:
        The classifier (must be in eval mode for normal inference, but
        the hook works in both modes).
    target_layer:
        The convolutional layer to inspect.  Use
        ``model.get_gradcam_target_layer()`` on a :class:`BaseClassifier`.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks.
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ---- hook callbacks ------------------------------------------------
    def _save_activation(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ---- public API ----------------------------------------------------
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap for a single image or batch.

        Parameters
        ----------
        input_tensor:
            ``(1, C, H, W)`` or ``(B, C, H, W)`` float tensor.
        target_class:
            Class index to explain.  If *None*, the predicted class
            (argmax of logits) is used.

        Returns
        -------
        np.ndarray
            If ``B == 1``: ``(H_feat, W_feat)`` float32 in ``[0, 1]``.
            If ``B > 1``: ``(B, H_feat, W_feat)`` float32 in ``[0, 1]``.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward.
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)  # (B, num_classes)

        if target_class is None:
            target_class_idx = logits.argmax(dim=1)
        else:
            target_class_idx = torch.full(
                (logits.shape[0],), target_class, dtype=torch.long, device=device
            )

        # Backward for the target class score.
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            one_hot[i, target_class_idx[i]] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        heatmaps = self._compute_cam(self._activations, self._gradients)
        return heatmaps.squeeze() if heatmaps.shape[0] == 1 else heatmaps

    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """Standard Grad-CAM: global-average-pool the gradients, weight
        the activation maps, ReLU, and normalise.

        Parameters
        ----------
        activations:
            ``(B, C_feat, H, W)``
        gradients:
            ``(B, C_feat, H, W)``

        Returns
        -------
        np.ndarray
            ``(B, H, W)`` float32 in ``[0, 1]``.
        """
        # α_k = GAP of gradients over spatial dims.
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # Per-sample normalisation.
        cam_np = cam.cpu().numpy()
        results = []
        for c in cam_np:
            mn, mx = c.min(), c.max()
            if mx - mn > 0:
                c = (c - mn) / (mx - mn)
            else:
                c = np.zeros_like(c)
            results.append(c)
        return np.stack(results).astype(np.float32)

    def remove_hooks(self) -> None:
        """Detach the forward/backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __del__(self) -> None:
        self.remove_hooks()


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM++
# ──────────────────────────────────────────────────────────────────────


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ (Chattopadhyay *et al.*, 2018).

    Uses second- and third-order gradient information to compute
    pixel-level weights, often producing tighter saliency maps for
    objects that span a large fraction of the image.
    """

    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """Grad-CAM++ weighting scheme."""
        # Second and third powers of gradients.
        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)

        # Per-channel global sum of activations.
        sum_act = activations.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # α^{kc}_{ij} (pixel-wise importance weights).
        denom = 2.0 * grad2 + sum_act * grad3 + 1e-8
        alpha = grad2 / denom

        # Weight = sum over spatial of (alpha * ReLU(dY/dA)).
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        results = []
        for c in cam_np:
            mn, mx = c.min(), c.max()
            if mx - mn > 0:
                c = (c - mn) / (mx - mn)
            else:
                c = np.zeros_like(c)
            results.append(c)
        return np.stack(results).astype(np.float32)
