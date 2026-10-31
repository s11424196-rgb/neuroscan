"""
Training loop, learning-rate scheduling, early stopping, and calibration.

:class:`Trainer` is the primary entry-point.  It is configured via a
:class:`TrainingConfig` dataclass and supports:

* Class-weighted cross-entropy for imbalanced datasets.
* Mixed-precision training (``torch.amp``).
* Cosine annealing with warm restarts.
* Early stopping by validation F1.
* Best-model checkpointing.
* Monte Carlo dropout for uncertainty estimation.
* Temperature scaling for post-training calibration.
* Per-epoch metric logging (loss, accuracy, F1, per-class P/R).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All hyper-parameters and settings for a training run.

    Attributes
    ----------
    epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size for DataLoader.
    learning_rate:
        Initial learning rate.
    weight_decay:
        L2 regularisation coefficient.
    class_weights:
        Optional per-class weights for the loss function (list of floats).
    use_amp:
        Enable automatic mixed-precision training.
    patience:
        Early-stopping patience (epochs without val F1 improvement).
    cosine_t0:
        T_0 for ``CosineAnnealingWarmRestarts``.
    cosine_t_mult:
        T_mult multiplier for warm restarts.
    mc_dropout_passes:
        Number of stochastic forward passes for MC-dropout uncertainty.
    temperature_scaling:
        Whether to apply post-training temperature scaling.
    checkpoint_dir:
        Where to save model checkpoints.
    num_workers:
        DataLoader workers.
    seed:
        Random seed.
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    """

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    class_weights: Optional[list[float]] = None
    use_amp: bool = True
    patience: int = 7
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    mc_dropout_passes: int = 10
    temperature_scaling: bool = True
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        """Return the resolved :class:`torch.device`."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ──────────────────────────────────────────────────────────────────────
# Temperature scaling (post-training calibration)
# ──────────────────────────────────────────────────────────────────────


class TemperatureScaler(nn.Module):
    """Learnable temperature parameter applied after the logit layer.

    Optimised on the validation set *after* training to minimise
    negative log-likelihood without changing model predictions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Optimise the temperature on the validation set.

        Returns
        -------
        float
            The learned temperature value.
        """
        model.eval()
        self.to(device)

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits_cat = torch.cat(all_logits).to(device)
        labels_cat = torch.cat(all_labels).to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(logits_cat), labels_cat)
            loss.backward()
            return loss

        optimizer.step(_closure)
        temp_val = self.temperature.item()
        logger.info("Learned temperature: %.4f", temp_val)
        return temp_val


# ──────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────


class _EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 7) -> None:
        self.patience = patience
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Update with the latest metric.  Returns *True* when training
        should stop."""
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    """Metrics logged for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    val_f1_macro: float
    val_precision_per_class: list[float]
    val_recall_per_class: list[float]
    lr: float
    elapsed_seconds: float


class Trainer:
    """Complete training pipeline for brain MRI classifiers.

    Parameters
    ----------
    model:
        A :class:`BaseClassifier` (or any ``nn.Module``).
    config:
        :class:`TrainingConfig` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = config.resolve_device()
        self.model.to(self.device)

        self.history: list[EpochMetrics] = []
        self.best_model_state: Optional[dict] = None
        self.best_f1: float = 0.0
        self.temperature_scaler: Optional[TemperatureScaler] = None

        # Loss function.
        weight_tensor: Optional[torch.Tensor] = None
        if config.class_weights is not None:
            weight_tensor = torch.tensor(
                config.class_weights, dtype=torch.float32
            ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimiser.
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.cosine_t0,
            T_mult=config.cosine_t_mult,
        )

        # Mixed precision.
        self.scaler: Optional[torch.amp.GradScaler] = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[EpochMetrics]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader:
            Training-set DataLoader.
        val_loader:
            Validation-set DataLoader.

        Returns
        -------
        list[EpochMetrics]
            Per-epoch metrics.
        """
        early_stop = _EarlyStopping(patience=self.config.patience)
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc, val_f1, val_prec, val_rec = self._validate(
                val_loader
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                val_f1_macro=val_f1,
                val_precision_per_class=val_prec,
                val_recall_per_class=val_rec,
                lr=current_lr,
                elapsed_seconds=elapsed,
            )
            self.history.append(metrics)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.3f  val_f1=%.3f  lr=%.2e  (%.1fs)",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
                current_lr,
                elapsed,
            )

            # Checkpoint on best F1.
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(
                    self.model.state_dict(),
                    ckpt_dir / "best_model.pt",
                )
                logger.info("  ↳ New best model saved (F1=%.4f)", val_f1)

            if early_stop.step(val_f1):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        # Restore best model.
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model (F1=%.4f).", self.best_f1)

        # Optional temperature scaling.
        if self.config.temperature_scaling:
            self.temperature_scaler = TemperatureScaler()
            self.temperature_scaler.fit(
                self.model, val_loader, self.device
            )

        return self.history

    # ------------------------------------------------------------------
    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    # ------------------------------------------------------------------
    def _validate(
        self, loader: DataLoader
    ) -> tuple[float, float, float, list[float], list[float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        n = max(len(all_labels), 1)
        avg_loss = total_loss / n
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        prec = precision_score(
            all_labels, all_preds, average=None, zero_division=0
        ).tolist()
        rec = recall_score(
            all_labels, all_preds, average=None, zero_division=0
        ).tolist()
        return avg_loss, acc, f1, prec, rec

    # ------------------------------------------------------------------
    # Test / evaluate
    # ------------------------------------------------------------------

    def test(
        self,
        test_loader: DataLoader,
    ) -> dict:
        """Run inference on the test set and return raw predictions.

        Returns
        -------
        dict
            Keys: ``"labels"``, ``"preds"``, ``"probs"``.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                logits = self.model(images)

                # Apply temperature scaling if available.
                if self.temperature_scaler is not None:
                    logits = self.temperature_scaler(logits)

                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.tolist())

        return {
            "labels": all_labels,
            "preds": all_preds,
            "probs": np.concatenate(all_probs, axis=0),
        }

    # ------------------------------------------------------------------
    # MC-dropout uncertainty
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        n_passes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run MC-dropout forward passes and return mean probabilities
        and uncertainty (std of predicted class probability).

        Parameters
        ----------
        images:
            ``(B, C, H, W)`` tensor.
        n_passes:
            Override ``config.mc_dropout_passes``.

        Returns
        -------
        (mean_probs, uncertainty)
            ``mean_probs``: ``(B, num_classes)`` float32.
            ``uncertainty``: ``(B,)`` float32 – std of the top-class
            probability across passes.
        """
        n = n_passes or self.config.mc_dropout_passes
        self.model.eval()
        # Enable dropout layers.
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        images = images.to(self.device)
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for _ in range(n):
                logits = self.model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        stacked = np.stack(all_probs, axis=0)  # (n, B, C)
        mean_probs = stacked.mean(axis=0)  # (B, C)
        top_class = mean_probs.argmax(axis=1)  # (B,)
        # Std of the top class probability across passes.
        uncertainty = np.array(
            [stacked[:, i, top_class[i]].std() for i in range(images.size(0))]
        )
        return mean_probs, uncertainty

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a full training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
                "history": [
                    {
                        "epoch": m.epoch,
                        "train_loss": m.train_loss,
                        "val_loss": m.val_loss,
                        "train_acc": m.train_acc,
                        "val_acc": m.val_acc,
                        "val_f1_macro": m.val_f1_macro,
                    }
                    for m in self.history
                ],
                "temperature": (
                    self.temperature_scaler.temperature.item()
                    if self.temperature_scaler is not None
                    else None
                ),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_f1 = ckpt.get("best_f1", 0.0)
        logger.info("Checkpoint loaded from %s", path)
