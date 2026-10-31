"""Evaluation metrics, calibration, bootstrap confidence intervals, and model card generation.

Computes comprehensive diagnostic metrics for multi-class medical classification:
- Macro and per-class Precision, Recall, F1
- Accuracy and Balanced Accuracy
- ROC Curves and One-vs-Rest AUROC
- Expected Calibration Error (ECE)
- Non-parametric bootstrap confidence intervals (e.g. 95% CI)
- ModelCard dataclass and markdown report generation
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    probabilities:
        (N, C) array of predicted class probabilities.
    labels:
        (N,) array of true integer labels.
    num_bins:
        Number of equal-width probability bins between 0 and 1.

    Returns
    -------
    float
        ECE score in [0, 1].
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics.

    Parameters
    ----------
    y_true:
        Ground truth integer labels.
    y_pred:
        Predicted integer class labels.
    y_prob:
        Optional predicted probabilities (N, C).
    class_names:
        Class names corresponding to indices.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics.
    """
    num_classes = len(class_names) if class_names else len(np.unique(y_true))
    classes = class_names or [f"class_{i}" for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, c_name in enumerate(classes):
        per_class[c_name] = {
            "precision": float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
            "recall": float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
            "f1": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    results: Dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": classes,
    }

    if y_prob is not None:
        try:
            if y_prob.shape[1] > 2:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            else:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            results["auroc_macro"] = auc
        except Exception:
            results["auroc_macro"] = None

        results["ece"] = compute_ece(y_prob, y_true)

    return results


# Backward compatibility alias
generate_evaluation_report = compute_metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap percentile confidence intervals for primary metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_samples: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]

        # Ensure multi-class presence in sample
        if len(np.unique(sample_true)) < 2:
            continue

        metrics_samples["accuracy"].append(float(accuracy_score(sample_true, sample_pred)))
        metrics_samples["balanced_accuracy"].append(
            float(balanced_accuracy_score(sample_true, sample_pred))
        )
        metrics_samples["macro_f1"].append(
            float(f1_score(sample_true, sample_pred, average="macro", zero_division=0))
        )

    alpha = (1.0 - ci_level) / 2.0
    intervals: Dict[str, Tuple[float, float]] = {}

    for metric, values in metrics_samples.items():
        if len(values) > 0:
            low = float(np.percentile(values, alpha * 100))
            high = float(np.percentile(values, (1.0 - alpha) * 100))
            intervals[metric] = (round(low, 4), round(high, 4))
        else:
            intervals[metric] = (0.0, 0.0)

    return intervals


def plot_confusion_matrix(
    cm: np.ndarray | List[List[int]],
    class_names: List[str],
    output_path: Optional[str | Path] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot and optionally save a styled confusion matrix."""
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.divide(cm_arr, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


@dataclass
class ModelCard:
    """Model Card documenting metadata, evaluation metrics, and intended use."""

    model_name: str
    task: str
    backbone: str
    version: str
    date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    intended_use: str = "Decision support and educational screening of brain MRI scans."
    out_of_scope_use: str = (
        "Primary diagnosis, medical device use, or any clinical workflow without doctor oversight."
    )
    dataset_summary: str = "Trained with patient-level split to prevent data leakage."
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, Any] = field(default_factory=dict)
    ethical_considerations: str = (
        "Dataset may carry demographic and scanner bias. Review all low-confidence predictions."
    )

    def to_markdown(self) -> str:
        """Render model card as formatted markdown."""
        md = [
            f"# Model Card: {self.model_name} (v{self.version})",
            "",
            f"- **Task**: {self.task}",
            f"- **Architecture**: {self.backbone}",
            f"- **Date**: {self.date}",
            "",
            "## Intended Use",
            self.intended_use,
            "",
            "## Out-of-Scope / Non-Goals",
            self.out_of_scope_use,
            "",
            "## Training Data & Methodology",
            self.dataset_summary,
            "",
            "## Performance Metrics",
            f"- **Accuracy**: {self.metrics.get('accuracy', 0.0):.4f}",
            f"- **Macro F1**: {self.metrics.get('macro_f1', 0.0):.4f}",
            f"- **Balanced Accuracy**: {self.metrics.get('balanced_accuracy', 0.0):.4f}",
        ]
        if "auroc_macro" in self.metrics and self.metrics["auroc_macro"] is not None:
            md.append(f"- **Macro AUROC**: {self.metrics['auroc_macro']:.4f}")
        if "ece" in self.metrics and self.metrics["ece"] is not None:
            md.append(f"- **Expected Calibration Error (ECE)**: {self.metrics['ece']:.4f}")

        if self.confidence_intervals:
            md.append("")
            md.append("### 95% Confidence Intervals (Bootstrap)")
            for k, (lo, hi) in self.confidence_intervals.items():
                md.append(f"- **{k}**: [{lo:.4f}, {hi:.4f}]")

        md.extend([
            "",
            "## Ethical Considerations & Disclaimers",
            self.ethical_considerations,
            "",
            "> ⚠️ **Disclaimer**: For research and educational use only. Not for clinical diagnosis.",
        ])

        return "\n".join(md)

    def save(self, filepath: str | Path) -> None:
        """Save model card as markdown file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
