"""
Training entry point for Brain Tumour and Alzheimer's MRI classifiers.

Features:
- Reproducible random seed setting
- Patient-aware train / validation / test splits (stratified by class)
- Configurable backbones (ResNet-50, EfficientNet-B0, DenseNet-121)
- Mixed-precision training and early stopping
- Test set evaluation with Bootstrap 95% Confidence Intervals
- Confusion matrix and ROC curve plot exports
- Model Card documentation generation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Brain MRI Classifier")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["tumour", "alzheimer"],
        help="Classification task: 'tumour' or 'alzheimer'",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["resnet50", "efficientnet_b0", "densenet121"],
        help="Feature extractor backbone architecture",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (structured as class_name/image.*)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/registry/experiment_1",
        help="Directory to save checkpoints and evaluation reports",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing training task '%s' with backbone '%s'", args.task, args.backbone)

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from src.ml.models.classifier import ModelFactory, TUMOUR_CLASSES, ALZHEIMER_CLASSES
        from src.ml.preprocessing.transforms import get_train_transforms, get_val_transforms
        from src.ml.training.dataset import BrainMRIDataset, PatientAwareSplitter
        from src.ml.training.trainer import Trainer, TrainingConfig
        from src.ml.evaluation.metrics import (
            ModelCard,
            bootstrap_confidence_intervals,
            compute_metrics,
            plot_confusion_matrix,
        )
    except ImportError as exc:
        logger.error("Required deep learning dependencies are not installed: %s", exc)
        logger.info("Please install project requirements via: pip install -e '.[gpu]'")
        return

    # Select target device
    device_str = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    logger.info("Using compute device: %s", device)

    # 1. Dataset & Patient-Aware Split
    train_transform = get_train_transforms(image_size=224)
    val_transform = get_val_transforms(image_size=224)

    base_dataset = BrainMRIDataset(root_dir=args.data_dir, transform=None)
    logger.info("Discovered %d samples across %d classes", len(base_dataset), len(base_dataset.class_names))

    splitter = PatientAwareSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    split_indices = splitter.split(base_dataset)
    logger.info(
        "Split sizes: Train=%d, Val=%d, Test=%d (Split by patient ID)",
        len(split_indices.train),
        len(split_indices.val),
        len(split_indices.test),
    )

    # Wrap subsets with respective augmentations
    class TransformSubset(Subset):
        def __init__(self, subset, transform):
            super().__init__(subset.dataset, subset.indices)
            self.transform = transform

        def __getitem__(self, idx):
            img, label, patient_id = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label, patient_id

    train_subset = TransformSubset(Subset(base_dataset, split_indices.train), train_transform)
    val_subset = TransformSubset(Subset(base_dataset, split_indices.val), val_transform)
    test_subset = TransformSubset(Subset(base_dataset, split_indices.test), val_transform)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Instantiate Model
    model = ModelFactory.create(
        task=args.task,
        backbone=args.backbone,
        pretrained=True,
    )

    # 3. Training
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=output_path,
        seed=args.seed,
    )

    trainer = Trainer(model=model, config=config, device=device)
    logger.info("Starting training loop...")
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    # 4. Final Test Set Evaluation
    logger.info("Running evaluation on held-out test split...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test Results: Macro F1 = %.4f | Accuracy = %.4f", test_metrics.get("macro_f1", 0.0), test_metrics.get("accuracy", 0.0))

    # Compute bootstrap confidence intervals
    y_true = np.array(test_metrics["y_true"])
    y_pred = np.array(test_metrics["y_pred"])
    y_prob = np.array(test_metrics.get("y_prob")) if "y_prob" in test_metrics else None

    ci_results = bootstrap_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=base_dataset.class_names,
        n_bootstraps=300,
        ci_level=0.95,
        seed=args.seed,
    )
    logger.info("95%% Bootstrap CIs: %s", ci_results)

    # 5. Save Plots & Model Card
    cm_plot_path = output_path / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        class_names=base_dataset.class_names,
        output_path=cm_plot_path,
        title=f"Test Confusion Matrix ({args.task.capitalize()} - {args.backbone})",
    )

    model_card = ModelCard(
        model_name=f"NeuroScan-{args.task.capitalize()}",
        task=args.task,
        backbone=args.backbone,
        version="0.1.0",
        metrics=test_metrics,
        confidence_intervals=ci_results,
    )
    model_card.save(output_path / "MODEL_CARD.md")

    # Save metrics JSON
    metrics_summary = {
        "task": args.task,
        "backbone": args.backbone,
        "metrics": {k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred", "y_prob")},
        "confidence_intervals_95": ci_results,
    }
    with open(output_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Training and evaluation completed successfully! Artifacts saved to %s", output_path)


if __name__ == "__main__":
    main()
