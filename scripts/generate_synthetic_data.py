"""
Generate synthetic brain MRI datasets for pipeline verification and end-to-end testing.

Creates realistic grayscale brain MRI-like images with synthetic pathology:
- Tumour task: glioma, meningioma, pituitary, no_tumour
- Alzheimer's task: non_demented, very_mild, mild, moderate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.training.dataset import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MRI dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Root directory for synthetic output.",
    )
    parser.add_argument(
        "--num_per_class",
        type=int,
        default=50,
        help="Number of images per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print(f"Generating synthetic datasets in: {out_root.resolve()}")

    # 1. Tumour dataset
    tumour_dir = out_root / "tumour"
    print(f"Creating tumour dataset at {tumour_dir} ({args.num_per_class} per class)...")
    generate_synthetic_dataset(
        output_dir=tumour_dir,
        task="tumour",
        num_per_class=args.num_per_class,
        seed=args.seed,
    )

    # 2. Alzheimer's dataset
    alz_dir = out_root / "alzheimer"
    print(f"Creating Alzheimer's dataset at {alz_dir} ({args.num_per_class} per class)...")
    generate_synthetic_dataset(
        output_dir=alz_dir,
        task="alzheimer",
        num_per_class=args.num_per_class,
        seed=args.seed + 1,
    )

    print("Synthetic data generation completed successfully!")


if __name__ == "__main__":
    main()

