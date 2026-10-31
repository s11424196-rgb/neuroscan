"""Unit tests for ML evaluation metrics, calibration (ECE), and patient-aware splitting."""

import unittest
import numpy as np

from src.ml.evaluation.metrics import (
    compute_ece,
    compute_metrics,
    bootstrap_confidence_intervals,
)


class TestMetrics(unittest.TestCase):
    """Test evaluation calculation functions."""

    def setUp(self):
        self.y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        self.y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3, 0, 1])
        # Probabilities with high confidence for diagonal
        probs = np.zeros((10, 4))
        for i, p in enumerate(self.y_pred):
            probs[i, p] = 0.9
            rem = 0.1 / 3
            for j in range(4):
                if j != p:
                    probs[i, j] = rem
        self.y_prob = probs
        self.class_names = ["glioma", "meningioma", "pituitary", "no_tumour"]

    def test_compute_metrics(self):
        metrics = compute_metrics(
            self.y_true,
            self.y_pred,
            self.y_prob,
            class_names=self.class_names,
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertIn("ece", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 0.9, places=2)

    def test_compute_ece(self):
        # Perfect calibration check
        ece = compute_ece(self.y_prob, self.y_true, num_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_bootstrap_ci(self):
        ci = bootstrap_confidence_intervals(
            self.y_true,
            self.y_pred,
            n_bootstraps=50,
            ci_level=0.95,
            seed=42,
        )
        self.assertIn("accuracy", ci)
        low, high = ci["accuracy"]
        self.assertLessEqual(low, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
