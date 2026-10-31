"""Unit tests for NeuroScan AI API schemas, health check, and mock services."""

import unittest
from unittest.mock import MagicMock

from src.utils.constants import (
    AnalysisType,
    TumourClass,
    AlzheimerClass,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.api.models.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ScanUploadResponse,
)


class TestApiSchemas(unittest.TestCase):
    """Test validation of Pydantic models in API schemas."""

    def test_health_response(self):
        resp = HealthResponse(
            status="healthy",
            version="0.1.0",
            models_loaded=True,
            gpu_available=False,
        )
        self.assertEqual(resp.status, "healthy")
        self.assertEqual(resp.version, "0.1.0")

    def test_prediction_request(self):
        req = PredictionRequest(analysis_type="tumour")
        self.assertEqual(req.analysis_type, "tumour")

    def test_scan_upload_response(self):
        resp = ScanUploadResponse(
            scan_id="test-1234",
            filename="scan_01.dcm",
            file_type="dicom",
            status="uploaded",
        )
        self.assertEqual(resp.scan_id, "test-1234")
        self.assertEqual(resp.file_type, "dicom")


class TestConstants(unittest.TestCase):
    """Verify consistency of clinical classes and thresholds."""

    def test_tumour_classes(self):
        expected = {"glioma", "meningioma", "pituitary", "no_tumour"}
        actual = {c.value for c in TumourClass}
        self.assertEqual(expected, actual)

    def test_alzheimer_classes(self):
        expected = {"non_demented", "very_mild", "mild", "moderate"}
        actual = {c.value for c in AlzheimerClass}
        self.assertEqual(expected, actual)

    def test_threshold(self):
        self.assertGreater(DEFAULT_CONFIDENCE_THRESHOLD, 0.5)
        self.assertLess(DEFAULT_CONFIDENCE_THRESHOLD, 1.0)


if __name__ == "__main__":
    unittest.main()
