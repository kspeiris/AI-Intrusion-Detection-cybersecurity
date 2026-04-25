import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app  # noqa: E402
from config import RAW_FEATURE_COUNT  # noqa: E402


class ApiTests(unittest.TestCase):
    def setUp(self):
        app.load_runtime_state()
        self.client = TestClient(app.app)

    def test_health_reports_loaded(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["loaded"])
        self.assertIsNone(payload["error"])

    def test_home_exposes_model_metadata(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["loaded"])
        self.assertIn("model", payload)
        self.assertIn("threshold", payload)

    def test_predict_rejects_wrong_feature_count(self):
        response = self.client.post("/predict", json={"features": [0.0] * 5})

        self.assertEqual(response.status_code, 422)

    def test_predict_accepts_valid_feature_count(self):
        response = self.client.post(
            "/predict",
            json={"features": [0.0] * RAW_FEATURE_COUNT},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["result"], {"Attack", "Normal"})
        self.assertIsInstance(payload["prediction"], int)
        self.assertIsInstance(payload["probability"], float)


if __name__ == "__main__":
    unittest.main()
