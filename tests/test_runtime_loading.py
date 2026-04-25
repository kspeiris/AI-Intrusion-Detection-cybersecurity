import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app
import realtime_detector


class RuntimeLoadingTests(unittest.TestCase):
    def test_api_runtime_reset_on_load_failure(self):
        app.runtime_state.update(
            {
                "loaded": True,
                "error": None,
                "metadata": {"stale": True},
                "model": object(),
                "scaler": object(),
                "selector": object(),
                "model_name": "Stale Model",
                "threshold": 0.99,
            }
        )

        with patch("app.joblib.load", side_effect=RuntimeError("broken artifact")):
            with self.assertRaisesRegex(RuntimeError, "broken artifact"):
                app.load_runtime_state()

        self.assertFalse(app.runtime_state["loaded"])
        self.assertEqual(app.runtime_state["error"], "broken artifact")
        self.assertIsNone(app.runtime_state["metadata"])
        self.assertIsNone(app.runtime_state["model"])
        self.assertIsNone(app.runtime_state["scaler"])
        self.assertIsNone(app.runtime_state["selector"])
        self.assertIsNone(app.runtime_state["model_name"])
        self.assertIsNone(app.runtime_state["threshold"])

    def test_detector_uses_best_model_from_metadata(self):
        metadata = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))

        realtime_detector.load_detector_artifacts()

        self.assertTrue(realtime_detector.detector_state["loaded"])
        self.assertEqual(
            realtime_detector.detector_state["model_name"],
            metadata["best_model_name"],
        )
        self.assertEqual(
            realtime_detector.detector_state["threshold"],
            metadata["best_threshold"],
        )

    def test_detector_reset_on_load_failure(self):
        realtime_detector.detector_state.update(
            {
                "loaded": True,
                "error": None,
                "model": object(),
                "scaler": object(),
                "selector": object(),
                "encoders": object(),
                "model_name": "Stale Model",
                "threshold": 0.77,
            }
        )

        with patch("realtime_detector.joblib.load", side_effect=RuntimeError("detector load failed")):
            with self.assertRaisesRegex(RuntimeError, "detector load failed"):
                realtime_detector.load_detector_artifacts()

        self.assertFalse(realtime_detector.detector_state["loaded"])
        self.assertEqual(realtime_detector.detector_state["error"], "detector load failed")
        self.assertIsNone(realtime_detector.detector_state["model"])
        self.assertIsNone(realtime_detector.detector_state["scaler"])
        self.assertIsNone(realtime_detector.detector_state["selector"])
        self.assertIsNone(realtime_detector.detector_state["encoders"])
        self.assertIsNone(realtime_detector.detector_state["model_name"])
        self.assertIsNone(realtime_detector.detector_state["threshold"])


if __name__ == "__main__":
    unittest.main()
