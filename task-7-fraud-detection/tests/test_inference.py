"""Integration tests for the scorer and the Flask API.

Skipped automatically when no trained bundle is present, so the suite still runs
on a fresh clone before `python -m fraudguard.train` has been run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from fraudguard.config import MODEL_DIR  # noqa: E402

BUNDLE = MODEL_DIR / "fraud_pipeline.joblib"
HAVE_BUNDLE = BUNDLE.exists()


def feature_vector(**overrides) -> dict:
    """A zeroed 30-feature vector with sensible Time/Amount defaults."""
    vector = {f"V{i}": 0.0 for i in range(1, 29)}
    vector.update({"Time": 41_280.0, "Amount": 100.0})
    vector.update(overrides)
    return vector


@unittest.skipUnless(HAVE_BUNDLE, "no trained bundle; run `python -m fraudguard.train` first")
class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fraudguard.inference import FraudDetector

        cls.detector = FraudDetector()

    def test_metadata(self):
        meta = self.detector.metadata
        self.assertEqual(len(meta["features"]), 30)
        self.assertGreater(meta["threshold"], 0.0)
        self.assertLessEqual(meta["threshold"], 1.0)

    def test_score_returns_every_field(self):
        result = self.detector.score(feature_vector())
        payload = result.to_dict()
        for key in ("fraud_probability", "risk_score", "verdict", "recommendation",
                    "anomaly_isolation_forest", "contributing_features", "scored_at"):
            self.assertIn(key, payload)
        self.assertIn(result.verdict, {"Low", "Medium", "High", "Critical"})
        self.assertTrue(0.0 <= result.fraud_probability <= 1.0)
        self.assertTrue(0.0 <= result.risk_score <= 1.0)

    def test_contributions_ranked_and_bounded(self):
        contributions = self.detector.score(feature_vector(Amount=9999.0)).contributing_features
        self.assertGreaterEqual(len(contributions), 1)
        magnitudes = [abs(c["z_score"]) for c in contributions]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_single_and_batch_agree(self):
        vector = feature_vector(Amount=1_850.0, V14=-1.9)
        single = self.detector.score(vector)
        batch = self.detector.score_dataframe(pd.DataFrame([vector])).iloc[0]
        self.assertAlmostEqual(single.fraud_probability, float(batch["fraud_probability"]), places=6)
        self.assertEqual(single.verdict, batch["verdict"])

    def test_batch_scoring_shape(self):
        frame = pd.DataFrame([feature_vector(Amount=a) for a in (5.0, 100.0, 2_000.0)])
        scored = self.detector.score_dataframe(frame)
        self.assertEqual(len(scored), 3)
        self.assertTrue((scored["fraud_probability"].between(0, 1)).all())

    def test_high_amount_is_not_silently_ignored(self):
        low = self.detector.score(feature_vector(Amount=2.0))
        high = self.detector.score(feature_vector(Amount=9_000.0))
        self.assertGreaterEqual(high.risk_score, low.risk_score)

    def test_missing_feature_is_rejected(self):
        vector = feature_vector()
        vector.pop("V14")
        with self.assertRaises(ValueError):
            self.detector.score_dataframe(pd.DataFrame([vector]))

    def test_non_finite_input_is_handled(self):
        scored = self.detector.score(feature_vector(Amount=0.0))
        self.assertTrue(np.isfinite(scored.fraud_probability))


@unittest.skipUnless(HAVE_BUNDLE, "no trained bundle; run `python -m fraudguard.train` first")
class TestFlaskApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import flask_app

        flask_app.app.config.update(TESTING=True)
        cls.client = flask_app.app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_pages_render(self):
        for path in ("/", "/score", "/batch"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(b"FraudGuard", response.data)

    def test_predict_endpoint(self):
        response = self.client.post("/api/predict", json=feature_vector(Amount=1_850.0, V14=-1.9))
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload["verdict"], {"Low", "Medium", "High", "Critical"})
        self.assertIn("contributing_features", payload)

    def test_predict_rejects_missing_features(self):
        response = self.client.post("/api/predict", json={"Time": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing", response.get_json())

    def test_batch_endpoint(self):
        rows = [feature_vector(Amount=a) for a in (5.0, 500.0, 5_000.0)]
        response = self.client.post("/api/batch", json={"transactions": rows})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 3)

    def test_batch_endpoint_rejects_empty(self):
        self.assertEqual(self.client.post("/api/batch", json={}).status_code, 400)

    def test_score_form_post(self):
        response = self.client.post("/score", data=feature_vector(Amount=1_850.0, V14=-1.9))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"risk", response.data)

    def test_unknown_page_is_404(self):
        self.assertEqual(self.client.get("/definitely-not-a-page").status_code, 404)

    def test_figure_route_rejects_traversal(self):
        self.assertEqual(self.client.get("/figures/..%2f..%2fmetrics.json").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
