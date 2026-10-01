"""Real-time scoring service shared by the notebook, Streamlit dashboard and Flask API."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, MIN_PRECISION, MODEL_DIR
from .models import absolute_percentile, score_anomaly

BUNDLE_PATH = MODEL_DIR / "fraud_pipeline.joblib"

VERDICT_BANDS = (
    (0.85, "Critical", "#ef4444", "Block the transaction and step up authentication immediately."),
    (0.60, "High", "#f97316", "Hold for manual review by the fraud analyst queue."),
    (0.30, "Medium", "#eab308", "Allow with step-up checks; monitor the cardholder's session."),
    (0.00, "Low", "#22c55e", "No anomaly signal; approve on the standard authorisation path."),
)


@dataclass
class TransactionScore:
    """Full explainable verdict for a single transaction."""

    fraud_probability: float
    risk_score: float
    verdict: str
    color: str
    recommendation: str
    is_fraud: bool
    threshold: float
    above_threshold: bool
    anomaly_isolation_forest: float
    anomaly_local_outlier_factor: float | None
    ensemble_score: float
    contributing_features: list[dict] = field(default_factory=list)
    scored_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

class FraudDetector:
    """Loads the trained bundle once and scores transactions on demand.

    The same object backs the Streamlit dashboard, the Flask API and the notebook,
    so every surface reports identical numbers for identical input.
    """

    def __init__(self, bundle_path: str | Path = BUNDLE_PATH) -> None:
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle missing at {self.bundle_path}. "
                "Run `python -m fraudguard.train` to train the pipeline first."
            )
        self.bundle = joblib.load(self.bundle_path)
        self.xgb = self.bundle["xgboost"]
        self.scaler = self.bundle["scaler"]
        self.feature_names: list[str] = self.bundle["feature_names"]
        self.isolation_forest = self.bundle.get("isolation_forest")
        self.lof = self.bundle.get("lof")
        self.if_reference = self.bundle.get("if_reference")
        self.lof_reference = self.bundle.get("lof_reference")
        self.threshold = float(self.bundle.get("threshold", 0.5))
        self.blend = self.bundle.get("blend", {"weights": {"xgb": 1.0, "lof": 0.0}})
        self.dataset = self.bundle.get("dataset", {})
        self.version = self.bundle.get("version", "unknown")
        self._reference = self._reference_distribution()

    @property
    def metadata(self) -> dict:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "features": list(self.feature_names),
            "dataset": self.dataset,
            "blend": self.blend,
            "min_precision": MIN_PRECISION,
        }

    def _reference_distribution(self) -> pd.DataFrame:
        """Per-feature median and MAD of legitimate training transactions.

        The scaler's `scale_` is the population std, so robust z-scores built from
        these stats flag genuinely unusual values without letting a few extreme
        legitimate purchases dominate the scale.
        """
        scaler = self.scaler
        medians = np.asarray(scaler.mean_, dtype=np.float64)
        spread = np.asarray(scaler.scale_, dtype=np.float64)
        return pd.DataFrame({"median": medians, "spread": np.maximum(spread, 1e-9)})

    def _validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        return frame[self.feature_names].astype(float)

    def _explain(self, raw: np.ndarray, top: int = 4) -> list[dict]:
        """Rank features by robust deviation from the legitimate baseline."""
        z = (raw - self._reference["median"].to_numpy()) / self._reference["spread"].to_numpy()
        order = np.argsort(np.abs(z))[::-1][:top]
        return [
            {
                "feature": self.feature_names[i],
                "value": round(float(raw[i]), 3),
                "typical": round(float(self._reference["median"].iloc[i]), 3),
                "z_score": round(float(z[i]), 2),
                "direction": "high" if z[i] > 0 else "low",
            }
            for i in order
        ]

    def _band(self, risk: float) -> tuple[str, str, str]:
        for floor, verdict, color, recommendation in VERDICT_BANDS:
            if risk >= floor:
                return verdict, color, recommendation
        return "Low", "#22c55e", VERDICT_BANDS[-1][3]

    def score_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring: adds probability, risk score, verdict and explanations."""
        validated = self._validate(frame)
        raw = validated.to_numpy(dtype=np.float64)
        scaled = self.scaler.transform(raw).astype(np.float32)

        probability = self.xgb.predict_proba(scaled)[:, 1]
        if_score = (
            absolute_percentile(score_anomaly(self.isolation_forest.model, scaled), self.if_reference)
            if self.isolation_forest is not None else None
        )
        lof_score = (
            absolute_percentile(score_anomaly(self.lof.model, scaled), self.lof_reference)
            if self.lof is not None else None
        )

        w_xgb = float(self.blend.get("weights", {}).get("xgb", 1.0))
        w_lof = float(self.blend.get("weights", {}).get("lof", 0.0))
        if lof_score is None:
            ensemble = probability
        else:
            ensemble = w_xgb * probability + w_lof * lof_score

        risk = np.where(lof_score is None, probability, 0.6 * probability + 0.25 * if_score + 0.15 * lof_score)
        results = []
        for i in range(len(frame)):
            verdict, color, recommendation = self._band(float(risk[i]))
            results.append(
                {
                    "fraud_probability": round(float(probability[i]), 6),
                    "risk_score": round(float(risk[i]), 6),
                    "ensemble_score": round(float(ensemble[i]), 6),
                    "anomaly_isolation_forest": round(float(if_score[i]), 6) if if_score is not None else None,
                    "anomaly_local_outlier_factor": round(float(lof_score[i]), 6) if lof_score is not None else None,
                    "verdict": verdict,
                    "verdict_color": color,
                    "recommendation": recommendation,
                    "is_fraud": bool(probability[i] >= self.threshold),
                    "above_threshold": bool(ensemble[i] >= self.threshold),
                    "contributing_features": self._explain(raw[i]),
                }
            )
        return pd.DataFrame(results, index=frame.index)

    def score(self, transaction: dict) -> TransactionScore:
        """Score one transaction given as a feature -> value mapping."""
        row = {feature: float(transaction.get(feature, 0.0)) for feature in self.feature_names}
        scored = self.score_dataframe(pd.DataFrame([row])).iloc[0].to_dict()
        return TransactionScore(
            fraud_probability=scored["fraud_probability"],
            risk_score=scored["risk_score"],
            verdict=scored["verdict"],
            color=scored["verdict_color"],
            recommendation=scored["recommendation"],
            is_fraud=scored["is_fraud"],
            threshold=self.threshold,
            above_threshold=scored["above_threshold"],
            anomaly_isolation_forest=scored["anomaly_isolation_forest"],
            anomaly_local_outlier_factor=scored["anomaly_local_outlier_factor"],
            ensemble_score=scored["ensemble_score"],
            contributing_features=scored["contributing_features"],
            scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def sample_transactions(self, n: int = 200, seed: int = 7) -> pd.DataFrame:
        """Pull a few real rows from the dataset so the UI can demo live scoring."""
        from .data import load_dataset

        df, _ = load_dataset()
        return df.sample(n=min(n, len(df)), random_state=seed)[FEATURES].reset_index(drop=True)

    def feature_ranges(self) -> dict:
        """Typical ranges per feature, used to seed sensible UI input defaults."""
        return {
            "Time": {"min": 0.0, "max": 172_800.0, "default": 41_280.0, "step": 60.0,
                     "help": "Seconds elapsed since the first transaction in the file (2-day window)."},
            "Amount": {"min": 0.0, "max": 25_000.0, "default": 120.0, "step": 1.0,
                       "help": "Transaction value in the dataset's currency units."},
        }


def sigmoid_confidence(probability: float) -> float:
    """Symmetric confidence around the 0.5 boundary, scaled to 0-1 for the gauge."""
    return float(min(1.0, max(0.0, abs(probability - 0.5) * 2 + 0.5)))


def expected_loss(recall: float, precision: float, fraud_rate: float, avg_amount: float) -> float:
    """Rough money view: expected value of frauds stopped minus review cost of false alarms.

    Assumes each caught fraud loses the full transaction amount, each false alarm costs a
    fixed review, and misses are the remainder. Useful for comparing thresholds with
    operations stakeholders who do not read precision/recall tables.
    """
    review_cost = 2.0
    true_positive_rate = recall * min(1.0, precision / max(precision, 1e-9)) if precision > 0 else 0.0
    caught_per_1k = true_positive_rate * fraud_rate * 1000
    false_alarms_per_1k = (recall * fraud_rate * 1000) * (1 - precision) / max(precision, 1e-9) if precision > 0 else 0.0
    return float((caught_per_1k * avg_amount) - (false_alarms_per_1k * review_cost))


def is_finite_number(value: object) -> bool:
    """Guard against NaN/inf leaking into the scoring path from user input."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
