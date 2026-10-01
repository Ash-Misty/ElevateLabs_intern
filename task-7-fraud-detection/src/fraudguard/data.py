"""Dataset acquisition, validation and inspection.

The project targets the Kaggle "Credit Card Fraud Detection" dataset
(``mlg-ulb/creditcardfraud``, 284,807 transactions, 0.172% fraud). Because that
file requires Kaggle credentials and is ~148 MB, :func:`ensure_dataset` falls back
to a high-fidelity synthetic generator that reproduces the exact 30-feature schema,
the PCA-like correlation structure and the fraud anatomy of the original file, so
the whole pipeline runs end to end without manual steps.

To train on the real data, drop ``creditcard.csv`` into ``dataset/`` or export it
with the Kaggle CLI::

    kaggle competitions download -c creditcardfraud -p dataset --unzip
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, KAGGLE_COLUMNS, RANDOM_STATE, REAL_DATASET, SYNTHETIC_DATASET, TARGET

FRAUD_SIGNATURES = {
    "V14": -1.9, "V12": -1.6, "V17": -1.8, "V10": -1.3, "V7": -1.1,
    "V16": -0.9, "V18": -0.8, "V1": -0.6, "V3": -0.5, "V11": -0.4,
}

HARD_FRAUD_FRACTION = 0.34


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance metadata attached to every trained artifact."""

    path: Path
    rows: int
    fraud_count: int
    fraud_rate: float
    source: str
    is_real: bool

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "rows": int(self.rows),
            "fraud_count": int(self.fraud_count),
            "fraud_rate": float(self.fraud_rate),
            "source": self.source,
            "is_real": bool(self.is_real),
        }


def _kaggle_download(target: Path) -> bool:
    """Attempt ``kaggle competitions download``; return True on success."""
    if not shutil.which("kaggle"):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle", "competitions", "download",
        "-c", "creditcardfraud", "-p", str(target.parent), "--unzip", "--force",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and target.exists()


def _synthesize(n_rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    """Generate a Kaggle-shaped synthetic dataset with realistic fraud structure.

    Three properties are deliberately reproduced so the pipeline faces a realistic
    difficulty ceiling rather than a trivially separable toy problem:

    1. a latent-factor correlation structure, because V1-V28 are PCA projections
       and therefore never independent;
    2. night-time clustering of fraud, which is the clearest real-world signal;
    3. a ``HARD_FRAUD_FRACTION`` of frauds that carry only a faint signature, which
       caps achievable recall and makes precision/recall trade-offs real.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_rows * fraud_rate)), 50)
    n_legit = max(n_rows - n_fraud, n_fraud)

    loadings = rng.normal(0, 0.45, size=(28, 10))
    loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)

    def build(n: int, latent_shift: np.ndarray, amount_mu: float, amount_sigma: float) -> pd.DataFrame:
        latent = rng.normal(0, 1, size=(n, 10)) + latent_shift
        v = latent @ loadings.T + rng.normal(0, 0.62, size=(n, 28))
        frame = pd.DataFrame(v.astype(np.float32), columns=[f"V{i}" for i in range(1, 29)])
        frame["Amount"] = rng.lognormal(amount_mu, amount_sigma, size=n).clip(0, 25_691).astype(np.float32)
        return frame

    legit = build(n_legit, np.zeros(10), 3.1, 1.45)
    fraud = build(n_fraud, rng.normal(0.30, 0.22, size=10), 4.1, 1.25)
    fraud["Amount"] = (
        rng.lognormal(4.5, 1.15, size=n_fraud).clip(0, 25_691).astype(np.float32)
    )

    shift = np.zeros(28)
    for column, magnitude in FRAUD_SIGNATURES.items():
        shift[int(column[1:]) - 1] = magnitude
    fraud.loc[:, [f"V{i}" for i in range(1, 29)]] += shift.astype(np.float32)

    v_columns = [f"V{i}" for i in range(1, 29)]
    n_hard = int(n_fraud * HARD_FRAUD_FRACTION)
    if n_hard:
        hard_idx = rng.choice(n_fraud, size=n_hard, replace=False)
        easy_idx = np.setdiff1d(np.arange(n_fraud), hard_idx)
        hard_features = build(n_hard, rng.normal(0.05, 0.10, size=10), 3.6, 1.35)[v_columns].to_numpy()
        fraud.iloc[hard_idx, fraud.columns.get_indexer(v_columns)] = (
            hard_features + (shift * 0.18).astype(np.float32)
        )
        fraud.loc[fraud.index[easy_idx], "Amount"] = (
            rng.lognormal(4.8, 1.05, size=len(easy_idx)).clip(0, 25_691).astype(np.float32)
        )
        fraud.loc[fraud.index[hard_idx], "Amount"] = (
            rng.lognormal(3.3, 1.30, size=n_hard).clip(0, 25_691).astype(np.float32)
        )

    def timestamps(n: int, fraud_like: bool) -> np.ndarray:
        if fraud_like:
            night = rng.random(n) < 0.62
            t = np.where(night, rng.uniform(0, 26_400, n), rng.uniform(26_400, 172_800, n))
        else:
            hour = rng.beta(1.6, 1.9, size=n) * 24
            t = np.clip(np.floor(hour) * 3_600 + rng.uniform(0, 3_600, n), 0, 172_799)
        return t.astype(np.float32)

    legit.insert(0, "Time", timestamps(n_legit, fraud_like=False))
    fraud.insert(0, "Time", timestamps(n_fraud, fraud_like=True))
    legit["Class"] = 0
    fraud["Class"] = 1

    df = pd.concat([legit, fraud], ignore_index=True)

    dupes = df.sample(n=int(len(df) * 0.012), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[KAGGLE_COLUMNS]


FULL_SIZE_ROWS = 284_807


def ensure_dataset(
    max_rows: int | None = None,
    fraud_rate: float = 0.00172,
    seed: int = RANDOM_STATE,
) -> DatasetInfo:
    """Return a ready-to-train dataset, downloading or synthesising as needed."""
    if not REAL_DATASET.exists() and _kaggle_download(REAL_DATASET):
        pass

    target_rows = FULL_SIZE_ROWS if max_rows is None else max_rows
    stale = SYNTHETIC_DATASET.exists() and _row_count(SYNTHETIC_DATASET) < FULL_SIZE_ROWS

    if REAL_DATASET.exists():
        path, source, is_real = REAL_DATASET, "Kaggle creditcardfraud (creditcard.csv)", True
    elif SYNTHETIC_DATASET.exists() and not stale:
        path, source, is_real = SYNTHETIC_DATASET, "Synthetic Kaggle-schema generator", False
    else:
        df = _synthesize(target_rows, fraud_rate, seed)
        if max_rows is None or not stale:
            SYNTHETIC_DATASET.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(SYNTHETIC_DATASET, index=False)
        path, source, is_real = SYNTHETIC_DATASET, "Synthetic Kaggle-schema generator", False

    df = pd.read_csv(path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    fraud_count = int(df["Class"].sum())
    return DatasetInfo(
        path=path,
        rows=int(len(df)),
        fraud_count=fraud_count,
        fraud_rate=fraud_count / max(len(df), 1),
        source=source,
        is_real=is_real,
    )


def _row_count(path: Path) -> int:
    """Count data rows in a CSV without loading it into memory."""
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def load_dataset(
    max_rows: int | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load the dataset and normalise dtypes/column order."""
    info = ensure_dataset(max_rows=max_rows, seed=seed)
    df = pd.read_csv(info.path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[KAGGLE_COLUMNS], info


def clean_dataframe(df: pd.DataFrame, target: str = "Class") -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and coerce types; return the clean frame and a report."""
    report = {
        "rows_in": int(len(df)),
        "duplicates_removed": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }
    clean = df.drop_duplicates().reset_index(drop=True)
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=[target]).reset_index(drop=True)
    report["rows_out"] = int(len(clean))
    report["fraud_rate"] = float(clean[target].mean())
    return clean, report


def split_features_target(
    df: pd.DataFrame,
    target: str = TARGET,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the canonical feature matrix and the binary target.

    Only ``FEATURES`` (``Time``, ``V1``-``V28``, ``Amount``) are returned, even if
    the frame carries extra derived columns. The Kaggle schema is a contract shared
    with the CSV files users upload and the API payloads callers send, so a stray
    exploratory column must never be able to leak into training and silently change
    the number of inputs the deployed model expects.
    """
    expected = list(features or FEATURES)
    extra = [c for c in df.columns if c not in expected and c != target]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if extra:
        print(f"split_features_target: ignoring non-schema column(s) {extra}")
    X = df[expected]
    y = df[target].astype(int)
    return X, y


def describe_dataset(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """Per-column summary statistics used by the notebook and dashboard."""
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
                "skew": float(series.skew()),
                "fraud_mean": float(series[df[target] == 1].mean()),
                "legit_mean": float(series[df[target] == 0].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def missing_free(df: pd.DataFrame) -> bool:
    """True when the frame has no NaNs in any column."""
    return bool(df.isna().sum().sum() == 0)
