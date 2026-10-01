"""Leakage-safe preprocessing: stratified splitting, scaling and SMOTE balancing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitBundle:
    """Container for the three-way train/valid/test split plus fitted transforms."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_balanced: np.ndarray
    y_train_balanced: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    stats: dict

    @property
    def imbalance_ratio(self) -> float:
        return float((self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1))


def fit_scaler(X: pd.DataFrame | np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the (already PCA-like) feature matrix."""
    scaler = StandardScaler()
    scaler.fit(np.asarray(X, dtype=np.float64))
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Apply a fitted scaler, returning float32 to keep memory bounded."""
    return scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    valid_size: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 3-way split. Returns X_train/X_valid/X_test and matching y.

    The test fold is carved out first and then never touched again; the validation
    fold is taken from the remaining training pool. Sizing is expressed as a share of
    the *whole* dataset, so ``test_size=0.2, valid_size=0.1`` yields 70/10/20.
    """
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_valid = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_pool, y_train_pool, test_size=relative_valid,
        random_state=random_state, stratify=y_train_pool,
    )
    return (
        X_train.to_numpy(dtype=np.float64),
        y_train.to_numpy(),
        X_valid.to_numpy(dtype=np.float64),
        y_valid.to_numpy(),
        X_test.to_numpy(dtype=np.float64),
        y_test.to_numpy(),
    )


def balance_with_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float | str = 1.0,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the fraud class with SMOTE on the training fold only."""
    X_bal, y_bal = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    ).fit_resample(X_train, y_train)
    return np.asarray(X_bal, dtype=np.float32), np.asarray(y_bal)


def build_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    valid_size: float = 0.10,
    smote_strategy: float | str = 1.0,
    random_state: int = 42,
    apply_scaling: bool = True,
) -> SplitBundle:
    """Full preprocessing chain: split -> scale -> SMOTE (train only).

    SMOTE is deliberately applied *after* the split so synthetic fraud samples are
    never created from validation/test rows, which would leak label information.
    """
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_splits(
        X, y, test_size=test_size, valid_size=valid_size, random_state=random_state
    )

    if apply_scaling:
        scaler = fit_scaler(X_tr)
        X_tr_s, X_va_s, X_te_s = (
            apply_scaler(scaler, X_tr), apply_scaler(scaler, X_va), apply_scaler(scaler, X_te)
        )
    else:
        scaler = StandardScaler().fit(np.zeros((1, X.shape[1])))
        X_tr_s, X_va_s, X_te_s = (
            X_tr.astype(np.float32), X_va.astype(np.float32), X_te.astype(np.float32)
        )

    k = int(min(5, max(1, (y_tr == 1).sum() - 1)))
    X_bal, y_bal = balance_with_smote(
        X_tr_s, y_tr, sampling_strategy=smote_strategy, random_state=random_state, k_neighbors=k
    )

    stats = {
        "rows_before_split": int(len(X)),
        "train_rows": int(len(y_tr)),
        "valid_rows": int(len(y_va)),
        "test_rows": int(len(y_te)),
        "train_fraud": int((y_tr == 1).sum()),
        "test_fraud": int((y_te == 1).sum()),
        "valid_fraud": int((y_va == 1).sum()),
        "rows_after_smote": int(len(y_bal)),
        "smote_strategy": str(smote_strategy),
        "scaled": bool(apply_scaling),
        "original_fraud_rate": float(y.mean()),
        "balanced_fraud_rate": float(np.mean(y_bal)),
    }

    return SplitBundle(
        X_train=X_tr_s, y_train=y_tr,
        X_valid=X_va_s, y_valid=y_va,
        X_test=X_te_s, y_test=y_te,
        X_train_balanced=X_bal, y_train_balanced=y_bal,
        scaler=scaler,
        feature_names=list(X.columns),
        stats=stats,
    )


def smote_manifold_diagnostics(X_syn: np.ndarray, X_real_fraud: np.ndarray) -> dict:
    """Quantify how far SMOTE samples drift from the real fraud manifold.

    Real fraud transactions are rare and often non-compact, so synthetic points
    can land off-manifold and dilute precision. These diagnostics are surfaced in
    the notebook to justify keeping threshold-based decisions on the real test set.
    """
    real_mean, syn_mean = X_real_fraud.mean(axis=0), X_syn.mean(axis=0)
    real_std = X_real_fraud.std(axis=0) + 1e-9
    standardized_shift = (syn_mean - real_mean) / real_std

    rng = np.random.default_rng(0)
    syn_sample = X_syn[rng.choice(len(X_syn), size=min(500, len(X_syn)), replace=False)].astype(np.float64)
    real_sample = X_real_fraud[
        rng.choice(len(X_real_fraud), size=min(500, len(X_real_fraud)), replace=False)
    ].astype(np.float64)
    distances = np.linalg.norm(syn_sample[:, None, :] - real_sample[None, :, :], axis=2).min(axis=1)

    return {
        "mean_standardized_shift": float(np.abs(standardized_shift).mean()),
        "max_standardized_shift": float(np.abs(standardized_shift).max()),
        "median_distance_to_real_fraud": float(np.median(distances)),
        "p90_distance_to_real_fraud": float(np.percentile(distances, 90)),
    }
