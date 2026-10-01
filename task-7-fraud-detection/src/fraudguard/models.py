"""Model zoo: anomaly detectors, XGBoost classifier and reference baselines."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier


@dataclass
class AnomalyResult:
    """Scores and labels from an unsupervised detector, plus the fitted model."""

    name: str
    scores: np.ndarray
    predictions: np.ndarray
    fit_rows: int
    model: object = field(default=None, repr=False)
    params: dict = field(default_factory=dict)


def score_anomaly(model, X: np.ndarray) -> np.ndarray:
    """Raw anomaly score for unseen rows (higher = more anomalous).

    Both detectors expose `score_samples`, where *lower* means more typical, hence
    the negation. Works on train, validation and test rows alike because both
    estimators are constructed with `novelty=True` semantics.
    """
    return -np.asarray(model.score_samples(np.asarray(X, dtype=np.float32)), dtype=np.float64)


def fit_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.00172,
    n_estimators: int = 300,
    random_state: int = 42,
    n_jobs: int = -1,
) -> AnomalyResult:
    """Isolation Forest via random axis-aligned cuts; lower score = more anomalous."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        bootstrap=False,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X)
    scores = score_anomaly(model, X)
    predictions = (model.predict(X) == -1).astype(int)
    return AnomalyResult(
        name="IsolationForest",
        scores=scores,
        predictions=predictions,
        fit_rows=int(X.shape[0]),
        model=model,
        params={"n_estimators": n_estimators, "contamination": contamination},
    )


def fit_local_outlier_factor(
    X: np.ndarray,
    contamination: float = 0.00172,
    n_neighbors: int = 20,
    random_state: int = 42,
) -> AnomalyResult:
    """LOF with `novelty=True` so it can score unseen rows after fitting.

    LOF compares each point's local density with that of its k nearest neighbours,
    which suits *local* outlier structures that a global forest split can miss.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        algorithm="auto",
    )
    model.fit(X)
    scores = score_anomaly(model, X)
    predictions = (model.predict(X) == -1).astype(int)
    return AnomalyResult(
        name="LocalOutlierFactor",
        scores=scores,
        predictions=predictions,
        fit_rows=int(X.shape[0]),
        model=model,
        params={"n_neighbors": n_neighbors, "contamination": contamination},
    )


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    n_estimators: int = 700,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    use_scale_pos_weight: bool = True,
) -> XGBClassifier:
    """Gradient-boosted trees tuned for extreme imbalance.

    `scale_pos_weight` re-weights the positive class so the loss cares about the
    ~0.17% fraud signal; SMOTE handles the representation side. Early stopping uses
    the untouched validation fold with AUC-PR as the stopping metric.
    """
    if use_scale_pos_weight and scale_pos_weight is None:
        neg = float((y_train == 0).sum())
        pos = float((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1.0)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight if use_scale_pos_weight else None,
        eval_metric="aucpr",
        tree_method="hist",
        enable_categorical=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(np.asarray(X_valid, dtype=np.float32), np.asarray(y_valid))],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def fit_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Linear baseline with balanced class weights, for comparison against XGBoost."""
    with warnings.catch_warnings():
        # sklearn 1.5 + scipy>=1.13 emit a spurious "Unknown solver options: iprint" warning.
        warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
        model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
        model.fit(np.asarray(X_train, dtype=np.float64), y_train)
    return model


def fit_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Bagged-tree baseline; slow to train but a useful sanity check."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=random_state,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    return model


def minmax_normalise(scores: np.ndarray) -> np.ndarray:
    """Scale raw anomaly scores to [0, 1] so detectors can be blended with probabilities."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores.astype(np.float64) - lo) / (hi - lo)


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Rank-normalise scores to [0, 1]; robust to heavy tails in LOF/IF outputs."""
    order = np.argsort(np.argsort(scores))
    return order / max(len(scores) - 1, 1)


def build_reference(scores: np.ndarray, n: int = 4_000, seed: int = 0) -> np.ndarray:
    """Sorted sample of training anomaly scores, kept for absolute calibration.

    A raw anomaly score only means something relative to a population. Persisting the
    training distribution lets a *single* unseen transaction be placed on that
    population's scale, instead of being ranked against whatever else is in the batch.
    """
    values = np.asarray(scores, dtype=np.float64)
    if len(values) > n:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), size=n, replace=False)]
    return np.sort(values)


def absolute_percentile(raw_scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map raw anomaly scores to their percentile within a fixed reference population."""
    if reference is None or len(reference) == 0:
        return np.zeros_like(np.asarray(raw_scores, dtype=np.float64))
    raw = np.asarray(raw_scores, dtype=np.float64)
    return np.searchsorted(reference, raw, side="right") / float(len(reference))
