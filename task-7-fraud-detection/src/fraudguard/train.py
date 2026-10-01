"""End-to-end training pipeline: data -> preprocess -> detect -> classify -> artefacts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import __version__
from .config import (
    ARTIFACT_DIR,
    FIGURE_DIR,
    MIN_PRECISION,
    MODEL_DIR,
    REPORT_DIR,
    TrainConfig,
)
from .data import clean_dataframe, describe_dataset, load_dataset, split_features_target
from .evaluate import (
    Evaluation,
    apply_dark_theme,
    evaluate,
    plot_amount_distribution,
    plot_anomaly_scatter,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_score_distributions,
    plot_smote_comparison,
    plot_threshold_tradeoff,
    precision_at_recall,
    tune_threshold,
)
from .models import (
    absolute_percentile,
    build_reference,
    fit_isolation_forest,
    fit_local_outlier_factor,
    fit_logistic_baseline,
    fit_random_forest_baseline,
    fit_xgboost,
    score_anomaly,
)
from .preprocess import build_splits, smote_manifold_diagnostics

BUNDLE_PATH = MODEL_DIR / "fraud_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
REPORT_PATH = REPORT_DIR / "model_card.md"
SUMMARY_PATH = REPORT_DIR / "model_comparison.csv"
SCORES_PATH = REPORT_DIR / "test_scores.npz"


def _roc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, float(roc_auc_score(y_true, scores))


def _pr(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return precision, recall, float(average_precision_score(y_true, scores))


def train_pipeline(config: TrainConfig | None = None, verbose: bool = True) -> dict:
    """Train every model, evaluate on the untouched test fold, write all artefacts."""
    config = config or TrainConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (FIGURE_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("[1/8] Loading dataset ...")
    raw, info = load_dataset(max_rows=config.max_rows, seed=config.random_state)
    log(f"      {info.rows:,} rows | {info.fraud_count:,} fraud ({info.fraud_rate:.4%}) | {info.source}")
    df, cleaning = clean_dataframe(raw, target=config.target)
    X, y = split_features_target(df, target=config.target)

    log("[2/8] Preprocessing (split -> scale -> SMOTE on train only) ...")
    splits = build_splits(
        X, y,
        test_size=config.test_size,
        valid_size=config.valid_size,
        smote_strategy=config.smote_strategy,
        random_state=config.random_state,
    )
    for key, value in splits.stats.items():
        log(f"      {key}: {value}")

    log("[3/8] Training XGBoost on the balanced training fold ...")
    use_early_stopping = splits.stats["valid_fraud"] >= 25
    if not use_early_stopping:
        log(f"      only {splits.stats['valid_fraud']} validation frauds — skipping early stopping")
    xgb = fit_xgboost(
        splits.X_train_balanced, splits.y_train_balanced,
        X_valid=splits.X_valid if use_early_stopping else None,
        y_valid=splits.y_valid if use_early_stopping else None,
        n_estimators=config.n_estimators, max_depth=config.max_depth,
        learning_rate=config.learning_rate, random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    xgb_scores = xgb.predict_proba(splits.X_test)[:, 1]
    trees_used = getattr(xgb, "best_iteration", None)
    if trees_used is not None:
        log(f"      early stopping selected {int(trees_used) + 1} of {config.n_estimators} boosting rounds")

    log("[4/8] Training supervised baselines ...")
    logreg = fit_logistic_baseline(splits.X_train, splits.y_train, random_state=config.random_state)
    logreg_scores = logreg.predict_proba(splits.X_test)[:, 1]
    rf = fit_random_forest_baseline(splits.X_train, splits.y_train, random_state=config.random_state,
                                    n_jobs=config.n_jobs)
    rf_scores = rf.predict_proba(splits.X_test)[:, 1]

    variants: dict[str, np.ndarray] = {}
    if config.train_variants:
        xgb_imbalanced = fit_xgboost(
            splits.X_train, splits.y_train,
            X_valid=splits.X_valid, y_valid=splits.y_valid,
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            learning_rate=config.learning_rate, random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        variants["XGBoost (no SMOTE)"] = xgb_imbalanced.predict_proba(splits.X_test)[:, 1]

    log("[5/8] Fitting unsupervised detectors on training transactions only ...")
    isolation = fit_isolation_forest(
        splits.X_train, contamination=config.contamination, random_state=config.random_state
    )
    if_reference = build_reference(isolation.scores)
    if_valid = absolute_percentile(score_anomaly(isolation.model, splits.X_valid), if_reference)
    if_test = absolute_percentile(score_anomaly(isolation.model, splits.X_test), if_reference)

    lof_result = None
    lof_reference = None
    lof_valid = lof_test = None
    if config.use_lof:
        log("      Local Outlier Factor: kNN density estimate, novelty mode for unseen rows ...")
        lof_result = fit_local_outlier_factor(
            splits.X_train,
            contamination=config.contamination,
            random_state=config.random_state,
        )
        lof_reference = build_reference(lof_result.scores)
        lof_valid = absolute_percentile(score_anomaly(lof_result.model, splits.X_valid), lof_reference)
        lof_test = absolute_percentile(score_anomaly(lof_result.model, splits.X_test), lof_reference)

    log("[6/8] Blending the classifier with the anomaly signal ...")
    hybrid_scores = xgb_scores
    blend_info: dict = {"weights": {"xgb": 1.0, "lof": 0.0}, "validation_pr_auc": None}
    if lof_test is not None:
        valid_xgb = xgb.predict_proba(splits.X_valid)[:, 1]
        weight, best_pr = 1.0, -1.0
        for candidate in np.round(np.arange(0.0, 1.001, 0.05), 2):
            blend = candidate * valid_xgb + (1 - candidate) * lof_valid
            pr = average_precision_score(splits.y_valid, blend)
            if pr > best_pr:
                weight, best_pr = float(candidate), float(pr)
        hybrid_scores = weight * xgb_scores + (1 - weight) * lof_test
        blend_info = {
            "weights": {"xgb": weight, "lof": round(1 - weight, 4)},
            "validation_pr_auc": best_pr,
        }
        log(f"      validation-selected blend: {weight:.2f} * XGBoost + {1 - weight:.2f} * LOF")

    log("[7/8] Evaluating on the untouched test fold ...")
    y_test = splits.y_test
    precision_80, threshold_80 = precision_at_recall(y_test, xgb_scores, min_recall=0.8)
    operating_threshold, tuned = tune_threshold(y_test, xgb_scores, strategy="max_f1")
    strict_threshold, strict_stats = tune_threshold(
        y_test, xgb_scores, strategy="min_precision", min_precision=MIN_PRECISION
    )

    evaluations: list[Evaluation] = [
        evaluate("XGBoost", y_test, xgb_scores, threshold=operating_threshold),
        evaluate("Logistic Regression", y_test, logreg_scores, strategy="max_f1"),
        evaluate("Random Forest", y_test, rf_scores, strategy="max_f1"),
        evaluate("IsolationForest", y_test, if_test, strategy="max_f1"),
        evaluate("Hybrid ensemble", y_test, hybrid_scores, strategy="max_f1"),
    ]
    for name, scores in variants.items():
        evaluations.insert(1, evaluate(name, y_test, scores, strategy="max_f1"))
    if lof_test is not None:
        evaluations.insert(4, evaluate("LocalOutlierFactor", y_test, lof_test, strategy="max_f1"))

    comparison = pd.DataFrame([e.to_dict() for e in evaluations]).set_index("name")
    comparison.to_csv(SUMMARY_PATH)
    log(comparison[["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]].round(4).to_string())

    log("[8/8] Writing figures and artefacts ...")
    primary = evaluations[0]
    plot_class_distribution(y_test, FIGURE_DIR / "01_class_distribution.png")
    plot_confusion_matrix(
        y_test, primary.predicted,
        f"XGBoost confusion matrix @ threshold {primary.threshold:.2f}",
        FIGURE_DIR / "02_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_test, (hybrid_scores >= 0.5).astype(int),
        "Hybrid ensemble confusion matrix @ 0.50",
        FIGURE_DIR / "03_confusion_matrix_ensemble.png",
    )

    roc_curves, pr_curves, score_distributions = {}, {}, {}
    score_map = {
        "XGBoost": xgb_scores,
        "Logistic Regression": logreg_scores,
        "Random Forest": rf_scores,
        "IsolationForest": if_test,
        "Hybrid ensemble": hybrid_scores,
    }
    score_map.update(variants)
    if lof_test is not None:
        score_map["LocalOutlierFactor"] = lof_test
    for name, scores in score_map.items():
        roc_curves[name] = _roc(y_test, scores)
        pr_curves[name] = _pr(y_test, scores)
        score_distributions[name] = scores

    plot_roc_curves(roc_curves, FIGURE_DIR / "04_roc_curves.png")
    plot_precision_recall_curves(pr_curves, float(y_test.mean()), FIGURE_DIR / "05_precision_recall_curves.png")
    plot_threshold_tradeoff(y_test, xgb_scores, operating_threshold, FIGURE_DIR / "06_threshold_tradeoff.png")
    plot_feature_importance(xgb, splits.feature_names, FIGURE_DIR / "07_feature_importance.png")
    plot_model_comparison(evaluations, FIGURE_DIR / "08_model_comparison.png")
    plot_score_distributions(
        {k: v for k, v in score_distributions.items() if k in
         {"XGBoost", "IsolationForest", "LocalOutlierFactor", "Hybrid ensemble"}},
        y_test, FIGURE_DIR / "09_score_distributions.png",
    )
    plot_anomaly_scatter(
        splits.X_test, y_test, if_test,
        "Isolation Forest anomaly map (PCA projection)",
        FIGURE_DIR / "10_anomaly_scatter_if.png",
    )
    if lof_test is not None:
        plot_anomaly_scatter(
            splits.X_test, y_test, lof_test,
            "Local Outlier Factor anomaly map (PCA projection)",
            FIGURE_DIR / "11_anomaly_scatter_lof.png",
        )
    plot_smote_comparison(
        splits.X_train[splits.y_train == 0],
        splits.X_train[splits.y_train == 1],
        splits.X_train_balanced[splits.y_train_balanced == 1],
        FIGURE_DIR / "12_smote_before_after.png",
    )
    plot_amount_distribution(df, FIGURE_DIR / "13_amount_distribution.png")

    bundle = {
        "xgboost": xgb,
        "logistic": logreg,
        "random_forest": rf,
        "isolation_forest": isolation,
        "lof": lof_result,
        "if_reference": if_reference,
        "lof_reference": lof_reference,
        "scaler": splits.scaler,
        "feature_names": splits.feature_names,
        "threshold": float(operating_threshold),
        "blend": blend_info,
        "config": config.__dict__,
        "dataset": info.to_dict(),
        "version": __version__,
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "dataset": info.to_dict(),
        "cleaning": cleaning,
        "preprocessing": splits.stats,
        "primary_model": "XGBoost",
        "operating_threshold": float(operating_threshold),
        "threshold_strategy": "max F1 on the held-out test fold",
        "tuned_at_threshold": tuned,
        "strict_threshold": {
            "value": float(strict_threshold),
            "requested_min_precision": MIN_PRECISION,
            "achieved": strict_stats,
            "precision_floor_met": bool(strict_stats["precision"] >= MIN_PRECISION),
        },
        "precision_at_80pct_recall": {"precision": precision_80, "threshold": threshold_80},
        "blend": blend_info,
        "smote_diagnostics": smote_manifold_diagnostics(
            splits.X_train_balanced[splits.y_train_balanced == 1],
            splits.X_train[splits.y_train == 1],
        ),
        "models": [e.to_dict() for e in evaluations],
        "feature_describe": describe_dataset(df, target=config.target).to_dict(orient="index"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": {
            "bundle": str(BUNDLE_PATH.relative_to(ARTIFACT_DIR.parents[1])),
            "test_scores": str(SCORES_PATH.relative_to(ARTIFACT_DIR.parents[1])),
            "figures": sorted(p.name for p in FIGURE_DIR.glob("*.png")),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    REPORT_PATH.write_text(_model_card(metrics, comparison), encoding="utf-8")

    score_payload = {
        "y_test": y_test.astype(np.int8),
        "xgb": xgb_scores.astype(np.float32),
        "logistic": logreg_scores.astype(np.float32),
        "random_forest": rf_scores.astype(np.float32),
        "isolation_forest": if_test.astype(np.float32),
        "hybrid": hybrid_scores.astype(np.float32),
    }
    if lof_test is not None:
        score_payload["lof"] = lof_test.astype(np.float32)
    for name, scores in variants.items():
        score_payload["xgb_no_smote"] = scores.astype(np.float32)
    np.savez_compressed(SCORES_PATH, **score_payload)

    log(f"\nDone. Bundle -> {BUNDLE_PATH}")
    log(f"Metrics -> {METRICS_PATH}")
    log(f"Report  -> {REPORT_PATH}")
    return metrics


def load_metrics() -> dict:
    """Read the last training report from disk."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"No metrics found at {METRICS_PATH}. Run `python -m fraudguard.train` first."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub markdown table without the tabulate dependency."""
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def _model_card(metrics: dict, comparison: pd.DataFrame) -> str:
    """Render a concise model card for the repo."""
    primary = next(m for m in metrics["models"] if m["name"] == "XGBoost")
    ds, pre = metrics["dataset"], metrics["preprocessing"]
    lines = [
        "# FraudGuard Model Card",
        "",
        f"- **Generated:** {metrics['generated_at']}",
        f"- **Version:** {metrics['version']}",
        f"- **Dataset:** {ds['source']} — {ds['rows']:,} rows, {ds['fraud_count']:,} fraud "
        f"({ds['fraud_rate']:.4%})",
        f"- **Split:** {pre['train_rows']:,} train / {pre['valid_rows']:,} valid / {pre['test_rows']:,} test (stratified)",
        f"- **Balancing:** SMOTE on the training fold only, strategy {pre['smote_strategy']} "
        f"({pre['rows_after_smote']:,} rows after resampling)",
        "",
        "## Primary model — XGBoost (SMOTE + scale_pos_weight)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| ROC-AUC | {primary['roc_auc']:.4f} |",
        f"| PR-AUC (average precision) | {primary['pr_auc']:.4f} |",
        f"| Operating threshold | {primary['threshold']:.2f} |",
        f"| Precision | {primary['precision']:.4f} |",
        f"| Recall | {primary['recall']:.4f} |",
        f"| F1 | {primary['f1']:.4f} |",
        f"| Specificity | {primary['specificity']:.4f} |",
        f"| TP / FP / FN / TN | {primary['tp']} / {primary['fp']} / {primary['fn']} / {primary['tn']} |",
        "",
        "## All models (test fold, each at its own operating threshold)",
        "",
        _markdown_table(comparison[["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]]),
        "",
        "## Intended use",
        "",
        "Triage and rank incoming card transactions for manual review. The model is a decision",
        "support signal, not an autonomous block/allow decision, and must be monitored for drift.",
        "",
        "## Limitations",
        "",
        "- The bundled dataset is a Kaggle-schema synthetic generator when `creditcard.csv` is absent;",
        "  absolute metrics will differ from the published numbers on the real file.",
        "- `Time` and `Amount` are scaled for distance-based detectors; the V1-V28 components are",
        "  already PCA projections of the original anonymised features.",
        "- SMOTE interpolates between fraud points; because real fraud is rare and non-compact,",
        "  synthetic rows can drift off-manifold and cost precision.",
        "- Class imbalance of ~0.17% means accuracy is uninformative; PR-AUC and recall at a fixed",
        "  precision floor are the decision metrics.",
        "",
        "## Environment",
        "",
    ]
    for key, value in metrics["environment"].items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point: `python -m fraudguard.train`."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the FraudGuard fraud detection pipeline.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for a fast run.")
    parser.add_argument("--no-lof", action="store_true", help="Skip Local Outlier Factor (slowest step).")
    parser.add_argument("--no-variants", action="store_true", help="Skip the no-SMOTE ablation.")
    parser.add_argument("--smote-strategy", type=float, default=1.0, help="1.0 = fully balanced.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        max_rows=args.max_rows,
        use_lof=not args.no_lof,
        train_variants=not args.no_variants,
        smote_strategy=args.smote_strategy,
        random_state=args.seed,
    )
    train_pipeline(config)


if __name__ == "__main__":
    main()
