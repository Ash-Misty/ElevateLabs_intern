"""Metrics, threshold selection and publication-quality figures."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PALETTE = {
    "navy": "#0b1020",
    "surface": "#131c33",
    "grid": "#25324f",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "blue": "#3b82f6",
    "violet": "#8b5cf6",
    "pink": "#ec4899",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "cyan": "#06b6d4",
}

MODEL_COLORS = {
    "XGBoost": PALETTE["blue"],
    "XGBoost (no SMOTE)": PALETTE["cyan"],
    "Logistic Regression": PALETTE["muted"],
    "Random Forest": PALETTE["green"],
    "IsolationForest": PALETTE["violet"],
    "LocalOutlierFactor": PALETTE["amber"],
    "Hybrid ensemble": PALETTE["pink"],
}

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


@dataclass
class Evaluation:
    """Threshold-free and threshold-based metrics for one model."""

    name: str
    roc_auc: float
    pr_auc: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    tn: int
    fp: int
    fn: int
    tp: int
    predicted: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "threshold": round(self.threshold, 4),
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "specificity": round(self.specificity, 6),
            "tn": int(self.tn), "fp": int(self.fp),
            "fn": int(self.fn), "tp": int(self.tp),
        }


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, min_recall: float = 0.8) -> tuple[float, float]:
    """Highest precision achievable while keeping recall >= min_recall (analyst view)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    mask = recall >= min_recall
    if not mask.any():
        return 0.0, 0.0
    best = int(np.argmax(precision[mask]))
    threshold = float(thresholds[best]) if best < len(thresholds) else 1.0
    return float(precision[mask][best]), threshold


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str = "max_f1",
    min_precision: float | None = None,
) -> tuple[float, dict]:
    """Pick an operating threshold.

    Strategies:
      - ``max_f1``    : best harmonic mean of precision/recall
      - ``max_recall`` : catch as much fraud as possible
      - ``min_precision`` : highest precision with a precision floor (fewest false alarms)
    """
    rows = []
    for threshold in THRESHOLD_GRID:
        predicted = (scores >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {"threshold": float(threshold), "precision": precision, "recall": recall,
             "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        )

    table = np.array([r["threshold"] for r in rows])
    f1s = np.array([r["f1"] for r in rows])
    precisions = np.array([r["precision"] for r in rows])
    recalls = np.array([r["recall"] for r in rows])

    if strategy == "min_precision" and min_precision is not None:
        eligible = np.where(precisions >= min_precision)[0]
        index = int(eligible[np.argmax(recalls[eligible])]) if len(eligible) else int(np.argmax(f1s))
    elif strategy == "max_recall":
        eligible = np.where(precisions > 0)[0]
        index = int(eligible[np.argmax(recalls[eligible])]) if len(eligible) else int(np.argmax(f1s))
    else:
        index = int(np.argmax(f1s))

    return float(table[index]), {k: float(v) for k, v in rows[index].items()}


def evaluate(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    strategy: str = "max_f1",
    min_precision: float | None = None,
) -> Evaluation:
    """Compute the full metric suite at a tuned or supplied threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    if threshold is None:
        threshold, _ = tune_threshold(y_true, scores, strategy=strategy, min_precision=min_precision)

    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return Evaluation(
        name=name,
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        threshold=float(threshold),
        accuracy=float((tp + tn) / max(tp + tn + fp + fn, 1)),
        precision=float(precision_score(y_true, predicted, zero_division=0)),
        recall=float(recall_score(y_true, predicted, zero_division=0)),
        f1=float(f1_score(y_true, predicted, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        predicted=predicted,
        probabilities=scores,
    )


def apply_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Evaluation:
    """Evaluate at a fixed threshold (used for IF/LOF contamination labels)."""
    return evaluate("fixed", y_true, scores, threshold=threshold)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=10)
    ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=10)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.6)


def _save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(PALETTE["navy"])
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=PALETTE["navy"])
    plt.close(fig)


def plot_class_distribution(y: np.ndarray, path) -> None:
    """Log-scale count + share panel showing the 0.17% fraud reality."""
    counts = np.bincount(y.astype(int), minlength=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    bars = ax.bar(["Legitimate", "Fraud"], counts, color=[PALETTE["blue"], PALETTE["red"]], width=0.55)
    ax.set_yscale("log")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:,}",
                ha="center", color=PALETTE["text"], fontsize=11, fontweight="bold")
    _style_axes(ax, "Transaction count (log scale)", "Class", "Transactions")

    ax = axes[1]
    ax.bar(["Legitimate", "Fraud"], counts / counts.sum() * 100,
           color=[PALETTE["blue"], PALETTE["red"]], width=0.55)
    for i, value in enumerate(counts / counts.sum() * 100):
        ax.text(i, value + 1.6, f"{value:.3f}%", ha="center",
                color=PALETTE["text"], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    _style_axes(ax, "Share of dataset (%)", "Class", "Percent")
    _save(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, predicted: np.ndarray, title: str, path) -> np.ndarray:
    """Annotated confusion matrix heatmap with counts and row-normalised shares."""
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    normalised = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = ["Legitimate", "Fraud"]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        normalised, annot=cm, fmt="d", cmap=sns.dark_palette("#3b82f6", as_cmap=True),
        xticklabels=labels, yticklabels=labels, cbar=False,
        annot_kws={"size": 17, "weight": "bold"}, linewidths=2, linecolor=PALETTE["navy"],
        ax=ax,
    )
    ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted", color=PALETTE["muted"], fontsize=10)
    ax.set_ylabel("Actual", color=PALETTE["muted"], fontsize=10)
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    _save(fig, path)
    return cm


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path) -> None:
    """Overlaid ROC curves for every scored model."""
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1.2, label="Random (AUC 0.50)")
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, PALETTE["cyan"]),
                linewidth=2.2, label=f"{name} — AUC {auc:.4f}")
    _style_axes(ax, "ROC curve — false positive rate vs true positive rate",
                "False positive rate", "True positive rate")
    legend = ax.legend(loc="lower right", fontsize=9, frameon=True)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _save(fig, path)


def plot_precision_recall_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]], baseline: float, path
) -> None:
    """PR curves with the no-skill baseline, the view that matters for rare events."""
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.axhline(baseline, linestyle="--", color=PALETTE["muted"], linewidth=1.2,
               label=f"No-skill baseline (PR-AUC {baseline:.4f})")
    for name, (precision, recall, ap) in curves.items():
        ax.plot(recall, precision, color=MODEL_COLORS.get(name, PALETTE["cyan"]),
                linewidth=2.2, label=f"{name} — AP {ap:.4f}")
    _style_axes(ax, "Precision-Recall curve (imbalanced data)", "Recall", "Precision")
    legend = ax.legend(loc="lower left", fontsize=9, frameon=True)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_threshold_tradeoff(y_true: np.ndarray, scores: np.ndarray, chosen: float, path) -> None:
    """Precision/recall/F1 as the decision threshold sweeps 0.05 → 0.95."""
    thresholds, precision, recall, f1 = [], [], [], []
    for threshold in THRESHOLD_GRID:
        predicted = (scores >= threshold).astype(int)
        thresholds.append(threshold)
        precision.append(precision_score(y_true, predicted, zero_division=0))
        recall.append(recall_score(y_true, predicted, zero_division=0))
        f1.append(f1_score(y_true, predicted, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds, precision, color=PALETTE["blue"], linewidth=2, label="Precision")
    ax.plot(thresholds, recall, color=PALETTE["pink"], linewidth=2, label="Recall")
    ax.plot(thresholds, f1, color=PALETTE["green"], linewidth=2.4, label="F1")
    ax.axvline(chosen, color=PALETTE["amber"], linestyle="--", linewidth=1.8,
               label=f"Operating point = {chosen:.2f}")
    _style_axes(ax, "Threshold trade-off on the held-out test set", "Decision threshold", "Score")
    legend = ax.legend(fontsize=9, frameon=True)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_feature_importance(model, feature_names: list[str], path, top: int = 15) -> None:
    """XGBoost gain-based importance for the most influential features."""
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1][:top][::-1]
    colors = [MODEL_COLORS.get("XGBoost", PALETTE["blue"]) if v >= importance[order].mean() else PALETTE["muted"]
              for v in importance[order]]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    ax.barh([feature_names[i] for i in order], importance[order], color=colors)
    _style_axes(ax, f"XGBoost feature importance (top {len(order)})", "Gain", "Feature")
    _save(fig, path)


def plot_anomaly_scatter(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    title: str,
    path,
    limit: int = 6000,
    seed: int = 42,
) -> None:
    """PCA projection of transactions coloured by detector score and true label."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    legit_idx = np.flatnonzero(y == 0)
    fraud_idx = np.flatnonzero(y == 1)
    sample = np.concatenate([
        rng.choice(legit_idx, size=min(limit, len(legit_idx)), replace=False),
        fraud_idx,
    ])
    coords = PCA(n_components=2, random_state=seed).fit_transform(X[sample].astype(np.float64))
    labels, score_sample = y[sample], scores[sample]

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    legit_mask = labels == 0
    ax.scatter(coords[legit_mask, 0], coords[legit_mask, 1], s=9, alpha=0.35,
               c=score_sample[legit_mask], cmap="coolwarm", vmin=0, vmax=1,
               linewidths=0, label="Legitimate")
    ax.scatter(coords[~legit_mask, 0], coords[~legit_mask, 1], s=64, alpha=0.95,
               c=PALETTE["red"], edgecolors="white", linewidths=0.6, label="Fraud", zorder=5)
    _style_axes(ax, title, "PC1", "PC2")
    legend = ax.legend(fontsize=9, frameon=True, loc="best")
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_model_comparison(evaluations: list[Evaluation], path) -> None:
    """Grouped bar chart of ROC-AUC / PR-AUC / F1 / recall across models."""
    names = [e.name for e in evaluations]
    metrics = {
        "ROC-AUC": [e.roc_auc for e in evaluations],
        "PR-AUC": [e.pr_auc for e in evaluations],
        "F1": [e.f1 for e in evaluations],
        "Recall": [e.recall for e in evaluations],
        "Precision": [e.precision for e in evaluations],
    }
    x = np.arange(len(names))
    width = 0.16

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.7), 5.2))
    for i, (label, values) in enumerate(metrics.items()):
        ax.bar(x + (i - 2) * width, values, width,
               label=label, color=[PALETTE["blue"], PALETTE["violet"], PALETTE["green"],
                                   PALETTE["pink"], PALETTE["amber"]][i], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=16, ha="right", fontsize=9.5)
    _style_axes(ax, "Model comparison at each model's operating threshold", "", "Score")
    ax.set_ylim(0, 1.12)
    legend = ax.legend(fontsize=9, frameon=True, ncol=5, loc="upper right")
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_score_distributions(
    scores_by_model: dict[str, np.ndarray], y_true: np.ndarray, path
) -> None:
    """Legitimate vs fraud score histograms, clipped to the 99th percentile for readability."""
    fig, axes = plt.subplots(1, len(scores_by_model), figsize=(4.4 * len(scores_by_model), 4.2), squeeze=False)
    for ax, (name, scores) in zip(axes[0], scores_by_model.items()):
        hi = float(np.percentile(scores, 99)) or 1.0
        bins = np.linspace(0, hi, 70)
        ax.hist(scores[y_true == 0], bins=bins, color=PALETTE["blue"],
                alpha=0.65, label="Legitimate", density=True)
        fraud_scores = scores[y_true == 1]
        if len(fraud_scores) and np.ptp(fraud_scores[: len(fraud_scores)]) > 0:
            ax.hist(fraud_scores, bins=bins, color=PALETTE["red"],
                    alpha=0.75, label="Fraud", density=True)
        else:
            ax.hist(fraud_scores, bins=bins, color=PALETTE["red"], alpha=0.75, label="Fraud")
        _style_axes(ax, f"{name} score distribution", "Score (0-1)", "Density")
        legend = ax.legend(fontsize=8, frameon=True)
        legend.get_frame().set_facecolor(PALETTE["surface"])
        legend.get_frame().set_edgecolor(PALETTE["grid"])
        for text in legend.get_texts():
            text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_smote_comparison(
    X_legit: np.ndarray, X_fraud: np.ndarray, X_synthetic: np.ndarray, path, seed: int = 42
) -> None:
    """Real legitimate, real fraud and SMOTE-synthesised fraud in a joint PCA view."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)

    def take(array: np.ndarray, n: int) -> np.ndarray:
        if len(array) == 0:
            return array[:0]
        return array[rng.choice(len(array), size=min(n, len(array)), replace=False)]

    legit = take(np.asarray(X_legit, dtype=np.float32), 4000)
    fraud = take(np.asarray(X_fraud, dtype=np.float32), min(1500, len(X_fraud)))
    synthetic = take(np.asarray(X_synthetic, dtype=np.float32), 4000)

    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(np.vstack([legit, fraud, synthetic]))
    n_legit, n_fraud = len(legit), len(fraud)
    a, b = n_legit, n_legit + n_fraud

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(projected[:a, 0], projected[:a, 1], s=11, alpha=0.35,
               c=PALETTE["blue"], linewidths=0, label="Real legitimate")
    ax.scatter(projected[a:b, 0], projected[a:b, 1], s=58, alpha=0.95,
               c=PALETTE["red"], edgecolors="white", linewidths=0.5, label="Real fraud")
    ax.scatter(projected[b:, 0], projected[b:, 1], s=16, alpha=0.45,
               c=PALETTE["amber"], linewidths=0, label="SMOTE synthetic fraud")
    _style_axes(
        ax,
        "SMOTE fills the fraud region between real fraud samples (PCA view)",
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
    )
    legend = ax.legend(fontsize=8.5, frameon=True)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def plot_amount_distribution(df, path, target: str = "Class") -> None:
    """Amount distributions for legitimate vs fraud transactions."""
    legit = df.loc[df[target] == 0, "Amount"]
    fraud = df.loc[df[target] == 1, "Amount"]
    hi = float(np.percentile(legit, 99.5))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(0, hi, 60)
    ax.hist(legit, bins=bins, color=PALETTE["blue"], alpha=0.6, density=True, label="Legitimate")
    if len(fraud) and float(np.ptp(fraud)) > 0:
        ax.hist(fraud, bins=bins, color=PALETTE["red"], alpha=0.75, density=True, label="Fraud")
    else:
        ax.hist(fraud, bins=bins, color=PALETTE["red"], alpha=0.75, label="Fraud")
    _style_axes(ax, "Transaction amount: fraud skews to higher values", "Amount (clipped at p99.5)", "Density")
    legend = ax.legend(fontsize=9, frameon=True)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    legend.get_frame().set_edgecolor(PALETTE["grid"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    _save(fig, path)


def apply_dark_theme() -> None:
    """Apply the shared dark plotting theme across the notebook."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor": PALETTE["navy"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["navy"],
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["muted"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "axes.edgecolor": PALETTE["grid"],
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.size": 11,
        "figure.dpi": 110,
    })
