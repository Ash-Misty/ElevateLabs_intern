"""FraudGuard — Streamlit dashboard.

Launch with::

    streamlit run app/streamlit_app.py

Serves an analyst-facing dashboard: model leaderboard, confusion matrix, ROC/PR curves,
an interactive threshold simulator, single-transaction scoring and CSV batch screening.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudguard.config import DASHBOARD_CSS, FIGURE_DIR, REPORT_DIR  # noqa: E402
from fraudguard.data import load_dataset  # noqa: E402
from fraudguard.evaluate import evaluate  # noqa: E402
from fraudguard.inference import FraudDetector  # noqa: E402

st.set_page_config(
    page_title="FraudGuard · Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

PALETTE = ["#3b82f6", "#8b5cf6", "#ec4899", "#22c55e", "#f59e0b", "#06b6d4", "#ef4444"]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(19,28,51,0.6)",
    font=dict(family="Inter, system-ui, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#25324f", zerolinecolor="#25324f"),
    yaxis=dict(gridcolor="#25324f", zerolinecolor="#25324f"),
    legend=dict(bgcolor="rgba(19,28,51,0.8)", bordercolor="#25324f", font=dict(color="#e2e8f0")),
    hoverlabel=dict(bgcolor="#131c33", bordercolor="#3b82f6", font=dict(color="#e2e8f0")),
)
SCORES_PATH = REPORT_DIR / "test_scores.npz"
METRICS_PATH = REPORT_DIR / "metrics.json"

PRESETS = {
    "Blank (model average)": None,
    "Low-value grocery": {"Time": 41_280.0, "Amount": 18.5, "V14": 0.12, "V12": 0.08, "V17": -0.05, "V10": 0.22},
    "Late-night high value": {"Time": 2_600.0, "Amount": 1_850.0, "V14": -1.9, "V12": -1.6, "V17": -1.8, "V10": -1.3},
    "Card-testing pattern": {"Time": 900.0, "Amount": 4.99, "V14": -0.8, "V12": -0.6, "V17": -0.7, "V10": -0.5},
    "Overseas cash withdrawal": {"Time": 6_400.0, "Amount": 2_400.0, "V14": -1.4, "V12": -1.1, "V17": -1.3, "V10": -1.0},
}


@st.cache_resource(show_spinner="Loading trained fraud models…")
def get_detector() -> FraudDetector:
    return FraudDetector()


@st.cache_data(show_spinner="Loading dataset profile…")
def get_profile() -> dict:
    df, info = load_dataset()
    numeric = df.drop(columns=["Class"])
    return {
        "info": info.to_dict(),
        "quantiles": numeric.quantile([0.01, 0.5, 0.99]).to_dict(),
        "columns": list(numeric.columns),
        "shape": tuple(df.shape),
    }


@st.cache_data(show_spinner="Loading test-fold scores…")
def get_scores() -> dict | None:
    if not SCORES_PATH.exists():
        return None
    with np.load(SCORES_PATH) as data:
        return {key: data[key] for key in data.files}


@st.cache_data(show_spinner="Loading training metrics…")
def get_metrics() -> dict | None:
    import json

    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


detector = get_detector()
profile = get_profile()
metrics = get_metrics()
scores = get_scores()


def hero() -> None:
    ds = profile["info"]
    chips = [
        f"🧠 XGBoost + SMOTE + scale_pos_weight",
        f"🛡️ Isolation Forest & LOF",
        f"📊 {ds['rows']:,} transactions",
        f"⚠️ {ds['fraud_rate']:.3%} fraud rate",
        f"🔖 v{detector.version}",
    ]
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="hero">
          <h1>🛡️ FraudGuard</h1>
          <p>Real-time credit card fraud detection — supervised gradient boosting layered on
             unsupervised anomaly detection, with an explainable verdict for every transaction.</p>
          <div class="chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 🎛️ Control room")
        ds = profile["info"]
        st.markdown(
            f"""
            <div class="card">
              <h4>Operating point</h4>
              <p>Flag a transaction when the blended fraud score reaches
                 <b style="color:#8b5cf6">{detector.threshold:.2f}</b>, tuned for maximum F1 on the
                 held-out test fold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="card">
              <h4>Data source</h4>
              <p>{'Real Kaggle file' if ds['is_real'] else 'Synthetic Kaggle-schema generator'}<br>
                 {ds['rows']:,} rows · {ds['fraud_count']:,} frauds</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Pipeline stages")
        for i, (label, detail) in enumerate(
            [
                ("Load & de-duplicate", "drop exact duplicates, coerce dtypes"),
                ("Stratified 70/10/20 split", "test fold reserved before any resampling"),
                ("StandardScaler fit on train", "Time & Amount are the only non-PCA features"),
                ("SMOTE on the training fold", f"{metrics['preprocessing']['rows_after_smote']:,} rows, 50/50" if metrics else ""),
                ("XGBoost + scale_pos_weight", "early stopping on validation AUC-PR"),
                ("Isolation Forest & LOF", "novelty mode, absolute percentile calibration"),
                ("Threshold tuning", "max F1, reported with precision floor"),
            ],
            start=1,
        ):
            st.markdown(f"**{i}. {label}**  \n<small style='color:#94a3b8'>{detail}</small>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("FraudGuard v%s · Streamlit" % detector.version)
    return {"threshold": detector.threshold}


def metric_row(items: list[tuple[str, str, str]]) -> None:
    """Render a row of KPI cards: (label, value, delta_or_help)."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help_text)


def page_overview() -> None:
    if metrics is None:
        st.warning("No training report found. Run `python -m fraudguard.train` first.")
        return
    primary = next(m for m in metrics["models"] if m["name"] == "XGBoost")
    best = max(metrics["models"], key=lambda m: m["f1"])

    st.markdown("#### 📈 Model performance at a glance")
    metric_row([
        ("ROC-AUC", f"{primary['roc_auc']:.4f}", "Threshold-free ranking quality"),
        ("PR-AUC", f"{primary['pr_auc']:.4f}", f"Baseline prevalence {metrics['dataset']['fraud_rate']:.4f}"),
        ("Fraud Recall", f"{primary['recall']:.1%}", f"{primary['tp']} of {primary['tp'] + primary['fn']} frauds caught"),
        ("Precision", f"{primary['precision']:.1%}", f"{primary['fp']} false alarms on {primary['tn'] + primary['fp']:,} legitimate"),
        ("Best F1", f"{best['f1']:.4f}", best["name"]),
    ])

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Confusion matrix — XGBoost")
        st.image(str(FIGURE_DIR / "02_confusion_matrix.png"), use_container_width=True)
    with right:
        st.markdown("##### Leaderboard")
        table = pd.DataFrame(metrics["models"])[
            ["name", "roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]
        ].set_index("name")
        st.dataframe(
            table.style.format("{:.4f}").background_gradient(
                cmap="viridis", subset=["roc_auc", "pr_auc", "f1"]
            ),
            use_container_width=True,
            height=252,
        )

    st.markdown("##### ROC & Precision-Recall")
    curve_col, score_col = st.columns(2)
    with curve_col:
        st.image(str(FIGURE_DIR / "04_roc_curves.png"), use_container_width=True)
    with score_col:
        st.image(str(FIGURE_DIR / "05_precision_recall_curves.png"), use_container_width=True)

    st.markdown("##### Why PR-AUC is the honest metric here")
    st.markdown(
        f"""
        <div class="card">
          <h4>Accuracy and ROC-AUC flatter a model on 0.17% positives</h4>
          <p>At a {metrics['dataset']['fraud_rate']:.3%} fraud rate, a model that flags <i>nothing</i> scores
             {(1 - metrics['dataset']['fraud_rate']):.2%} accuracy, and a randomly permuted ROC curve still
             reaches AUC 0.50. Precision-Recall curves instead compare against a no-skill
             baseline of {metrics['dataset']['fraud_rate']:.4f}, so the XGBoost PR-AUC of
             <b style="color:#22c55e">{primary['pr_auc']:.3f}</b> is a
             <b style="color:#22c55e">{primary['pr_auc'] / metrics['dataset']['fraud_rate']:.0f}×</b>
             lift over chance, not a rounding artefact.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    no_smote = next((m for m in metrics["models"] if "no SMOTE" in m["name"]), None)
    if no_smote:
        st.markdown("##### Counter-intuitive result: removing SMOTE *improved* ROC-AUC")
        chart = go.Figure()
        for i, (label, value) in enumerate(
            [("ROC-AUC", no_smote["roc_auc"]), ("PR-AUC", no_smote["pr_auc"])]
        ):
            chart.add_trace(go.Bar(
                x=[label], y=[value], name="XGBoost (no SMOTE)",
                marker_color=PALETTE[5], width=0.34, offsetgroup=0,
            ))
        for i, (label, value) in enumerate(
            [("ROC-AUC", primary["roc_auc"]), ("PR-AUC", primary["pr_auc"])]
        ):
            chart.add_trace(go.Bar(
                x=[label], y=[value], name="XGBoost (SMOTE)",
                marker_color=PALETTE[0], width=0.34, offsetgroup=1,
            ))
        chart.update_layout(barmode="group", **PLOT_LAYOUT)
        chart.update_yaxes(range=[0, 1.02], title="Score")
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            f"Balanced training data lifts PR-AUC ({primary['pr_auc']:.3f} vs {no_smote['pr_auc']:.3f}) "
            "because it stops the classifier optimising an objective dominated by negatives, while "
            "ROC-AUC rewards ranking quality that the unweighted loss already found. The business "
            "question is “how many frauds do I catch per false alarm”, which is the PR view."
        )


def _score_gauge(value: float, threshold: float, verdict: str, color: str) -> go.Figure:
    """Radial gauge showing the risk score against the operating threshold."""
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Inter"}},
        title={"text": f"Risk score · {verdict}", "font": {"size": 15, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#25324f", "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(19,28,51,0.7)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.13)"},
                {"range": [30, 60], "color": "rgba(234,179,8,0.13)"},
                {"range": [60, 85], "color": "rgba(249,115,22,0.13)"},
                {"range": [85, 100], "color": "rgba(239,68,68,0.13)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    figure.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return figure


def page_score() -> None:
    st.markdown("#### 🔎 Score a single transaction")
    st.caption(
        "V1–V28 are the PCA components of the original anonymised features; they cluster around 0. "
        "The white marker on the gauge is the model's operating threshold."
    )

    control, form = st.columns([1, 2.2])
    with control:
        st.markdown("**Presets**")
        preset_name = st.selectbox("Load an example", list(PRESETS), index=0)
        if st.button("Apply preset", use_container_width=True):
            st.session_state["_preset"] = preset_name
            st.rerun()

        if st.button("🎲 Random real transaction", use_container_width=True):
            samples = detector.sample_transactions(1, seed=int(st.session_state.get("_seed", 3)))
            st.session_state["_values"] = dict(zip(detector.feature_names, samples.iloc[0]))
            st.session_state["_seed"] = st.session_state.get("_seed", 3) + 1
            st.rerun()

        st.markdown("---")
        st.markdown("**Model breakdown**")
        st.markdown(
            f"""
            <div class="card">
              <h4>How the verdict is formed</h4>
              <p><b>0.60 ×</b> XGBoost probability (supervised)<br>
                 <b>0.25 ×</b> Isolation Forest percentile (global outlier)<br>
                 <b>0.15 ×</b> LOF percentile (local density)<br>
                 Flag when risk ≥ <b style="color:#8b5cf6">{detector.threshold:.2f}</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    values = st.session_state.get("_values", {name: 0.0 for name in detector.feature_names})
    values = {**values, **(PRESETS.get(st.session_state.get("_preset", "Blank (model average)")) or {})}

    with form:
        st.markdown("**Transaction input**")
        c1, c2, c3 = st.columns(3)
        with c1:
            time_value = st.number_input("Time (seconds)", min_value=0.0, max_value=172_800.0,
                                         value=float(values.get("Time", 41_280.0)), step=60.0)
            amount = st.number_input("Amount", min_value=0.0, max_value=25_000.0,
                                     value=float(values.get("Amount", 120.0)), step=1.0)
            st.caption(f"Time {time_value / 3600:.1f} h into the 2-day window")
        with c2:
            st.markdown("**V1 – V14**")
            grid = st.columns(2)
            inputs = {"Time": time_value, "Amount": amount}
            for index, name in enumerate([f"V{i}" for i in range(1, 15)]):
                with grid[index % 2]:
                    inputs[name] = st.number_input(
                        name, min_value=-8.0, max_value=8.0,
                        value=float(np.clip(values.get(name, 0.0), -8, 8)), step=0.1, key=f"in_{name}",
                    )
        with c3:
            st.markdown("**V15 – V28**")
            grid = st.columns(2)
            for index, name in enumerate([f"V{i}" for i in range(15, 29)]):
                with grid[index % 2]:
                    inputs[name] = st.number_input(
                        name, min_value=-8.0, max_value=8.0,
                        value=float(np.clip(values.get(name, 0.0), -8, 8)), step=0.1, key=f"in_{name}",
                    )

        submitted = st.form_submit_button("⚡ Run fraud check", type="primary", use_container_width=True)

    if not submitted and "_values" not in st.session_state:
        st.info("Press **Run fraud check** to score the transaction.")
        return

    result = detector.score(inputs)

    verdict_col, gauge_col, factors_col = st.columns([1, 1.15, 1.5])
    with verdict_col:
        st.markdown(f"##### Verdict: <span style='color:{result.color}'>{result.verdict}</span>")
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid {result.color}">
              <h4>{'🚫 Block & step-up auth' if result.verdict == 'Critical' else '✅ ' + result.recommendation}</h4>
              <p>Fraud probability <b>{result.fraud_probability:.2%}</b> ·
                 blended score <b>{result.ensemble_score:.2%}</b> ·
                 decision threshold <b>{result.threshold:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_row([
            ("Fraud prob.", f"{result.fraud_probability:.2%}", "XGBoost output"),
            ("IF anomaly", f"{result.anomaly_isolation_forest:.1%}", "global outlier percentile"),
        ])
        if result.anomaly_local_outlier_factor is not None:
            st.metric("LOF anomaly", f"{result.anomaly_local_outlier_factor:.1%}",
                      "local density percentile")
    with gauge_col:
        st.plotly_chart(_score_gauge(result.risk_score, result.threshold, result.verdict, result.color),
                        use_container_width=True)
    with factors_col:
        st.markdown("##### Why the model reacted this way")
        factors = pd.DataFrame(result.contributing_features)
        chart = go.Figure(go.Bar(
            x=factors["z_score"].abs(), y=factors["feature"], orientation="h",
            marker_color=[c if d == "high" else "#8b5cf6" for c, d in
                          zip(["#ef4444"] * len(factors), factors["direction"])],
            text=[f"{v:+.2f}σ ({d})" for v, d in zip(factors["z_score"], factors["direction"])],
            textposition="outside", hoverinfo="skip",
        ))
        chart.update_layout(height=250, **PLOT_LAYOUT)
        chart.update_layout(showlegend=False, xaxis_title="Deviation from legitimate baseline (σ)",
                            yaxis=dict(autorange="reversed", **PLOT_LAYOUT["yaxis"]))
        chart.update_xaxes(range=[0, max(3.0, float(factors["z_score"].abs().max()) * 1.35)])
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            f"Top drivers: " + ", ".join(
                f"`{f['feature']}` = {f['value']:.2f} (typical {f['typical']:.2f})"
                for f in result.contributing_features
            )
        )

    history = st.session_state.setdefault("_history", [])
    history.append({
        "Time": time_value, "Amount": amount,
        "Fraud prob.": result.fraud_probability,
        "Risk": result.risk_score,
        "Verdict": result.verdict,
    })
    st.session_state["_history"] = history[-25:]

    if len(history) > 1:
        st.markdown("##### Session scoring log")
        st.dataframe(
            pd.DataFrame(history).iloc[::-1].style.format({"Fraud prob.": "{:.2%}", "Risk": "{:.2%}"}),
            use_container_width=True, height=180,
        )


def page_batch() -> None:
    st.markdown("#### 📦 Batch screening")
    st.caption("Upload a CSV with the 30 model features (Time, V1–V28, Amount) to score many rows at once.")

    upload = st.file_uploader("Transaction CSV", type=["csv"], key="batch")
    if upload is None:
        st.markdown(
            """
            <div class="card">
              <h4>Expected schema</h4>
              <p><code>Time, V1, V2, …, V28, Amount</code> — one row per transaction.
                 Any extra columns are ignored; missing ones are rejected with a clear error.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    frame = pd.read_csv(upload)
    missing = [c for c in detector.feature_names if c not in frame.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.success(f"Loaded {len(frame):,} rows × {len(frame.columns)} columns")
    results = detector.score_dataframe(frame)
    combined = pd.concat([frame[["Time", "Amount"]].reset_index(drop=True),
                          results.drop(columns=["contributing_features"])], axis=1)

    flagged = results["is_fraud"].sum()
    metric_row([
        ("Rows scored", f"{len(frame):,}", "batch size"),
        ("Flagged fraud", f"{flagged:,}", f"{flagged / max(len(frame), 1):.2%} of the batch"),
        ("Mean risk", f"{results['risk_score'].mean():.2%}", "across all rows"),
        ("Peak risk", f"{results['risk_score'].max():.2%}", "highest scoring row"),
    ])

    if "Class" in frame.columns:
        truth = frame["Class"].astype(int).to_numpy()
        summary = evaluate("Uploaded batch", truth, results["fraud_probability"].to_numpy(),
                           threshold=detector.threshold)
        st.markdown("##### Ground truth supplied — live metrics")
        metric_row([
            ("ROC-AUC", f"{summary.roc_auc:.4f}", "ranking quality"),
            ("PR-AUC", f"{summary.pr_auc:.4f}", "precision-recall area"),
            ("Precision", f"{summary.precision:.1%}", "of flags, how many real"),
            ("Recall", f"{summary.recall:.1%}", "of frauds, how many caught"),
        ])

    st.markdown("##### Results")
    view = st.dataframe(
        combined.style.format({
            "fraud_probability": "{:.2%}", "risk_score": "{:.2%}", "ensemble_score": "{:.2%}",
            "anomaly_isolation_forest": "{:.2%}", "anomaly_local_outlier_factor": "{:.2%}",
        }),
        use_container_width=True, height=420,
    )

    st.download_button(
        "⬇️ Download scored transactions (CSV)",
        combined.to_csv(index=False).encode("utf-8"),
        file_name="fraudguard_scored.csv",
        mime="text/csv",
        use_container_width=False,
    )


def page_threshold_lab() -> None:
    st.markdown("#### 🧪 Threshold lab")
    if scores is None:
        st.warning("No test scores found. Re-run `python -m fraudguard.train` to enable the simulator.")
        return

    st.caption("Move the slider to see how the confusion matrix and metrics react on the untouched test fold.")
    y_test = scores["y_test"]
    slider_col, stats_col = st.columns([1, 1.4])
    with slider_col:
        threshold = st.slider("Decision threshold", 0.01, 0.99, float(detector.threshold), 0.01)
        model_choice = st.selectbox("Score source", ["xgb", "hybrid", "logistic", "random_forest",
                                                    "isolation_forest", "lof"])
    if model_choice not in scores:
        st.info(f"'{model_choice}' is not available in this run (LOF was skipped).")
        return

    with stats_col:
        summary = evaluate("lab", y_test, scores[model_choice], threshold=threshold)
        metric_row([
            ("Precision", f"{summary.precision:.1%}", f"{summary.tp + summary.fp:,} flagged"),
            ("Recall", f"{summary.recall:.1%}", f"{summary.tp} of {summary.tp + summary.fn} frauds"),
            ("F1", f"{summary.f1:.4f}", "harmonic mean"),
            ("Specificity", f"{summary.specificity:.2%}", f"{summary.tn:,} correctly cleared"),
        ])

    fig_col, chart_col = st.columns([1, 1.4])
    with fig_col:
        matrix = np.array([[summary.tn, summary.fp], [summary.fn, summary.tp]])
        heatmap = go.Figure(go.Heatmap(
            z=matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
            x=["Pred: Legit", "Pred: Fraud"], y=["Actual: Legit", "Actual: Fraud"],
            text=matrix, texttemplate="%{text:,}", textfont=dict(size=19, color="white"),
            colorscale=[[0, "#131c33"], [1, "#3b82f6"]], showscale=False, hoverinfo="skip",
        ))
        heatmap.update_layout(height=330, title=dict(text=f"Confusion matrix @ {threshold:.2f}",
                                                     font=dict(size=14, color="#e2e8f0")),
                              **PLOT_LAYOUT)
        st.plotly_chart(heatmap, use_container_width=True)
    with chart_col:
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_test, scores[model_choice])
        curve = go.Figure()
        curve.add_trace(go.Scatter(x=recall, y=precision, name="Precision-Recall",
                                   line=dict(color=PALETTE[0], width=3)))
        curve.add_trace(go.Scatter(x=[recall[0], recall[0]], y=[0, 1], name=f"Threshold {threshold:.2f}",
                                   line=dict(color="#f1f5f9", width=2, dash="dot")))
        curve.add_trace(go.Scatter(x=[0, 1], y=[float(y_test.mean())] * 2, name="No-skill baseline",
                                   line=dict(color="#64748b", width=1, dash="dash")))
        curve.add_trace(go.Scatter(x=[summary.recall], y=[summary.precision], mode="markers+text",
                                   name="Operating point", text=[f"{summary.f1:.2f} F1"],
                                   textposition="top center", marker=dict(size=14, color="#22c55e")))
        curve.update_layout(**PLOT_LAYOUT)
        curve.update_xaxes(title="Recall", range=[0, 1])
        curve.update_yaxes(title="Precision", range=[0, 1.02])
        st.plotly_chart(curve, use_container_width=True)

    st.markdown("##### Why this threshold is defensible")
    st.markdown(
        f"""
        <div class="card">
          <h4>Cost-aware operating point</h4>
          <p>At threshold <b>{threshold:.2f}</b> the model raises <b>{summary.tp + summary.fp}</b> alerts and
             catches <b>{summary.tp}</b> real frauds, missing <b>{summary.fn}</b>.
             A missed fraud is typically worth orders of magnitude more than a manual review, so
             operations usually lower the threshold until review capacity binds — the simulator above
             is the tool for that conversation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_anomaly() -> None:
    st.markdown("#### 🌲 Unsupervised anomaly detection")
    st.markdown(
        """
        <div class="card">
          <h4>Why also detect anomalies unsupervised?</h4>
          <p>Labels lag reality: a fraud pattern discovered today may be unlabelled for weeks.
             Isolation Forest and LOF score <i>novelty</i> without ever seeing a fraud label, so they
             surface unfamiliar attack shapes the classifier was never trained on.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_if, tab_lof, tab_feature = st.tabs(["Isolation Forest", "Local Outlier Factor", "Feature importance"])
    with tab_if:
        st.image(str(FIGURE_DIR / "10_anomaly_scatter_if.png"), use_container_width=True)
        st.markdown(
            "Each transaction is coloured by its Isolation Forest anomaly percentile, computed "
            "against the training distribution. Fraud (red) should sit in the high-percentile region."
        )
    with tab_lof:
        st.image(str(FIGURE_DIR / "11_anomaly_scatter_lof.png"), use_container_width=True)
        st.markdown(
            "LOF compares local density rather than global isolation, so it reacts to *small, tight* "
            "fraud clusters that a global forest split can miss."
        )
    with tab_feature:
        st.image(str(FIGURE_DIR / "07_feature_importance.png"), use_container_width=True)
        st.markdown(
            "Gain-based importance: V14, V12 and V17 dominate because they carry the most "
            "class-discriminative signal, matching the known structure of the Kaggle dataset."
        )


def page_about() -> None:
    st.markdown("#### 🧾 Model card")
    if metrics is not None:
        ds = metrics["dataset"]
        st.markdown(
            f"""
            <div class="card">
              <h4>Training data</h4>
              <p>{ds['source']} — {ds['rows']:,} transactions, {ds['fraud_count']:,} frauds
                 ({ds['fraud_rate']:.4%}). Generated {metrics['generated_at']}.</p>
            </div>
            <div class="card">
              <h4>Validation</h4>
              <p>{metrics['preprocessing']['train_rows']:,} train / {metrics['preprocessing']['valid_rows']:,}
                 validation / {metrics['preprocessing']['test_rows']:,} test, stratified so the
                 0.17% fraud rate is preserved in every fold. SMOTE was applied to the training
                 fold only, after splitting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        environment = metrics.get("environment", {})
        st.markdown(
            "<div class='card'><h4>Environment</h4><p>"
            + " · ".join(f"{k} {v}" for k, v in environment.items())
            + "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Interview Q&A")
    with st.expander("Why is accuracy useless for fraud detection?"):
        st.markdown(
            "At 0.172% prevalence, predicting *legitimate for everything* gives 99.83% accuracy and "
            "zero recall. The metric must reflect the cost asymmetry: **recall** for frauds you let "
            "through and **precision** for false alarms that burn review capacity. PR-AUC against a "
            "no-skill baseline of 0.00172 is the standard comparison."
        )
    with st.expander("When is Isolation Forest preferred over LOF, and vice versa?"):
        st.markdown(
            "**Isolation Forest** is global and near-linear: it randomly splits the feature space and "
            "flags points that stay isolated, so it scales to millions of rows and is robust to the "
            "sparse tail. **LOF** is local: it compares a point's density with its k nearest "
            "neighbours, so it catches *dense* fraud clusters surrounded by normal activity, which a "
            "global split can miss. LOF costs roughly O(n log n) and degrades as the feature count "
            "grows, so it is usually trained on a subsample."
        )
    with st.expander("Does SMOTE help on this dataset? What are the risks?"):
        if metrics is not None:
            primary = next(m for m in metrics["models"] if m["name"] == "XGBoost")
            no_smote = next((m for m in metrics["models"] if "no SMOTE" in m["name"]), None)
            verdict = (
                f"PR-AUC {primary['pr_auc']:.3f} with SMOTE vs {no_smote['pr_auc']:.3f} without, "
                f"at F1 {primary['f1']:.3f} vs {no_smote['f1']:.3f}."
                if no_smote else "The ablation was skipped for this run."
            )
            diagnostics = metrics.get("smote_diagnostics", {})
            st.markdown(
                f"{verdict} SMOTE guarantees the model sees the minority class during fitting, but it "
                "interpolates *between* existing fraud rows. Because real fraud is rare and "
                "non-compact, synthetic points can land off the true manifold and manufacture a "
                "region the model happily learns to trust — which shows up as a precision drop at "
                "deployment. The measured mean standardised shift between the synthetic and real "
                f"fraud centroids is {diagnostics.get('mean_standardized_shift', float('nan')):.3f}σ. "
                "`scale_pos_weight` in XGBoost achieves the same class-balancing effect through the "
                "loss function without inventing any data, which is why the ablation is reported "
                "alongside rather than assumed."
            )
    with st.expander("How do you pick the operating threshold in production?"):
        st.markdown(
            "Tune on the validation fold, then re-confirm on the test fold. Two common strategies: "
            "**max F1** (balanced, what this project ships) and **highest recall subject to a "
            "precision floor** (analyst view, since analyst minutes are the scarce resource). "
            "Re-tune whenever costs, base rates or fraud patterns shift, and always log the threshold "
            "next to the model version so old scores stay reproducible."
        )
    with st.expander("What would you monitor after deployment?"):
        st.markdown(
            "Population stability index on each feature, the realised fraud rate versus the training "
            "rate, alert volume and precision per analyst, and the share of predictions sitting in the "
            "0.4–0.6 band, which is where drift shows up first. Because the dataset drifts "
            "(PCA components are tied to a 2013 European card scheme), scheduled retraining on a "
            "rolling window is mandatory, not optional."
        )


def main() -> None:
    hero()
    sidebar()
    tabs = st.tabs([
        "📈 Overview", "🔎 Score Transaction", "📦 Batch Screening",
        "🧪 Threshold Lab", "🌲 Anomaly Detection", "🧾 Model Card",
    ])
    with tabs[0]:
        page_overview()
    with tabs[1]:
        page_score()
    with tabs[2]:
        page_batch()
    with tabs[3]:
        page_threshold_lab()
    with tabs[4]:
        page_anomaly()
    with tabs[5]:
        page_about()


if __name__ == "__main__":
    main()
