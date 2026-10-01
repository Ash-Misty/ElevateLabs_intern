"""FraudGuard — Flask web application (HTML UI + REST API).

Launch with::

    python app/flask_app.py            # http://127.0.0.1:5000

Routes
------
``GET  /``                analyst dashboard
``GET  /score``           single-transaction scoring form
``POST /score``           render the verdict
``GET  /batch``           CSV batch screening
``POST /api/predict``     JSON scoring of one transaction
``POST /api/batch``       JSON scoring of many transactions
``GET  /api/metrics``     training report
``GET  /api/simulate``    threshold sweep on the held-out test fold
``GET  /api/health``      liveness probe
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudguard.evaluate import evaluate  # noqa: E402
from fraudguard.inference import FraudDetector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

_detector: FraudDetector | None = None
_metrics: dict | None = None
_scores: dict | None = None


def detector() -> FraudDetector:
    """Lazily load the trained bundle once per worker process."""
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector


def training_metrics() -> dict:
    global _metrics
    if _metrics is None:
        path = REPORT_DIR / "metrics.json"
        _metrics = __import__("json").loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _metrics


def test_scores() -> dict:
    global _scores
    if _scores is None:
        path = REPORT_DIR / "test_scores.npz"
        if path.exists():
            with np.load(path) as data:
                _scores = {key: data[key] for key in data.files}
        else:
            _scores = {}
    return _scores


@app.context_processor
def inject_globals() -> dict:
    return {"feature_names": detector().feature_names, "app_version": detector().version}


def _request_transaction(form) -> dict:
    """Pull the 30 model features out of a submitted form, ignoring junk keys."""
    transaction = {}
    for feature in detector().feature_names:
        raw = form.get(feature, "0")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        transaction[feature] = value if np.isfinite(value) else 0.0
    return transaction


@app.route("/")
def dashboard():
    """Landing dashboard: headline metrics, curves, leaderboard."""
    metrics = training_metrics()
    models = metrics.get("models", [])
    primary = next((m for m in models if m["name"] == "XGBoost"), {})
    best = max(models, key=lambda m: m["f1"], default={})
    data = training_metrics().get("dataset", {})
    return render_template(
        "index.html",
        primary=primary,
        best=best,
        models=models,
        dataset=data,
        figure_dir=FIGURE_DIR,
        threshold=detector().threshold,
    )


@app.route("/score", methods=["GET", "POST"])
def score():
    """Score one transaction and explain the verdict."""
    model = detector()
    result = None
    transaction = {feature: 0.0 for feature in model.feature_names}
    if request.method == "POST":
        transaction = _request_transaction(request.form)
        result = model.score(transaction).to_dict()
    else:
        transaction["Time"] = 41_280.0
        transaction["Amount"] = 120.0
    return render_template("score.html", result=result, transaction=transaction)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Screen an uploaded CSV of transactions."""
    model = detector()
    table, summary = None, None
    error = None
    if request.method == "POST":
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            error = "Choose a CSV file first."
        else:
            try:
                frame = pd.read_csv(io.BytesIO(upload.read()))
                missing = [c for c in model.feature_names if c not in frame.columns]
                if missing:
                    error = f"Missing columns: {', '.join(missing)}"
                else:
                    scored = model.score_dataframe(frame)
                    frame = frame.copy()
                    for column in scored.columns:
                        frame[column] = scored[column]
                    table = frame.head(500)
                    summary = {
                        "rows": len(frame),
                        "flagged": int(scored["is_fraud"].sum()),
                        "flag_rate": float(scored["is_fraud"].mean()),
                        "mean_risk": float(scored["risk_score"].mean()),
                        "peak_risk": float(scored["risk_score"].max()),
                    }
                    buffer = io.StringIO()
                    scored.assign(
                        Time=frame["Time"].iloc[: len(scored)].to_numpy(),
                        Amount=frame["Amount"].iloc[: len(scored)].to_numpy(),
                    ).to_csv(buffer, index=False)
                    summary["_download"] = buffer.getvalue()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                error = f"Could not read that file: {exc}"
    return render_template("batch.html", table=table, summary=summary, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Score one transaction supplied as JSON."""
    payload = request.get_json(silent=True) or {}
    model = detector()
    missing = [f for f in model.feature_names if f not in payload]
    if missing:
        return jsonify({
            "error": "missing features",
            "missing": missing,
            "hint": "Send all 30 features: Time, V1-V28, Amount.",
        }), 400
    try:
        return jsonify(model.score(payload).to_dict())
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Score many transactions supplied as JSON or a CSV upload."""
    model = detector()
    frame: pd.DataFrame | None = None
    if request.files.get("file"):
        frame = pd.read_csv(io.BytesIO(request.files["file"].read()))
    else:
        payload = request.get_json(silent=True) or {}
        rows = payload.get("transactions", payload if isinstance(payload, list) else [])
        if rows:
            frame = pd.DataFrame(rows)
    if frame is None or frame.empty:
        return jsonify({"error": "no transactions supplied"}), 400
    missing = [c for c in model.feature_names if c not in frame.columns]
    if missing:
        return jsonify({"error": "missing features", "missing": missing}), 400
    return jsonify(model.score_dataframe(frame).to_dict(orient="records"))


@app.route("/api/metrics")
def api_metrics():
    """Full training report, as produced by `python -m fraudguard.train`."""
    return jsonify(training_metrics())


@app.route("/api/simulate")
def api_simulate():
    """Recompute metrics at an arbitrary threshold on the held-out test fold."""
    data = test_scores()
    if not data:
        return jsonify({"error": "no test scores available"}), 404
    source = request.args.get("model", "xgb")
    if source not in data:
        return jsonify({"error": f"unknown score source '{source}'", "available": list(data)}), 400
    try:
        threshold = float(request.args.get("threshold", detector().threshold))
    except ValueError:
        return jsonify({"error": "threshold must be a number"}), 400

    summary = evaluate("sim", data["y_test"], data[source], threshold=threshold)
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(data["y_test"], data[source])
    payload = summary.to_dict()
    payload.pop("predicted", None)
    payload["curve"] = {
        "precision": [round(float(v), 5) for v in precision[:: max(1, len(precision) // 200)]],
        "recall": [round(float(v), 5) for v in recall[:: max(1, len(recall) // 200)]],
    }
    payload["baseline"] = float(data["y_test"].mean())
    return jsonify(payload)


@app.route("/api/curves")
def api_curves():
    """ROC and PR curve points for every model, for the dashboard charts."""
    data = test_scores()
    if not data:
        return jsonify({"error": "no test scores available"}), 404
    from sklearn.metrics import precision_recall_curve, roc_curve

    y_true = data["y_test"]
    out: dict = {"roc": {}, "pr": {}, "baseline": float(y_true.mean())}
    for key, scores in data.items():
        if key == "y_test":
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        step = max(1, len(fpr) // 250)
        out["roc"][key] = {
            "fpr": [round(float(v), 5) for v in fpr[::step]],
            "tpr": [round(float(v), 5) for v in tpr[::step]],
        }
        out["pr"][key] = {
            "precision": [round(float(v), 5) for v in precision[::step]],
            "recall": [round(float(v), 5) for v in recall[::step]],
        }
    return jsonify(out)


@app.route("/api/health")
def api_health():
    """Liveness probe used by the container healthcheck."""
    return jsonify({"status": "ok", "version": detector().version, "threshold": detector().threshold})


@app.route("/download")
def download():
    """Return the last scored batch as a CSV attachment."""
    payload = request.args.get("data")
    if not payload:
        return "no data", 400
    return send_file(
        io.BytesIO(payload.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraudguard_scored.csv",
    )


@app.route("/figures/<path:name>")
def figure(name: str):
    """Serve a generated figure from the artifacts directory."""
    target = (FIGURE_DIR / name).resolve()
    if not str(target).startswith(str(FIGURE_DIR.resolve())) or not target.exists():
        return "not found", 404
    return send_file(target, mimetype="image/png", max_age=3600)


@app.errorhandler(413)
def too_large(_error):
    return render_template("error.html", code=413, message="That file is larger than the 32 MB limit."), 413


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="That page does not exist."), 404


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"FraudGuard v{detector().version} running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
