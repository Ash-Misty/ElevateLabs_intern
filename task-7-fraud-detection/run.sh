#!/usr/bin/env bash
# One-command setup and launch for FraudGuard.
#   ./run.sh setup      install dependencies
#   ./run.sh train      train every model and write the artifacts
#   ./run.sh web        train if needed, then serve the Flask app on :5000
#   ./run.sh streamlit  serve the Streamlit dashboard on :8501
#   ./run.sh notebook   train if needed, then open JupyterLab
#   ./run.sh test       run the smoke tests
set -euo pipefail

COMMAND="${1:-web}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/../venv/bin/python"
[[ -x "${VENV_PYTHON}" ]] || VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

cd "${PROJECT_ROOT}"
BUNDLE="artifacts/models/fraud_pipeline.joblib"
export PYTHONPATH="${PROJECT_ROOT}/src"

train_if_needed() {
  if [[ ! -f "${BUNDLE}" ]]; then
    echo "No trained model found, training first..."
    "${VENV_PYTHON}" -m fraudguard.train
  fi
}

case "${COMMAND}" in
  setup)     "${VENV_PYTHON}" -m pip install --upgrade pip && "${VENV_PYTHON}" -m pip install -r requirements.txt ;;
  train)     "${VENV_PYTHON}" -m fraudguard.train ;;
  web)       train_if_needed && echo "Flask app on http://127.0.0.1:5000" && "${VENV_PYTHON}" app/flask_app.py 5000 ;;
  streamlit) train_if_needed && "${VENV_PYTHON}" -m streamlit run app/streamlit_app.py --server.port 8501 ;;
  notebook)  train_if_needed && "${VENV_PYTHON}" -m jupyter lab --notebook-dir notebooks ;;
  test)      "${VENV_PYTHON}" -m unittest discover -s tests -v ;;
  *) echo "Unknown command '${COMMAND}'. Use: setup | train | web | streamlit | notebook | test" && exit 1 ;;
esac
