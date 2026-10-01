"""Central configuration for the FraudGuard credit-card fraud detection project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

REAL_DATASET = DATASET_DIR / "creditcard.csv"
SYNTHETIC_DATASET = DATASET_DIR / "creditcard_synthetic.csv"

FEATURES: list[str] = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"
KAGGLE_COLUMNS: list[str] = [*FEATURES, TARGET]
CONTAMINATION = 0.00172
RANDOM_STATE = 42
LOF_N_NEIGHBORS = 20
LOF_FIT_SAMPLE = 60_000
MIN_PRECISION = 0.75
BLEND_GRID = [round(0.05 * i, 2) for i in range(1, 20)]

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { background: radial-gradient(circle at 12% -10%, #1b2a4a 0%, #0b1020 45%, #070a14 100%); }
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.hero {
  padding: 2.1rem 2.4rem; border-radius: 22px; margin-bottom: 1.1rem;
  background: linear-gradient(120deg, #2563eb 0%, #7c3aed 52%, #db2777 100%);
  box-shadow: 0 18px 44px rgba(37, 99, 235, 0.32);
}
.hero h1 { color: #fff; margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { color: rgba(255,255,255,0.90); margin: 0.45rem 0 0; font-size: 1.02rem; }
.hero .chips { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.chip {
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  color: #fff; padding: 0.22rem 0.7rem; border-radius: 999px; font-size: 0.76rem; font-weight: 600;
}
.card {
  background: rgba(17, 24, 43, 0.78); border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 18px; padding: 1.15rem 1.3rem; margin-bottom: 0.9rem;
  backdrop-filter: blur(6px);
}
.card h4 { margin: 0 0 0.35rem 0; color: #e2e8f0; font-size: 0.98rem; font-weight: 700; }
.card p { margin: 0; color: #94a3b8; font-size: 0.85rem; line-height: 1.5; }
.section-title {
  color: #f1f5f9; font-size: 1.32rem; font-weight: 700; margin: 1.4rem 0 0.2rem 0;
  border-left: 4px solid #7c3aed; padding-left: 0.7rem;
}
.muted { color: #94a3b8; font-size: 0.88rem; }
.verdict-Low { color: #22c55e; } .verdict-Medium { color: #eab308; }
.verdict-High { color: #f97316; } .verdict-Critical { color: #ef4444; }
[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800; }
div[data-testid="stSidebar"] { background: rgba(11, 16, 32, 0.92); }
</style>
"""


@dataclass(frozen=True)
class TrainConfig:
    """Runtime knobs for the training pipeline."""

    test_size: float = 0.20
    valid_size: float = 0.10
    random_state: int = RANDOM_STATE
    smote_strategy: float = 1.0
    contamination: float = CONTAMINATION
    n_estimators: int = 700
    max_depth: int = 6
    learning_rate: float = 0.08
    n_jobs: int = -1
    max_rows: int | None = None
    lof_sample: int = LOF_FIT_SAMPLE
    use_lof: bool = True
    train_variants: bool = True
    features: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURES))
    target: str = TARGET
