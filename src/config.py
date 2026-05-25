"""Paths and constants shared across the package.

All other modules import paths from here so the repo can be relocated
without hunting for hard-coded strings.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "Processed"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"

RAW_MATCHES_CSV = RAW_DATA_DIR / "matches_laliga.csv"
CLEANED_CSV = PROCESSED_DIR / "cleaned_data.csv"
ENGINEERED_CSV = PROCESSED_DIR / "engineered_data.csv"
LABEL_ENCODINGS_CSV = PROCESSED_DIR / "label_encodings_data.csv"

# The production model. hyper_para_tuning.ipynb writes this; the API reads it.
FINAL_MODEL_PKL = RESULTS_DIR / "final_model.pkl"

# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 5
OUTCOME_TO_LABEL = {-1: "Loss", 0: "Draw", 1: "Win"}
LABEL_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_LABEL.items()}

# XGBoost trains on {0,1,2}; we map back to {-1,0,1}.
XGB_TO_OUTCOME = {0: -1, 1: 0, 2: 1}
OUTCOME_TO_XGB = {v: k for k, v in XGB_TO_OUTCOME.items()}

ROLLING_FEATURES = [
    "gf", "ga", "xg", "xga", "poss",
    "sh", "sot", "dist", "fk", "pk", "pkatt",
]

CATEGORICAL_COLS = ["team", "opponent", "venue", "formation", "opp formation"]

# Feature order the model was trained on (matches feature_cols in
# hyper_para_tuning.ipynb). Order matters — XGBoost is positional.
MODEL_FEATURE_COLS = [
    "season", "year", "month",
    "team_encoded", "opponent_encoded", "venue_encoded",
    "formation_encoded", "opp formation_encoded", "is_home",
    "avg_gf_5", "avg_ga_5", "avg_xg_5", "avg_xga_5", "avg_poss_5",
    "avg_sh_5", "avg_sot_5", "avg_dist_5", "avg_fk_5", "avg_pk_5", "avg_pkatt_5",
    "avg_goal_diff", "avg_xgoal_diff", "avg_shot_acc", "avg_pk_acc",
    "form_5", "points_5",
    "attendance",
    "day_of_week_encoded",
]

# Standardisation of team naming, used at preprocessing and inference time.
TEAM_NAME_MAPPING = {
    "Alavés": "Alaves",
    "Atlético Madrid": "Atletico Madrid",
    "Betis": "Real Betis",
    "Cádiz": "Cadiz",
    "Málaga": "Malaga",
    "Leganés": "Leganes",
    "Almería": "Almeria",
}
