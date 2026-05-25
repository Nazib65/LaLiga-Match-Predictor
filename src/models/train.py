"""Train the production XGBoost model and persist it to results/final_model.pkl.

This is a slimmed, reproducible version of the winning configuration found
in notebooks/hyper_para_tuning.ipynb (XGBoost + Optuna), so the same model
the API serves can be rebuilt with a single command:

    python -m src.models.train
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from src import config

log = logging.getLogger(__name__)

# Best params found by hyper_para_tuning.ipynb (XGBoost + Optuna, 50 trials).
# Pinned so re-training is deterministic. To re-tune, re-run that notebook
# and update these values.
BEST_XGB_PARAMS = {
    "n_estimators": 493,
    "max_depth": 5,
    "learning_rate": 0.04947455333172755,
    "subsample": 0.63197823231335,
    "colsample_bytree": 0.7435557078373036,
    "min_child_weight": 8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

TRAIN_FRACTION = 0.8


def train(
    engineered_csv: Path = config.ENGINEERED_CSV,
    out_path: Path = config.FINAL_MODEL_PKL,
) -> dict:
    """Train XGBoost on the engineered dataset and write a model package."""
    log.info("Loading engineered data from %s", engineered_csv)
    df = pd.read_csv(engineered_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[config.MODEL_FEATURE_COLS].copy()
    y = df["outcome"].map(config.OUTCOME_TO_XGB)  # XGBoost wants {0,1,2}

    split_idx = int(len(df) * TRAIN_FRACTION)
    split_date = df["date"].iloc[split_idx]
    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    log.info(
        "Train: %d rows (%s → %s)  Test: %d rows (%s → %s)",
        len(X_train), df.loc[train_mask, "date"].min().date(), df.loc[train_mask, "date"].max().date(),
        len(X_test), df.loc[test_mask, "date"].min().date(), df.loc[test_mask, "date"].max().date(),
    )

    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, y_pred_proba)
    log.info("Test accuracy=%.4f  f1_macro=%.4f  log_loss=%.4f", acc, f1_macro, ll)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pkg = {
        "model": model,
        "model_name": "XGBoost (Optuna)",
        "feature_cols": config.MODEL_FEATURE_COLS,
        "best_params": BEST_XGB_PARAMS,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "log_loss": float(ll),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    joblib.dump(pkg, out_path)
    log.info("Saved model package to %s", out_path)
    return pkg


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
