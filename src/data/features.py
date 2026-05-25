"""Feature engineering.

Deterministic port of notebooks/feature_engineering.ipynb. Produces the
modeling table (Processed/engineered_data.csv) and the label-encoding
reference table (Processed/label_encodings_data.csv).

The same `build_rolling_features` function is used at inference time —
the API computes last-5-match rolling stats for the home/away team from
the historical engineered dataset rather than re-engineering from raw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config

log = logging.getLogger(__name__)

KEEP_COLUMNS = [
    # identifiers
    "date", "team", "opponent", "season", "year", "month", "day_of_week",
    # target
    "outcome",
    # encoded categoricals
    "team_encoded", "opponent_encoded", "venue_encoded",
    "formation_encoded", "opp formation_encoded", "is_home",
    # rolling averages
    "avg_gf_5", "avg_ga_5", "avg_xg_5", "avg_xga_5", "avg_poss_5",
    "avg_sh_5", "avg_sot_5", "avg_dist_5", "avg_fk_5", "avg_pk_5", "avg_pkatt_5",
    # derived
    "avg_goal_diff", "avg_xgoal_diff", "avg_shot_acc", "avg_pk_acc",
    "form_5", "points_5",
    # context
    "attendance", "round", "comp", "time", "day_of_week_encoded",
]


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["team"] = df["team"].replace(config.TEAM_NAME_MAPPING)
    df["opponent"] = df["opponent"].replace(config.TEAM_NAME_MAPPING)
    return df


def build_rolling_features(df: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    """Add last-`window`-match rolling averages per team.

    Uses shift(1) so the current match is excluded — no leakage. Caller must
    have already sorted by ['team','date'].
    """
    out = df.copy()
    for feat in config.ROLLING_FEATURES:
        out[f"avg_{feat}_{window}"] = (
            out.groupby("team")[feat]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return out


def _add_derived(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df["avg_goal_diff"] = df[f"avg_gf_{window}"] - df[f"avg_ga_{window}"]
    df["avg_xgoal_diff"] = df[f"avg_xg_{window}"] - df[f"avg_xga_{window}"]
    df["avg_shot_acc"] = np.where(
        df[f"avg_sh_{window}"] > 0, df[f"avg_sot_{window}"] / df[f"avg_sh_{window}"], 0
    )
    df["avg_pk_acc"] = np.where(
        df[f"avg_pkatt_{window}"] > 0, df[f"avg_pk_{window}"] / df[f"avg_pkatt_{window}"], 0
    )

    # form (win-rate) and points over last `window`
    df["form_5"] = (
        df.groupby("team")["outcome"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).apply(lambda x: (x == 1).mean()))
        .reset_index(level=0, drop=True)
    )
    points = df.groupby("team")["outcome"].shift(1).map({1: 3, 0: 1, -1: 0})
    df["points_5"] = (
        points.groupby(df["team"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    return df


def build_features(
    cleaned_csv: Path = config.CLEANED_CSV,
    engineered_csv: Path = config.ENGINEERED_CSV,
    encodings_csv: Path = config.LABEL_ENCODINGS_CSV,
    window: int = config.ROLLING_WINDOW,
) -> pd.DataFrame:
    """Build the engineered modeling table and label-encoding reference."""
    log.info("Loading cleaned data from %s", cleaned_csv)
    df = pd.read_csv(cleaned_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = _normalize_team_names(df)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    df["outcome"] = df["result"].map({"W": 1, "D": 0, "L": -1})

    df = build_rolling_features(df, window=window)
    df = df.dropna(subset=[f"avg_{f}_{window}" for f in config.ROLLING_FEATURES])
    df = _add_derived(df, window=window)

    # categorical encodings — persist for inference
    encoders: dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if "day_of_week" in df.columns:
        le = LabelEncoder()
        df["day_of_week_encoded"] = le.fit_transform(df["day_of_week"].astype(str))
        encoders["day_of_week"] = le

    df["is_home"] = (df["venue"] == "Home").astype(int)

    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    engineered = df[keep].dropna(subset=["form_5", "points_5"]).reset_index(drop=True)

    engineered_csv.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(engineered_csv, index=False)
    log.info("Wrote engineered data (%s) to %s", engineered.shape, engineered_csv)

    rows = [
        {"column": col, "original_value": cls, "encoded_value": i}
        for col, enc in encoders.items()
        for i, cls in enumerate(enc.classes_)
    ]
    pd.DataFrame(rows).to_csv(encodings_csv, index=False)
    log.info("Wrote label encodings to %s", encodings_csv)

    return engineered


def load_encoding_lookup(path: Path = config.LABEL_ENCODINGS_CSV) -> dict[str, dict[str, int]]:
    """Return {column: {original_value: encoded_value}} from the persisted CSV."""
    enc = pd.read_csv(path)
    return {
        col: dict(zip(g["original_value"].astype(str), g["encoded_value"].astype(int)))
        for col, g in enc.groupby("column")
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_features()
