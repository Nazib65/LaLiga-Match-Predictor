"""Raw-data preprocessing.

Deterministic port of notebooks/data_preprocessing.ipynb. Given the
fbref-style match CSV at config.RAW_MATCHES_CSV, produces config.CLEANED_CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

DROP_COLUMNS = ["Unnamed: 0", "notes", "match report"]

NUMERIC_INT_COLS = ["gf", "ga", "pk", "pkatt", "attendance", "sh", "sot", "fk", "season"]
NUMERIC_FLOAT_COLS = ["xg", "xga", "poss", "dist"]


def preprocess(
    raw_csv: Path = config.RAW_MATCHES_CSV,
    out_csv: Path = config.CLEANED_CSV,
) -> pd.DataFrame:
    """Read raw, clean, write cleaned CSV. Returns the cleaned DataFrame."""
    log.info("Loading raw data from %s", raw_csv)
    df = pd.read_csv(raw_csv)
    log.info("Raw shape: %s", df.shape)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()

    for col in NUMERIC_INT_COLS + NUMERIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # int-like cols use nullable Int64 when NaNs present, else int
    for col in NUMERIC_INT_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype("float64") if df[col].isnull().any() else df[col].astype("Int64")
            )
    for col in NUMERIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    if "attendance" in df.columns and df["attendance"].isna().any():
        median = df["attendance"].median()
        n_missing = int(df["attendance"].isna().sum())
        df["attendance"] = df["attendance"].fillna(median)
        log.info("Filled %d missing attendance values with median=%.1f", n_missing, median)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote cleaned data (%s) to %s", df.shape, out_csv)
    return df


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess()
