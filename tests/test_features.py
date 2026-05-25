"""Sanity checks for feature engineering helpers."""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.features import build_rolling_features, load_encoding_lookup


def test_rolling_uses_only_prior_matches():
    """The current match's value must not leak into its own rolling stat."""
    df = pd.DataFrame({
        "team": ["A"] * 6,
        "date": pd.date_range("2020-01-01", periods=6, freq="7D"),
        "gf": [1, 2, 3, 4, 5, 6],
        "ga": [0, 0, 0, 0, 0, 0],
        "xg": [0.5] * 6,
        "xga": [0.5] * 6,
        "poss": [50] * 6,
        "sh": [10] * 6,
        "sot": [3] * 6,
        "dist": [18] * 6,
        "fk": [0] * 6,
        "pk": [0] * 6,
        "pkatt": [0] * 6,
    }).sort_values(["team", "date"]).reset_index(drop=True)

    out = build_rolling_features(df, window=3)

    # Row 0 has no prior matches, so avg_gf_3 must be NaN.
    assert pd.isna(out.loc[0, "avg_gf_3"])
    # Row 3's avg over the prior 3 matches (rows 0,1,2) = mean(1,2,3) = 2.0
    assert out.loc[3, "avg_gf_3"] == 2.0


def test_label_encodings_lookup_round_trip():
    lookup = load_encoding_lookup(config.LABEL_ENCODINGS_CSV)
    assert "team" in lookup and "opponent" in lookup
    # Every encoded value must be a non-negative int.
    for col, table in lookup.items():
        assert all(isinstance(v, int) and v >= 0 for v in table.values()), col
