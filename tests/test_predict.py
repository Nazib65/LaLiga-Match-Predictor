"""Round-trip test for the Predictor.

Picks a real (team, opponent, date) triple from the engineered dataset
and asks the model to predict it. We don't assert on the predicted class
(the model is ~50% accurate); we only assert the response shape and that
probabilities sum to 1.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from src import config
from src.models.predict import MatchInput, Predictor


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    if not config.FINAL_MODEL_PKL.exists():
        pytest.skip("results/final_model.pkl not present — run `python -m src.cli train` first")
    return Predictor()


@pytest.fixture(scope="module")
def sample_match() -> MatchInput:
    eng = pd.read_csv(config.ENGINEERED_CSV)
    eng["date"] = pd.to_datetime(eng["date"])
    # Pick a row whose team has at least 5 prior matches in the dataset.
    eng = eng.sort_values("date")
    row = eng.iloc[-1]  # latest match — guaranteed to have history
    return MatchInput(
        team=row["team"],
        opponent=row["opponent"],
        match_date=row["date"].date(),
        venue="Home" if row["is_home"] else "Away",
        formation="4-3-3",
        opp_formation="4-4-2",
        attendance=float(row["attendance"]),
    )


def test_predictor_returns_valid_distribution(predictor: Predictor, sample_match: MatchInput):
    result = predictor.predict(sample_match)
    assert result["predicted_outcome"] in {"Win", "Draw", "Loss"}
    assert result["outcome_code"] in {-1, 0, 1}
    probs = result["probabilities"]
    assert set(probs) == {"Win", "Draw", "Loss"}
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5)


def test_unknown_team_raises(predictor: Predictor):
    bad = MatchInput(
        team="Nonexistent FC",
        opponent="Barcelona",
        match_date=date(2025, 5, 1),
        venue="Home",
        formation="4-3-3",
        opp_formation="4-4-2",
        attendance=50000,
    )
    with pytest.raises(ValueError, match="No historical matches"):
        predictor.predict(bad)
