"""Inference service.

Loads the trained model package and the historical engineered dataset
(used as a feature store for last-5-match rolling stats), and exposes a
single `Predictor` class with a `predict(match)` method.

The historical dataset is loaded once at construction; the same instance
is reused across API requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.features import load_encoding_lookup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchInput:
    """Minimal fields a caller must supply to request a prediction."""

    team: str
    opponent: str
    match_date: _date
    venue: str  # "Home" or "Away" — from team's perspective
    formation: str
    opp_formation: str
    attendance: float | None = None


class Predictor:
    """Load model + feature store once, serve predictions repeatedly."""

    def __init__(
        self,
        model_path: Path = config.FINAL_MODEL_PKL,
        history_path: Path = config.ENGINEERED_CSV,
        encodings_path: Path = config.LABEL_ENCODINGS_CSV,
    ) -> None:
        log.info("Loading model from %s", model_path)
        pkg = joblib.load(model_path)
        self.model = pkg["model"]
        self.feature_cols: list[str] = pkg.get("feature_cols", config.MODEL_FEATURE_COLS)
        self.model_meta = {k: v for k, v in pkg.items() if k not in {"model"}}

        log.info("Loading match history from %s", history_path)
        hist = pd.read_csv(history_path)
        hist["date"] = pd.to_datetime(hist["date"])
        self.history = hist.sort_values(["team", "date"]).reset_index(drop=True)

        self.encodings = load_encoding_lookup(encodings_path)
        log.info("Predictor ready: %d historical rows, encodings for %s",
                 len(self.history), list(self.encodings))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize_team(name: str) -> str:
        return config.TEAM_NAME_MAPPING.get(name, name)

    def _encode(self, column: str, value: str) -> int:
        """Look up the encoded id; raise a clear error if the value is unseen."""
        table = self.encodings.get(column, {})
        if str(value) not in table:
            known = ", ".join(sorted(table)[:10])
            raise ValueError(
                f"Unknown {column!r}={value!r}. Model has not seen this value. "
                f"Examples of known values: {known}..."
            )
        return table[str(value)]

    def _team_rolling(self, team: str, before: pd.Timestamp) -> dict[str, float]:
        """Compute last-N-match rolling stats for `team` strictly before `before`."""
        window = config.ROLLING_WINDOW
        team_hist = self.history[
            (self.history["team"] == team) & (self.history["date"] < before)
        ].tail(window)
        if team_hist.empty:
            raise ValueError(
                f"No historical matches for team={team!r} before {before.date()}. "
                f"Cannot compute rolling features."
            )

        # The engineered dataset already stores avg_*_5; we average those over
        # the most recent rows to approximate the rolling state going into the
        # next match. Using the latest row directly is equivalent when N>=window.
        latest = team_hist.iloc[-1]
        rolling = {col: float(latest[col]) for col in [
            "avg_gf_5", "avg_ga_5", "avg_xg_5", "avg_xga_5", "avg_poss_5",
            "avg_sh_5", "avg_sot_5", "avg_dist_5", "avg_fk_5", "avg_pk_5", "avg_pkatt_5",
            "avg_goal_diff", "avg_xgoal_diff", "avg_shot_acc", "avg_pk_acc",
            "form_5", "points_5",
        ]}
        return rolling

    def _default_attendance(self) -> float:
        return float(self.history["attendance"].median())

    # --------------------------------------------------------------- predict
    def build_feature_row(self, match: MatchInput) -> pd.DataFrame:
        team = self._normalize_team(match.team)
        opponent = self._normalize_team(match.opponent)
        match_dt = pd.Timestamp(match.match_date)

        rolling = self._team_rolling(team, before=match_dt)

        venue = match.venue.capitalize()
        if venue not in {"Home", "Away"}:
            raise ValueError(f"venue must be 'Home' or 'Away', got {match.venue!r}")

        attendance = match.attendance if match.attendance is not None else self._default_attendance()

        row: dict[str, Any] = {
            "season": match_dt.year if match_dt.month >= 7 else match_dt.year - 1,
            "year": match_dt.year,
            "month": match_dt.month,
            "team_encoded": self._encode("team", team),
            "opponent_encoded": self._encode("opponent", opponent),
            "venue_encoded": self._encode("venue", venue),
            "formation_encoded": self._encode("formation", match.formation),
            "opp formation_encoded": self._encode("opp formation", match.opp_formation),
            "is_home": 1 if venue == "Home" else 0,
            "attendance": attendance,
            "day_of_week_encoded": self._encode("day_of_week", match_dt.day_name()),
            **rolling,
        }
        return pd.DataFrame([[row[c] for c in self.feature_cols]], columns=self.feature_cols)

    def predict(self, match: MatchInput) -> dict[str, Any]:
        X = self.build_feature_row(match)
        proba = self.model.predict_proba(X)[0]
        pred_xgb = int(np.argmax(proba))
        outcome = config.XGB_TO_OUTCOME[pred_xgb]
        return {
            "predicted_outcome": config.OUTCOME_TO_LABEL[outcome],
            "outcome_code": outcome,
            "probabilities": {
                config.OUTCOME_TO_LABEL[config.XGB_TO_OUTCOME[i]]: float(p)
                for i, p in enumerate(proba)
            },
            "model_name": self.model_meta.get("model_name", "XGBoost"),
            "model_trained_at": self.model_meta.get("trained_at"),
        }
