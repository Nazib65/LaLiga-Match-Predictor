"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    team: str = Field(..., min_length=1, examples=["Real Madrid"])
    opponent: str = Field(..., min_length=1, examples=["Barcelona"])
    match_date: _date = Field(..., examples=["2025-10-26"])
    venue: Literal["Home", "Away"] = Field(..., description="Venue from the team's perspective")
    formation: str = Field(..., examples=["4-3-3"])
    opp_formation: str = Field(..., examples=["4-4-2"])
    attendance: float | None = Field(
        None, ge=0, description="Optional. If omitted, the historical median is used."
    )

    @field_validator("team", "opponent", "formation", "opp_formation")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class OutcomeProbabilities(BaseModel):
    Loss: float
    Draw: float
    Win: float


class PredictResponse(BaseModel):
    predicted_outcome: Literal["Loss", "Draw", "Win"]
    outcome_code: int
    probabilities: OutcomeProbabilities
    model_name: str
    model_trained_at: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str | None = None
