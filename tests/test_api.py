"""Smoke tests for the FastAPI app via TestClient (in-process, no socket)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not config.FINAL_MODEL_PKL.exists():
        pytest.skip("results/final_model.pkl not present — run `python -m src.cli train` first")
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client: TestClient):
    eng = pd.read_csv(config.ENGINEERED_CSV)
    eng["date"] = pd.to_datetime(eng["date"])
    row = eng.sort_values("date").iloc[-1]
    payload = {
        "team": row["team"],
        "opponent": row["opponent"],
        "match_date": row["date"].date().isoformat(),
        "venue": "Home" if row["is_home"] else "Away",
        "formation": "4-3-3",
        "opp_formation": "4-4-2",
        "attendance": float(row["attendance"]),
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_outcome"] in {"Win", "Draw", "Loss"}
    assert set(body["probabilities"]) == {"Win", "Draw", "Loss"}


def test_predict_rejects_unknown_team(client: TestClient):
    r = client.post("/predict", json={
        "team": "Nonexistent FC",
        "opponent": "Barcelona",
        "match_date": "2025-05-01",
        "venue": "Home",
        "formation": "4-3-3",
        "opp_formation": "4-4-2",
        "attendance": 50000,
    })
    assert r.status_code == 422
