"""FastAPI app exposing the prediction service.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   – liveness + model-loaded check
    POST /predict  – return outcome probabilities for an upcoming match
    GET  /         – brief landing JSON
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    OutcomeProbabilities,
    PredictRequest,
    PredictResponse,
)
from src.models.predict import MatchInput, Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("laliga.api")

# Mutable state attached to the app via lifespan.
_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    try:
        _predictor = Predictor()
        log.info("Predictor loaded successfully")
    except Exception as exc:  # pragma: no cover - logged and re-raised
        log.exception("Failed to load predictor: %s", exc)
        _predictor = None
    yield
    _predictor = None


app = FastAPI(
    title="La Liga Match Outcome Predictor",
    version="1.0.0",
    description="Predicts Win/Draw/Loss for an upcoming La Liga match.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "laliga-predictor", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _predictor is not None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_name=(_predictor.model_meta.get("model_name") if loaded else None),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = _predictor.predict(
            MatchInput(
                team=req.team,
                opponent=req.opponent,
                match_date=req.match_date,
                venue=req.venue,
                formation=req.formation,
                opp_formation=req.opp_formation,
                attendance=req.attendance,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - last-resort safety
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc

    return PredictResponse(
        predicted_outcome=result["predicted_outcome"],
        outcome_code=result["outcome_code"],
        probabilities=OutcomeProbabilities(**result["probabilities"]),
        model_name=result["model_name"],
        model_trained_at=result.get("model_trained_at"),
    )
