# La Liga Match Predictor

Predicts the outcome (Win / Draw / Loss) of an upcoming La Liga match from
team identities, formations, venue, attendance, and the home team's recent
form. Trained on fbref data spanning **2019‑08 → 2025‑09** (4,644 matches).

The production model is **XGBoost** tuned with Optuna — test-set accuracy
**~49.8 %**, macro-F1 **~0.45** vs a majority-class baseline of 36.9 %.

---

## Quick start

### 1. Install
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### 2. Serve the API
A trained model ships in `results/final_model.pkl`, so the server runs out of the box:
```bash
python -m src.cli serve --port 8000
# or:  uvicorn src.api.main:app --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

### 3. Hit `/predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "team": "Real Madrid",
    "opponent": "Barcelona",
    "match_date": "2025-10-26",
    "venue": "Home",
    "formation": "4-3-3",
    "opp_formation": "4-4-2",
    "attendance": 78000
  }'
```

Response:
```json
{
  "predicted_outcome": "Win",
  "outcome_code": 1,
  "probabilities": {"Loss": 0.18, "Draw": 0.24, "Win": 0.58},
  "model_name": "XGBoost (Optuna)",
  "model_trained_at": "2026-05-24T12:00:00Z"
}
```

### 4. Docker
```bash
docker build -t laliga-predictor .
docker run --rm -p 8000:8000 laliga-predictor
```

---

## Re-training

Everything below is reproducible from `raw_data/matches_laliga.csv`:

```bash
python -m src.cli preprocess   # raw_data -> Processed/cleaned_data.csv
python -m src.cli features     # cleaned -> engineered_data.csv + label_encodings_data.csv
python -m src.cli train        # train XGBoost -> results/final_model.pkl
# or all three at once:
python -m src.cli pipeline
```

Hyperparameters are pinned in [src/models/train.py](src/models/train.py)
(`BEST_XGB_PARAMS`) from the Optuna search in
[notebooks/hyper_para_tuning.ipynb](notebooks/hyper_para_tuning.ipynb). To
re-tune, re-run that notebook and update the constants.

---

## Repository layout

```
.
├── src/                       production code
│   ├── config.py              paths + modeling constants
│   ├── data/
│   │   ├── preprocess.py      raw -> cleaned
│   │   └── features.py        cleaned -> engineered
│   ├── models/
│   │   ├── train.py           train XGBoost -> results/final_model.pkl
│   │   └── predict.py         Predictor class used by the API
│   ├── api/
│   │   ├── main.py            FastAPI app
│   │   └── schemas.py         Pydantic request/response models
│   └── cli.py                 `python -m src.cli ...` entrypoint
├── notebooks/                 research notebooks (preserved, not deployed)
├── raw_data/                  fbref CSV
├── Processed/                 cleaned + engineered tables + label encodings
├── Models/                    individual model artifacts (RF, LR, XGB, NN)
├── Evaluation/                plots + classification reports
├── results/                   final tuned model + tuning artifacts
├── tests/                     pytest suite
├── Dockerfile / .dockerignore
├── Makefile
├── requirements.txt           pinned, Python 3.12
└── pyproject.toml
```

---

## API reference

| Method | Path       | Purpose                                            |
|--------|------------|----------------------------------------------------|
| GET    | `/health`  | liveness + whether the model loaded                |
| POST   | `/predict` | outcome + per-class probabilities for a match      |
| GET    | `/docs`    | Swagger UI                                         |

`/predict` validates inputs with Pydantic. Unknown teams / formations
return **422** with a human-readable error.

---

## How the predictor builds features

Rolling form features (`avg_*_5`, `form_5`, `points_5`) for the home team
are read from `Processed/engineered_data.csv`, which acts as a feature
store. To incorporate new matches, re-run the pipeline; the feature store
is regenerated and the API picks it up on the next start.

Categorical fields (`team`, `opponent`, `venue`, `formation`,
`opp formation`, `day_of_week`) use the label encodings persisted in
`Processed/label_encodings_data.csv`. Unseen values are rejected at
inference rather than silently mis-encoded.

---

## Tests

```bash
pytest -q
```

The suite covers feature engineering determinism, the encoding lookup,
and a happy-path round-trip through `Predictor`.

---

## Notes / caveats

- The model was trained on La Liga matches only — predictions for other
  competitions are not supported.
- Class accuracy is best for Win/Loss; Draw recall is intrinsically low
  (a known property of soccer outcome prediction, not a bug).
- `attendance` is optional in `/predict`; the historical median is used
  when not supplied.
