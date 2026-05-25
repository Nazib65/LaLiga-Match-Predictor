# La Liga predictor — operational shortcuts.
# Works in PowerShell, bash, and CI runners (no fancy shell features).

PY ?= python
PIP ?= pip
IMAGE ?= laliga-predictor
PORT ?= 8000

.PHONY: help install preprocess features train pipeline serve test docker docker-run clean

help:
	@echo "Targets:"
	@echo "  install      install pinned dependencies"
	@echo "  preprocess   raw_data -> Processed/cleaned_data.csv"
	@echo "  features     cleaned -> engineered_data.csv + label_encodings_data.csv"
	@echo "  train        train XGBoost -> results/final_model.pkl"
	@echo "  pipeline     preprocess + features + train"
	@echo "  serve        run FastAPI on :$(PORT)"
	@echo "  test         pytest tests/"
	@echo "  docker       build the docker image"
	@echo "  docker-run   run the image on :$(PORT)"
	@echo "  clean        remove caches"

install:
	$(PIP) install -r requirements.txt

preprocess:
	$(PY) -m src.cli preprocess

features:
	$(PY) -m src.cli features

train:
	$(PY) -m src.cli train

pipeline:
	$(PY) -m src.cli pipeline

serve:
	$(PY) -m src.cli serve --host 0.0.0.0 --port $(PORT)

test:
	$(PY) -m pytest -q

docker:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 --name $(IMAGE) $(IMAGE)

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ *.egg-info
