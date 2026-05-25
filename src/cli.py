"""Command-line entrypoint.

Usage:
    python -m src.cli preprocess
    python -m src.cli features
    python -m src.cli train
    python -m src.cli pipeline       # preprocess + features + train
    python -m src.cli serve [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import logging

from src.data.features import build_features
from src.data.preprocess import preprocess
from src.models.train import train


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="laliga", description="La Liga predictor CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preprocess", help="Clean raw_data/matches_laliga.csv -> Processed/cleaned_data.csv")
    sub.add_parser("features", help="Build engineered_data.csv + label_encodings_data.csv")
    sub.add_parser("train", help="Train the production XGBoost model")
    sub.add_parser("pipeline", help="Run preprocess -> features -> train")

    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.cmd == "preprocess":
        preprocess()
    elif args.cmd == "features":
        build_features()
    elif args.cmd == "train":
        train()
    elif args.cmd == "pipeline":
        preprocess()
        build_features()
        train()
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("src.api.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
