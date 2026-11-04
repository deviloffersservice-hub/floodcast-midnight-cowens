"""FLOODCAST — Midnight Cowens. Train the model and/or write predictions.csv.

Usage
-----
  python solution.py train                                  # train on data/processed/train.csv
  python solution.py predict --test data/processed/test.csv # -> predictions.csv
  python solution.py predict --test test.csv --train train.csv --out predictions.csv

`--train` is optional at prediction time. When the organizers' train.csv is supplied, its
weather rows warm up the rolling-rainfall windows and its discharge gives per-station
climatology for stations our model has never seen. Test discharge is never read.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from floodcast.features import KNOWN_FEATURES, UNSEEN_FEATURES, build_features, compute_station_stats  # noqa: E402
from floodcast.model import OBJECTIVES, Ensemble, FloodcastModel, fit  # noqa: E402
from floodcast.schema import ID_COL, TARGET, make_submission, read_table, validate_submission  # noqa: E402

MODEL_DIR = ROOT / "models"
PROC = ROOT / "data" / "processed"
# Boosting rounds per objective, from early stopping on 2021-2022 (scripts/evaluate.py).
KNOWN_ROUNDS = {"log_l2": 300, "log_specific": 250, "tweedie": 300}
UNSEEN_ROUNDS = {"log_l2": 300, "log_specific": 250, "tweedie": 300}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("floodcast")


def train(train_path: Path, rounds_scale: float = 1.0) -> FloodcastModel:
    data = read_table(train_path, require_target=True)
    data = data[data[TARGET].notna() & (data[TARGET] >= 0)]
    warm_path = PROC / "warmup_weather.parquet"
    stats = compute_station_stats(data)
    frame = data.assign(_lab=True)
    if warm_path.exists():
        warm = pd.read_parquet(warm_path).assign(_lab=False)
        warm[ID_COL] = warm[ID_COL].astype(str)
        frame = pd.concat([warm, frame], ignore_index=True)
    feats = build_features(frame, stats)
    feats = feats[feats["_lab"].astype(bool)].reset_index(drop=True)
    log.info("training on %d rows, %d stations", len(feats), feats[ID_COL].nunique())
    known = Ensemble(tuple(fit(feats, KNOWN_FEATURES, o, rounds=int(KNOWN_ROUNDS[o] * rounds_scale))
                           for o in OBJECTIVES))
    unseen = Ensemble(tuple(fit(feats, UNSEEN_FEATURES, o, rounds=int(UNSEEN_ROUNDS[o] * rounds_scale))
                            for o in OBJECTIVES))
    model = FloodcastModel(known, unseen, stats)
    model.save(MODEL_DIR)
    log.info("saved model to %s", MODEL_DIR)
    return model


def predict(test_path: Path, out_path: Path, train_path: Path | None) -> pd.DataFrame:
    test = read_table(test_path)
    history = read_table(train_path, require_target=False) if train_path else None
    model = FloodcastModel.load(MODEL_DIR)
    pred = model.predict(test, history)
    out = make_submission(test, pred)
    validate_submission(out, test)
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    log.info("wrote %d predictions to %s", len(out), out_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", type=Path, default=PROC / "train.csv")
    t.add_argument("--rounds-scale", type=float, default=1.0, help="multiply boosting rounds")
    p = sub.add_parser("predict")
    p.add_argument("--test", type=Path, default=PROC / "test.csv")
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.train, args.rounds_scale)
    else:
        predict(args.test, args.out, args.train)


if __name__ == "__main__":
    main()
