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
from floodcast.model import Ensemble, FloodcastModel, fit  # noqa: E402
from floodcast.schema import (  # noqa: E402
    ID_COL, TARGET, make_submission, read_table, validate_history, validate_submission, validate_test_input,
)

MODEL_DIR = ROOT / "models"
PROC = ROOT / "data" / "processed"
# Ensemble members: objective -> (boosting rounds, LightGBM overrides of model.BASE_PARAMS).
# Rounds from early stopping; the overrides won on both time folds (2012-19 -> 2020-22 and
# 2012-22 -> 2023-25) in a capacity search, for known stations and in station-group CV.
KNOWN_MEMBERS = {
    "log_l2": (1550, {"num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.6}),
    "log_specific": (447, {}),
    "tweedie": (1320, {}),
}
UNSEEN_MEMBERS = {"log_specific": (330, {"num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.6})}
# Median-to-mean correction for stations without a flow record (station-group CV, chosen on
# two time folds: 2012-19 -> 2020-22 and 2012-22 -> 2023-25; both favoured 1.25-1.30).
UNSEEN_SCALE = 1.25

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("floodcast")


def train(train_path: Path, rounds_scale: float = 1.0, warm_path: Path = PROC / "warmup_weather.parquet") -> FloodcastModel:
    data = read_table(train_path, require_target=True)
    data = data[data[TARGET].notna() & (data[TARGET] >= 0)]
    stats = compute_station_stats(data)
    frame = data.assign(_lab=True)
    if warm_path.exists():
        warm = pd.read_parquet(warm_path).assign(_lab=False)
        warm[ID_COL] = warm[ID_COL].astype(str)
        frame = pd.concat([warm, frame], ignore_index=True)
    feats = build_features(frame, stats)
    feats = feats[feats["_lab"].astype(bool)].reset_index(drop=True)
    log.info("training on %d rows, %d stations", len(feats), feats[ID_COL].nunique())
    known = Ensemble(tuple(fit(feats, KNOWN_FEATURES, o, rounds=int(n * rounds_scale), params=p)
                           for o, (n, p) in KNOWN_MEMBERS.items()))
    unseen = Ensemble(tuple(fit(feats, UNSEEN_FEATURES, o, rounds=int(n * rounds_scale), params=p)
                            for o, (n, p) in UNSEEN_MEMBERS.items()))
    model = FloodcastModel(known, unseen, stats, UNSEEN_SCALE)
    model.save(MODEL_DIR)
    log.info("saved model to %s", MODEL_DIR)
    return model


def predict(test_path: Path, out_path: Path, train_path: Path | None) -> pd.DataFrame:
    test = read_table(test_path)
    validate_test_input(test)
    history = read_table(train_path, require_target=False) if train_path else None
    if history is not None:
        validate_history(history, test)
    model = FloodcastModel.load(MODEL_DIR)
    pred = model.predict(test, history)
    out = make_submission(test, pred)
    checks = validate_submission(out, test)
    log.info("submission checks passed: %s", checks)
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    log.info("wrote %d predictions to %s", len(out), out_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", type=Path, default=PROC / "train.csv")
    t.add_argument("--rounds-scale", type=float, default=1.0, help="multiply boosting rounds")
    t.add_argument("--warmup", type=Path, default=PROC / "warmup_weather.parquet",
                   help="weather before the first training day (warms up rolling windows)")
    p = sub.add_parser("predict")
    p.add_argument("--test", type=Path, default=PROC / "test.csv")
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.train, args.rounds_scale, args.warmup)
    else:
        predict(args.test, args.out, args.train)


if __name__ == "__main__":
    main()
