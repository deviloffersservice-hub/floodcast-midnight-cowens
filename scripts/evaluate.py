"""Model selection: temporal holdout for known stations, station-group CV for unseen stations.

Known-station setting : train on 2012-2022, score 2023+ on the same stations.
Unseen-station setting: hold out 1/K of stations entirely, train the no-history model on the
                        rest (2012-2022), score the held-out stations on 2023+.
Writes reports/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.features import KNOWN_FEATURES, UNSEEN_FEATURES, build_features, compute_station_stats  # noqa: E402
from floodcast.metrics import score  # noqa: E402
from floodcast.model import OBJECTIVES, fit  # noqa: E402
from floodcast.schema import DATE_COL, ID_COL, TARGET  # noqa: E402

PROC = ROOT / "data" / "processed"
EARLY_STOP_FROM = "2021-01-01"
N_FOLDS = 4
KNOWN_ONLY = "--known-only" in sys.argv


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_parquet(PROC / "full_daily.parquet")
    full[ID_COL] = full[ID_COL].astype(str)
    train_mask = (full[DATE_COL] >= "2012-01-01") & (full[DATE_COL] <= "2022-12-31")
    stats = compute_station_stats(full[train_mask])
    feats = build_features(full, stats)
    is_train = feats[DATE_COL].between("2012-01-01", "2022-12-31") & feats[TARGET].notna()
    is_test = (feats[DATE_COL] > "2022-12-31") & feats[TARGET].notna()
    return feats[is_train].reset_index(drop=True), feats[is_test].reset_index(drop=True)


def climatology_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    clim = train.groupby([ID_COL, "month"])[TARGET].mean().rename("clim")
    return test.join(clim, on=[ID_COL, "month"])["clim"].fillna(train[TARGET].median()).to_numpy()


def fit_with_es(train: pd.DataFrame, features: list[str], objective: str):
    early = train[train[DATE_COL] < EARLY_STOP_FROM]
    late = train[train[DATE_COL] >= EARLY_STOP_FROM]
    probe = fit(early, features, objective, valid_feats=late)
    rounds = max(probe.booster.best_iteration, 50)
    return fit(train, features, objective, rounds=rounds), rounds


ENSEMBLES = {"ens_logspec_tweedie": ("log_specific", "tweedie"), "ens_all": tuple(OBJECTIVES)}


def add_ensembles(report: dict, prefix: str, preds: dict, test: pd.DataFrame) -> None:
    for name, members in ENSEMBLES.items():
        avg = np.mean([preds[m] for m in members], axis=0)
        report[f"{prefix}_{name}"] = score(test[TARGET], avg, test[ID_COL])
        print(prefix, name, report[f"{prefix}_{name}"], flush=True)


def main() -> None:
    train, test = load()
    report: dict = {"n_stations": int(train[ID_COL].nunique()), "n_train": len(train), "n_test": len(test)}
    report["baseline_monthly_climatology"] = score(test[TARGET], climatology_baseline(train, test), test[ID_COL])

    known_preds = {}
    for obj in OBJECTIVES:
        model, rounds = fit_with_es(train, KNOWN_FEATURES, obj)
        known_preds[obj] = model.predict(test)
        report[f"known_{obj}"] = {**score(test[TARGET], known_preds[obj], test[ID_COL]), "rounds": rounds}
        print(obj, "known", report[f"known_{obj}"], flush=True)
    add_ensembles(report, "known", known_preds, test)
    blend_frame = test[[ID_COL, DATE_COL, TARGET]].assign(
        clim=climatology_baseline(train, test), **{f"p_{k}": v for k, v in known_preds.items()})
    blend_frame.to_parquet(PROC / "known_holdout_preds.parquet", index=False)

    stations = train[ID_COL].unique()
    unseen_preds = {}
    for obj in ([] if KNOWN_ONLY else OBJECTIVES):
        preds = np.full(len(test), np.nan)
        fold_rounds = []
        for tr_idx, te_idx in GroupKFold(N_FOLDS).split(stations, groups=stations):
            tr_st, te_st = set(stations[tr_idx]), set(stations[te_idx])
            model, n_rounds = fit_with_es(train[train[ID_COL].isin(tr_st)], UNSEEN_FEATURES, obj)
            fold_rounds.append(n_rounds)
            mask = test[ID_COL].isin(te_st).to_numpy()
            preds[mask] = model.predict(test[mask])
        unseen_preds[obj] = preds
        report[f"unseen_{obj}"] = {**score(test[TARGET], preds, test[ID_COL]), "rounds": int(np.median(fold_rounds))}
        print(obj, "unseen", report[f"unseen_{obj}"], flush=True)
    if not KNOWN_ONLY:
        add_ensembles(report, "unseen", unseen_preds, test)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / ("metrics_known_only.json" if KNOWN_ONLY else "metrics.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
