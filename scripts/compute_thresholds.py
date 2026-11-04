"""Compute per-station risk thresholds and likely-range intervals (training years only).

models/thresholds.csv : obs_* thresholds from observed 2012-2022 discharge and model_*
                        thresholds from the final model's 2012-2022 predictions.
models/intervals.csv  : split-conformal 10th/90th percentile log1p residuals. Probe models
                        are trained on 2012-2020 and scored on 2021-2022, so the 2023+
                        holdout is never used. '__unseen__' comes from station-group CV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.features import KNOWN_FEATURES, UNSEEN_FEATURES, build_features, compute_station_stats  # noqa: E402
from floodcast.model import Ensemble, FloodcastModel, fit  # noqa: E402
from floodcast.risk import GLOBAL_KEY, INTERVAL_FILE, LEVELS, THRESHOLD_FILE, UNSEEN_KEY, compute_threshold_table  # noqa: E402
from floodcast.schema import DATE_COL, ID_COL, TARGET, coerce_frame  # noqa: E402
from solution import KNOWN_ROUNDS, UNSEEN_ROUNDS  # noqa: E402

MODELS, PROC = ROOT / "models", ROOT / "data" / "processed"
TRAIN_START, TRAIN_END, CALIB_START = "2012-01-01", "2022-12-31", "2021-01-01"
LOW_Q, HIGH_Q = 0.10, 0.90
MIN_CALIB_DAYS = 200
N_FOLDS = 4


def load_full() -> pd.DataFrame:
    full = pd.read_parquet(PROC / "full_daily.parquet")
    full[ID_COL] = full[ID_COL].astype(str)
    return coerce_frame(full)


def thresholds(full: pd.DataFrame, model: FloodcastModel) -> pd.DataFrame:
    period = full[DATE_COL].between(TRAIN_START, TRAIN_END)
    labelled = full[period & full[TARGET].notna()].reset_index(drop=True)
    warm = full[full[DATE_COL] < TRAIN_START].drop(columns=[TARGET])
    labelled = labelled.assign(pred=model.predict(labelled.drop(columns=[TARGET]), history=warm))
    obs = compute_threshold_table(labelled, TARGET).add_prefix("obs_").rename(columns={f"obs_{ID_COL}": ID_COL})
    mod = compute_threshold_table(labelled, "pred").add_prefix("model_").rename(columns={f"model_{ID_COL}": ID_COL})
    return obs.merge(mod, on=ID_COL)


def _residuals(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.log1p(y) - np.log1p(np.clip(p, 0, None))


def intervals(full: pd.DataFrame) -> pd.DataFrame:
    fit_rows = full[DATE_COL].between(TRAIN_START, "2020-12-31") & full[TARGET].notna()
    cal_rows = full[DATE_COL].between(CALIB_START, TRAIN_END) & full[TARGET].notna()
    stats = compute_station_stats(full[fit_rows])
    feats = build_features(full, stats)
    tr, cal = feats[fit_rows.to_numpy()], feats[cal_rows.to_numpy()].reset_index(drop=True)

    known = Ensemble(tuple(fit(tr, KNOWN_FEATURES, o, rounds=n) for o, n in KNOWN_ROUNDS.items()))
    res = cal.assign(r=_residuals(cal[TARGET].to_numpy(), known.predict(cal)))
    grp = res.groupby(ID_COL).r
    per = pd.DataFrame({"log_lo": grp.quantile(LOW_Q), "log_hi": grp.quantile(HIGH_Q), "n": grp.size()})
    per = per[per.n >= MIN_CALIB_DAYS].drop(columns="n")
    rows = [per.reset_index(), pd.DataFrame([{ID_COL: GLOBAL_KEY, "log_lo": res.r.quantile(LOW_Q),
                                              "log_hi": res.r.quantile(HIGH_Q)}])]

    stations = tr[ID_COL].unique()
    unseen_r = []
    for tr_idx, te_idx in GroupKFold(N_FOLDS).split(stations, groups=stations):
        tr_st, te_st = set(stations[tr_idx]), set(stations[te_idx])
        ens = Ensemble(tuple(fit(tr[tr[ID_COL].isin(tr_st)], UNSEEN_FEATURES, o, rounds=n)
                             for o, n in UNSEEN_ROUNDS.items()))
        part = cal[cal[ID_COL].isin(te_st)]
        unseen_r.append(_residuals(part[TARGET].to_numpy(), ens.predict(part)))
    ur = np.concatenate(unseen_r)
    rows.append(pd.DataFrame([{ID_COL: UNSEEN_KEY, "log_lo": np.quantile(ur, LOW_Q), "log_hi": np.quantile(ur, HIGH_Q)}]))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    full = load_full()
    model = FloodcastModel.load(MODELS)
    thr = thresholds(full, model)
    thr.to_csv(MODELS / THRESHOLD_FILE, index=False, encoding="utf-8")
    print(thr.head(8).round(0).to_string())
    print("stations with warning threshold:", int(thr["model_warning"].notna().sum()), "/", len(thr))
    iv = intervals(full)
    iv.to_csv(MODELS / INTERVAL_FILE, index=False, encoding="utf-8")
    print(iv.tail(4).round(3).to_string())
    print("levels:", LEVELS)


if __name__ == "__main__":
    main()
