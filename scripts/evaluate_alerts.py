"""Alert skill and likely-range coverage on the 2023+ holdout; picks replay demo events.

Actual level    = observed discharge vs observed-flow thresholds.
Predicted level = predicted discharge vs thresholds of the chosen scheme:
  model_climate (used by the app) or observed (naive, reported for comparison).
Writes reports/alert_skill.json and reports/replay_events.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.risk import LEVELS, RiskTable, classify  # noqa: E402
from floodcast.schema import ID_COL  # noqa: E402

REPORTS, PROC = ROOT / "reports", ROOT / "data" / "processed"
RECOMMENDED = ("01646500", "07010000", "08158000", "02087500", "11447650")
N_EVENTS = 6
REPLAY_LEAD_DAYS = 4
TOLERANCE_DAYS = 2  # an alert within +/-2 days of the observed exceedance counts as on time


def _near(flags: pd.Series, days: int) -> pd.Series:
    """True where `flags` is True within +/- days (per station, daily rows sorted by date)."""
    return flags.rolling(2 * days + 1, center=True, min_periods=1).max().astype(bool)


def tolerant_skill(d: pd.DataFrame, actual: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    frame = d[[ID_COL, "date"]].assign(a=actual, p=pred, date=pd.to_datetime(d["date"])).sort_values([ID_COL, "date"])
    for k in range(1, len(LEVELS)):
        a, p = frame.a >= k, frame.p >= k
        near_p = p.groupby(frame[ID_COL]).transform(lambda s: _near(s, TOLERANCE_DAYS))
        near_a = a.groupby(frame[ID_COL]).transform(lambda s: _near(s, TOLERANCE_DAYS))
        hits, misses = int((a & near_p).sum()), int((a & ~near_p).sum())
        fa, good = int((p & ~near_a).sum()), int((p & near_a).sum())
        out[LEVELS[k]] = {"pod": hits / (hits + misses) if hits + misses else None,
                          "far": fa / (fa + good) if fa + good else None,
                          "observed_days": int(a.sum()), "alert_days": int(p.sum())}
    return out


def skill(actual: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for k in range(1, len(LEVELS)):
        a, p = actual >= k, pred >= k
        hits, misses, fa = int((a & p).sum()), int((a & ~p).sum()), int((~a & p).sum())
        out[LEVELS[k]] = {
            "observed_days": int(a.sum()), "alert_days": int(p.sum()), "hits": hits, "misses": misses,
            "false_alarms": fa,
            "pod": hits / (hits + misses) if hits + misses else None,
            "far": fa / (hits + fa) if hits + fa else None,
            "csi": hits / (hits + misses + fa) if hits + misses + fa else None,
        }
    return out


def main() -> None:
    risk = RiskTable.load(ROOT / "models")
    pred = pd.read_csv(ROOT / "predictions.csv", dtype={ID_COL: str})
    labels = pd.read_csv(PROC / "test_labels.csv", dtype={ID_COL: str})
    d = pred.merge(labels, on=[ID_COL, "date"], suffixes=("_pred", "_obs"), validate="one_to_one")
    d = d[d[ID_COL].isin(risk.model_thr[ID_COL])].reset_index(drop=True)
    ids = d[ID_COL].to_numpy()
    actual = classify(ids, d.discharge_cfs_obs.to_numpy(), risk.obs_thr)
    by_model = classify(ids, d.discharge_cfs_pred.to_numpy(), risk.model_thr)
    by_obs = classify(ids, d.discharge_cfs_pred.to_numpy(), risk.obs_thr)
    confusion = pd.crosstab(pd.Series(actual, name="actual"), pd.Series(by_model, name="predicted"))

    lo, hi = risk.bounds(ids, d.discharge_cfs_pred.to_numpy(), np.ones(len(d), dtype=bool))
    inside = (d.discharge_cfs_obs >= lo) & (d.discharge_cfs_obs <= hi)
    report = {
        "n_days": int(len(d)), "n_stations": int(d[ID_COL].nunique()),
        "model_climate_thresholds": skill(actual, by_model),
        "observed_thresholds_naive": skill(actual, by_obs),
        f"model_climate_within_{TOLERANCE_DAYS}_days": tolerant_skill(d, actual, by_model),
        "exact_level_match": float((actual == by_model).mean()),
        "within_one_level": float((np.abs(actual - by_model) <= 1).mean()),
        "likely_range_coverage_10_90": float(inside.mean()),
        "confusion_actual_rows_pred_cols": {LEVELS[int(i)]: {LEVELS[int(j)]: int(v) for j, v in row.items()}
                                            for i, row in confusion.iterrows()},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "alert_skill.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_actual_rows_pred_cols"}, indent=1))
    print(confusion)

    d = d.assign(actual=actual, predicted=by_model, date=pd.to_datetime(d["date"]))
    hits = d[(d.actual >= 2) & (d.predicted >= 2)]
    peaks = hits.loc[hits.groupby(ID_COL).discharge_cfs_obs.idxmax()]
    peaks = peaks.assign(recommended=peaks[ID_COL].isin(RECOMMENDED)).sort_values(
        ["recommended", "actual", "discharge_cfs_obs"], ascending=False).head(N_EVENTS)
    events = [{"station_id": r.station_id, "event_date": r.date.strftime("%Y-%m-%d"),
               "as_of": (r.date - pd.Timedelta(days=REPLAY_LEAD_DAYS)).strftime("%Y-%m-%d"),
               "observed_level": LEVELS[int(r.actual)], "predicted_level": LEVELS[int(r.predicted)],
               "observed_cfs": float(r.discharge_cfs_obs)} for r in peaks.itertuples()]
    (REPORTS / "replay_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(json.dumps(events, indent=1))


if __name__ == "__main__":
    main()
