"""Flood risk levels and likely-range bands for predicted discharge.

Levels (per station, from its own 2012-2022 record):
  0 normal    below the top-10 % daily flow
  1 elevated  >= top-10 % daily flow (the brief's "extreme day" definition)
  2 warning   >= 2-year flood   (Gumbel fit on annual maxima)
  3 danger    >= 5-year flood
  4 extreme   >= 20-year flood
Classification compares predicted flow with thresholds computed from the model's own
training-period predictions ("model climate", as GloFAS does), which cancels systematic
model bias. Observed-flow thresholds are kept for display.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ID_COL

LEVELS = ("normal", "elevated", "warning", "danger", "extreme")
LEVEL_COLORS = {"normal": "#2da44e", "elevated": "#d4a72c", "warning": "#fb8500",
                "danger": "#cf222e", "extreme": "#8250df", "unknown": "#8c959f"}
UNKNOWN = -1
RETURN_PERIODS = {"warning": 2, "danger": 5, "extreme": 20}
ELEVATED_QUANTILE = 0.90
MIN_YEARS = 3
MIN_THRESHOLD_CFS = 1.0  # dry/intermittent rivers: a 0 cfs threshold would flag every dry day
MIN_DAYS_PER_YEAR = 300
EULER_GAMMA = 0.5772156649
GLOBAL_KEY, UNSEEN_KEY = "__global__", "__unseen__"
THRESHOLD_FILE, INTERVAL_FILE = "thresholds.csv", "intervals.csv"


def gumbel_flow(annual_max: np.ndarray, period: float) -> float:
    """T-year flow from annual maxima, Gumbel distribution fitted by method of moments."""
    x = np.asarray(annual_max, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_YEARS:
        return float("nan")
    beta = x.std(ddof=1) * np.sqrt(6) / np.pi
    mu = x.mean() - EULER_GAMMA * beta
    return float(mu - beta * np.log(-np.log(1 - 1 / period)))


def station_thresholds(dates: pd.Series, flow: pd.Series) -> dict[str, float]:
    """Elevated/warning/danger/extreme flows for one station (monotone non-decreasing)."""
    q = pd.Series(np.asarray(flow, dtype=float), index=pd.to_datetime(np.asarray(dates))).dropna()
    if q.empty:
        return {k: float("nan") for k in LEVELS[1:]}
    years = q.groupby(q.index.year)
    full_years = years.size()[lambda s: s >= MIN_DAYS_PER_YEAR].index
    annual_max = years.max().loc[full_years].to_numpy()
    values = [float(q.quantile(ELEVATED_QUANTILE))] + [gumbel_flow(annual_max, RETURN_PERIODS[k])
                                                       for k in LEVELS[2:]]
    values = [max(v, MIN_THRESHOLD_CFS) if np.isfinite(v) else v for v in values]
    out, running = {}, -np.inf
    for name, v in zip(LEVELS[1:], values):
        running = max(running, v) if np.isfinite(v) else running
        out[name] = running if np.isfinite(v) else float("nan")
    return out


def compute_threshold_table(frame: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    rows = [{ID_COL: sid, **station_thresholds(g["date"], g[flow_col])} for sid, g in frame.groupby(ID_COL)]
    return pd.DataFrame(rows)


LIKELY_RANGE_COVERAGE = 0.80  # the likely range is the 10th-90th percentile band


@dataclass(frozen=True)
class RiskTable:
    model_thr: pd.DataFrame   # station_id + LEVELS[1:] in cfs (model climate) -> used to classify
    obs_thr: pd.DataFrame     # station_id + LEVELS[1:] in cfs (observed)      -> shown to users
    intervals: pd.DataFrame   # station_id, log_lo, log_hi (log1p residual quantiles)

    @classmethod
    def load(cls, model_dir: Path) -> "RiskTable":
        thr = pd.read_csv(model_dir / THRESHOLD_FILE, dtype={ID_COL: str}, encoding="utf-8")
        iv = pd.read_csv(model_dir / INTERVAL_FILE, dtype={ID_COL: str}, encoding="utf-8")
        cols = list(LEVELS[1:])
        model_thr = thr[[ID_COL, *[f"model_{c}" for c in cols]]].set_axis([ID_COL, *cols], axis=1)
        obs_thr = thr[[ID_COL, *[f"obs_{c}" for c in cols]]].set_axis([ID_COL, *cols], axis=1)
        return cls(model_thr, obs_thr, iv)

    def has_levels(self, station_id: str) -> bool:
        row = self.model_thr[self.model_thr[ID_COL] == station_id]
        return bool(len(row)) and bool(np.isfinite(row[list(LEVELS[1:])].to_numpy(dtype=float)).any())

    def levels(self, station_ids: pd.Series | np.ndarray, flows: np.ndarray) -> np.ndarray:
        """Risk level index per row; UNKNOWN (-1) when the station has no flow record."""
        return classify(np.asarray(station_ids, dtype=str), np.asarray(flows, dtype=float), self.model_thr)

    def observed_thresholds(self, station_id: str) -> dict[str, float]:
        row = self.obs_thr[self.obs_thr[ID_COL] == station_id]
        if row.empty:
            return {}
        return {k: float(row.iloc[0][k]) for k in LEVELS[1:] if np.isfinite(row.iloc[0][k])}

    def model_thresholds(self, station_id: str) -> dict[str, float]:
        row = self.model_thr[self.model_thr[ID_COL] == station_id]
        if row.empty:
            return {}
        return {k: float(row.iloc[0][k]) for k in LEVELS[1:] if np.isfinite(row.iloc[0][k])}

    def bounds(self, station_ids, flows: np.ndarray, has_history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """10th-90th percentile likely range (split-conformal, residuals from 2021-22); nominal
        coverage LIKELY_RANGE_COVERAGE."""
        ids = np.asarray(station_ids, dtype=str)
        iv = self.intervals.set_index(ID_COL)
        glob = iv.loc[GLOBAL_KEY]
        unseen = iv.loc[UNSEEN_KEY] if UNSEEN_KEY in iv.index else glob
        lo = np.where(has_history, glob.log_lo, unseen.log_lo).astype(float)
        hi = np.where(has_history, glob.log_hi, unseen.log_hi).astype(float)
        known = pd.Series(ids).isin(iv.index).to_numpy() & np.asarray(has_history, dtype=bool)
        if known.any():
            lo[known] = iv.loc[ids[known], "log_lo"].to_numpy(dtype=float)
            hi[known] = iv.loc[ids[known], "log_hi"].to_numpy(dtype=float)
        base = np.log1p(np.clip(np.asarray(flows, dtype=float), 0, None))
        # both bounds floored at 0 in log space, so low <= high and neither is negative
        return np.expm1(np.maximum(base + lo, 0.0)), np.expm1(np.maximum(base + hi, 0.0))


def classify(ids: np.ndarray, flows: np.ndarray, thr: pd.DataFrame) -> np.ndarray:
    table = thr.set_index(ID_COL)[list(LEVELS[1:])]
    out = np.full(len(ids), UNKNOWN, dtype=int)
    present = pd.Series(ids).isin(table.index).to_numpy()
    if not present.any():
        return out
    t = table.loc[ids[present]].to_numpy(dtype=float)
    has_any = np.isfinite(t).any(axis=1)
    t = np.where(np.isfinite(t), t, np.inf)
    lvl = (flows[present][:, None] >= t).sum(axis=1)
    out[np.flatnonzero(present)[has_any]] = lvl[has_any]
    return out


def confidence(lo: float, hi: float, point: float) -> str:
    """'high' / 'medium' / 'low' from the relative width of the likely range."""
    if not np.isfinite(point) or point <= 0:
        return "low"
    ratio = (hi + 1) / (lo + 1)
    return "high" if ratio < 2.0 else "medium" if ratio < 4.0 else "low"
