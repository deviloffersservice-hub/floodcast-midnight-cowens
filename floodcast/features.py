"""Feature engineering from the columns the FLOODCAST test file provides.

The test file has no past discharge, so river memory is approximated from weather history:
rolling rainfall totals, an antecedent precipitation index, soil-moisture trends and a
simple snow/melt proxy. Rolling windows are computed on a gap-filled daily calendar per
station, so irregular dates in a test file still get correct time windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DATE_COL, ID_COL, STATIC_COLS, WEATHER_COLS

RAIN_WINDOWS = (3, 7, 14, 30, 60, 90)
SOIL_WINDOWS = (7, 30)
API_DECAYS = (0.85, 0.95)
FREEZE_C = 0.0
STAT_COLS = ["st_log_q_mean", "st_log_q_p10", "st_log_q_p50", "st_log_q_p90", "st_log_q_p99", "st_q_per_area"]


def _api(rain: np.ndarray, k: float) -> np.ndarray:
    out = np.empty_like(rain)
    acc = 0.0
    for i, r in enumerate(rain):
        acc = acc * k + r
        out[i] = acc
    return out


def _station_dynamic(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(DATE_COL)
    full = g.set_index(DATE_COL)[WEATHER_COLS].groupby(level=0).mean()
    full = full.reindex(pd.date_range(full.index.min(), full.index.max(), freq="D"))
    rain = full["precipitation_mm"].fillna(0.0)
    tmean = (full["temp_max_c"] + full["temp_min_c"]) / 2
    feats = {}
    for w in RAIN_WINDOWS:
        feats[f"rain_sum_{w}d"] = rain.rolling(w, min_periods=1).sum()
    feats["rain_max_7d"] = rain.rolling(7, min_periods=1).max()
    for lag in (1, 2, 3):
        feats[f"rain_lag_{lag}"] = rain.shift(lag)
    for k in API_DECAYS:
        feats[f"api_{int(k * 100)}"] = _api(rain.to_numpy(dtype=float), k)
    soil = full["soil_moisture"].interpolate(limit=5)
    for w in SOIL_WINDOWS:
        feats[f"soil_mean_{w}d"] = soil.rolling(w, min_periods=1).mean()
    feats["soil_change_3d"] = soil - soil.shift(3)
    snowfall = rain.where(tmean < FREEZE_C, 0.0)
    feats["snow_90d"] = snowfall.rolling(90, min_periods=1).sum()
    feats["melt_dd_14d"] = tmean.clip(lower=0).rolling(14, min_periods=1).sum()
    feats["tmean_30d"] = tmean.rolling(30, min_periods=1).mean()
    dyn = pd.DataFrame(feats, index=full.index)
    dyn.index.name = DATE_COL
    return g.merge(dyn.reset_index(), on=DATE_COL, how="left")


def build_features(df: pd.DataFrame, station_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a copy of df with model features appended (row order preserved)."""
    base = df.reset_index(drop=True).assign(_row=lambda d: np.arange(len(d)))
    parts = [_station_dynamic(g) for _, g in base.groupby(ID_COL, sort=False)]
    out = pd.concat(parts).sort_values("_row").reset_index(drop=True)

    doy = out[DATE_COL].dt.day_of_year
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month"] = out[DATE_COL].dt.month
    out["temp_mean_c"] = (out.temp_max_c + out.temp_min_c) / 2
    out["temp_range_c"] = out.temp_max_c - out.temp_min_c
    log_area = np.log1p(out.drainage_area_sqmi)
    out["log_area"] = log_area
    out["log_rain_vol_30d"] = np.log1p(out.rain_sum_30d) + log_area
    out["log_rain_vol_90d"] = np.log1p(out.rain_sum_90d) + log_area
    out["log_api95_vol"] = np.log1p(out.api_95) + log_area

    stats = station_stats if station_stats is not None else pd.DataFrame(columns=[ID_COL, *STAT_COLS])
    out = out.drop(columns=[c for c in STAT_COLS if c in out.columns]).merge(stats[[ID_COL, *STAT_COLS]], on=ID_COL, how="left")
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def compute_station_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Per-station discharge climatology from training labels only (never test)."""
    q = train[[ID_COL, "discharge_cfs", "drainage_area_sqmi"]].dropna(subset=["discharge_cfs"])
    lq = np.log1p(q.discharge_cfs)
    grp = q.assign(lq=lq).groupby(ID_COL)
    stats = pd.DataFrame({
        "st_log_q_mean": grp.lq.mean(),
        "st_log_q_p10": grp.lq.quantile(0.10),
        "st_log_q_p50": grp.lq.quantile(0.50),
        "st_log_q_p90": grp.lq.quantile(0.90),
        "st_log_q_p99": grp.lq.quantile(0.99),
        "st_q_per_area": grp.discharge_cfs.mean() / grp.drainage_area_sqmi.first(),
    })
    return stats.reset_index()


DYNAMIC_FEATURES = [
    *WEATHER_COLS, "temp_mean_c", "temp_range_c",
    *[f"rain_sum_{w}d" for w in RAIN_WINDOWS], "rain_max_7d", "rain_lag_1", "rain_lag_2", "rain_lag_3",
    *[f"api_{int(k * 100)}" for k in API_DECAYS],
    *[f"soil_mean_{w}d" for w in SOIL_WINDOWS], "soil_change_3d",
    "snow_90d", "melt_dd_14d", "tmean_30d", "doy_sin", "doy_cos", "month",
]
STATIC_FEATURES = [*STATIC_COLS, "log_area", "log_rain_vol_30d", "log_rain_vol_90d", "log_api95_vol"]
UNSEEN_FEATURES = [*DYNAMIC_FEATURES, *STATIC_FEATURES]
KNOWN_FEATURES = [*UNSEEN_FEATURES, *STAT_COLS]
