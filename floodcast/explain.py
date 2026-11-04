"""Plain-language "why" for a prediction, from LightGBM's built-in SHAP contributions.

Uses the log-scale member of each ensemble (known: log(1+Q); unseen: log(Q/area)), so a
contribution c means the flow is multiplied by about exp(c) compared with an average day.
River size / usual-flow features are reported separately from weather drivers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FloodcastModel

GROUPS = {
    "recent_rain": ("precipitation_mm", "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_sum_3d", "rain_sum_7d",
                    "rain_max_7d"),
    "past_rain": ("rain_sum_14d", "rain_sum_30d", "rain_sum_60d", "rain_sum_90d", "api_85", "api_95",
                  "log_rain_vol_30d", "log_rain_vol_90d", "log_api95_vol"),
    "soil": ("soil_moisture", "soil_mean_7d", "soil_mean_30d", "soil_change_3d"),
    "snow_temp": ("temp_max_c", "temp_min_c", "temp_mean_c", "temp_range_c", "snow_90d", "melt_dd_14d", "tmean_30d"),
    "season": ("doy_sin", "doy_cos", "month"),
}
RIVER_GROUP = "river"
EXPLAIN_OBJECTIVE = {"known": "log_l2", "unseen": "log_specific"}
TOP_N = 3


def _member(model: FloodcastModel, kind: str):
    ens = model.known if kind == "known" else model.unseen
    wanted = EXPLAIN_OBJECTIVE[kind]
    return next((m for m in ens.members if m.objective == wanted), ens.members[0])


def group_contributions(model: FloodcastModel, feats: pd.DataFrame, has_stats: bool) -> dict[str, float]:
    """Sum of log-scale SHAP contributions per driver group for ONE feature row."""
    member = _member(model, "known" if has_stats else "unseen")
    contrib = member.booster.predict(feats[member.features].iloc[[0]], pred_contrib=True)[0][:-1]
    by_feature = dict(zip(member.features, contrib))
    out = {g: float(sum(by_feature.get(f, 0.0) for f in cols)) for g, cols in GROUPS.items()}
    grouped = {f for cols in GROUPS.values() for f in cols}
    out[RIVER_GROUP] = float(sum(v for f, v in by_feature.items() if f not in grouped))
    return out


def top_drivers(contribs: dict[str, float], n: int = TOP_N) -> list[tuple[str, float]]:
    """Weather/season drivers sorted by size, as (group, multiplier) with multiplier=exp(c)."""
    weather = [(g, c) for g, c in contribs.items() if g != RIVER_GROUP and np.isfinite(c)]
    weather.sort(key=lambda gc: abs(gc[1]), reverse=True)
    return [(g, float(np.exp(c))) for g, c in weather[:n]]
