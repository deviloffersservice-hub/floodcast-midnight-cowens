"""Early-warning logic shared by the app views (no UI code here, so it is unit-testable).

outlooks(): weather window -> per-station daily predicted flow, likely range, risk level,
            today's level, the peak in the look-ahead window and its drivers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .explain import group_contributions, top_drivers
from .forecast import OBS_COL
from .model import FloodcastModel
from .risk import LEVELS, UNKNOWN, RiskTable, confidence
from .schema import DATE_COL, ID_COL, coerce_frame

FAR_LEAD_DAYS = 7
FAR_WIDEN = 1.5  # widen the log-range for forecast days 8-16 (weather forecast less reliable)
DISPLAY_PAST_DAYS = 30


@dataclass(frozen=True)
class Outlook:
    station_id: str
    has_levels: bool
    daily: pd.DataFrame          # date, predicted, lo, hi, level, is_forecast, observed_cfs
    today_level: int
    peak_level: int
    peak_date: str
    peak_cfs: float
    peak_lo: float
    peak_hi: float
    confidence: str
    drivers: tuple[tuple[str, float], ...]

    @property
    def peak_level_name(self) -> str:
        return LEVELS[self.peak_level] if self.peak_level >= 0 else "unknown"

    @property
    def today_level_name(self) -> str:
        return LEVELS[self.today_level] if self.today_level >= 0 else "unknown"


def _widen(pred: np.ndarray, lo: np.ndarray, hi: np.ndarray, lead: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.log1p(pred)
    factor = np.where(lead > FAR_LEAD_DAYS, FAR_WIDEN, 1.0)
    lo_w = np.expm1(np.maximum(base - (base - np.log1p(lo)) * factor, 0.0))
    hi_w = np.expm1(base + (np.log1p(hi) - base) * factor)
    return lo_w, hi_w


def outlooks(model: FloodcastModel, risk: RiskTable, weather: pd.DataFrame, as_of: date,
             window_days: int = 16) -> dict[str, Outlook]:
    """Run the model over a weather window and summarise each station's outlook."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    weather = weather.reset_index(drop=True)  # callers often pass a filtered slice
    frame = coerce_frame(weather.drop(columns=[c for c in ("is_forecast",) if c in weather]))
    feats, has_stats = model.featurize(frame)
    pred = model.predict_features(feats, has_stats)
    ids = frame[ID_COL].to_numpy()
    lo, hi = risk.bounds(ids, pred, has_stats)
    today = pd.Timestamp(as_of)
    lead = ((frame[DATE_COL] - today).dt.days).to_numpy()
    lo, hi = _widen(pred, lo, hi, lead)
    levels = risk.levels(ids, pred)
    obs = weather[OBS_COL].to_numpy(dtype=float) if OBS_COL in weather else np.full(len(frame), np.nan)
    table = pd.DataFrame({ID_COL: ids, DATE_COL: frame[DATE_COL], "predicted": pred, "lo": lo, "hi": hi,
                          "level": levels, "is_forecast": lead > 0, "lead": lead, OBS_COL: obs})
    result = {}
    for sid, g in table.groupby(ID_COL, sort=False):
        g = g.sort_values(DATE_COL)
        show = g[g["lead"] >= -DISPLAY_PAST_DAYS]
        ahead = g[(g["lead"] >= 0) & (g["lead"] <= window_days)]
        if ahead.empty:
            ahead = g.tail(1)
        today_rows = g[g["lead"] == 0]
        today_level = int(today_rows["level"].iloc[0]) if len(today_rows) else int(ahead["level"].iloc[0])
        peak_idx = ahead.sort_values(["level", "predicted"], ascending=False).index[0]
        peak = table.loc[peak_idx]
        sid_has_stats = bool(has_stats[peak_idx])
        contribs = group_contributions(model, feats.loc[[peak_idx]], sid_has_stats)
        result[sid] = Outlook(
            station_id=sid, has_levels=risk.has_levels(sid), daily=show.reset_index(drop=True),
            today_level=today_level, peak_level=int(peak["level"]), peak_date=str(peak[DATE_COL])[:10],
            peak_cfs=float(peak["predicted"]), peak_lo=float(peak["lo"]), peak_hi=float(peak["hi"]),
            confidence=confidence(float(peak["lo"]), float(peak["hi"]), float(peak["predicted"])),
            drivers=tuple(top_drivers(contribs)),
        )
    return result


def what_if(weather: pd.DataFrame, station_id: str, extra_mm: float, day: int, as_of: date) -> pd.DataFrame:
    """Copy of `weather` with `extra_mm` of rain added at one station on forecast day `day`."""
    if extra_mm < 0:
        raise ValueError("extra rain must be >= 0 mm")
    target = pd.Timestamp(as_of) + pd.Timedelta(days=int(day))
    mask = (weather[ID_COL] == station_id) & (pd.to_datetime(weather[DATE_COL]) == target)
    if not mask.any():
        raise ValueError(f"day {day} is outside the forecast window")
    out = weather.copy()
    out.loc[mask, "precipitation_mm"] = out.loc[mask, "precipitation_mm"].fillna(0) + float(extra_mm)
    return out


def ranking(results: dict[str, Outlook]) -> list[Outlook]:
    """Stations ordered by peak risk (unknown last), then by today's level, then flow."""
    return sorted(results.values(), key=lambda o: (o.peak_level if o.peak_level != UNKNOWN else -2,
                                                   o.today_level, o.peak_cfs), reverse=True)
