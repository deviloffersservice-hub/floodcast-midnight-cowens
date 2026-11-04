"""Weather inputs for the early-warning views: live forecast, offline snapshot, replay.

Live: Open-Meteo archive (past PAST_DAYS, same source as training) + Open-Meteo forecast
API (latest days and next FORECAST_DAYS). The forecast API has no 0-7 cm soil moisture,
so soil is held at its last archived value (it changes slowly). One multi-location
request per API keeps the whole dashboard within the free quota; results are cached.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .schema import DATE_COL, ID_COL, STATIC_COLS, TARGET, WEATHER_COLS

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
SNAPSHOT_PATH = ROOT / "data" / "snapshot" / "forecast_snapshot.parquet"
FULL_DAILY = ROOT / "data" / "processed" / "full_daily.parquet"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
PAST_DAYS = 120
FORECAST_DAYS = 16
ARCHIVE_LAG_DAYS = 6
FORECAST_PAST_DAYS = 10
CACHE_TTL_SECONDS = 3 * 3600
HTTP_TIMEOUT = 45
RETRIES = 2
OBS_COL = "observed_cfs"
log = logging.getLogger("floodcast")


class ForecastUnavailable(RuntimeError):
    """Raised when neither the weather service nor a saved snapshot can provide data."""


def _get_json(url: str, params: dict) -> object:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(1 + attempt)
    raise ForecastUnavailable(f"{url}: {last}")


def _daily_frames(payload: object, station_ids: list[str]) -> list[pd.DataFrame]:
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != len(station_ids):
        raise ForecastUnavailable(f"expected {len(station_ids)} locations, got {len(items)}")
    frames = []
    for sid, item in zip(station_ids, items):
        daily = item.get("daily") if isinstance(item, dict) else None
        if not daily or "time" not in daily:
            raise ForecastUnavailable(f"no daily data for {sid}")
        frames.append(pd.DataFrame(daily).assign(**{ID_COL: sid}))
    return frames


def _coords(stations: pd.DataFrame) -> dict:
    return {"latitude": ",".join(f"{v:.5f}" for v in stations.latitude),
            "longitude": ",".join(f"{v:.5f}" for v in stations.longitude), "timezone": "auto"}


def _fetch_live(stations: pd.DataFrame, today: date) -> pd.DataFrame:
    ids = stations[ID_COL].tolist()
    arch_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    arch = _get_json(ARCHIVE_URL, {**_coords(stations), "start_date": str(today - timedelta(days=PAST_DAYS)),
                                   "end_date": str(arch_end),
                                   "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                                            "soil_moisture_0_to_7cm_mean"})
    fc = _get_json(FORECAST_URL, {**_coords(stations), "past_days": FORECAST_PAST_DAYS,
                                  "forecast_days": FORECAST_DAYS,
                                  "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min"})
    rename = {"time": DATE_COL, "precipitation_sum": "precipitation_mm", "temperature_2m_max": "temp_max_c",
              "temperature_2m_min": "temp_min_c", "soil_moisture_0_to_7cm_mean": "soil_moisture"}
    a = pd.concat(_daily_frames(arch, ids)).rename(columns=rename)
    f = pd.concat(_daily_frames(fc, ids)).rename(columns=rename)
    a[DATE_COL], f[DATE_COL] = pd.to_datetime(a[DATE_COL]), pd.to_datetime(f[DATE_COL])
    last_arch = a.dropna(subset=["precipitation_mm"]).groupby(ID_COL)[DATE_COL].max()
    f = f[f[DATE_COL] > f[ID_COL].map(last_arch).fillna(pd.Timestamp(arch_end))]
    frame = pd.concat([a, f], ignore_index=True).sort_values([ID_COL, DATE_COL])
    frame["soil_moisture"] = frame.groupby(ID_COL)["soil_moisture"].ffill()
    return frame[[ID_COL, DATE_COL, *WEATHER_COLS]].reset_index(drop=True)


def _cache_path(ids: list[str], today: date) -> Path:
    key = hashlib.sha1(("|".join(sorted(ids)) + str(today)).encode()).hexdigest()[:16]
    return CACHE_DIR / f"live_{key}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Fresh cached frame, or None if missing, expired or unreadable (a corrupt file is removed)."""
    try:
        if not path.exists() or time.time() - path.stat().st_mtime >= CACHE_TTL_SECONDS:
            return None
        frame = pd.read_parquet(path)
        return frame if {ID_COL, DATE_COL, *WEATHER_COLS} <= set(frame.columns) and len(frame) else None
    except Exception as exc:  # corrupt/partial cache must never break the app
        log.warning("ignoring unreadable cache %s: %s", path, exc)
        path.unlink(missing_ok=True)
        return None


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    except OSError as exc:  # read-only folder etc.: run without cache
        log.warning("could not write cache %s: %s", path, exc)


def _attach_static(frame: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in STATIC_COLS if c in frame], errors="ignore").merge(
        stations[[ID_COL, *STATIC_COLS]], on=ID_COL, how="left")


def live_weather(stations: pd.DataFrame, today: date | None = None) -> tuple[pd.DataFrame, str, str]:
    """Weather for each station: past PAST_DAYS + next FORECAST_DAYS.

    Returns (frame, source, issued) where source is 'live', 'cache' or 'snapshot'.
    frame columns: station_id, date, weather columns, static columns, is_forecast.
    """
    today = today or date.today()
    ids = stations[ID_COL].tolist()
    path = _cache_path(ids, today)
    try:
        frame = _read_cache(path)
        source = "cache"
        if frame is None:
            frame, source = _fetch_live(stations, today), "live"
            _write_cache(frame, path)
        issued = str(today)
    except (ForecastUnavailable, OSError) as exc:
        log.warning("live weather unavailable (%s); trying snapshot", exc)
        frame, issued = load_snapshot(ids)
        today, source = date.fromisoformat(issued), "snapshot"
    frame = frame.assign(**{"is_forecast": pd.to_datetime(frame[DATE_COL]) > pd.Timestamp(today)})
    return _attach_static(frame, stations), source, issued


def save_snapshot(frame: pd.DataFrame, issued: str) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[[ID_COL, DATE_COL, *WEATHER_COLS]].assign(issued=issued).to_parquet(SNAPSHOT_PATH, index=False)


def load_snapshot(ids: list[str]) -> tuple[pd.DataFrame, str]:
    if not SNAPSHOT_PATH.exists():
        raise ForecastUnavailable("no saved forecast snapshot")
    snap = pd.read_parquet(SNAPSHOT_PATH)
    snap = snap[snap[ID_COL].isin(ids)]
    if snap.empty:
        raise ForecastUnavailable("saved snapshot has none of the requested stations")
    issued = str(snap["issued"].iloc[0])
    return snap.drop(columns="issued").reset_index(drop=True), issued


def replay_weather(stations: pd.DataFrame, as_of: date, horizon: int = FORECAST_DAYS) -> pd.DataFrame:
    """Recorded weather around `as_of` (works offline), with observed discharge for comparison."""
    full = pd.read_parquet(FULL_DAILY)
    full[ID_COL] = full[ID_COL].astype(str)
    start, end = pd.Timestamp(as_of) - pd.Timedelta(days=PAST_DAYS), pd.Timestamp(as_of) + pd.Timedelta(days=horizon)
    sub = full[full[ID_COL].isin(stations[ID_COL]) & full[DATE_COL].between(start, end)]
    if sub.empty:
        raise ForecastUnavailable(f"no recorded weather around {as_of}")
    sub = sub.rename(columns={TARGET: OBS_COL})[[ID_COL, DATE_COL, *WEATHER_COLS, OBS_COL]]
    sub = sub.assign(is_forecast=sub[DATE_COL] > pd.Timestamp(as_of))
    return _attach_static(sub.reset_index(drop=True), stations)


def recent_observed(station_id: str, start: date, end: date) -> pd.DataFrame:
    """USGS daily observed discharge (best effort; empty frame on any failure)."""
    params = {"format": "json", "sites": station_id, "startDT": str(start), "endDT": str(end),
              "parameterCd": "00060", "statCd": "00003", "siteStatus": "all"}
    try:
        payload = _get_json(USGS_DV_URL, params)
        rows = [(v["dateTime"][:10], float(v["value"])) for ts in payload["value"]["timeSeries"]
                for block in ts["values"] for v in block["value"]]
    except (ForecastUnavailable, KeyError, TypeError, ValueError) as exc:
        log.warning("USGS observations unavailable for %s: %s", station_id, exc)
        rows = []
    obs = pd.DataFrame(rows, columns=[DATE_COL, OBS_COL])
    obs = obs[obs[OBS_COL] > -999].drop_duplicates(DATE_COL)
    obs[DATE_COL] = pd.to_datetime(obs[DATE_COL])
    return obs


__all__ = ["ForecastUnavailable", "live_weather", "replay_weather", "save_snapshot", "load_snapshot",
           "recent_observed", "FORECAST_DAYS", "PAST_DAYS", "OBS_COL"]
