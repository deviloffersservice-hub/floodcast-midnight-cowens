"""Add more USGS training stations (the brief allows it) to improve accuracy on unseen rivers.

  python scripts/add_stations.py select   # pick GAGES-II candidates, download USGS flow, keep good ones
  python scripts/add_stations.py meteo    # download Open-Meteo weather for kept stations (quota-aware)

Candidates are stratified by HUC02 region x basin size so the extra rivers are diverse. Stations
used by the practice kit (judge_test/) are excluded, so that kit stays a never-seen test.
Writes data/processed/stations_extra.csv; raw downloads go next to the existing ones.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_data import END, METEO_URL, METEO_VARS, START, USGS_DIR, METEO_DIR, fetch_usgs  # noqa: E402
from select_stations import load_gages2  # noqa: E402

EXTRA = ROOT / "data" / "processed" / "stations_extra.csv"
EXISTING = ROOT / "data" / "processed" / "stations.csv"
EXCLUDE = {"03451500", "08068000"}  # judge_test never-seen rivers
SEED = 2026
SIZE_BINS = 4
PER_CELL = 1
MIN_COVERAGE = 0.8  # share of 2012-01-01..2025-10-31 days with a usable flow value
BAD_QUALITY = "Ice|Eqp|Bkw"
METEO_PAUSE_S = 16
LIMIT_WAIT_S = {"Minutely": 65, "Hourly": 300, "Daily": 900}
TIMEOUT_S = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("add_stations")
COLS = ["station_id", "station_name", "latitude", "longitude", "state",
        "drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct"]


def _coverage(q: pd.DataFrame) -> float:
    days = pd.date_range("2012-01-01", "2025-10-31", freq="D")
    ok = q[~q.quality_cd.fillna("").str.contains(BAD_QUALITY)]
    return pd.to_datetime(ok.date).isin(days).sum() / len(days)


def select(max_candidates: int) -> None:
    g = load_gages2()
    taken = set(pd.read_csv(EXISTING, dtype={"station_id": str}).station_id) | EXCLUDE
    pool = g[(g.ACTIVE09 == "yes") & (g.FLOWYRS_1990_2009 >= 20) & ~g.station_id.isin(taken)].copy()
    pool["size_bin"] = pd.qcut(np.log10(pool.drainage_area_sqmi), SIZE_BINS, labels=False)
    picks = pool.sample(frac=1.0, random_state=SEED).groupby(["HUC02", "size_bin"]).head(PER_CELL)
    picks = picks.head(max_candidates)
    kept = []
    USGS_DIR.mkdir(parents=True, exist_ok=True)
    for row in picks.itertuples():
        path = USGS_DIR / f"{row.station_id}.parquet"
        try:
            q = pd.read_parquet(path) if path.exists() else fetch_usgs(row.station_id)
        except Exception as exc:  # one bad station must not stop the batch
            log.error("usgs %s failed: %s", row.station_id, exc)
            continue
        cov = _coverage(q) if len(q) else 0.0
        log.info("usgs %s %s coverage=%.2f", row.station_id, row.station_name[:40], cov)
        if cov >= MIN_COVERAGE:
            q.to_parquet(path, index=False)
            kept.append(row.station_id)
    out = picks[picks.station_id.isin(kept)][COLS].assign(in_gages2=True)
    out.to_csv(EXTRA, index=False)
    log.info("kept %d of %d candidates -> %s", len(out), len(picks), EXTRA)


def _meteo(lat: float, lon: float) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "daily": METEO_VARS, "timezone": "auto"}
    while True:
        resp = requests.get(METEO_URL, params=params, timeout=TIMEOUT_S)
        if resp.status_code == 429 or (resp.ok and not resp.text.strip()):
            reason = resp.text[:200]
            wait = next((s for k, s in LIMIT_WAIT_S.items() if k in reason), LIMIT_WAIT_S["Minutely"])
            log.warning("rate limited (%s); waiting %ss", reason.strip(), wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        d = resp.json()["daily"]
        return pd.DataFrame({"date": d["time"], "precipitation_mm": d["precipitation_sum"],
                             "temp_max_c": d["temperature_2m_max"], "temp_min_c": d["temperature_2m_min"],
                             "soil_moisture": d["soil_moisture_0_to_7cm_mean"]})


def meteo() -> None:
    METEO_DIR.mkdir(parents=True, exist_ok=True)
    for row in pd.read_csv(EXTRA, dtype={"station_id": str}).itertuples():
        path = METEO_DIR / f"{row.station_id}.parquet"
        if path.exists():
            continue
        try:
            df = _meteo(row.latitude, row.longitude)
        except Exception as exc:
            log.error("meteo %s failed: %s", row.station_id, exc)
            continue
        df.to_parquet(path, index=False)
        log.info("meteo %s rows=%d", row.station_id, len(df))
        time.sleep(METEO_PAUSE_S)


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    else:
        meteo()
