"""Download daily discharge (USGS NWIS) and weather (Open-Meteo archive) for selected stations.

Mirrors the FLOODCAST data recipe: USGS parameter 00060 (daily mean discharge, cfs) and
Open-Meteo daily precipitation, max/min temperature and 0-7 cm soil moisture at the gauge.
Each station is cached to its own parquet file, so the script can be resumed.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / "data" / "processed" / "stations.csv"
USGS_DIR = ROOT / "data" / "raw" / "usgs"
METEO_DIR = ROOT / "data" / "raw" / "meteo"

START, END = "2011-10-01", "2025-10-31"
USGS_URL = "https://waterservices.usgs.gov/nwis/dv/"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_VARS = "precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean"
MAX_RETRIES = 8
BACKOFF_SECONDS = 30
METEO_PAUSE_SECONDS = 16  # each station costs ~147 of the 600/minute quota
PRIORITY = ["01646500", "07010000", "08158000", "02087500", "11447650", "03294500", "06934500",
            "07289000", "14105700", "01570500", "02489500", "05587450", "03612500", "12340500", "09380000"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch")


def get_with_retry(url: str, params: dict) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            resp.json()  # empty/truncated bodies happen under load; treat as retryable
            return resp
        except (requests.RequestException, ValueError) as exc:
            wait = BACKOFF_SECONDS * (attempt + 1)
            log.warning("retry %d for %s in %ss (%s)", attempt + 1, url, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_usgs(site: str) -> pd.DataFrame:
    params = {"format": "json", "sites": site, "startDT": START, "endDT": END,
              "parameterCd": "00060", "statCd": "00003", "siteStatus": "all"}
    series = get_with_retry(USGS_URL, params).json()["value"]["timeSeries"]
    rows = []
    for ts in series:
        for block in ts["values"]:
            for v in block["value"]:
                rows.append((v["dateTime"][:10], float(v["value"]), ",".join(v.get("qualifiers", []))))
    df = pd.DataFrame(rows, columns=["date", "discharge_cfs", "quality_cd"])
    df = df[df.discharge_cfs > -999]  # -999999 is the USGS no-data sentinel
    return df.drop_duplicates("date").assign(station_id=site)


def fetch_meteo(lat: float, lon: float) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "daily": METEO_VARS, "timezone": "auto"}
    daily = get_with_retry(METEO_URL, params).json()["daily"]
    return pd.DataFrame({
        "date": daily["time"],
        "precipitation_mm": daily["precipitation_sum"],
        "temp_max_c": daily["temperature_2m_max"],
        "temp_min_c": daily["temperature_2m_min"],
        "soil_moisture": daily["soil_moisture_0_to_7cm_mean"],
    })


def main(kind: str) -> None:
    stations = pd.read_csv(STATIONS, dtype={"station_id": str})
    priority = stations.station_id.isin(PRIORITY)
    stations = pd.concat([stations[priority], stations[~priority]])
    out_dir = USGS_DIR if kind == "usgs" else METEO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in stations.itertuples():
        path = out_dir / f"{row.station_id}.parquet"
        if path.exists():
            continue
        try:
            df = fetch_usgs(row.station_id) if kind == "usgs" else fetch_meteo(row.latitude, row.longitude)
        except Exception as exc:  # keep going; one bad station must not stop the batch
            log.error("%s %s failed: %s", kind, row.station_id, exc)
            continue
        df.to_parquet(path, index=False)
        log.info("%s %s rows=%d", kind, row.station_id, len(df))
        if kind == "meteo":
            time.sleep(METEO_PAUSE_SECONDS)


if __name__ == "__main__":
    main(sys.argv[1])
