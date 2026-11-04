"""Build judge_test/: practice files in the exact format the organizers will use.

Window 2025-11-01 to 2026-10-31: AFTER every day the final model was trained on (it uses
data up to 2025-10-31), downloaded fresh from USGS (provisional data) + Open-Meteo:
  * 4 stations recommended in the brief (the 5th, Sacramento 11447650, has no USGS data after 2016)
  * Rolling Fork, KY (03301500)
  * 2 rivers the model has NEVER seen (not among its training stations): French Broad at
    Asheville, NC (03451500) and West Fork San Jacinto near Conroe, TX (08068000)

Files
  test.csv              official input format (no discharge column) - upload this
  test_excel_messy.csv  same rows as Excel on Windows might save them: leading zeros stripped,
                        dd/mm/yyyy dates, soil moisture in %, shuffled rows
  answers.csv           true USGS discharge for every test row (organizers keep this)
  train_optional.csv    earlier history (2023-01 to 2025-10) for the two never-seen rivers

Needs internet only (USGS + Open-Meteo); station coordinates and catchment attributes come from
data/processed/judge_kit_stations.csv, so it runs on any machine with the repo (e.g. Windows).
Downloads are cached in data/raw/judge/, so a rerun after an error resumes where it stopped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.schema import INPUT_COLS, STATIC_COLS, TARGET, TRAIN_COLS  # noqa: E402

OUT = ROOT / "judge_test"
RAW = ROOT / "data" / "raw" / "judge"
KIT_STATIONS = ROOT / "data" / "processed" / "judge_kit_stations.csv"  # coordinates + attributes
START, END = "2025-11-01", "2026-10-31"
HISTORY_START, HISTORY_END = "2023-01-01", "2025-10-31"
KNOWN = ["01646500", "07010000", "08158000", "02087500", "03301500"]
NEW = ["03451500", "08068000"]
USGS_URL = "https://waterservices.usgs.gov/nwis/dv/"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_VARS = "precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean"
BAD_QUALITY = "Ice|Eqp|Bkw"  # same cleaning as the training data
SHUFFLE_SEED = 7
TIMEOUT_S = 120
HTTP_TOO_MANY_REQUESTS = 429


def _usgs(sid: str, start: str, end: str) -> pd.DataFrame:
    params = {"format": "json", "sites": sid, "startDT": start, "endDT": end, "parameterCd": "00060",
              "statCd": "00003", "siteStatus": "all"}
    resp = requests.get(USGS_URL, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    rows = [(v["dateTime"][:10], float(v["value"]), ",".join(v.get("qualifiers", [])))
            for ts in resp.json()["value"]["timeSeries"] for block in ts["values"] for v in block["value"]]
    q = pd.DataFrame(rows, columns=["date", TARGET, "quality_cd"])
    return q[(q[TARGET] > -999) & ~q.quality_cd.str.contains(BAD_QUALITY)].drop_duplicates("date")


def _meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "daily": METEO_VARS,
              "timezone": "auto"}
    resp = requests.get(METEO_URL, params=params, timeout=TIMEOUT_S)
    if resp.status_code == HTTP_TOO_MANY_REQUESTS:
        raise SystemExit(f"Open-Meteo rate limit reached ({resp.text[:200]}). The daily limit resets at "
                         "00:00 UTC (05:30 IST), the hourly one on the hour: wait, then rerun (downloads resume).")
    resp.raise_for_status()
    d = resp.json()["daily"]
    return pd.DataFrame({"date": d["time"], "precipitation_mm": d["precipitation_sum"],
                         "temp_max_c": d["temperature_2m_max"], "temp_min_c": d["temperature_2m_min"],
                         "soil_moisture": d["soil_moisture_0_to_7cm_mean"]})


def _attributes() -> pd.DataFrame:
    """Coordinates + static attributes of the kit stations (model's station table / GAGES-II)."""
    attrs = pd.read_csv(KIT_STATIONS, dtype={"station_id": str}).set_index("station_id")
    missing = sorted(set(KNOWN + NEW) - set(attrs.index))
    if missing:
        raise ValueError(f"{KIT_STATIONS.name} has no attributes for {missing}")
    return attrs


def station_rows(stations: list[str], start: str, end: str) -> pd.DataFrame:
    """USGS flow + Open-Meteo weather + static attributes (each download cached)."""
    attrs = _attributes()
    frames = []
    RAW.mkdir(parents=True, exist_ok=True)
    for sid in stations:
        path = RAW / f"{sid}_{start}_{end}.csv"
        if not path.exists():
            site = attrs.loc[sid]
            weather = _meteo(float(site.latitude), float(site.longitude), start, end)
            weather.merge(_usgs(sid, start, end)[["date", TARGET]], on="date", how="left").to_csv(path, index=False)
        frames.append(pd.read_csv(path).assign(station_id=sid, **{c: attrs.loc[sid, c] for c in STATIC_COLS}))
    return pd.concat(frames, ignore_index=True).dropna(subset=[TARGET])[TRAIN_COLS]


def main() -> None:
    trained = set(pd.read_csv(ROOT / "data" / "processed" / "train_all.csv", dtype={"station_id": str},
                              usecols=["station_id"]).station_id)
    if trained & set(NEW):
        raise ValueError(f"{sorted(trained & set(NEW))} are training stations, not new ones")
    window = station_rows(KNOWN + NEW, START, END)
    missing = sorted(set(KNOWN + NEW) - set(window.station_id))
    if missing:
        raise ValueError(f"no rows for {missing}")

    OUT.mkdir(exist_ok=True)
    window[INPUT_COLS].to_csv(OUT / "test.csv", index=False, lineterminator="\n")
    window[["station_id", "date", TARGET]].to_csv(OUT / "answers.csv", index=False, lineterminator="\n")
    messy = window[INPUT_COLS].sample(frac=1, random_state=SHUFFLE_SEED).assign(
        station_id=lambda d: d.station_id.str.lstrip("0"),
        date=lambda d: pd.to_datetime(d.date).dt.strftime("%d/%m/%Y"),
        soil_moisture=lambda d: (d.soil_moisture * 100).round(2))
    messy.to_csv(OUT / "test_excel_messy.csv", index=False, lineterminator="\r\n")
    history = station_rows(NEW, HISTORY_START, HISTORY_END)
    history.to_csv(OUT / "train_optional.csv", index=False, lineterminator="\n")
    print(f"test.csv: {len(window)} rows, {window.station_id.nunique()} stations, {START} to {END}; "
          f"train_optional.csv: {len(history)} rows")


if __name__ == "__main__":
    main()
