"""Join USGS discharge + Open-Meteo weather + catchment attributes into FLOODCAST-format files.

Outputs (data/processed/):
  train.csv            rows up to TRAIN_END, with discharge_cfs
  test.csv             rows after TRAIN_END, WITHOUT discharge_cfs (same schema as the brief)
  test_labels.csv      held-out truth for local scoring only (never read by the model)
  station_metadata.csv station_id, station_name, latitude, longitude, state, river_name
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
TRAIN_START, TRAIN_END = "2012-01-01", "2022-12-31"

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from floodcast.schema import INPUT_COLS, STATIC_COLS, TARGET, TRAIN_COLS  # noqa: E402

SAMPLE_STATIONS = ("01646500", "02087500")
SAMPLE_PERIOD = ("2023-06-01", "2023-08-31")


ALL_END = "2025-10-31"  # final model: every labelled day we have (holdout years included)
# Extra GAGES-II stations (scripts/add_stations.py) are opt-in: in station-group CV the first
# 10 made unseen-station accuracy slightly worse on both time folds, so they are not used.
WITH_EXTRA = "--with-extra" in sys.argv


def join_downloads(stations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sid in stations.station_id:
        q_path, m_path = RAW / "usgs" / f"{sid}.parquet", RAW / "meteo" / f"{sid}.parquet"
        if not (q_path.exists() and m_path.exists()):
            print(f"skip {sid}: missing download")
            continue
        q = pd.read_parquet(q_path)
        q = q[~q.quality_cd.fillna("").str.contains("Ice|Eqp|Bkw")]
        m = pd.read_parquet(m_path).assign(station_id=sid)
        frames.append(m.merge(q[["date", TARGET]], on="date", how="left"))
    full = pd.concat(frames, ignore_index=True).merge(stations[["station_id", *STATIC_COLS]], on="station_id")
    full["date"] = pd.to_datetime(full.date)
    return full[TRAIN_COLS]


def write_all_years(full: pd.DataFrame) -> None:
    """train_all.csv (2012 -> ALL_END, core + extra stations) and its rolling-window warm-up."""
    extra_path = PROC / "stations_extra.csv"
    if WITH_EXTRA and extra_path.exists():
        extra = pd.read_csv(extra_path, dtype={"station_id": str})
        full = pd.concat([full, join_downloads(extra[~extra.station_id.isin(full.station_id)])], ignore_index=True)
    labelled = full[(full.date >= TRAIN_START) & (full.date <= ALL_END)].dropna(subset=[TARGET])
    labelled.to_csv(PROC / "train_all.csv", index=False, date_format="%Y-%m-%d")
    full[full.date < TRAIN_START].drop(columns=TARGET).to_parquet(PROC / "warmup_all.parquet", index=False)
    full.to_parquet(PROC / "full_daily_all.parquet", index=False)
    print(f"train_all: {len(labelled)} rows, {labelled.station_id.nunique()} stations")


def main() -> None:
    stations = pd.read_csv(PROC / "stations.csv", dtype={"station_id": str})
    full = join_downloads(stations)

    # Weather is kept for every day (rolling-window warm-up); labelled rows drive training/scoring.
    train = full[(full.date >= TRAIN_START) & (full.date <= TRAIN_END)]
    test = full[full.date > TRAIN_END]
    warmup = full[full.date < TRAIN_START].drop(columns=TARGET)

    PROC.mkdir(parents=True, exist_ok=True)
    train.dropna(subset=[TARGET]).to_csv(PROC / "train.csv", index=False, date_format="%Y-%m-%d")
    labelled_test = test.dropna(subset=[TARGET])
    labelled_test[INPUT_COLS].to_csv(PROC / "test.csv", index=False, date_format="%Y-%m-%d")
    labelled_test[["station_id", "date", TARGET]].to_csv(PROC / "test_labels.csv", index=False, date_format="%Y-%m-%d")
    full.to_parquet(PROC / "full_daily.parquet", index=False)
    warmup.to_parquet(PROC / "warmup_weather.parquet", index=False)
    stations.assign(river_name=stations.station_name)[
        ["station_id", "station_name", "latitude", "longitude", "state", "river_name"]
    ].to_csv(PROC / "station_metadata.csv", index=False)
    sample_dir = ROOT / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = labelled_test[labelled_test.station_id.isin(SAMPLE_STATIONS)
                           & labelled_test.date.between(*SAMPLE_PERIOD)]
    sample[INPUT_COLS].to_csv(sample_dir / "test_sample.csv", index=False, date_format="%Y-%m-%d")
    sample[["station_id", "date", TARGET]].to_csv(sample_dir / "test_sample_labels.csv", index=False,
                                                  date_format="%Y-%m-%d")
    write_all_years(full)
    print(f"stations={full.station_id.nunique()} train={train[TARGET].notna().sum()} test={len(labelled_test)}")
    print(full.groupby("station_id")[TARGET].agg(["count", "median", "max"]).sort_values("median").to_string())


if __name__ == "__main__":
    main()
