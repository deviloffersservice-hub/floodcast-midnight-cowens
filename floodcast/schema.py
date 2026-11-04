"""Input/output contract from the FLOODCAST brief, plus tolerant readers and strict validation.

Reading is tolerant (Excel-on-Windows quirks: BOM, stripped leading zeros, non-ISO dates,
soil moisture in %), but the output echoes station_id and date exactly as they appear in
test.csv, so the organizers' row matching always succeeds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL, DATE_COL, TARGET = "station_id", "date", "discharge_cfs"
RAW_ID, RAW_DATE = "_raw_station_id", "_raw_date"
WEATHER_COLS = ["precipitation_mm", "temp_max_c", "temp_min_c", "soil_moisture"]
STATIC_COLS = ["drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct"]
INPUT_COLS = [ID_COL, DATE_COL, *WEATHER_COLS, *STATIC_COLS]
TRAIN_COLS = [ID_COL, DATE_COL, TARGET, *WEATHER_COLS, *STATIC_COLS]
SUBMISSION_COLS = [ID_COL, DATE_COL, TARGET]
USGS_ID_LEN = 8
SOIL_PERCENT_THRESHOLD = 1.5  # m3/m3 is always <= ~0.6; larger values must be percentages
CSV_ENCODINGS = ("utf-8-sig", "cp1252")
# Inputs are rounded to a fixed precision so equivalent inputs (e.g. soil 46.4 % vs 0.464)
# become bit-identical floats and always fall on the same side of every tree split.
INPUT_DECIMALS = {"precipitation_mm": 2, "temp_max_c": 2, "temp_min_c": 2, "soil_moisture": 4,
                  "drainage_area_sqmi": 2, "mean_elevation_ft": 1, "slope_pct": 3, "forest_cover_pct": 3}
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y")

log = logging.getLogger("floodcast")


def normalize_station_id(values: pd.Series) -> pd.Series:
    """USGS ids keep leading zeros ('01646500'); spreadsheets often strip them."""
    ids = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_short_number = ids.str.fullmatch(r"\d+") & (ids.str.len() < USGS_ID_LEN)
    return ids.where(~is_short_number, ids.str.zfill(USGS_ID_LEN))


def parse_dates(raw: pd.Series) -> pd.Series:
    """Parse dates, trying explicit formats; among formats that parse everything, prefer the
    one giving the most contiguous daily series (resolves dd/mm vs mm/dd ambiguity)."""
    text = raw.astype(str).str.strip()
    candidates = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().all():
            uniq = np.sort(parsed.unique())
            gap = float(np.median(np.diff(uniq)) / np.timedelta64(1, "D")) if len(uniq) > 1 else 0.0
            candidates.append((gap, fmt, parsed))
    if candidates:
        gap, fmt, parsed = min(candidates, key=lambda c: c[0])
        if fmt != DATE_FORMATS[0]:
            log.warning("dates are not ISO (YYYY-MM-DD); parsed with format %s", fmt)
        return parsed.dt.normalize()
    parsed = pd.to_datetime(text, format="mixed", errors="coerce")
    bad = text[parsed.isna()]
    if len(bad):
        raise ValueError(f"unparseable dates, e.g. {bad.head(3).tolist()}")
    return parsed.dt.normalize()


def read_table(path: str | Path, require_target: bool = False) -> pd.DataFrame:
    """Read a train/test CSV (UTF-8 with/without BOM, or Windows-1252) and coerce types."""
    last_error: Exception | None = None
    for enc in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=True)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError(f"cannot decode {path}: {last_error}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return coerce_frame(df, require_target=require_target)


def coerce_frame(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    required = TRAIN_COLS if require_target else [ID_COL, DATE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.copy()
    if RAW_ID not in out:
        out[RAW_ID] = out[ID_COL].astype(str).str.strip()
        out[RAW_DATE] = out[DATE_COL].astype(str).str.strip()
    out[ID_COL] = normalize_station_id(out[ID_COL])
    out[DATE_COL] = out[DATE_COL] if pd.api.types.is_datetime64_any_dtype(out[DATE_COL]) else parse_dates(out[DATE_COL])
    out[DATE_COL] = out[DATE_COL].dt.normalize()
    for col in [*WEATHER_COLS, *STATIC_COLS, TARGET]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col != TARGET:
            log.warning("column %s missing; the model will treat it as unknown", col)
            out[col] = np.nan
    out["precipitation_mm"] = out["precipitation_mm"].clip(lower=0)
    soil = out["soil_moisture"]
    if soil.dropna().gt(SOIL_PERCENT_THRESHOLD).mean() > 0.5:
        log.warning("soil_moisture looks like percent; converting to m3/m3")
        out["soil_moisture"] = soil / 100.0
    for col, decimals in INPUT_DECIMALS.items():
        out[col] = out[col].round(decimals)
    return out


def make_submission(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Build predictions.csv rows that echo test.csv's own id/date text, in test order."""
    values = np.round(np.clip(np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0), 0.0, None), 3)
    return pd.DataFrame({ID_COL: test[RAW_ID].to_numpy(), DATE_COL: test[RAW_DATE].to_numpy(), TARGET: values})


def validate_submission(pred: pd.DataFrame, test: pd.DataFrame) -> None:
    """Apply every rejection rule from the brief before a file is written."""
    if list(pred.columns) != SUBMISSION_COLS:
        raise ValueError(f"columns must be exactly {SUBMISSION_COLS}, got {list(pred.columns)}")
    if len(pred) != len(test):
        raise ValueError(f"row count {len(pred)} != test rows {len(test)}")
    values = pred[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or infinity")
    if (values < 0).any():
        raise ValueError("predictions contain negative discharge")
    pred_keys = list(zip(pred[ID_COL].astype(str), pred[DATE_COL].astype(str)))
    test_keys = list(zip(test[RAW_ID].astype(str), test[RAW_DATE].astype(str)))
    if pred_keys != test_keys:
        raise ValueError("station_id/date pairs do not match test.csv row-for-row")
