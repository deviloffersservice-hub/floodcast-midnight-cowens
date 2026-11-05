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
ALT_SEPARATORS = (";", "\t")
NUMERIC_COLS = {*WEATHER_COLS, *STATIC_COLS, TARGET}
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


def _read_csv(path: str | Path, encoding: str) -> pd.DataFrame:
    """Comma CSV, or the semicolon/tab files that Excel writes in some locales
    (semicolon files usually also use a decimal comma: '1,5' -> 1.5)."""
    df = pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=True)
    if df.shape[1] != 1:
        return df
    header = str(df.columns[0])
    for sep in ALT_SEPARATORS:
        if sep in header:
            alt = pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=True, sep=sep)
            numeric = [c for c in alt.columns if c.strip().lower() in NUMERIC_COLS]
            return alt.assign(**{c: alt[c].str.replace(",", ".", regex=False) for c in numeric}) if sep == ";" else alt
    return df


def read_table(path: str | Path, require_target: bool = False) -> pd.DataFrame:
    """Read a train/test CSV (UTF-8 with/without BOM, or Windows-1252) and coerce types."""
    last_error: Exception | None = None
    for enc in CSV_ENCODINGS:
        try:
            df = _read_csv(path, enc)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError(f"cannot decode {Path(path).name}: {last_error}")
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
    """Build predictions.csv rows that echo test.csv's own id/date text, in test order.

    Documented output transformation: discharge cannot be negative, so values are clipped at 0 and
    rounded to 3 decimals. NaN/infinite model outputs are never filled in; they are rejected.
    """
    values = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(f"the model returned {int(bad.sum())} NaN/infinite prediction(s), e.g. for "
                         f"{_examples(zip(test[RAW_ID][bad], test[RAW_DATE][bad]))}; predictions.csv not written")
    values = np.round(np.clip(values, 0.0, None), 3)
    return pd.DataFrame({ID_COL: test[RAW_ID].to_numpy(), DATE_COL: test[RAW_DATE].to_numpy(), TARGET: values})


BLANK_VALUES = {"", "nan", "none", "null", "na", "n/a", "nat"}
MAX_EXAMPLES = 3


def _examples(items, k: int = MAX_EXAMPLES) -> str:
    items = list(items)
    shown = ", ".join("/".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in items[:k])
    return shown + (f" (+{len(items) - k} more)" if len(items) > k else "")


def _is_blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().str.lower().isin(BLANK_VALUES)


def validate_test_input(test: pd.DataFrame) -> None:
    """Reject malformed test records before predicting: blank station ids and repeated
    station_id/date pairs (each required key must appear exactly once). Dates were already
    checked by parse_dates. Stations the model has never seen are valid (the brief requires them)."""
    blank = _is_blank(test[RAW_ID])
    if blank.any():
        rows = (np.flatnonzero(blank.to_numpy()) + 2).tolist()  # +2: header line, 1-based
        raise ValueError(f"{len(rows)} test row(s) have a blank station_id (file lines {_examples(rows)})")
    dup = test.duplicated([ID_COL, DATE_COL], keep=False)
    if dup.any():
        repeated = test.loc[dup].drop_duplicates([ID_COL, DATE_COL])
        examples = zip(repeated[ID_COL], repeated[DATE_COL].dt.strftime("%Y-%m-%d"))
        raise ValueError(f"test.csv repeats {len(repeated)} station_id/date pair(s) across {int(dup.sum())} rows, "
                         f"e.g. {_examples(examples)}; each pair must appear once")


def validate_submission(pred: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Apply every rejection rule before a file is written; return a summary of what was checked.

    Rules: exact columns; no malformed records; numeric, finite, non-negative discharge; no station
    outside test.csv (unknown catchment); no date outside that station's test dates (unexpected
    horizon); one prediction per key (no duplicates); no missing keys; same rows and order as test.csv.
    """
    if list(pred.columns) != SUBMISSION_COLS:
        raise ValueError(f"columns must be exactly {SUBMISSION_COLS}, got {list(pred.columns)}")
    malformed = _is_blank(pred[ID_COL]) | _is_blank(pred[DATE_COL])
    if malformed.any():
        raise ValueError(f"{int(malformed.sum())} prediction row(s) have a blank station_id or date")
    numeric = pd.to_numeric(pred[TARGET], errors="coerce")
    not_numeric = numeric.isna() & ~_is_blank(pred[TARGET])
    if not_numeric.any():
        raise ValueError(f"{int(not_numeric.sum())} prediction(s) are not numbers, e.g. "
                         f"{_examples(pred.loc[not_numeric, TARGET])}")
    values = numeric.to_numpy(dtype=float)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        raise ValueError(f"{int(non_finite.sum())} prediction(s) are blank, NaN or infinity")
    if (values < 0).any():
        raise ValueError(f"{int((values < 0).sum())} prediction(s) are negative discharge")

    pred_keys = list(zip(pred[ID_COL].astype(str), pred[DATE_COL].astype(str)))
    test_keys = list(zip(test[RAW_ID].astype(str), test[RAW_DATE].astype(str)))
    required, given = set(test_keys), set(pred_keys)
    unknown = sorted({sid for sid, _ in pred_keys} - {sid for sid, _ in test_keys})
    if unknown:
        raise ValueError(f"{len(unknown)} station_id(s) are not in test.csv (unknown catchment): {_examples(unknown)}")
    unexpected = sorted(given - required)
    if unexpected:
        raise ValueError(f"{len(unexpected)} prediction(s) are for dates not requested in test.csv "
                         f"(unexpected horizon), e.g. {_examples(unexpected)}")
    dup = pd.Series(pred_keys).duplicated()
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate prediction(s) for the same station_id/date, e.g. "
                         f"{_examples(pd.Series(pred_keys)[dup])}")
    missing = sorted(required - given)
    if missing:
        raise ValueError(f"{len(missing)} required station_id/date pair(s) have no prediction (submission would be "
                         f"rejected), e.g. {_examples(missing)}")
    if len(pred) != len(test):
        raise ValueError(f"row count {len(pred)} != test rows {len(test)}")
    if pred_keys != test_keys:
        raise ValueError("station_id/date pairs do not match test.csv row-for-row")
    return {"rows": len(pred), "stations": len({sid for sid, _ in pred_keys}), "missing": 0, "duplicates": 0,
            "unknown_stations": 0, "unexpected_dates": 0, "non_finite": 0, "negative": 0}


PROBABILITY_HINTS = ("prob", "coverage")


def validate_risk_file(risk: pd.DataFrame, low: str, high: str, coverage: str) -> dict:
    """Interval and probability checks for predictions_with_risk.csv: the coverage field is present
    and a valid probability; bounds are finite, non-negative and ordered (low <= high); every
    probability-like column is within [0, 1]."""
    missing = [c for c in (low, high, coverage) if c not in risk]
    if missing:
        raise ValueError(f"risk file is missing required interval field(s): {missing}")
    lo, hi = risk[low].to_numpy(dtype=float), risk[high].to_numpy(dtype=float)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("likely range contains NaN or infinity")
    if (lo < 0).any():
        raise ValueError("likely range has negative bounds")
    if (lo > hi).any():
        raise ValueError(f"{int((lo > hi).sum())} likely range(s) are out of order (low > high)")
    for col in risk.columns:
        if col == coverage or any(h in col.lower() for h in PROBABILITY_HINTS):
            p = pd.to_numeric(risk[col], errors="coerce").to_numpy(dtype=float)
            if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
                raise ValueError(f"{col} must hold probabilities between 0 and 1")
    return {"intervals": len(risk), "intervals_ordered": True, "coverage": float(risk[coverage].iloc[0])}
