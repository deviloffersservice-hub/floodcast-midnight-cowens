"""Robustness tests: the input quirks a Windows/Excel test.csv can have, and the output rules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.schema import (  # noqa: E402
    RAW_DATE, RAW_ID, SUBMISSION_COLS, TARGET, make_submission, normalize_station_id, parse_dates,
    read_table, validate_submission,
)

MODEL = FloodcastModel.load(ROOT / "models")
SAMPLE = ROOT / "data" / "sample" / "test_sample.csv"


def _predict_file(path: Path) -> pd.DataFrame:
    test = read_table(path)
    out = make_submission(test, MODEL.predict(test))
    validate_submission(out, test)
    return out


@pytest.fixture()
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE, dtype=str)


def test_station_ids_get_leading_zeros_back():
    ids = pd.Series(["1646500", "01646500", "1646500.0", " 7010000 ", "ABC"])
    assert normalize_station_id(ids).tolist() == ["01646500", "01646500", "01646500", "07010000", "ABC"]


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"])
def test_date_formats_round_trip(fmt):
    days = pd.date_range("2023-01-01", periods=40, freq="D")
    parsed = parse_dates(pd.Series(days.strftime(fmt)))
    assert (parsed.to_numpy() == days.to_numpy()).all()


def test_prediction_matches_rules_and_echoes_input(sample, tmp_path):
    path = tmp_path / "test.csv"
    sample.to_csv(path, index=False)
    out = _predict_file(path)
    assert list(out.columns) == SUBMISSION_COLS
    assert len(out) == len(sample)
    assert (out[TARGET] >= 0).all() and np.isfinite(out[TARGET]).all()
    assert out["station_id"].tolist() == sample["station_id"].tolist()
    assert out["date"].tolist() == sample["date"].tolist()


def test_excel_style_file_gives_same_numbers(sample, tmp_path):
    """BOM + CRLF + stripped zeros + dd-mm-yyyy dates + soil in % -> identical predictions."""
    clean = tmp_path / "clean.csv"
    sample.to_csv(clean, index=False)
    excel = sample.copy()
    excel["station_id"] = excel["station_id"].str.lstrip("0")
    excel["date"] = pd.to_datetime(excel["date"]).dt.strftime("%d-%m-%Y")
    excel["soil_moisture"] = (excel["soil_moisture"].astype(float) * 100).round(4).astype(str)
    messy = tmp_path / "excel.csv"
    excel.to_csv(messy, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    a, b = _predict_file(clean), _predict_file(messy)
    np.testing.assert_allclose(a[TARGET].to_numpy(), b[TARGET].to_numpy(), rtol=1e-6, atol=1e-3)
    assert b["station_id"].tolist() == excel["station_id"].tolist()  # echoed as given


def test_shuffled_rows_keep_order_and_values(sample, tmp_path):
    clean = tmp_path / "clean.csv"
    sample.to_csv(clean, index=False)
    shuffled = sample.sample(frac=1.0, random_state=0)
    shuf = tmp_path / "shuffled.csv"
    shuffled.to_csv(shuf, index=False)
    a = _predict_file(clean).set_index(["station_id", "date"])[TARGET]
    b = _predict_file(shuf)
    assert b["date"].tolist() == shuffled["date"].tolist()
    np.testing.assert_allclose(b.set_index(["station_id", "date"])[TARGET].loc[a.index].to_numpy(), a.to_numpy())


def test_missing_values_and_unknown_station_still_predict(sample, tmp_path):
    broken = sample.copy()
    broken.loc[broken.index[:5], "precipitation_mm"] = ""
    broken.loc[broken.index[5:10], "soil_moisture"] = "NA"
    broken.loc[broken.index[:3], "station_id"] = "NEW_STATION"
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    out = _predict_file(path)
    assert len(out) == len(broken) and (out[TARGET] >= 0).all()


def test_single_row_file(sample, tmp_path):
    path = tmp_path / "one.csv"
    sample.head(1).to_csv(path, index=False)
    assert len(_predict_file(path)) == 1


def test_predictions_are_deterministic(sample, tmp_path):
    path = tmp_path / "t.csv"
    sample.to_csv(path, index=False)
    first, second = _predict_file(path), _predict_file(path)
    assert first[TARGET].tolist() == second[TARGET].tolist()


def test_validation_rejects_bad_outputs(sample, tmp_path):
    path = tmp_path / "t.csv"
    sample.to_csv(path, index=False)
    test = read_table(path)
    good = make_submission(test, np.ones(len(test)))
    with pytest.raises(ValueError):
        validate_submission(good.assign(**{TARGET: -1.0}), test)
    with pytest.raises(ValueError):
        validate_submission(good.iloc[:-1], test)
    with pytest.raises(ValueError):
        validate_submission(good[["date", "station_id", TARGET]], test)
    assert RAW_ID in test and RAW_DATE in test


def test_missing_required_column_is_clear_error(sample, tmp_path):
    path = tmp_path / "t.csv"
    sample.drop(columns=["date"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        read_table(path)
