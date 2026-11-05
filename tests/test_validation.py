"""Validation rules: malformed records, unknown catchments, unexpected horizons, NaN/infinity,
negative discharge, duplicates, missing keys, interval ordering and coverage/probability fields."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodcast.risk import RiskTable
from floodcast.schema import (
    DATE_COL, ID_COL, TARGET, coerce_frame, make_submission, validate_risk_file, validate_submission,
    validate_test_input,
)

ROWS = [("01646500", "2025-11-01"), ("01646500", "2025-11-02"), ("ST001", "2025-11-01")]


@pytest.fixture()
def test_frame() -> pd.DataFrame:
    return coerce_frame(pd.DataFrame(ROWS, columns=[ID_COL, DATE_COL]))


@pytest.fixture()
def good(test_frame) -> pd.DataFrame:
    return make_submission(test_frame, np.array([10.0, 20.0, 5.0]))


def test_valid_submission_returns_a_clean_report(good, test_frame):
    report = validate_submission(good, test_frame)
    assert report["rows"] == 3 and report["stations"] == 2 and report["missing"] == 0


def test_never_seen_station_is_allowed(test_frame):
    validate_test_input(test_frame)  # ST001 is not a training station: still valid


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.assign(**{ID_COL: ["01646500", "01646500", "ST999"]}), "not in test.csv"),
    (lambda d: d.assign(**{DATE_COL: ["2025-11-01", "2025-11-09", "2025-11-01"]}), "unexpected horizon"),
    (lambda d: d.assign(**{TARGET: [1.0, np.nan, 2.0]}), "NaN or infinity"),
    (lambda d: d.assign(**{TARGET: [1.0, np.inf, 2.0]}), "NaN or infinity"),
    (lambda d: d.assign(**{TARGET: [1.0, -3.0, 2.0]}), "negative"),
    (lambda d: d.assign(**{TARGET: ["1.0", "abc", "2"]}), "not numbers"),
    (lambda d: d.assign(**{ID_COL: ["01646500", "", "ST001"]}), "blank station_id or date"),
    (lambda d: pd.concat([d.iloc[:2], d.iloc[[1]]], ignore_index=True), "duplicate prediction"),
    (lambda d: d.iloc[:2], "have no prediction"),
    (lambda d: d.iloc[[1, 0, 2]].reset_index(drop=True), "row-for-row"),
])
def test_bad_submissions_are_rejected(good, test_frame, edit, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(edit(good), test_frame)


def test_nan_model_output_is_rejected_not_filled(test_frame):
    with pytest.raises(ValueError, match="NaN/infinite prediction"):
        make_submission(test_frame, np.array([1.0, np.nan, 2.0]))


def test_negative_model_output_is_clipped_to_zero(test_frame):
    assert make_submission(test_frame, np.array([-0.5, 1.0, 2.0]))[TARGET].tolist() == [0.0, 1.0, 2.0]


def test_blank_or_repeated_test_keys_are_rejected():
    blank = coerce_frame(pd.DataFrame([("", "2025-11-01"), ("01646500", "2025-11-02")], columns=[ID_COL, DATE_COL]))
    with pytest.raises(ValueError, match="blank station_id"):
        validate_test_input(blank)
    # '1646500' and '01646500' are the same station after the Excel leading-zero fix
    dup = coerce_frame(pd.DataFrame([("01646500", "2025-11-01"), ("1646500", "2025-11-01")],
                                    columns=[ID_COL, DATE_COL]))
    with pytest.raises(ValueError, match="repeats 1 station_id/date pair"):
        validate_test_input(dup)


def _risk(lo, hi, coverage=0.8, **extra):
    return pd.DataFrame({"lo": lo, "hi": hi, "cov": coverage, **extra})


@pytest.mark.parametrize("frame, message", [
    (_risk([1.0, 5.0], [2.0, 4.0]), "out of order"),
    (_risk([-1.0, 1.0], [2.0, 2.0]), "negative bounds"),
    (_risk([1.0, np.nan], [2.0, 2.0]), "NaN or infinity"),
    (_risk([1.0, 1.0], [2.0, 2.0], coverage=1.5), "between 0 and 1"),
    (_risk([1.0, 1.0], [2.0, 2.0], flood_probability=[0.2, -0.1]), "between 0 and 1"),
    (pd.DataFrame({"lo": [1.0], "hi": [2.0]}), "missing required interval field"),
])
def test_bad_intervals_are_rejected(frame, message):
    with pytest.raises(ValueError, match=message):
        validate_risk_file(frame, "lo", "hi", "cov")


def test_likely_range_is_ordered_even_for_tiny_flows():
    from pathlib import Path
    risk = RiskTable.load(Path(__file__).resolve().parents[1] / "models")
    ids = risk.intervals[ID_COL].to_numpy()
    for flow in (0.0, 0.01, 0.5, 10.0, 1e5):
        lo, hi = risk.bounds(ids, np.full(len(ids), flow), np.ones(len(ids), bool))
        assert (lo >= 0).all() and (lo <= hi).all()
