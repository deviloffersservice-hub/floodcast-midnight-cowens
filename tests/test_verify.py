"""Tests for the install check (scripts/verify_install.py) and kit scoring (scripts/score_kit.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_kit import KIT, kit_scores, to_markdown  # noqa: E402
from verify_install import compare  # noqa: E402


def _preds(values: list[float], dates: tuple[str, ...] = ("2025-01-01", "2025-01-02")) -> pd.DataFrame:
    return pd.DataFrame({"station_id": ["01646500"] * len(values), "date": list(dates), "discharge_cfs": values})


def test_identical_tables_have_no_rows_outside_tolerance():
    result = compare(_preds([10.0, 2000.0]), _preds([10.0, 2000.0]))
    assert result == {"keys_match": True, "max_abs_diff": 0.0, "n_outside_tol": 0}


def test_one_rounding_step_difference_is_tolerated():
    result = compare(_preds([10.0, 2000.0]), _preds([10.001, 2000.001]))
    assert result["keys_match"] and result["n_outside_tol"] == 0


def test_real_prediction_change_is_flagged():
    result = compare(_preds([10.0, 2000.0]), _preds([10.0, 2100.0]))
    assert result["n_outside_tol"] == 1
    assert abs(result["max_abs_diff"] - 100.0) < 1e-9


def test_different_keys_fail_even_with_equal_values():
    result = compare(_preds([10.0, 20.0]), _preds([10.0, 20.0], dates=("2025-01-01", "2025-01-03")))
    assert result["keys_match"] is False


def test_different_row_counts_fail():
    result = compare(_preds([10.0, 20.0]), _preds([10.0], dates=("2025-01-01",)))
    assert result["keys_match"] is False


@pytest.mark.skipif(not (KIT / "answers.csv").exists(), reason="practice kit not built")
def test_kit_scores_cover_every_answer_row_for_both_inputs():
    rows = kit_scores()
    n_answers = len(pd.read_csv(KIT / "answers.csv"))
    totals = [r for r in rows if r["group"] == "all"]
    assert [r["n"] for r in totals] == [n_answers] * len(totals) and len(totals) in (1, 2)
    assert all(np.isfinite([r["rmse"], r["rmse_extreme"], r["mae"], r["r2"]]).all() for r in rows)
    split = {r["group"]: r["n"] for r in rows if r["input"] == "test.csv"}
    assert split.get("known", 0) + split.get("new", 0) == n_answers
    assert to_markdown(rows).count("\n") == len(rows) + 1
