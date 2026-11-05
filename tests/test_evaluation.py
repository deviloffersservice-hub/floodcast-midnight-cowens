"""Extended evaluation (Batch tab): the 8 metric families."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app
from floodcast import evaluation

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "judge_test"
SAMPLE = ROOT / "data" / "sample" / "test_sample.csv"
HEADINGS = ("1 · Continuous", "Per river", "2 · Event detection", "3 · Operational alert burden",
            "4 · Probabilistic", "Reliability", "7 · Robustness", "8 · Efficiency")
needs_kit = pytest.mark.skipif(not (KIT / "answers.csv").exists(), reason="practice kit not built")


def _frame(pred_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    obs = np.concatenate([rng.gamma(2, 500, 200), rng.gamma(2, 50, 200)])
    ids = np.repeat(["A", "B"], 200)
    dates = np.tile(pd.date_range("2025-01-01", periods=200).strftime("%Y-%m-%d"), 2)
    pred = obs * pred_scale
    return evaluation.scored_frame(ids, dates, pred, obs, pred * 0.5, pred * 1.5, ids == "A")


def test_perfect_forecast_scores_perfectly():
    tables = evaluation.evaluate(_frame(), app.RISK)
    cont = tables["continuous"].set_index("Rivers")
    assert cont.loc["all", "NSE (all rows)"] == pytest.approx(1.0)
    assert cont.loc["all", "NSE (avg per river)"] == pytest.approx(1.0)
    assert set(cont.index) == {"all", "known", "new"}
    events = tables["events"].set_index("Rivers")
    assert events.loc["all", "AUPRC"] == pytest.approx(1.0)
    assert events.loc["all", "Recall @10% alert budget"] == pytest.approx(1.0)
    assert tables["uncertainty"].set_index("Rivers").loc["all", "Coverage (target 80%)"] == pytest.approx(1.0)


def test_flood_days_are_each_rivers_top_ten_percent():
    frame = _frame()
    share = frame.groupby("station_id").flood_day.mean()
    assert np.allclose(share, 0.10, atol=0.01)


def test_worse_forecast_scores_worse():
    good = evaluation.evaluate(_frame(1.0), app.RISK)["continuous"].set_index("Rivers")
    bad = evaluation.evaluate(_frame(2.0), app.RISK)["continuous"].set_index("Rivers")
    assert bad.loc["all", "NSE (avg per river)"] < good.loc["all", "NSE (avg per river)"]
    assert bad.loc["all", "MAE (cfs)"] > good.loc["all", "MAE (cfs)"]


def test_markdown_formats_and_handles_missing_values():
    table = pd.DataFrame([{"Rivers": "all", "Rows": 1234, "POD": 0.5, "FAR": float("nan"), "NSE": 0.12345}])
    md = evaluation.to_markdown(table)
    assert "| all | 1,234 | 50% | – | 0.123 |" in md


def test_peak_memory_is_measured():
    mb = evaluation.peak_memory_mb()
    assert math.isnan(mb) or mb > 0


def test_batch_without_answers_shows_efficiency_only():
    md = app.run_batch(str(SAMPLE), None)[5]
    assert "8 · Efficiency" in md and "Upload the answers file" in md
    assert "1 · Continuous" not in md


@needs_kit
def test_batch_with_answers_shows_all_families():
    md = app.run_batch(str(KIT / "test.csv"), None, str(KIT / "answers.csv"), True)[5]
    for heading in HEADINGS:
        assert heading in md, heading
    assert "| new |" in md and "| known |" in md


@needs_kit
def test_metrics_do_not_depend_on_row_order_or_excel_quirks():
    def body(md: str) -> str:
        return md.split("### 8 · Efficiency")[0]

    clean = app.run_batch(str(KIT / "test.csv"), None, str(KIT / "answers.csv"))[5]
    messy = app.run_batch(str(KIT / "test_excel_messy.csv"), None, str(KIT / "answers.csv"))[5]
    assert body(clean) == body(messy)
