"""App handler edge cases: bad uploads, manual input, language switching, officer dashboard."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pytest

import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "test_sample.csv"
KANNADA, HINDI = "ಕನ್ನಡ (Kannada)", "हिन्दी (Hindi)"
MANUAL = dict(lang_label="English", station_id="01646500", day="2024-05-15", rain_today=20, rain_7d=60,
              rain_30d=120, tmax=25, tmin=14, soil=0.32, area=11560, elev=1300, slope=8, forest=55)


def _preds(path: str) -> np.ndarray:
    return pd.read_csv(path)["discharge_cfs"].to_numpy()


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE, dtype=str)


@pytest.fixture(scope="module")
def baseline() -> np.ndarray:
    return _preds(app.run_batch(str(SAMPLE), None)[0])


@pytest.fixture(scope="module")
def dashboard():
    event = next(iter(app.EVENT_LABELS))
    return app.refresh_dashboard("English", "replay", event, 7)


@pytest.mark.parametrize("content, message", [
    ("", "empty"),
    ("station_id,date,precipitation_mm\n", "no data rows"),
    ("station_id,precipitation_mm\n01646500,3\n", "missing required columns"),
    ("station_id,date\n01646500,notadate\n", "unparseable dates"),
])
def test_batch_rejects_bad_files_with_clear_message(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(gr.Error, match=message):
        app.run_batch(str(path), None)


def test_batch_semicolon_decimal_comma_matches_comma_csv(tmp_path, sample, baseline):
    euro = sample.assign(**{c: sample[c].str.replace(".", ",", regex=False)
                            for c in ("precipitation_mm", "soil_moisture", "temp_max_c")})
    path = tmp_path / "euro.csv"
    euro.to_csv(path, index=False, sep=";")
    assert np.array_equal(_preds(app.run_batch(str(path), None)[0]), baseline)


def test_batch_tab_separated_matches_comma_csv(tmp_path, sample, baseline):
    path = tmp_path / "tab.csv"
    sample.to_csv(path, index=False, sep="\t")
    assert np.array_equal(_preds(app.run_batch(str(path), None)[0]), baseline)


def test_batch_keeps_duplicates_and_order(tmp_path, sample):
    messy = pd.concat([sample.sample(frac=1, random_state=3), sample.head(5)])
    path = tmp_path / "messy.csv"
    messy.to_csv(path, index=False)
    out = pd.read_csv(app.run_batch(str(path), None)[0], dtype=str)
    assert len(out) == len(messy)
    assert (out["date"].to_numpy() == messy["date"].to_numpy()).all()


def test_batch_missing_weather_still_valid(tmp_path, sample):
    blank = sample.assign(precipitation_mm="", temp_max_c="", temp_min_c="", soil_moisture="")
    path = tmp_path / "blank.csv"
    blank.to_csv(path, index=False)
    preds = _preds(app.run_batch(str(path), None)[0])
    assert np.isfinite(preds).all() and (preds >= 0).all()


def test_batch_risk_file_has_levels(sample):
    risk = pd.read_csv(app.run_batch(str(SAMPLE), None)[1])
    assert {"risk_level", "likely_low_cfs", "likely_high_cfs"} <= set(risk.columns)
    assert (risk["likely_low_cfs"] <= risk["likely_high_cfs"]).all()


@pytest.mark.parametrize("field, value", [("day", ""), ("day", None), ("day", "31/31/2024"),
                                          ("rain_today", None), ("rain_7d", 5), ("tmin", 40), ("area", -1)])
def test_manual_rejects_bad_input(field, value):
    with pytest.raises(gr.Error):
        app.run_manual(**{**MANUAL, field: value})


@pytest.mark.parametrize("lang", app.LANG_CHOICES)
def test_manual_answers_in_each_language(lang):
    text = app.run_manual(**{**MANUAL, "lang_label": lang})
    assert "cfs" in text and "{" not in text


def test_replay_check_in_all_languages():
    event = next(iter(app.EVENT_LABELS))
    for lang in app.LANG_CHOICES:
        card, fig, alert, link = app.check_river(lang, "replay", None, event)
        assert "fc-card" in card and "{" not in alert and 'class="fc-wa"' in link


def test_language_change_before_first_check_is_noop():
    assert all(u == gr.update() for u in app.refresh_public("English", "live", app.DEFAULT_RIVER, None, ""))
    assert all(u == gr.update() for u in app.refresh_officer(KANNADA, None, None, 0, 2))


def test_officer_relabel_translates_every_label():
    updates = app.relabel_officer(KANNADA, "replay")
    assert len(updates) == 10
    labels = [u.get("label") or u.get("value") for u in updates]
    assert all(isinstance(x, str) and x for x in labels)
    assert not any(x.isascii() for x in labels if x)


def test_officer_dashboard_rerenders_in_new_language(dashboard):
    state, first = dashboard[2], dashboard[4]["value"]
    fmap, table, note, fig, card, cap = app.refresh_officer(HINDI, state, first, 30, 2)
    assert any("ऀ" <= ch <= "ॿ" for ch in "".join(table.columns))
    assert ET.parse(cap).getroot().tag.endswith("alert")
    assert len(fig.data) > len(app.station_detail(HINDI, state, first, 0, 2)[0].data)  # what-if line added


def test_station_detail_needs_dashboard_first():
    with pytest.raises(gr.Error):
        app.station_detail("English", None, app.DEFAULT_RIVER, 0, 2)


KIT = ROOT / "judge_test"


@pytest.mark.skipif(not (KIT / "answers.csv").exists(), reason="practice kit not built")
def test_batch_scores_against_answers_and_messy_file_matches():
    clean = app.run_batch(str(KIT / "test.csv"), None, str(KIT / "answers.csv"))
    messy = app.run_batch(str(KIT / "test_excel_messy.csv"), None, str(KIT / "answers.csv"))
    assert "rows matched" in clean[4] and clean[4] == messy[4]
    assert f"{len(pd.read_csv(KIT / 'answers.csv')):,} of" in clean[4]


def test_batch_answers_without_target_is_rejected(tmp_path):
    bad = tmp_path / "answers.csv"
    bad.write_text("station_id,date\n02087500,2023-06-01\n", encoding="utf-8")
    with pytest.raises(gr.Error, match="discharge_cfs"):
        app.run_batch(str(SAMPLE), None, str(bad))


def test_batch_answers_for_other_rows_is_rejected(tmp_path):
    other = tmp_path / "answers.csv"
    other.write_text("station_id,date,discharge_cfs\n99999999,1999-01-01,5\n", encoding="utf-8")
    with pytest.raises(gr.Error, match="No answers matched"):
        app.run_batch(str(SAMPLE), None, str(other))
