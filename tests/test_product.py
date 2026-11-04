"""Tests for the early-warning product layer: risk levels, languages, alerts, forecast, app."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast import forecast  # noqa: E402
from floodcast.alerts import CAP_NS, alert_text, cap_xml, whatsapp_link  # noqa: E402
from floodcast.i18n import LANGS, STRINGS, actions, lang_code, t  # noqa: E402
from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.product import outlooks, ranking, what_if  # noqa: E402
from floodcast.risk import LEVELS, UNKNOWN, RiskTable, classify, confidence, gumbel_flow, station_thresholds  # noqa: E402

MODEL = FloodcastModel.load(ROOT / "models")
RISK = RiskTable.load(ROOT / "models")
STATIONS = pd.read_csv(ROOT / "models" / "stations.csv", dtype={"station_id": str})
EVENTS = json.loads((ROOT / "reports" / "replay_events.json").read_text(encoding="utf-8"))
REPLAY = EVENTS[0]


# ---------- languages ----------
def test_every_key_exists_in_every_language():
    keys = set(STRINGS["en"])
    for code, table in STRINGS.items():
        assert set(table) == keys, f"{code} missing {keys - set(table)}"


@pytest.mark.parametrize("code", list(STRINGS))
def test_actions_and_formatting(code):
    for level in [*LEVELS, "unknown"]:
        assert len(actions(level, code)) == 3
        assert t(f"level.{level}", code)
    assert "12" in t("card.peak", code, days=12)


def test_lang_code_accepts_labels_codes_and_garbage():
    assert [lang_code(x) for x in [*LANGS, "hi", None, "xx"]] == ["en", "hi", "kn", "hi", "en", "en"]


# ---------- risk ----------
def test_gumbel_increases_with_return_period():
    am = np.array([100, 150, 120, 300, 180, 90, 210.0])
    assert gumbel_flow(am, 2) < gumbel_flow(am, 5) < gumbel_flow(am, 20)
    assert np.isnan(gumbel_flow(am[:2], 2))


def test_thresholds_monotone_and_floored_for_dry_rivers():
    dates = pd.date_range("2012-01-01", "2016-12-31", freq="D")
    dry = station_thresholds(dates, np.zeros(len(dates)))
    assert all(v >= 1.0 for v in dry.values())
    wet = station_thresholds(dates, np.abs(np.sin(np.arange(len(dates)) / 30)) * 1000)
    vals = [wet[k] for k in LEVELS[1:]]
    assert vals == sorted(vals)


def test_classify_unknown_and_levels():
    thr = pd.DataFrame({"station_id": ["A"], "elevated": [10.0], "warning": [20.0], "danger": [30.0], "extreme": [40.0]})
    lv = classify(np.array(["A", "A", "A", "A", "B"]), np.array([5, 10, 25, 99, 50.0]), thr)
    assert lv.tolist() == [0, 1, 2, 4, UNKNOWN]


def test_bounds_contain_prediction_and_confidence_labels():
    sid = RISK.model_thr.station_id.iloc[0]
    lo, hi = RISK.bounds(np.array([sid, "NEW"]), np.array([1000.0, 1000.0]), np.array([True, False]))
    assert (lo <= 1000).all() and (hi >= 1000).all()
    assert (hi[1] / lo[1]) > (hi[0] / lo[0])  # unseen stations get a wider range
    assert confidence(900, 1100, 1000) == "high" and confidence(10, 1000, 100) == "low"


# ---------- alerts ----------
@pytest.mark.parametrize("code", list(STRINGS))
def test_alert_text_and_whatsapp_link(code):
    text = alert_text("Test River", "danger", "2025-04-06", 12345.0, code)
    assert "12,345" in text and t("level.danger", code) in text
    link = whatsapp_link(text)
    assert link.startswith("https://wa.me/?text=") and " " not in link
    assert unquote(link.split("=", 1)[1]) == text


@pytest.mark.parametrize("level", [*LEVELS, "unknown"])
def test_cap_xml_is_valid_cap(level):
    xml = cap_xml("03301500", "Rolling Fork", 37.7, -85.7, level, "2025-04-06", 26000, "kn")
    root = ET.fromstring(xml.split("\n", 1)[1].encode("utf-8"))
    ns = {"c": CAP_NS}
    assert root.tag == f"{{{CAP_NS}}}alert"
    for path in ["c:identifier", "c:sent", "c:status", "c:info/c:severity", "c:info/c:area/c:areaDesc"]:
        assert root.find(path, ns) is not None and root.find(path, ns).text
    assert root.find("c:info/c:language", ns).text == "kn-IN"


# ---------- product ----------
def _replay_weather(stations: pd.DataFrame) -> tuple[pd.DataFrame, date]:
    as_of = date.fromisoformat(REPLAY["as_of"])
    return forecast.replay_weather(stations, as_of), as_of


def test_replay_event_is_flagged_and_works_with_sliced_input():
    w, as_of = _replay_weather(STATIONS)
    sliced = w[w.station_id == REPLAY["station_id"]]  # non-zero-based index (regression test)
    o = outlooks(MODEL, RISK, sliced, as_of)[REPLAY["station_id"]]
    assert o.peak_level >= LEVELS.index("warning")
    assert o.peak_lo <= o.peak_cfs <= o.peak_hi
    assert len(o.drivers) == 3 and all(m > 0 for _, m in o.drivers)


def test_outlooks_for_all_stations_and_ranking_order():
    w, as_of = _replay_weather(STATIONS)
    res = outlooks(MODEL, RISK, w, as_of, window_days=7)
    order = ranking(res)
    assert len(res) == len(STATIONS)
    peaks = [o.peak_level for o in order]
    assert peaks == sorted(peaks, reverse=True)


def test_what_if_adds_rain_and_validates():
    w, as_of = _replay_weather(STATIONS[STATIONS.station_id == REPLAY["station_id"]])
    more = what_if(w, REPLAY["station_id"], 100, 2, as_of)
    assert more.precipitation_mm.sum() == pytest.approx(w.precipitation_mm.fillna(0).sum() + 100, rel=1e-6)
    with pytest.raises(ValueError):
        what_if(w, REPLAY["station_id"], -5, 2, as_of)
    with pytest.raises(ValueError):
        what_if(w, REPLAY["station_id"], 5, 99, as_of)
    with pytest.raises(ValueError):
        outlooks(MODEL, RISK, w, as_of, window_days=0)


def test_station_without_history_is_unknown_not_crash():
    w, as_of = _replay_weather(STATIONS[STATIONS.station_id == REPLAY["station_id"]])
    o = outlooks(MODEL, RISK, w.assign(station_id="NEWSTATION"), as_of)["NEWSTATION"]
    assert o.peak_level == UNKNOWN and o.peak_level_name == "unknown" and o.peak_cfs >= 0


# ---------- forecast (no network) ----------
def _fake_daily(n_days: int, start: str, with_soil: bool) -> dict:
    days = pd.date_range(start, periods=n_days, freq="D").strftime("%Y-%m-%d").tolist()
    daily = {"time": days, "precipitation_sum": [1.0] * n_days, "temperature_2m_max": [20.0] * n_days,
             "temperature_2m_min": [10.0] * n_days}
    if with_soil:
        daily["soil_moisture_0_to_7cm_mean"] = [0.3] * n_days
    return {"daily": daily}


def test_fetch_live_merges_archive_and_forecast(monkeypatch):
    today = date(2026, 11, 5)
    st = STATIONS.head(2)
    arch = [_fake_daily(115, "2026-07-08", True)] * 2
    fc = [_fake_daily(26, "2026-10-26", False)] * 2
    calls = iter([arch, fc])
    monkeypatch.setattr(forecast, "_get_json", lambda url, params: next(calls))
    frame = forecast._fetch_live(st, today)
    for _, g in frame.groupby("station_id"):
        assert g.date.is_unique and g.date.is_monotonic_increasing
        assert g.soil_moisture.notna().all()  # forecast days carry the last archived soil value
        assert g.date.max() == pd.Timestamp("2026-11-20")


def test_live_falls_back_to_snapshot_then_errors(monkeypatch, tmp_path):
    def boom(url, params):
        raise forecast.ForecastUnavailable("offline")
    monkeypatch.setattr(forecast, "_get_json", boom)
    monkeypatch.setattr(forecast, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(forecast, "SNAPSHOT_PATH", tmp_path / "missing.parquet")
    with pytest.raises(forecast.ForecastUnavailable):
        forecast.live_weather(STATIONS.head(1), today=date(2026, 11, 5))
    snap = pd.DataFrame({"station_id": STATIONS.station_id.iloc[0], "date": pd.date_range("2026-10-01", periods=40),
                         "precipitation_mm": 1.0, "temp_max_c": 20.0, "temp_min_c": 10.0, "soil_moisture": 0.3})
    monkeypatch.setattr(forecast, "SNAPSHOT_PATH", tmp_path / "snap.parquet")
    forecast.save_snapshot(snap, "2026-10-25")
    frame, source, issued = forecast.live_weather(STATIONS.head(1), today=date(2026, 11, 5))
    assert source == "snapshot" and issued == "2026-10-25"
    assert frame.is_forecast.sum() == (snap.date > pd.Timestamp("2026-10-25")).sum()


def test_payload_with_wrong_location_count_is_rejected():
    with pytest.raises(forecast.ForecastUnavailable):
        forecast._daily_frames([_fake_daily(3, "2026-01-01", True)], ["a", "b"])
    with pytest.raises(forecast.ForecastUnavailable):
        forecast._daily_frames({"error": True}, ["a"])


# ---------- app handlers (offline replay paths) ----------
@pytest.fixture(scope="module")
def app_module():
    import app
    return app


@pytest.mark.parametrize("lang", list(LANGS))
def test_app_public_replay_all_languages(app_module, lang):
    ev = next(iter(app_module.EVENT_LABELS))
    card, fig, alert, link = app_module.check_river(lang, "replay", None, ev)
    assert t("card.title", lang_code(lang)) in card and "wa.me" in link and alert


def test_app_dashboard_detail_and_cap(app_module):
    ev = next(iter(app_module.EVENT_LABELS))
    fmap, table, state, note, pick = app_module.refresh_dashboard("English", "replay", ev, 7)
    assert len(table) == len(STATIONS)
    fig, card, cap = app_module.station_detail("English", state, pick["value"], 50, 3)
    ET.fromstring(Path(cap).read_text(encoding="utf-8").split("\n", 1)[1].encode("utf-8"))


def test_app_errors_are_user_friendly(app_module):
    import gradio as gr
    with pytest.raises(gr.Error):
        app_module.station_detail("English", None, "x", 0, 1)
    with pytest.raises(gr.Error):
        app_module.check_river("English", "replay", None, "not an event")
    with pytest.raises(gr.Error):
        app_module.run_manual("English", "x", "2024-05-15", 50, 10, 120, 25, 14, 0.3, 500, 1, 1, 1)
    with pytest.raises(gr.Error):
        app_module.run_batch(None, None)
