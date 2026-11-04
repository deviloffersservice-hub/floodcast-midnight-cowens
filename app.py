"""FLOODCAST early-warning app (Gradio) — Team Midnight Cowens. Runs locally: python app.py

Tabs
  1. Am I safe?        resident view: one river, colour-coded risk, what it means, what to do,
                       shareable WhatsApp alert, in English / Hindi / Kannada.
  2. Officer dashboard  all stations on a map, ranked by risk, hydrograph with flood bands,
                       likely range, why, what-if rainfall, CAP alert export.
  3. Batch             test.csv -> predictions.csv (official format) + optional risk file.
  4. Manual input      type one day's values.
Modes: live 16-day forecast (Open-Meteo; cached; offline snapshot fallback) or replay of a
past flood from stored data (fully offline).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from floodcast.alerts import CFS_PER_CMS, alert_text, cap_xml, whatsapp_link  # noqa: E402
from floodcast.forecast import (  # noqa: E402
    FORECAST_DAYS, OBS_COL, ForecastUnavailable, live_weather, recent_observed, replay_weather,
)
from floodcast.i18n import LANGS, lang_code, t  # noqa: E402
from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.product import outlooks, ranking, what_if  # noqa: E402
from floodcast.risk import LEVELS, RiskTable  # noqa: E402
from floodcast.schema import (  # noqa: E402
    DATE_COL, ID_COL, STATIC_COLS, TARGET, coerce_frame, make_submission, read_table, validate_submission,
)
from floodcast.views import card_html, hydrograph, ranking_table, risk_map  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL = FloodcastModel.load(ROOT / "models")
RISK = RiskTable.load(ROOT / "models")
STATIONS = pd.read_csv(ROOT / "models" / "stations.csv", dtype={ID_COL: str}).sort_values("station_name")
STATION_LABELS = {f"{r.station_name} · {r.state} ({r.station_id})": r.station_id for r in STATIONS.itertuples()}
LABEL_OF = {v: k for k, v in STATION_LABELS.items()}
NAME_OF = STATIONS.set_index(ID_COL)["station_name"].to_dict()
EVENTS_PATH = ROOT / "reports" / "replay_events.json"
EVENTS = json.loads(EVENTS_PATH.read_text(encoding="utf-8")) if EVENTS_PATH.exists() else []
EVENT_LABELS = {f"{NAME_OF.get(e['station_id'], e['station_id'])} — {pd.Timestamp(e['event_date']):%B %Y} "
                f"(as of {e['as_of']})": e for e in EVENTS}
DEFAULT_RIVER = LABEL_OF.get("01646500", next(iter(STATION_LABELS)))
LANG_CHOICES = list(LANGS)
MANUAL_HISTORY_DAYS = 30
log = logging.getLogger("app")


def _mode_choices(lang: str) -> list[tuple[str, str]]:
    return [(t("ui.mode_live", lang), "live"), (t("ui.mode_replay", lang), "replay")]


def _weather(mode: str, stations: pd.DataFrame, event_label: str | None, lang: str):
    """(weather frame, as_of date, note text) for live forecast or replay."""
    if mode == "replay":
        event = EVENT_LABELS.get(event_label or "")
        if event is None:
            raise gr.Error("Choose a past flood to replay.")
        as_of = date.fromisoformat(event["as_of"])
        return replay_weather(stations, as_of), as_of, t("note.replay", lang, date=str(as_of))
    try:
        weather, source, issued = live_weather(stations)
    except ForecastUnavailable as exc:
        raise gr.Error(t("err.network", lang)) from exc
    if source == "snapshot":
        return weather, date.fromisoformat(issued), t("note.snapshot", lang, date=issued)
    return weather, date.today(), t("note.live", lang, source=source) + " " + t("note.far", lang)


def _with_observed(weather: pd.DataFrame, sid: str, as_of: date) -> pd.DataFrame:
    if OBS_COL in weather:
        return weather
    obs = recent_observed(sid, as_of - timedelta(days=45), as_of)
    return weather.merge(obs, on=DATE_COL, how="left") if not obs.empty else weather.assign(**{OBS_COL: np.nan})


# ---------------- Tab 1: Am I safe? ----------------
def check_river(lang_label: str, mode: str, river_label: str, event_label: str):
    lang = lang_code(lang_label)
    if mode == "replay":
        event = EVENT_LABELS.get(event_label or "")
        if event is None:
            raise gr.Error("Choose a past flood to replay.")
        sid = event["station_id"]
    else:
        sid = STATION_LABELS.get(river_label or "")
        if sid is None:
            raise gr.Error("Choose a river station.")
    stations = STATIONS[STATIONS[ID_COL] == sid]
    weather, as_of, note = _weather(mode, stations, event_label, lang)
    if mode == "live":
        weather = _with_observed(weather, sid, as_of)
    o = outlooks(MODEL, RISK, weather, as_of)[sid]
    river = NAME_OF.get(sid, sid)
    card = card_html(o, river, RISK, lang, note, FORECAST_DAYS)
    fig = hydrograph(o, RISK, lang, river, pd.Timestamp(as_of))
    alert = alert_text(river, o.peak_level_name, o.peak_date, o.peak_cfs, lang)
    link = (f'<a href="{whatsapp_link(alert)}" target="_blank" rel="noopener" style="display:inline-block;'
            f'background:#25d366;color:#fff;padding:8px 14px;border-radius:8px;text-decoration:none;'
            f'font-weight:600">{t("share.whatsapp", lang)}</a>')
    return card, fig, alert, link


def relabel_public(lang_label: str, mode: str):
    lang = lang_code(lang_label)
    return (gr.update(label=t("ui.mode", lang), choices=_mode_choices(lang), value=mode),
            gr.update(label=t("ui.river", lang)), gr.update(label=t("ui.event", lang)),
            gr.update(value=t("ui.check", lang)), gr.update(label=t("share.title", lang)))


def toggle_mode(mode: str):
    return gr.update(visible=mode != "replay"), gr.update(visible=mode == "replay")


# ---------------- Tab 2: Officer dashboard ----------------
def refresh_dashboard(lang_label: str, mode: str, event_label: str, window: float):
    lang = lang_code(lang_label)
    weather, as_of, note = _weather(mode, STATIONS, event_label, lang)
    results = outlooks(MODEL, RISK, weather, as_of, window_days=int(window))
    ordered = ranking(results)
    state = {"weather": weather, "as_of": as_of, "note": note, "window": int(window)}
    choices = [LABEL_OF[o.station_id] for o in ordered if o.station_id in LABEL_OF]
    return (risk_map(results, STATIONS, lang), ranking_table(ordered, STATIONS, lang), state, note,
            gr.update(choices=choices, value=choices[0] if choices else None))


def station_detail(lang_label: str, state: dict | None, station_label: str, extra_mm: float, day: float):
    lang = lang_code(lang_label)
    if not state or "weather" not in state:
        raise gr.Error("Press 'Update dashboard' first.")
    sid = STATION_LABELS.get(station_label or "")
    if sid is None:
        raise gr.Error("Choose a station from the list.")
    weather = state["weather"][state["weather"][ID_COL] == sid]
    as_of, window = state["as_of"], state["window"]
    o = outlooks(MODEL, RISK, weather, as_of, window_days=window)[sid]
    alt = None
    extra = float(extra_mm or 0)
    if extra > 0:
        try:
            alt_w = what_if(weather, sid, extra, int(day), as_of)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        alt = outlooks(MODEL, RISK, alt_w, as_of, window_days=window)[sid].daily
    river = NAME_OF.get(sid, sid)
    fig = hydrograph(o, RISK, lang, river, pd.Timestamp(as_of), alt=alt)
    card = card_html(o, river, RISK, lang, state["note"], window, officer=True)
    st = STATIONS[STATIONS[ID_COL] == sid].iloc[0]
    xml = cap_xml(sid, river, float(st.latitude), float(st.longitude), o.peak_level_name, o.peak_date,
                  o.peak_cfs, lang)
    path = Path(tempfile.mkdtemp()) / f"cap_alert_{sid}.xml"
    path.write_text(xml, encoding="utf-8")
    return fig, card, str(path)


# ---------------- Tab 3: Batch ----------------
def _level_rank(name: str) -> int:
    return LEVELS.index(name) if name in LEVELS else -1


def run_batch(test_file, train_file):
    if test_file is None:
        raise gr.Error("Upload test.csv first.")
    try:
        test = read_table(test_file)
        history = read_table(train_file) if train_file is not None else None
        feats, has_stats = MODEL.featurize(test, history)
        pred = MODEL.predict_features(feats, has_stats)
    except ValueError as exc:
        raise gr.Error(f"Could not read the file: {exc}") from exc
    out = make_submission(test, pred)
    validate_submission(out, test)
    tmp = Path(tempfile.mkdtemp())
    out.to_csv(tmp / "predictions.csv", index=False, encoding="utf-8", lineterminator="\n")
    lo, hi = RISK.bounds(test[ID_COL].to_numpy(), out[TARGET].to_numpy(), has_stats)
    levels = RISK.levels(test[ID_COL].to_numpy(), out[TARGET].to_numpy())
    risk = out.assign(risk_level=[LEVELS[k] if k >= 0 else "unknown" for k in levels],
                      likely_low_cfs=np.round(lo, 1), likely_high_cfs=np.round(hi, 1))
    risk.to_csv(tmp / "predictions_with_risk.csv", index=False, encoding="utf-8", lineterminator="\n")
    summary = risk.groupby(ID_COL).agg(rows=(TARGET, "size"), mean_cfs=(TARGET, "mean"), max_cfs=(TARGET, "max"),
                                       highest_risk=("risk_level", lambda s: max(s, key=_level_rank))).reset_index()
    return str(tmp / "predictions.csv"), str(tmp / "predictions_with_risk.csv"), summary.round(1), out.head(200)


# ---------------- Tab 4: Manual ----------------
def run_manual(lang_label, station_id, day, rain_today, rain_7d, rain_30d, tmax, tmin, soil, area, elev, slope, forest):
    lang = lang_code(lang_label)
    try:
        target_day = pd.Timestamp(str(day)[:10])
    except ValueError as exc:
        raise gr.Error("Date must be YYYY-MM-DD.") from exc
    values = (rain_today, rain_7d, rain_30d, tmax, tmin, soil, area)
    if any(v is None for v in values):
        raise gr.Error("Fill in rain, temperatures, soil moisture and drainage area.")
    if min(rain_today, rain_7d, rain_30d, area) < 0 or rain_7d < rain_today or rain_30d < rain_7d:
        raise gr.Error("Rain totals must be non-negative and today ≤ 7-day ≤ 30-day; area must be ≥ 0.")
    days = pd.date_range(target_day - timedelta(days=MANUAL_HISTORY_DAYS - 1), target_day, freq="D")
    rain = np.where(days >= target_day - timedelta(days=6), (rain_7d - rain_today) / 6,
                    (rain_30d - rain_7d) / (MANUAL_HISTORY_DAYS - 7))
    rain[-1] = rain_today
    sid = str(station_id or "").strip() or "custom"
    frame = coerce_frame(pd.DataFrame({ID_COL: sid, DATE_COL: days, "precipitation_mm": rain, "temp_max_c": tmax,
                                       "temp_min_c": tmin, "soil_moisture": soil, "drainage_area_sqmi": area,
                                       "mean_elevation_ft": elev, "slope_pct": slope, "forest_cover_pct": forest}))
    feats, has_stats = MODEL.featurize(frame)
    pred = float(MODEL.predict_features(feats, has_stats)[-1])
    level_idx = int(RISK.levels(np.array([frame[ID_COL].iloc[-1]]), np.array([pred]))[0])
    level = LEVELS[level_idx] if level_idx >= 0 else "unknown"
    mode = "known-station model" if has_stats[-1] else "unseen-station model"
    return (f"### {t(f'level.{level}', lang)} — {pred:,.1f} cfs ({pred / CFS_PER_CMS:,.1f} m³/s)\n\n"
            f"{t(f'meaning.{level}', lang)}\n\n_Used the {mode}._")


def fill_station(choice: str):
    sid = STATION_LABELS.get(choice or "")
    if sid is None:
        return [gr.update()] * 5
    st = STATIONS[STATIONS[ID_COL] == sid].iloc[0]
    return [sid, *[None if pd.isna(st[c]) else float(st[c]) for c in STATIC_COLS]]


def build() -> gr.Blocks:
    first_event = next(iter(EVENT_LABELS), None)
    with gr.Blocks(title="FLOODCAST — Midnight Cowens") as demo:
        gr.Markdown(f"# {t('ui.title')}\n{t('ui.subtitle')}  \n_{t('disclaimer')}_")
        lang = gr.Dropdown(LANG_CHOICES, value=LANG_CHOICES[0], label="Language · भाषा · ಭಾಷೆ")
        with gr.Tab("Am I safe? · क्या मैं सुरक्षित हूँ? · ನಾನು ಸುರಕ್ಷಿತವೇ?"):
            with gr.Row():
                mode = gr.Radio(_mode_choices("en"), value="live", label=t("ui.mode"))
                river = gr.Dropdown(list(STATION_LABELS), value=DEFAULT_RIVER, label=t("ui.river"))
                event = gr.Dropdown(list(EVENT_LABELS), value=first_event, label=t("ui.event"), visible=False)
            check = gr.Button(t("ui.check"), variant="primary")
            card = gr.HTML()
            chart = gr.Plot()
            with gr.Row():
                alert_box = gr.Textbox(label=t("share.title"), lines=4, buttons=["copy"], interactive=False)
                wa = gr.HTML()
            mode.change(toggle_mode, mode, [river, event])
            lang.change(relabel_public, [lang, mode], [mode, river, event, check, alert_box])
            check.click(check_river, [lang, mode, river, event], [card, chart, alert_box, wa])
        with gr.Tab("Officer dashboard · अधिकारी डैशबोर्ड · ಅಧಿಕಾರಿ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್"):
            state = gr.State()
            with gr.Row():
                o_mode = gr.Radio(_mode_choices("en"), value="replay" if first_event else "live", label="Mode")
                o_event = gr.Dropdown(list(EVENT_LABELS), value=first_event, label="Past flood to replay")
                window = gr.Slider(1, FORECAST_DAYS, value=7, step=1, label="Look-ahead window (days)")
            refresh = gr.Button("Update dashboard", variant="primary")
            note = gr.Markdown()
            with gr.Row():
                fmap = gr.Plot()
                table = gr.Dataframe(label="Stations ranked by risk", wrap=True, max_height=460)
            pick = gr.Dropdown([], label="Station detail")
            with gr.Row():
                extra = gr.Number(value=0, label="What if? Extra rain (mm)", minimum=0)
                day = gr.Slider(1, FORECAST_DAYS, value=2, step=1, label="on forecast day")
            detail_btn = gr.Button("Show station / run what-if")
            d_chart = gr.Plot()
            d_card = gr.HTML()
            cap_file = gr.File(label="CAP 1.2 alert (XML) — for NDMA SACHET / IPAWS-style systems")
            refresh.click(refresh_dashboard, [lang, o_mode, o_event, window], [fmap, table, state, note, pick])
            detail_btn.click(station_detail, [lang, state, pick, extra, day], [d_chart, d_card, cap_file])
            pick.change(station_detail, [lang, state, pick, extra, day], [d_chart, d_card, cap_file])
        with gr.Tab("Batch: test.csv → predictions.csv"):
            gr.Markdown("Upload `test.csv` (columns `station_id, date, precipitation_mm, temp_max_c, temp_min_c, "
                        "soil_moisture, drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct`). "
                        "Optional `train.csv` warms up rainfall history and teaches each station's typical flow.")
            with gr.Row():
                f_test = gr.File(label="test.csv", file_types=[".csv"], type="filepath")
                f_train = gr.File(label="train.csv (optional)", file_types=[".csv"], type="filepath")
            b = gr.Button("Predict", variant="primary")
            with gr.Row():
                out_file = gr.File(label="predictions.csv (official format)")
                risk_file = gr.File(label="predictions_with_risk.csv (adds risk level + likely range)")
            out_summary = gr.Dataframe(label="Per-station summary")
            out_table = gr.Dataframe(label="Preview (first 200 rows)")
            b.click(run_batch, [f_test, f_train], [out_file, risk_file, out_summary, out_table])
        with gr.Tab("Manual input"):
            with gr.Row():
                m_pick = gr.Dropdown(list(STATION_LABELS), label="Prefill from a known station (optional)")
                m_sid = gr.Textbox(label="station_id", value="01646500")
                m_day = gr.Textbox(label="date (YYYY-MM-DD)", value="2024-05-15")
            with gr.Row():
                r0 = gr.Number(label="Rain today (mm)", value=20, minimum=0)
                r7 = gr.Number(label="Rain last 7 days incl. today (mm)", value=60, minimum=0)
                r30 = gr.Number(label="Rain last 30 days incl. today (mm)", value=120, minimum=0)
            with gr.Row():
                tx = gr.Number(label="Max temp (°C)", value=25)
                tn = gr.Number(label="Min temp (°C)", value=14)
                sm = gr.Number(label="Soil moisture 0–7 cm (m³/m³, or %)", value=0.32, minimum=0)
            with gr.Row():
                ar = gr.Number(label="Drainage area (sq mi)", value=11560, minimum=0)
                el = gr.Number(label="Mean elevation (ft)", value=1300)
                sl = gr.Number(label="Slope (%)", value=8)
                fo = gr.Number(label="Forest cover (%)", value=55)
            m_btn = gr.Button("Predict", variant="primary")
            m_out = gr.Markdown()
            m_pick.change(fill_station, m_pick, [m_sid, ar, el, sl, fo])
            m_btn.click(run_manual, [lang, m_sid, m_day, r0, r7, r30, tx, tn, sm, ar, el, sl, fo], m_out)
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
