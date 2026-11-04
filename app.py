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
from floodcast.metrics import score  # noqa: E402
from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.product import outlooks, ranking, what_if  # noqa: E402
from floodcast.risk import LEVELS, RiskTable  # noqa: E402
from floodcast.schema import (  # noqa: E402
    DATE_COL, ID_COL, STATIC_COLS, TARGET, coerce_frame, make_submission, read_table, validate_submission,
)
from floodcast.views import CARD_CSS, card_html, hydrograph, ranking_table, risk_map  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL = FloodcastModel.load(ROOT / "models")
RISK = RiskTable.load(ROOT / "models")
STATIONS = pd.read_csv(ROOT / "models" / "stations.csv", dtype={ID_COL: str}).sort_values("station_name")


def _station_label(name: str, state: str, sid: str) -> str:
    words = str(name).upper().replace(",", " ").replace(".", " ").split()
    state = str(state).strip()
    is_redundant = not state.isalpha() or (words and words[-1] == state.upper())  # some states are FIPS codes
    place = name if is_redundant else f"{name} · {state}"
    return f"{place} ({sid})"


STATION_LABELS = {_station_label(r.station_name, r.state, r.station_id): r.station_id for r in STATIONS.itertuples()}
LABEL_OF = {v: k for k, v in STATION_LABELS.items()}
NAME_OF = STATIONS.set_index(ID_COL)["station_name"].to_dict()
EVENTS_PATH = ROOT / "reports" / "replay_events.json"
EVENTS = json.loads(EVENTS_PATH.read_text(encoding="utf-8")) if EVENTS_PATH.exists() else []
EVENT_LABELS = {f"{NAME_OF.get(e['station_id'], e['station_id'])} — {pd.Timestamp(e['event_date']):%B %Y} "
                f"(as of {e['as_of']})": e for e in EVENTS}
DEFAULT_RIVER = LABEL_OF.get("01646500", next(iter(STATION_LABELS)))
LANG_CHOICES = list(LANGS)
_default_row = STATIONS[STATIONS[ID_COL] == STATION_LABELS[DEFAULT_RIVER]].iloc[0]
DEFAULT_STATIC = [round(float(_default_row[c]), 2) for c in STATIC_COLS]
EXAMPLE_DIR = ROOT / "judge_test"
EXAMPLE_FILES = [str(EXAMPLE_DIR / f) for f in ("test.csv", "test_excel_messy.csv", "answers.csv",
                                                "train_optional.csv") if (EXAMPLE_DIR / f).exists()]
MANUAL_HISTORY_DAYS = 30
APP_CSS = CARD_CSS + """
.gradio-container{width:100% !important;max-width:1500px !important;margin:0 auto !important}
.gradio-container, .gradio-container *{font-size:max(1em, 12px)}
footer{display:none !important}
/* Gradio's own faint helper text (slider 1/16 ends, upload "- or -") at readable contrast */
.gradio-container span.min_value{color:var(--body-text-color) !important}
.gradio-container span.max_value{color:var(--body-text-color) !important}
.gradio-container span.or{color:var(--body-text-color) !important}
.dark button[role="tab"].selected{color:#93c5fd !important}
"""
APP_THEME = gr.themes.Default(primary_hue="blue").set(  # darker blues: white button text at >= 4.5:1 contrast
    button_primary_background_fill="*primary_700", button_primary_background_fill_hover="*primary_800",
    button_primary_background_fill_dark="*primary_700", button_primary_background_fill_hover_dark="*primary_600",
    color_accent="*primary_700")
log = logging.getLogger("app")


def _mode_choices(lang: str) -> list[tuple[str, str]]:
    return [(t("ui.mode_live", lang), "live"), (t("ui.mode_replay", lang), "replay")]


def _event(event_label: str | None) -> dict:
    event = EVENT_LABELS.get(event_label or "")
    if event is None:
        raise gr.Error("Choose a past flood to replay.")
    return event


def _weather(mode: str, stations: pd.DataFrame, event_label: str | None, lang: str):
    """(weather frame, as_of date, meta) for live forecast or replay; meta rebuilds the note per language."""
    if mode == "replay":
        as_of = date.fromisoformat(_event(event_label)["as_of"])
        return replay_weather(stations, as_of), as_of, {"kind": "replay", "date": str(as_of)}
    try:
        weather, source, issued = live_weather(stations)
    except ForecastUnavailable as exc:
        raise gr.Error(t("err.network", lang)) from exc
    if source == "snapshot":
        return weather, date.fromisoformat(issued), {"kind": "snapshot", "date": issued}
    return weather, date.today(), {"kind": "live", "source": source}


def _note(meta: dict, lang: str) -> str:
    if meta["kind"] == "replay":
        return t("note.replay", lang, date=meta["date"])
    if meta["kind"] == "snapshot":
        return t("note.snapshot", lang, date=meta["date"])
    return t("note.live", lang, source=meta["source"]) + " " + t("note.far", lang)


def _with_observed(weather: pd.DataFrame, sid: str, as_of: date) -> pd.DataFrame:
    if OBS_COL in weather:
        return weather
    obs = recent_observed(sid, as_of - timedelta(days=45), as_of)
    return weather.merge(obs, on=DATE_COL, how="left") if not obs.empty else weather.assign(**{OBS_COL: np.nan})


# ---------------- Tab 1: Am I safe? ----------------
def check_river(lang_label: str, mode: str, river_label: str, event_label: str):
    lang = lang_code(lang_label)
    if mode == "replay":
        sid = _event(event_label)["station_id"]
    else:
        sid = STATION_LABELS.get(river_label or "")
        if sid is None:
            raise gr.Error("Choose a river station.")
    stations = STATIONS[STATIONS[ID_COL] == sid]
    weather, as_of, meta = _weather(mode, stations, event_label, lang)
    if mode == "live":
        weather = _with_observed(weather, sid, as_of)
    o = outlooks(MODEL, RISK, weather, as_of)[sid]
    river = NAME_OF.get(sid, sid)
    card = card_html(o, river, RISK, lang, _note(meta, lang), FORECAST_DAYS)
    fig = hydrograph(o, RISK, lang, river, pd.Timestamp(as_of))
    alert = alert_text(river, o.peak_level_name, o.peak_date, o.peak_cfs, lang)
    link = f'<a class="fc-wa" href="{whatsapp_link(alert)}" target="_blank" rel="noopener">{t("share.whatsapp", lang)}</a>'
    return card, fig, alert, link


def relabel_public(lang_label: str, mode: str):
    lang = lang_code(lang_label)
    return (gr.update(label=t("ui.mode", lang), choices=_mode_choices(lang), value=mode),
            gr.update(label=t("ui.river", lang)), gr.update(label=t("ui.event", lang)),
            gr.update(value=t("ui.check", lang)), gr.update(label=t("share.title", lang)))


def refresh_public(lang_label: str, mode: str, river_label: str, event_label: str, current_card: str):
    """Re-render an already shown result in the new language (no-op before the first check)."""
    if not current_card:
        return gr.update(), gr.update(), gr.update(), gr.update()
    return check_river(lang_label, mode, river_label, event_label)


def toggle_mode(mode: str):
    return gr.update(visible=mode != "replay"), gr.update(visible=mode == "replay")


# ---------------- Tab 2: Officer dashboard ----------------
def _dashboard_outputs(state: dict, lang: str):
    results = outlooks(MODEL, RISK, state["weather"], state["as_of"], window_days=state["window"])
    ordered = ranking(results)
    choices = [LABEL_OF[o.station_id] for o in ordered if o.station_id in LABEL_OF]
    return risk_map(results, STATIONS, lang), ranking_table(ordered, STATIONS, lang), _note(state["meta"], lang), choices


def refresh_dashboard(lang_label: str, mode: str, event_label: str, window: float):
    lang = lang_code(lang_label)
    weather, as_of, meta = _weather(mode, STATIONS, event_label, lang)
    state = {"weather": weather, "as_of": as_of, "meta": meta, "window": int(window)}
    fmap, table, note, choices = _dashboard_outputs(state, lang)
    return fmap, table, state, note, gr.update(choices=choices, value=choices[0] if choices else None)


def station_detail(lang_label: str, state: dict | None, station_label: str, extra_mm: float, day: float):
    lang = lang_code(lang_label)
    if not state or "weather" not in state:
        raise gr.Error(t("officer.hint", lang))
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
    card = card_html(o, river, RISK, lang, _note(state["meta"], lang), window, officer=True)
    st = STATIONS[STATIONS[ID_COL] == sid].iloc[0]
    xml = cap_xml(sid, river, float(st.latitude), float(st.longitude), o.peak_level_name, o.peak_date,
                  o.peak_cfs, lang)
    path = Path(tempfile.mkdtemp()) / f"cap_alert_{sid}.xml"
    path.write_text(xml, encoding="utf-8")
    return fig, card, str(path)


def relabel_officer(lang_label: str, mode: str):
    lang = lang_code(lang_label)
    return (gr.update(label=t("officer.mode", lang), choices=_mode_choices(lang), value=mode),
            gr.update(label=t("officer.event", lang)), gr.update(label=t("officer.window", lang)),
            gr.update(value=t("officer.refresh", lang)), gr.update(label=t("officer.table", lang)),
            gr.update(label=t("officer.station", lang)), gr.update(label=t("officer.whatif", lang)),
            gr.update(label=t("officer.whatif_day", lang)), gr.update(value=t("officer.detail_btn", lang)),
            gr.update(label=t("officer.cap", lang)))


def refresh_officer(lang_label: str, state: dict | None, station_label: str, extra_mm: float, day: float):
    """Re-render the dashboard in the new language from stored data (no new download)."""
    if not state or "weather" not in state:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    lang = lang_code(lang_label)
    fmap, table, note, _ = _dashboard_outputs(state, lang)
    if station_label in STATION_LABELS:
        fig, card, cap = station_detail(lang_label, state, station_label, extra_mm, day)
        return fmap, table, note, fig, card, cap
    return fmap, table, note, gr.update(), gr.update(), gr.update()


# ---------------- Tab 3: Batch ----------------
def _level_rank(name: str) -> int:
    return LEVELS.index(name) if name in LEVELS else -1


def score_markdown(test: pd.DataFrame, pred: np.ndarray, answers: pd.DataFrame) -> str:
    """Brief's metrics (RMSE, RMSE on each station's top-10% days, MAE, R²) against an answer key."""
    keyed = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), DATE_COL: test[DATE_COL].to_numpy(), "pred": pred})
    truth = answers.dropna(subset=[TARGET]).drop_duplicates([ID_COL, DATE_COL])[[ID_COL, DATE_COL, TARGET]]
    both = keyed.merge(truth, on=[ID_COL, DATE_COL], how="inner")
    if both.empty:
        raise gr.Error("No answers matched the test rows: check station_id and date in the answers file.")
    m = score(both[TARGET].to_numpy(), both["pred"].to_numpy(), both[ID_COL].to_numpy())
    return (f"### Accuracy against the answers ({len(both):,} of {len(test):,} rows matched)\n"
            f"| RMSE | RMSE on top-10% days | MAE | R² |\n|---|---|---|---|\n"
            f"| {m['rmse']:,.0f} cfs | {m['rmse_extreme']:,.0f} cfs | {m['mae']:,.0f} cfs | {m['r2']:.3f} |")


def run_batch(test_file, train_file, answers_file=None):
    if test_file is None:
        raise gr.Error("Upload test.csv first.")
    try:
        test = read_table(test_file)
        history = read_table(train_file) if train_file is not None else None
        answers = read_table(answers_file) if answers_file is not None else None
    except pd.errors.EmptyDataError as exc:
        raise gr.Error("The uploaded file is empty.") from exc
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise gr.Error(f"Could not read the file: {exc}") from exc
    if test.empty:
        raise gr.Error("test.csv has a header but no data rows.")
    if answers is not None and TARGET not in answers:
        raise gr.Error(f"The answers file needs a '{TARGET}' column.")
    feats, has_stats = MODEL.featurize(test, history)
    pred = MODEL.predict_features(feats, has_stats)
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
    scored = score_markdown(test, out[TARGET].to_numpy(), answers) if answers is not None else ""
    return (str(tmp / "predictions.csv"), str(tmp / "predictions_with_risk.csv"), summary.round(1), out.head(200),
            scored)


# ---------------- Tab 4: Manual ----------------
def run_manual(lang_label, station_id, day, rain_today, rain_7d, rain_30d, tmax, tmin, soil, area, elev, slope, forest):
    lang = lang_code(lang_label)
    try:
        target_day = pd.Timestamp(str(day or "").strip()[:10])
    except ValueError as exc:
        raise gr.Error("Date must be YYYY-MM-DD.") from exc
    if pd.isna(target_day):
        raise gr.Error("Date must be YYYY-MM-DD.")
    values = (rain_today, rain_7d, rain_30d, tmax, tmin, soil, area)
    if any(v is None for v in values):
        raise gr.Error("Fill in rain, temperatures, soil moisture and drainage area.")
    if min(rain_today, rain_7d, rain_30d, area) < 0 or rain_7d < rain_today or rain_30d < rain_7d:
        raise gr.Error("Rain totals must be non-negative and today ≤ 7-day ≤ 30-day; area must be ≥ 0.")
    if tmin > tmax:
        raise gr.Error("Min temperature cannot be above max temperature.")
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


def header_md(lang: str) -> str:
    return f"# {t('ui.title', lang)}\n{t('ui.subtitle', lang)}  \n_{t('disclaimer', lang)}_"


TAB_KEYS = ("ui.tab_public", "ui.tab_officer", "ui.tab_batch", "ui.tab_manual")


def relabel_page(lang_label: str):
    lang = lang_code(lang_label)
    return (header_md(lang), *[gr.Tab(label=t(k, lang)) for k in TAB_KEYS])


def _show():
    return gr.Column(visible=True)


def build() -> gr.Blocks:
    first_event = next(iter(EVENT_LABELS), None)
    with gr.Blocks(title="FLOODCAST — Midnight Cowens", fill_width=True) as demo:
        with gr.Row(equal_height=True):
            header = gr.Markdown(header_md("en"), min_width=420, scale=3)
            lang = gr.Dropdown(LANG_CHOICES, value=LANG_CHOICES[0], label="Language · भाषा · ಭಾಷೆ", min_width=240,
                               scale=1)
        tabs = []
        with gr.Tab(t(TAB_KEYS[0])) as tab:
            tabs.append(tab)
            with gr.Row(equal_height=True):
                mode = gr.Radio(_mode_choices("en"), value="live", label=t("ui.mode"), min_width=320)
                river = gr.Dropdown(list(STATION_LABELS), value=DEFAULT_RIVER, label=t("ui.river"), min_width=320)
                event = gr.Dropdown(list(EVENT_LABELS), value=first_event, label=t("ui.event"), visible=False,
                                    min_width=320)
            check = gr.Button(t("ui.check"), variant="primary")
            with gr.Column(visible=False) as public_out:
                card = gr.HTML()
                chart = gr.Plot(show_label=False)
                with gr.Row():
                    alert_box = gr.Textbox(label=t("share.title"), lines=4, buttons=["copy"], interactive=False,
                                           min_width=320, scale=3)
                    wa = gr.HTML(min_width=200)
            public_in, public_res = [lang, mode, river, event], [card, chart, alert_box, wa]
            mode.change(toggle_mode, mode, [river, event])
            lang.change(relabel_public, [lang, mode], [mode, river, event, check, alert_box]).then(
                refresh_public, [*public_in, card], public_res)
            check.click(check_river, public_in, public_res).success(_show, None, public_out)
            demo.load(check_river, public_in, public_res).success(_show, None, public_out)
        with gr.Tab(t(TAB_KEYS[1])) as tab:
            tabs.append(tab)
            state = gr.State()
            with gr.Row(equal_height=True):
                o_mode = gr.Radio(_mode_choices("en"), value="replay" if first_event else "live",
                                  label=t("officer.mode"), min_width=320)
                o_event = gr.Dropdown(list(EVENT_LABELS), value=first_event, label=t("officer.event"), min_width=320)
                window = gr.Slider(1, FORECAST_DAYS, value=7, step=1, label=t("officer.window"), min_width=260)
            refresh = gr.Button(t("officer.refresh"), variant="primary")
            note = gr.Markdown(t("officer.hint"))
            with gr.Row():
                fmap = gr.Plot(min_width=420, show_label=False, scale=5)
                table = gr.Dataframe(label=t("officer.table"), wrap=True, max_height=520, min_width=480, scale=6,
                                     column_widths=["34%", "16%", "19%", "14%", "17%"], interactive=False)
            with gr.Row(equal_height=True):
                pick = gr.Dropdown([], label=t("officer.station"), min_width=320, scale=2)
                extra = gr.Number(value=0, label=t("officer.whatif"), minimum=0, min_width=200)
                day = gr.Slider(1, FORECAST_DAYS, value=2, step=1, label=t("officer.whatif_day"), min_width=220)
            detail_btn = gr.Button(t("officer.detail_btn"))
            d_chart = gr.Plot(show_label=False)
            d_card = gr.HTML()
            cap_file = gr.File(label=t("officer.cap"))
            dash_in, dash_out = [lang, o_mode, o_event, window], [fmap, table, state, note, pick]
            detail_in, detail_out = [lang, state, pick, extra, day], [d_chart, d_card, cap_file]
            refresh.click(refresh_dashboard, dash_in, dash_out)
            demo.load(refresh_dashboard, dash_in, dash_out)
            detail_btn.click(station_detail, detail_in, detail_out)
            pick.change(station_detail, detail_in, detail_out)
            lang.change(relabel_officer, [lang, o_mode],
                        [o_mode, o_event, window, refresh, table, pick, extra, day, detail_btn, cap_file]).then(
                refresh_officer, detail_in, [fmap, table, note, *detail_out])
            o_mode.change(lambda m: gr.update(visible=m == "replay"), o_mode, o_event)
        with gr.Tab(t(TAB_KEYS[2])) as tab:
            tabs.append(tab)
            gr.Markdown("Upload `test.csv` (columns `station_id, date, precipitation_mm, temp_max_c, temp_min_c, "
                        "soil_moisture, drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct`). "
                        "Optional `train.csv` warms up rainfall history and teaches each station's typical flow.")
            with gr.Row():
                f_test = gr.File(label="test.csv", file_types=[".csv"], type="filepath", min_width=260)
                f_train = gr.File(label="train.csv (optional)", file_types=[".csv"], type="filepath", min_width=260)
                f_answers = gr.File(label="answers.csv (optional: shows accuracy)", file_types=[".csv"],
                                    type="filepath", min_width=260)
            b = gr.Button("Predict", variant="primary")
            with gr.Column(visible=False) as batch_out:
                score_md = gr.Markdown()
                with gr.Row():
                    out_file = gr.File(label="predictions.csv (official format)", min_width=300)
                    risk_file = gr.File(label="predictions_with_risk.csv (adds risk level + likely range)",
                                        min_width=300)
                out_summary = gr.Dataframe(label="Per-station summary", wrap=True)
                out_table = gr.Dataframe(label="Preview (first 200 rows)", wrap=True, max_height=420)
            b.click(run_batch, [f_test, f_train, f_answers], [out_file, risk_file, out_summary, out_table, score_md]
                    ).success(_show, None, batch_out)
            if EXAMPLE_FILES:
                with gr.Accordion("Practice files (download, then upload above)", open=False):
                    gr.Markdown("2024-07 → 2025-06, never used in training: 5 known rivers + 2 rivers the model has "
                                "never seen. `test_excel_messy.csv` is the same data saved the way Excel might "
                                "(zeros stripped, dd/mm/yyyy dates, soil in %); it gives identical predictions.")
                    gr.File(value=EXAMPLE_FILES, file_count="multiple", interactive=False, show_label=False)
        with gr.Tab(t(TAB_KEYS[3])) as tab:
            tabs.append(tab)
            with gr.Row():
                m_pick = gr.Dropdown(list(STATION_LABELS), value=DEFAULT_RIVER, min_width=320,
                                     label="Prefill from a known station (optional)")
                m_sid = gr.Textbox(label="station_id", value=STATION_LABELS[DEFAULT_RIVER], min_width=160)
                m_day = gr.Textbox(label="date (YYYY-MM-DD)", value="2024-05-15", min_width=160)
            with gr.Row():
                r0 = gr.Number(label="Rain today (mm)", value=20, minimum=0, min_width=180)
                r7 = gr.Number(label="Rain last 7 days incl. today (mm)", value=60, minimum=0, min_width=180)
                r30 = gr.Number(label="Rain last 30 days incl. today (mm)", value=120, minimum=0, min_width=180)
            with gr.Row():
                tx = gr.Number(label="Max temp (°C)", value=25, min_width=180)
                tn = gr.Number(label="Min temp (°C)", value=14, min_width=180)
                sm = gr.Number(label="Soil moisture 0–7 cm (m³/m³, or %)", value=0.32, minimum=0, min_width=180)
            with gr.Row():
                ar = gr.Number(label="Drainage area (sq mi)", value=DEFAULT_STATIC[0], minimum=0, min_width=160)
                el = gr.Number(label="Mean elevation (ft)", value=DEFAULT_STATIC[1], min_width=160)
                sl = gr.Number(label="Slope (%)", value=DEFAULT_STATIC[2], min_width=160)
                fo = gr.Number(label="Forest cover (%)", value=DEFAULT_STATIC[3], min_width=160)
            m_btn = gr.Button("Predict", variant="primary")
            m_out = gr.Markdown()
            m_pick.change(fill_station, m_pick, [m_sid, ar, el, sl, fo])
            m_btn.click(run_manual, [lang, m_sid, m_day, r0, r7, r30, tx, tn, sm, ar, el, sl, fo], m_out)
        lang.change(relabel_page, lang, [header, *tabs])
    return demo


demo = build()


def launch_app(port: int = 7860, open_browser: bool = True) -> None:
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, css=APP_CSS, theme=APP_THEME)


if __name__ == "__main__":
    launch_app()
