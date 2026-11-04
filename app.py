"""FLOODCAST web app (Gradio) — Midnight Cowens.

Tabs
  1. Batch: upload test.csv (optional train.csv) -> predictions.csv in the official format.
  2. Live station: pick a USGS station + date range; weather is pulled from Open-Meteo,
     the model predicts discharge and, where available, USGS observations are overlaid.
  3. Manual: type one day's inputs (plus recent rainfall totals) and get one prediction.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.schema import (  # noqa: E402
    DATE_COL, ID_COL, STATIC_COLS, TARGET, coerce_frame, make_submission, read_table, validate_submission,
)

ROOT = Path(__file__).resolve().parent
MODEL = FloodcastModel.load(ROOT / "models")
STATIONS = pd.read_csv(ROOT / "models" / "stations.csv", dtype={ID_COL: str})
STATIONS = STATIONS[STATIONS[ID_COL].isin(MODEL.station_stats[ID_COL])].sort_values("drainage_area_sqmi", ascending=False)
DEFAULT_STATION = "01646500"
WARMUP_DAYS = 120
MAX_LIVE_DAYS = 730
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USGS_URL = "https://waterservices.usgs.gov/nwis/dv/"
HTTP_TIMEOUT = 60
log = logging.getLogger("app")

DISCLAIMER = ("Research prototype for the FLOODCAST hackathon. Not an official flood warning — "
              "always follow guidance from your local hydrological / disaster-management authority.")


def _station_choices() -> list[str]:
    return [f"{r.station_id} — {r.station_name}" for r in STATIONS.itertuples()]


def _default_choice() -> str:
    return next((c for c in _station_choices() if c.startswith(DEFAULT_STATION)), _station_choices()[0])


def _plot(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[DATE_COL], df["predicted_cfs"], label="Predicted", color="#1f6feb", lw=1.6)
    if "observed_cfs" in df and df["observed_cfs"].notna().any():
        ax.plot(df[DATE_COL], df["observed_cfs"], label="Observed (USGS)", color="#d1242f", lw=1.0, alpha=0.8)
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- Tab 1: batch ----------
def run_batch(test_file, train_file):
    if test_file is None:
        raise gr.Error("Upload test.csv first.")
    try:
        test = read_table(test_file)
        history = read_table(train_file) if train_file is not None else None
        pred = MODEL.predict(test, history)
    except ValueError as exc:
        raise gr.Error(f"Could not read the file: {exc}") from exc
    out = make_submission(test, pred)
    validate_submission(out, test)
    path = Path(tempfile.mkdtemp()) / "predictions.csv"
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    first = test[ID_COL].iloc[0]
    mask = (test[ID_COL] == first).to_numpy()
    sub = pd.DataFrame({DATE_COL: test.loc[mask, DATE_COL].to_numpy(), "predicted_cfs": out.loc[mask, TARGET].to_numpy()})
    fig = _plot(sub.sort_values(DATE_COL), f"Station {first}")
    summary = out.groupby(ID_COL)[TARGET].agg(["count", "mean", "max"]).round(1).reset_index()
    return str(path), out.head(200), summary, fig


# ---------- Tab 2: live ----------
def _fetch_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
              "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean",
              "timezone": "auto"}
    resp = requests.get(METEO_URL, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    d = resp.json()["daily"]
    return pd.DataFrame({DATE_COL: d["time"], "precipitation_mm": d["precipitation_sum"],
                         "temp_max_c": d["temperature_2m_max"], "temp_min_c": d["temperature_2m_min"],
                         "soil_moisture": d["soil_moisture_0_to_7cm_mean"]})


def _fetch_usgs(site: str, start: date, end: date) -> pd.DataFrame:
    params = {"format": "json", "sites": site, "startDT": str(start), "endDT": str(end),
              "parameterCd": "00060", "statCd": "00003", "siteStatus": "all"}
    try:
        resp = requests.get(USGS_URL, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        rows = [(v["dateTime"][:10], float(v["value"]))
                for ts in resp.json()["value"]["timeSeries"] for b in ts["values"] for v in b["value"]]
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("USGS observations unavailable for %s: %s", site, exc)
        rows = []
    obs = pd.DataFrame(rows, columns=[DATE_COL, "observed_cfs"])
    obs = obs[obs.observed_cfs > -999].drop_duplicates(DATE_COL)
    obs[DATE_COL] = pd.to_datetime(obs[DATE_COL])
    return obs


def run_live(choice: str, start: str, end: str):
    if not choice:
        raise gr.Error("Pick a station.")
    sid = choice.split(" — ")[0]
    st = STATIONS[STATIONS[ID_COL] == sid].iloc[0]
    try:
        d0, d1 = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    except ValueError as exc:
        raise gr.Error("Dates must be YYYY-MM-DD.") from exc
    if d1 < d0 or (d1 - d0).days > MAX_LIVE_DAYS:
        raise gr.Error(f"End must be after start and the range at most {MAX_LIVE_DAYS} days.")
    try:
        wx = _fetch_weather(st.latitude, st.longitude, d0 - timedelta(days=WARMUP_DAYS), d1)
    except requests.RequestException as exc:
        raise gr.Error(f"Weather service unavailable ({exc}). Try again in a minute.") from exc
    frame = coerce_frame(wx.assign(**{ID_COL: sid, **{c: st[c] for c in STATIC_COLS}}))
    pred = MODEL.predict(frame)
    res = frame[[DATE_COL]].assign(predicted_cfs=pred.round(1))
    res = res[res[DATE_COL] >= pd.Timestamp(d0)]
    res = res.merge(_fetch_usgs(sid, d0, d1), on=DATE_COL, how="left")
    fig = _plot(res, f"{st.station_name} ({sid})")
    table = res.assign(**{DATE_COL: res[DATE_COL].dt.strftime("%Y-%m-%d")})
    return fig, table


# ---------- Tab 3: manual ----------
def run_manual(station_id, day, rain_today, rain_7d, rain_30d, tmax, tmin, soil, area, elev, slope, forest):
    try:
        target_day = pd.Timestamp(str(day)[:10])
    except ValueError as exc:
        raise gr.Error("Date must be YYYY-MM-DD.") from exc
    if any(v is None or v < 0 for v in (rain_today, rain_7d, rain_30d, area)):
        raise gr.Error("Rainfall and drainage area must be non-negative numbers.")
    days = pd.date_range(target_day - timedelta(days=29), target_day, freq="D")
    prev7 = max(rain_7d - rain_today, 0) / 6
    prev30 = max(rain_30d - rain_7d, 0) / 23
    rain = np.where(days >= target_day - timedelta(days=6), prev7, prev30)
    rain[-1] = rain_today
    frame = pd.DataFrame({ID_COL: str(station_id).strip() or "custom", DATE_COL: days,
                          "precipitation_mm": rain, "temp_max_c": tmax, "temp_min_c": tmin,
                          "soil_moisture": soil, "drainage_area_sqmi": area, "mean_elevation_ft": elev,
                          "slope_pct": slope, "forest_cover_pct": forest})
    pred = MODEL.predict(coerce_frame(frame))[-1]
    known = str(station_id).strip().zfill(8) in set(MODEL.station_stats[ID_COL])
    mode = "known-station model (uses this station's history)" if known else "unseen-station model"
    return f"### Predicted discharge: **{pred:,.1f} cfs**  ({pred / 35.3147:,.1f} m³/s)\n\nUsed the {mode}."


def fill_station(choice: str):
    if not choice:
        return [gr.update()] * 5
    st = STATIONS[STATIONS[ID_COL] == choice.split(" — ")[0]].iloc[0]
    return [st[ID_COL], *[None if pd.isna(st[c]) else float(st[c]) for c in STATIC_COLS]]


with gr.Blocks(title="FLOODCAST — Midnight Cowens") as demo:
    gr.Markdown("# FLOODCAST — River discharge predictor\n**Team Midnight Cowens.** Predicts daily river "
                "discharge (cubic feet per second) from rainfall, temperature, soil moisture and catchment "
                f"attributes. CPU-only LightGBM.\n\n_{DISCLAIMER}_")
    with gr.Tab("1 · Batch (test.csv → predictions.csv)"):
        gr.Markdown("Upload a CSV with columns `station_id, date, precipitation_mm, temp_max_c, temp_min_c, "
                    "soil_moisture, drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct`. "
                    "Optionally add train.csv: its weather warms up rainfall history and its discharge "
                    "teaches the model each station's typical flow.")
        with gr.Row():
            f_test = gr.File(label="test.csv", file_types=[".csv"], type="filepath")
            f_train = gr.File(label="train.csv (optional)", file_types=[".csv"], type="filepath")
        b = gr.Button("Predict", variant="primary")
        out_file = gr.File(label="predictions.csv")
        out_plot = gr.Plot(label="First station")
        out_summary = gr.Dataframe(label="Per-station summary")
        out_table = gr.Dataframe(label="Preview (first 200 rows)")
        b.click(run_batch, [f_test, f_train], [out_file, out_table, out_summary, out_plot])
    with gr.Tab("2 · Live station"):
        gr.Markdown("Pulls real weather from Open-Meteo for the station location, predicts every day, and "
                    "overlays USGS observed discharge when available.")
        with gr.Row():
            dd = gr.Dropdown(_station_choices(), label="USGS station", value=_default_choice())
            s_in = gr.Textbox(label="Start (YYYY-MM-DD)", value="2024-01-01")
            e_in = gr.Textbox(label="End (YYYY-MM-DD)", value="2024-12-31")
        b2 = gr.Button("Run", variant="primary")
        live_plot = gr.Plot()
        live_table = gr.Dataframe()
        b2.click(run_live, [dd, s_in, e_in], [live_plot, live_table])
    with gr.Tab("3 · Manual input"):
        with gr.Row():
            pick = gr.Dropdown(_station_choices(), label="Prefill from a known station (optional)")
            sid = gr.Textbox(label="station_id", value="01646500")
            day = gr.Textbox(label="date (YYYY-MM-DD)", value="2024-05-15")
        with gr.Row():
            rain_t = gr.Number(label="Rain today (mm)", value=20)
            rain_7 = gr.Number(label="Rain last 7 days incl. today (mm)", value=60)
            rain_30 = gr.Number(label="Rain last 30 days incl. today (mm)", value=120)
        with gr.Row():
            tmax = gr.Number(label="Max temp (°C)", value=25)
            tmin = gr.Number(label="Min temp (°C)", value=14)
            soil = gr.Number(label="Soil moisture 0–7 cm (m³/m³, or %)", value=0.32)
        with gr.Row():
            area = gr.Number(label="Drainage area (sq mi)", value=11560)
            elev = gr.Number(label="Mean elevation (ft)", value=1300)
            slope = gr.Number(label="Slope (%)", value=8)
            forest = gr.Number(label="Forest cover (%)", value=55)
        b3 = gr.Button("Predict", variant="primary")
        md = gr.Markdown()
        pick.change(fill_station, pick, [sid, area, elev, slope, forest])
        b3.click(run_manual, [sid, day, rain_t, rain_7, rain_30, tmax, tmin, soil, area, elev, slope, forest], md)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
