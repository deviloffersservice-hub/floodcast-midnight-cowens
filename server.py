"""Run:  python server.py   then open http://127.0.0.1:8000  (the Gradio app is at /gradio/)

HOST and PORT environment variables override the address (the Docker image uses 0.0.0.0:7860).
"""
import base64
import os
import io
import tempfile
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app as core
from floodcast.views import CARD_CSS

LANG = core.LANG_CHOICES[0]  # English only
api = FastAPI()
STATE = {}


def run(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except Exception as e:
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(400, msg)


def parse_date(text):
    try:
        return date.fromisoformat(str(text or "").strip()[:10])
    except ValueError:
        raise HTTPException(400, "Choose a valid date.")


def fig_json(fig):
    if hasattr(fig, "to_plotly_json"):
        import plotly.io as pio
        return {"type": "plotly", "json": pio.to_json(fig)}
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "png", "data": data}


def table_json(df):
    df = getattr(df, "data", df)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.astype(object)
    df = df.where(df.notna(), "")
    return {"columns": [str(c) for c in df.columns], "rows": df.values.tolist()}


def replay_check(sid, as_of):
    """Replay any station on any stored date (not only the preset floods)."""
    stations = core.STATIONS[core.STATIONS[core.ID_COL] == sid]
    weather = core.replay_weather(stations, as_of)
    meta = {"kind": "replay", "date": str(as_of)}
    o = core.outlooks(core.MODEL, core.RISK, weather, as_of)[sid]
    river = core.NAME_OF.get(sid, sid)
    note = core._note(meta, LANG)
    card = core.card_html(o, river, core.RISK, LANG, note, core.FORECAST_DAYS)
    fig = core.hydrograph(o, core.RISK, LANG, river, pd.Timestamp(as_of))
    alert = core.alert_text(river, o.peak_level_name, o.peak_date, o.peak_cfs, LANG)
    return card, fig, alert


def replay_dashboard(as_of, window):
    weather = core.replay_weather(core.STATIONS, as_of)
    meta = {"kind": "replay", "date": str(as_of)}
    state = {"weather": weather, "as_of": as_of, "meta": meta, "window": window}
    fmap, table, note, choices = core._dashboard_outputs(state, LANG)
    return fmap, table, state, note, choices


class Check(BaseModel):
    mode: str
    river: str | None = None
    date: str | None = None


class Dash(BaseModel):
    mode: str
    date: str | None = None
    window: int = 7


class Detail(BaseModel):
    station: str
    extra: float = 0
    day: int = 2


class Manual(BaseModel):
    sid: str
    day: str
    rain0: float | None = None
    rain7: float | None = None
    rain30: float | None = None
    tmax: float | None = None
    tmin: float | None = None
    soil: float | None = None
    area: float | None = None
    elev: float | None = None
    slope: float | None = None
    forest: float | None = None


@api.get("/api/meta")
def meta():
    events = []
    for label, e in core.EVENT_LABELS.items():
        events.append({
            "label": label,
            "station": core.LABEL_OF.get(e["station_id"], ""),
            "date": e["as_of"],
        })
    last = date.fromisoformat(core.TRAINED_UNTIL) - timedelta(days=core.FORECAST_DAYS)
    return {
        "stations": list(core.STATION_LABELS),
        "events": events,
        "default_river": core.DEFAULT_RIVER,
        "date_max": str(last),
    }


@api.get("/api/station")
def station(label: str):
    return run(core.fill_station, label)


@api.get("/card.css")
def card_css():
    return PlainTextResponse(CARD_CSS, media_type="text/css")


@api.get("/plotly.js")
def plotly_js():
    from plotly.offline import get_plotlyjs
    return PlainTextResponse(get_plotlyjs(), media_type="application/javascript")


@api.post("/api/check")
def check(b: Check):
    if b.mode == "replay":
        sid = core.STATION_LABELS.get(b.river or "")
        if sid is None:
            raise HTTPException(400, "Choose a station from the list.")
        card, fig, alert = run(replay_check, sid, parse_date(b.date))
    else:
        card, fig, alert, _ = run(core.check_river, LANG, "live", b.river, None)
    return {"card": card, "fig": fig_json(fig), "alert": alert}


@api.post("/api/dashboard")
def dashboard(b: Dash):
    if b.mode == "replay":
        out = run(replay_dashboard, parse_date(b.date), b.window)
        fmap, table, state, note, choices = out
    else:
        out = run(core.refresh_dashboard, LANG, "live", None, b.window)
        fmap, table, state, note, upd = out
        choices = upd.get("choices", []) if isinstance(upd, dict) else []
        choices = [c[0] if isinstance(c, (list, tuple)) else c for c in choices]
    STATE["s"] = state
    return {
        "map": fig_json(fmap),
        "table": table_json(table),
        "note": note,
        "stations": choices,
    }


@api.post("/api/detail")
def detail(b: Detail):
    state = STATE.get("s")
    fig, card, cap = run(core.station_detail, LANG, state, b.station, b.extra, b.day)
    xml = Path(cap).read_text(encoding="utf-8")
    return {"fig": fig_json(fig), "card": card, "cap": xml}


@api.post("/api/batch")
def batch(
    test: UploadFile = File(...),
    train: UploadFile | None = File(None),
    answers: UploadFile | None = File(None),
    robustness: bool = Form(False),
):
    tmp = Path(tempfile.mkdtemp())

    def save(f):
        if f is None or not f.filename:
            return None
        p = tmp / Path(f.filename).name
        p.write_bytes(f.file.read())
        return str(p)

    paths = (save(test), save(train), save(answers))
    pred, risk, summary, preview, score, metrics = run(core.run_batch, *paths, robustness)
    return {
        "predictions": Path(pred).read_text(encoding="utf-8"),
        "risk": Path(risk).read_text(encoding="utf-8"),
        "summary": table_json(summary),
        "preview": table_json(preview),
        "score": score or "",
        "metrics": metrics or "",
    }


@api.post("/api/manual")
def manual(b: Manual):
    args = [
        LANG, b.sid, b.day,
        b.rain0, b.rain7, b.rain30,
        b.tmax, b.tmin, b.soil,
        b.area, b.elev, b.slope, b.forest,
    ]
    text = run(core.run_manual, *args)
    return {"md": text}


# The original Gradio app (all 3 languages) under /gradio/, mounted before the catch-all static site.
api = gr.mount_gradio_app(api, core.demo, path="/gradio", css=core.APP_CSS, theme=core.APP_THEME)
api.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True))

if __name__ == "__main__":
    uvicorn.run(api, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))