"""Rendering helpers for the app: risk card (HTML), hydrograph, map and ranking table."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .alerts import CFS_PER_CMS, fmt_cfs
from .forecast import OBS_COL
from .i18n import actions, t
from .product import Outlook
from .risk import LEVEL_COLORS, LEVELS, RiskTable
from .schema import DATE_COL, ID_COL

BAND_OPACITY = 0.12
MAP_ZOOM = 2.6
MAP_CENTER = {"lat": 39.0, "lon": -98.0}


def _e(text: object) -> str:
    return html.escape(str(text))


def level_badge(level: str, lang: str) -> str:
    color = LEVEL_COLORS.get(level, LEVEL_COLORS["unknown"])
    return (f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;'
            f'font-weight:700">{_e(t(f"level.{level}", lang))}</span>')


def drivers_html(o: Outlook, lang: str) -> str:
    items = []
    for group, mult in o.drivers:
        phrase = t("driver.up", lang, x=f"{mult:.1f}") if mult >= 1 else t("driver.down", lang, x=f"{1 / mult:.1f}")
        items.append(f"<li>{_e(t(f'driver.{group}', lang))}: {_e(phrase)}</li>")
    return "<ul style='margin:4px 0 0 18px'>" + "".join(items) + "</ul>" if items else ""


def card_html(o: Outlook, river: str, risk: RiskTable, lang: str, note: str, days: int) -> str:
    level = o.peak_level_name
    color = LEVEL_COLORS.get(level, LEVEL_COLORS["unknown"])
    acts = "".join(f"<li>{_e(a)}</li>" for a in actions(level, lang))
    obs_thr = risk.observed_thresholds(o.station_id)
    thr_line = ""
    if "warning" in obs_thr and "danger" in obs_thr:
        thr_line = (f"<div style='color:#57606a;font-size:13px;margin-top:6px'>"
                    f"{_e(t('card.thresholds', lang, warning=fmt_cfs(obs_thr['warning']), danger=fmt_cfs(obs_thr['danger'])))}</div>")
    peak = t("card.peak_line", lang, level=t(f"level.{level}", lang), date=o.peak_date,
             cfs=fmt_cfs(o.peak_cfs), cms=fmt_cfs(o.peak_cfs / CFS_PER_CMS))
    rng = t("card.range", lang, lo=fmt_cfs(o.peak_lo), hi=fmt_cfs(o.peak_hi), conf=t(f"conf.{o.confidence}", lang))
    why = (f"<div style='margin-top:10px'><b>{_e(t('card.why', lang))}</b>{drivers_html(o, lang)}</div>"
           if o.drivers else "")
    return f"""
<div style="border:3px solid {color};border-radius:14px;overflow:hidden;font-family:system-ui,sans-serif">
  <div style="background:{color};color:#fff;padding:14px 18px">
    <div style="font-size:14px;opacity:.9">{_e(t('card.title', lang))} · {_e(river)}</div>
    <div style="font-size:34px;font-weight:800;letter-spacing:.5px">{_e(t(f'level.{level}', lang))}</div>
    <div style="font-size:15px">{_e(t('card.peak', lang, days=days))}: {_e(peak)}</div>
  </div>
  <div style="padding:14px 18px;background:var(--background-fill-primary, #fff)">
    <div>{_e(t('card.today', lang))}: {level_badge(o.today_level_name, lang)}</div>
    <div style="margin-top:10px"><b>{_e(t('card.meaning', lang))}</b><br>{_e(t(f'meaning.{level}', lang))}</div>
    <div style="margin-top:10px"><b>{_e(t('card.actions', lang))}</b><ol style="margin:4px 0 0 18px">{acts}</ol></div>
    <div style="margin-top:10px;color:#57606a;font-size:14px">{_e(rng)}</div>
    {why}{thr_line}
    <div style="margin-top:10px;font-size:13px;color:#57606a">{_e(note)}</div>
    <div style="margin-top:6px;font-size:12px;color:#8c959f">{_e(t('disclaimer', lang))}</div>
  </div>
</div>"""


def hydrograph(o: Outlook, risk: RiskTable, lang: str, title: str, as_of: pd.Timestamp,
               alt: pd.DataFrame | None = None) -> go.Figure:
    d = o.daily
    fig = go.Figure()
    thr = risk.model_thresholds(o.station_id)
    top = float(np.nanmax([d["hi"].max(), d["predicted"].max(), d[OBS_COL].max() if OBS_COL in d else 0]))
    edges = [0.0] + [thr[k] for k in LEVELS[1:] if k in thr] + [max(top * 1.15, 1.0)]
    names = ["normal"] + [k for k in LEVELS[1:] if k in thr]
    for name, lo_e, hi_e in zip(names, edges[:-1], edges[1:]):
        if hi_e > lo_e:
            fig.add_hrect(y0=lo_e, y1=hi_e, fillcolor=LEVEL_COLORS[name], opacity=BAND_OPACITY, line_width=0,
                          annotation_text=t(f"level.{name}", lang), annotation_position="top left",
                          annotation_font_size=10)
    fig.add_trace(go.Scatter(x=pd.concat([d[DATE_COL], d[DATE_COL][::-1]]), y=pd.concat([d["hi"], d["lo"][::-1]]),
                             fill="toself", fillcolor="rgba(31,111,235,0.15)", line_width=0,
                             name=t("chart.range", lang), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d[DATE_COL], y=d["predicted"], mode="lines", line={"color": "#1f6feb", "width": 2.5},
                             name=t("chart.predicted", lang)))
    if alt is not None:
        fig.add_trace(go.Scatter(x=alt[DATE_COL], y=alt["predicted"], mode="lines",
                                 line={"color": "#bf3989", "width": 2, "dash": "dash"}, name="What-if"))
    if OBS_COL in d and d[OBS_COL].notna().any():
        fig.add_trace(go.Scatter(x=d[DATE_COL], y=d[OBS_COL], mode="lines+markers",
                                 line={"color": "#d1242f", "width": 1.2}, marker={"size": 4},
                                 name=t("chart.observed", lang)))
    fig.add_vline(x=as_of, line_dash="dot", line_color="#24292f")
    fig.add_annotation(x=as_of, y=1.02, yref="paper", text=t("chart.today", lang), showarrow=False)
    fig.update_layout(title=title, height=420, margin={"l": 50, "r": 20, "t": 50, "b": 40},
                      yaxis_title=t("chart.flow", lang), legend={"orientation": "h", "y": -0.15},
                      yaxis={"range": [0, edges[-1]]})
    return fig


def risk_map(results: dict[str, Outlook], stations: pd.DataFrame, lang: str) -> go.Figure:
    st = stations.set_index(ID_COL)
    rows = [(st.loc[sid, "latitude"], st.loc[sid, "longitude"], o.peak_level_name,
             f"{st.loc[sid, 'station_name']}<br>{t(f'level.{o.peak_level_name}', lang)} · {o.peak_date}"
             f"<br>{fmt_cfs(o.peak_cfs)} cfs") for sid, o in results.items() if sid in st.index]
    fig = go.Figure()
    for level in ["unknown", *LEVELS]:
        pts = [r for r in rows if r[2] == level]
        if not pts:
            continue
        fig.add_trace(go.Scattermap(lat=[p[0] for p in pts], lon=[p[1] for p in pts], mode="markers",
                                    marker={"size": 11 + 3 * max(LEVELS.index(level) if level in LEVELS else 0, 0),
                                            "color": LEVEL_COLORS[level]},
                                    text=[p[3] for p in pts], hoverinfo="text", name=t(f"level.{level}", lang)))
    fig.update_layout(map={"style": "open-street-map", "center": MAP_CENTER, "zoom": MAP_ZOOM},
                      height=460, margin={"l": 0, "r": 0, "t": 30, "b": 0}, title=t("officer.map", lang),
                      legend={"orientation": "h", "y": -0.02})
    return fig


def ranking_table(ordered: list[Outlook], stations: pd.DataFrame, lang: str) -> pd.DataFrame:
    names = stations.set_index(ID_COL)["station_name"]
    return pd.DataFrame({
        t("col.station", lang): [f"{names.get(o.station_id, o.station_id)} ({o.station_id})" for o in ordered],
        t("col.level", lang): [t(f"level.{o.peak_level_name}", lang) for o in ordered],
        t("col.peak_date", lang): [o.peak_date for o in ordered],
        t("col.peak_cfs", lang): [fmt_cfs(o.peak_cfs) for o in ordered],
        t("col.today_level", lang): [t(f"level.{o.today_level_name}", lang) for o in ordered],
    })
