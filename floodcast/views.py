"""Rendering helpers for the app: risk card (HTML), hydrograph, map and ranking table.

Readability rules: no text below 12 px, theme-aware colours (light and dark mode), and no
labels drawn inside chart bands (thin bands made them overlap); levels are a legend key.
"""
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

BAND_OPACITY = 0.14
MAP_ZOOM = 2.6
MAP_CENTER = {"lat": 39.0, "lon": -98.0}
FONT = {"family": "system-ui, -apple-system, Segoe UI, Noto Sans, Noto Sans Devanagari, Noto Sans Kannada, sans-serif",
        "size": 13}
SMALL_FONT = 12
# Darker shades of the level colours for backgrounds behind white text (WCAG AA, contrast >= 4.5:1).
# The brighter LEVEL_COLORS stay for map markers and chart bands.
LEVEL_TEXT_BG = {"normal": "#1a7f37", "elevated": "#9a6700", "warning": "#bc4c00", "danger": "#cf222e",
                 "extreme": "#8250df", "unknown": "#57606a"}
CHART_HEIGHT = 470
MAP_HEIGHT = 500

# Card styles: CSS variables come from the Gradio theme, so text keeps contrast in light/dark mode.
MUTED = "color:var(--body-text-color-subdued, #57606a)"
CARD_CSS = """
.fc-card{border:3px solid var(--fc-c);border-radius:14px;overflow:hidden;font-size:15px;line-height:1.5;
  overflow-wrap:anywhere;word-break:normal;max-width:100%}
.fc-head{background:var(--fc-c);color:#fff;padding:14px 18px}
.fc-head .fc-sub{font-size:14px;opacity:.95}
.fc-head .fc-level{font-size:34px;font-weight:800;letter-spacing:.5px;line-height:1.2}
.fc-head .fc-peak{font-size:16px}
.fc-body{padding:14px 18px;background:var(--block-background-fill, #fff);color:var(--body-text-color, #1b2430)}
.fc-body .fc-sec{margin-top:12px}
.fc-body ol,.fc-body ul{margin:4px 0 0 22px;padding:0}
.fc-muted{color:var(--body-text-color-subdued, #57606a);font-size:14px}
.fc-small{color:var(--body-text-color-subdued, #57606a);font-size:13px}
.fc-badge{color:#fff;padding:2px 10px;border-radius:12px;font-weight:700;white-space:nowrap}
.fc-card .fc-head,.fc-card .fc-head *,.fc-card .fc-badge{color:#fff !important}
.fc-card .fc-body{color:var(--body-text-color, #1b2430)}
/* Secondary text: the theme text colour at 75% strength, so contrast holds in light and dark mode.
   One selector per rule: Gradio prefixes comma lists unevenly. */
.fc-card .fc-muted{color:color-mix(in srgb, var(--body-text-color, #1b2430) 75%, transparent) !important}
.fc-card .fc-small{color:color-mix(in srgb, var(--body-text-color, #1b2430) 75%, transparent) !important}
.fc-wa{display:inline-block;background:#0b7a3e;color:#fff !important;padding:10px 16px;border-radius:8px;
  text-decoration:none;font-weight:700;font-size:15px}
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def level_badge(level: str, lang: str) -> str:
    color = LEVEL_TEXT_BG.get(level, LEVEL_TEXT_BG["unknown"])
    return f'<span class="fc-badge" style="background:{color}">{_e(t(f"level.{level}", lang))}</span>'


def drivers_html(o: Outlook, lang: str) -> str:
    items = []
    for group, mult in o.drivers:
        phrase = t("driver.up", lang, x=f"{mult:.1f}") if mult >= 1 else t("driver.down", lang, x=f"{1 / mult:.1f}")
        items.append(f"<li>{_e(t(f'driver.{group}', lang))}: {_e(phrase)}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else ""


def card_html(o: Outlook, river: str, risk: RiskTable, lang: str, note: str, days: int,
              officer: bool = False) -> str:
    level = o.peak_level_name
    color = LEVEL_TEXT_BG.get(level, LEVEL_TEXT_BG["unknown"])
    acts = "".join(f"<li>{_e(a)}</li>" for a in actions(level, lang))
    obs_thr = risk.observed_thresholds(o.station_id)
    thr_line = ""
    if officer and "warning" in obs_thr and "danger" in obs_thr:
        thr_line = (f"<div class='fc-sec fc-small'>"
                    f"{_e(t('card.thresholds', lang, warning=fmt_cfs(obs_thr['warning']), danger=fmt_cfs(obs_thr['danger'])))}"
                    f"<br>{_e(t('card.method', lang))}</div>")
    peak = t("card.peak_line", lang, level=t(f"level.{level}", lang), date=o.peak_date,
             cfs=fmt_cfs(o.peak_cfs), cms=fmt_cfs(o.peak_cfs / CFS_PER_CMS))
    rng = t("card.range", lang, lo=fmt_cfs(o.peak_lo), hi=fmt_cfs(o.peak_hi), conf=t(f"conf.{o.confidence}", lang))
    why = f"<div class='fc-sec'><b>{_e(t('card.why', lang))}</b>{drivers_html(o, lang)}</div>" if o.drivers else ""
    return f"""
<div class="fc-card" style="--fc-c:{color}">
  <div class="fc-head">
    <div class="fc-sub">{_e(t('card.title', lang))} · {_e(river)}</div>
    <div class="fc-level">{_e(t(f'level.{level}', lang))}</div>
    <div class="fc-peak">{_e(t('card.peak', lang, days=days))}: {_e(peak)}</div>
  </div>
  <div class="fc-body">
    <div>{_e(t('card.today', lang))}: {level_badge(o.today_level_name, lang)}</div>
    <div class="fc-sec"><b>{_e(t('card.meaning', lang))}</b><br>{_e(t(f'meaning.{level}', lang))}</div>
    <div class="fc-sec"><b>{_e(t('card.actions', lang))}</b><ol>{acts}</ol></div>
    <div class="fc-sec fc-muted">{_e(rng)}</div>
    {why}{thr_line}
    <div class="fc-sec fc-small">{_e(note)}</div>
    <div class="fc-small" style="margin-top:6px">{_e(t('disclaimer', lang))}</div>
  </div>
</div>"""


def _base_layout(fig: go.Figure, title: str, height: int) -> None:
    fig.update_layout(template="plotly_white", font=FONT, title={"text": title, "font": {"size": 16}, "x": 0.01},
                      height=height, autosize=True, hoverlabel={"font": {"size": 13}},
                      legend={"font": {"size": SMALL_FONT}, "orientation": "h", "yanchor": "top", "y": -0.14,
                              "xanchor": "left", "x": 0})


def hydrograph(o: Outlook, risk: RiskTable, lang: str, title: str, as_of: pd.Timestamp,
               alt: pd.DataFrame | None = None) -> go.Figure:
    d = o.daily
    fig = go.Figure()
    thr = risk.model_thresholds(o.station_id)
    obs_max = d[OBS_COL].max() if OBS_COL in d and d[OBS_COL].notna().any() else 0.0
    alt_max = alt["predicted"].max() if alt is not None and len(alt) else 0.0
    top = float(np.nanmax([d["hi"].max(), d["predicted"].max(), obs_max, alt_max, 1.0]))
    y_top = top * 1.12
    edges = [0.0] + [thr[k] for k in LEVELS[1:] if k in thr] + [max(y_top, 1.0)]
    names = ["normal"] + [k for k in LEVELS[1:] if k in thr]
    for name, lo_e, hi_e in zip(names, edges[:-1], edges[1:]):
        if hi_e > lo_e and lo_e < y_top:
            fig.add_hrect(y0=lo_e, y1=min(hi_e, y_top), fillcolor=LEVEL_COLORS[name], opacity=BAND_OPACITY,
                          line_width=0, layer="below")
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t(f"level.{name}", lang),
                                     marker={"symbol": "square", "size": 12, "color": LEVEL_COLORS[name]},
                                     hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pd.concat([d[DATE_COL], d[DATE_COL][::-1]]), y=pd.concat([d["hi"], d["lo"][::-1]]),
                             fill="toself", fillcolor="rgba(31,111,235,0.18)", line_width=0,
                             name=t("chart.range", lang), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d[DATE_COL], y=d["predicted"], mode="lines", line={"color": "#1f6feb", "width": 2.8},
                             name=t("chart.predicted", lang), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} cfs<extra></extra>"))
    if alt is not None and len(alt):
        fig.add_trace(go.Scatter(x=alt[DATE_COL], y=alt["predicted"], mode="lines",
                                 line={"color": "#bf3989", "width": 2.4, "dash": "dash"}, name=t("chart.whatif", lang),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} cfs<extra></extra>"))
    if OBS_COL in d and d[OBS_COL].notna().any():
        fig.add_trace(go.Scatter(x=d[DATE_COL], y=d[OBS_COL], mode="lines+markers",
                                 line={"color": "#d1242f", "width": 1.4}, marker={"size": 5},
                                 name=t("chart.observed", lang), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} cfs<extra></extra>"))
    fig.add_vline(x=as_of, line_dash="dot", line_color="#24292f", line_width=2)
    fig.add_annotation(x=as_of, y=1.0, yref="paper", yanchor="bottom", text=t("chart.today", lang),
                       showarrow=False, font={"size": SMALL_FONT + 1})
    _base_layout(fig, title, CHART_HEIGHT)
    fig.update_layout(margin={"l": 60, "r": 20, "t": 60, "b": 110}, yaxis_title=t("chart.flow", lang),
                      yaxis={"range": [0, y_top], "tickfont": {"size": SMALL_FONT}, "tickformat": ",~s"},
                      xaxis={"type": "date", "tickfont": {"size": SMALL_FONT}, "tickformat": "%d %b"})
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
        rank = LEVELS.index(level) if level in LEVELS else 0
        fig.add_trace(go.Scattermap(lat=[p[0] for p in pts], lon=[p[1] for p in pts], mode="markers",
                                    marker={"size": 12 + 3 * rank, "color": LEVEL_COLORS[level]},
                                    text=[p[3] for p in pts], hoverinfo="text", name=t(f"level.{level}", lang)))
    _base_layout(fig, t("officer.map", lang), MAP_HEIGHT)
    fig.update_layout(map={"style": "open-street-map", "center": MAP_CENTER, "zoom": MAP_ZOOM},
                      margin={"l": 0, "r": 0, "t": 44, "b": 60}, legend={"y": -0.02})
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
