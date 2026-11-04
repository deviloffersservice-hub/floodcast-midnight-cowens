"""Build the 5-slide presentation (reports/slides.pdf): Problem -> Data -> Model -> Results -> Limitations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
INK, ACCENT, MUTED, OBS = "#1b2430", "#1f6feb", "#57606a", "#d1242f"
PLOT_STATIONS = ("07010000", "01646500")
FOOTER = "FLOODCAST · Team Midnight Cowens"


def _page(title: str, subtitle: str) -> plt.Figure:
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.05, 0.9, title, fontsize=26, weight="bold", color=INK)
    fig.text(0.05, 0.855, subtitle, fontsize=14, color=MUTED)
    fig.text(0.05, 0.03, FOOTER, fontsize=11, color=MUTED)
    return fig


def _bullets(fig: plt.Figure, items: list[str], x: float = 0.06, y: float = 0.76, step: float = 0.07,
             size: int = 15) -> None:
    for b in items:
        fig.text(x, y, "•  " + b, fontsize=size, color=INK)
        y -= step


def _image(fig: plt.Figure, path: Path, rect: list[float]) -> None:
    if path.exists():
        ax = fig.add_axes(rect)
        ax.imshow(mpimg.imread(path))
        ax.axis("off")


def _fmt(m: dict) -> list[str]:
    return [f"{m['rmse']:,.0f}", f"{m['rmse_extreme']:,.0f}", f"{m['mae']:,.0f}", f"{m['r2']:.3f}"]


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    final = json.loads((REPORTS / "final_metrics.json").read_text(encoding="utf-8"))
    alerts = json.loads((REPORTS / "alert_skill.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(ROOT / "predictions.csv", dtype={"station_id": str})
    labels = pd.read_csv(PROC / "test_labels.csv", dtype={"station_id": str})
    meta = pd.read_csv(PROC / "station_metadata.csv", dtype={"station_id": str}).set_index("station_id")
    both = pred.merge(labels, on=["station_id", "date"], suffixes=("_pred", "_obs"))
    both["date"] = pd.to_datetime(both["date"])
    tol = alerts["model_climate_within_2_days"]

    with PdfPages(REPORTS / "slides.pdf") as pdf:
        fig = _page("A number saves no one. A warning people act on does.", "1 · Problem & users")
        _bullets(fig, [
            "Task: predict daily river discharge (cfs) per station from rain, temperature, soil moisture + 4 catchment attributes",
            "No past discharge in test.csv: the river's 'memory' must come from weather history",
            "District disaster officer: which rivers need attention this week, and how sure are we?",
            "Riverside resident: am I safe, what should I do, in my language (English, Hindi, Kannada)",
            "Evidence: in a Bihar study of 810 households, getting an alert doubled the chance of protective action",
            "So we built both: an accurate model and an early-warning product on top of it",
        ], step=0.085)
        pdf.savefig(fig); plt.close("all")

        fig = _page("Data: the three sources from the brief", "2 · Data")
        _bullets(fig, [
            f"USGS NWIS daily discharge (param 00060): {metrics['n_stations']} stations incl. the 5 recommended + big rivers",
            "Open-Meteo archive at each gauge: precipitation, Tmax, Tmin, soil moisture 0–7 cm (m³/m³)",
            "Catchment attributes: USGS GAGES-II (StreamStats API was offline): area, elevation, slope, forest",
            f"Train 2012–2022 ({metrics['n_train']:,} station-days) | held-out test 2023-01 → 2025-10 "
            f"({metrics['n_test']:,} station-days), never used for any choice of thresholds or intervals",
            "Cleaning: drop ice/equipment-affected days and USGS no-data codes; never fill missing targets",
            "Live product: Open-Meteo 16-day forecast (free, cached, offline snapshot fallback)",
        ], step=0.085)
        pdf.savefig(fig); plt.close("all")

        fig = _page("Model + product", "3 · CPU LightGBM ensembles → risk levels → alerts")
        _bullets(fig, [
            "Features: rolling rain 3–90 d, antecedent-precipitation index, soil trend, snow/melt, season",
            "Known stations: 3-objective ensemble + station climatology; unseen: flow per sq mi × area",
            "Risk levels per river: top-10% day, 2-, 5-, 20-year flood (Gumbel), bias-corrected (GloFAS method)",
            "Likely range (split-conformal), plain-language 'why' (SHAP), what-if rainfall, CAP 1.2 export",
            "Runs offline on an 8 GB laptop; deterministic; 43 automated tests",
        ], y=0.78, step=0.06, size=13)
        _image(fig, REPORTS / "screens" / "resident_view.jpg", [0.04, 0.06, 0.45, 0.42])
        _image(fig, REPORTS / "screens" / "officer_dashboard.jpg", [0.51, 0.06, 0.45, 0.42])
        pdf.savefig(fig); plt.close("all")

        fig = _page("Results on the 2023–2025 holdout", "4 · Accuracy (raw cfs) and warning skill")
        rows = [["Monthly climatology baseline", *_fmt(metrics["baseline_monthly_climatology"])],
                ["Ours — known stations", *_fmt(final["known"])],
                ["Ours — unseen stations (CV)", *_fmt(metrics["unseen_log_specific"])]]
        ax_t = fig.add_axes([0.05, 0.62, 0.5, 0.2]); ax_t.axis("off")
        tbl = ax_t.table(cellText=rows, colLabels=["", "RMSE", "RMSE top-10%", "MAE", "R²"], loc="center",
                         cellLoc="center", colWidths=[0.36, 0.15, 0.19, 0.15, 0.12])
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
        lv = ["elevated", "warning", "danger"]
        arows = [[k.title(), f"{tol[k]['pod']:.0%}", f"{tol[k]['far']:.0%}"] for k in lv]
        ax_a = fig.add_axes([0.6, 0.62, 0.36, 0.2]); ax_a.axis("off")
        atbl = ax_a.table(cellText=arows, colLabels=["Alert level", "Caught (POD)", "False alarms"],
                          loc="center", cellLoc="center")
        atbl.auto_set_font_size(False); atbl.set_fontsize(11); atbl.scale(1, 1.8)
        fig.text(0.05, 0.575, f"Warning skill uses a ±2-day timing tolerance · exact level right on {alerts['exact_level_match']:.0%} of days, "
                           f"within one level {alerts['within_one_level']:.0%}.", fontsize=11, color=MUTED)
        fig.text(0.05, 0.545, f"Likely range covers {alerts['likely_range_coverage_10_90']:.0%} of observations "
                             f"(target 80%). Replay: Rolling Fork, KY flagged EXTREME 4 days ahead of the April 2025 flood.",
                 fontsize=11, color=MUTED)
        for i, sid in enumerate(PLOT_STATIONS):
            d = both[both.station_id == sid].sort_values("date")
            if d.empty:
                continue
            ax = fig.add_axes([0.05 + i * 0.47, 0.08, 0.43, 0.42])
            ax.plot(d.date, d.discharge_cfs_obs, color=OBS, lw=1, label="Observed (USGS)")
            ax.plot(d.date, d.discharge_cfs_pred, color=ACCENT, lw=1.4, label="Predicted")
            ax.set_title(str(meta.loc[sid, "station_name"])[:45], fontsize=12)
            ax.set_ylabel("cfs"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        pdf.savefig(fig); plt.close("all")

        fig = _page("Limitations & responsible use", "5 · Limitations")
        _bullets(fig, [
            "Big regulated rivers (Mississippi, Ohio) follow upstream rain and dam releases, not gauge weather",
            "Peaks are under-predicted in magnitude; risk levels correct for this, but rare 20-year floods are mostly missed",
            "Dam-controlled rivers (e.g. Colorado at Austin) are poorly captured day-to-day",
            "Likely range is too narrow in 2023–25 (72% vs 80% target): a shift from the training years",
            "Replay uses recorded weather (perfect forecast); real forecasts add error, more so on days 8–16",
            "Hindi/Kannada text is machine-assisted: native review before real use. Research prototype; dial 112",
        ], step=0.085)
        pdf.savefig(fig); plt.close("all")
    print(f"wrote {REPORTS / 'slides.pdf'}")


if __name__ == "__main__":
    main()
