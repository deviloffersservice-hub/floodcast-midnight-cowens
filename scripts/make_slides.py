"""Build the 5-slide presentation (reports/slides.pdf) from the trained model and metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
INK, ACCENT, MUTED, OBS = "#1b2430", "#1f6feb", "#57606a", "#d1242f"
PLOT_STATIONS = ("07010000", "01646500")


def _slide(pdf: PdfPages, title: str, bullets: list[str], subtitle: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.05, 0.9, title, fontsize=30, weight="bold", color=INK)
    if subtitle:
        fig.text(0.05, 0.85, subtitle, fontsize=15, color=MUTED)
    y = 0.76
    for b in bullets:
        fig.text(0.06, y, "•  " + b, fontsize=16, color=INK, wrap=True)
        y -= 0.075
    fig.text(0.05, 0.03, "FLOODCAST · Team Midnight Cowens", fontsize=11, color=MUTED)
    return fig


def _fmt(m: dict) -> list[str]:
    return [f"{m['rmse']:,.0f}", f"{m['rmse_extreme']:,.0f}", f"{m['mae']:,.0f}", f"{m['r2']:.3f}"]


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    final = json.loads((REPORTS / "final_metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(ROOT / "predictions.csv", dtype={"station_id": str})
    labels = pd.read_csv(PROC / "test_labels.csv", dtype={"station_id": str})
    meta = pd.read_csv(PROC / "station_metadata.csv", dtype={"station_id": str}).set_index("station_id")
    both = pred.merge(labels, on=["station_id", "date"], suffixes=("_pred", "_obs"))
    both["date"] = pd.to_datetime(both["date"])

    with PdfPages(REPORTS / "slides.pdf") as pdf:
        pdf.savefig(_slide(pdf, "Predict the surge before it strikes", [
            "Task: predict daily river discharge (cubic feet per second) for every station/date in test.csv",
            "Inputs only: rainfall, max/min temperature, soil moisture + 4 catchment attributes",
            "No past discharge in test.csv, so river 'memory' must come from weather history",
            "Score: 40% RMSE + 30% RMSE on each station's top-10% days + 20% MAE + 10% R²",
            "Must work on stations never seen in training, and on a CPU-only laptop",
        ], "1 · Problem")); plt.close("all")

        pdf.savefig(_slide(pdf, "Data: the three sources from the brief", [
            f"USGS NWIS daily discharge (param 00060): {metrics['n_stations']} stations, incl. the 5 recommended",
            "Open-Meteo archive at each gauge: precipitation, Tmax, Tmin, soil moisture 0–7 cm",
            "Catchment attributes: USGS GAGES-II (StreamStats API was offline) — area, elevation, slope, forest",
            f"Train 2012–2022 ({metrics['n_train']:,} station-days)  |  "
            f"held-out test 2023-01 → 2025-10 ({metrics['n_test']:,} station-days)",
            "Cleaning: drop ice/equipment-affected days, USGS no-data codes; never fill missing targets",
        ], "2 · Data")); plt.close("all")

        pdf.savefig(_slide(pdf, "Model: two CPU LightGBM ensembles", [
            "Features: rolling rain 3–90 d, antecedent-precipitation index, soil trend, snow/melt proxy, season",
            "Known stations: + station discharge climatology learned from training labels",
            "Unseen stations: predict discharge per sq mi, multiply by drainage area (extrapolates to big rivers)",
            "Known-station ensemble averages log-Q, log-Q/area and Tweedie objectives (peaks vs. typical days)",
            "Every choice made on validation: unseen = log-Q/area alone (R² 0.74 vs 0.65 for an ensemble)",
            "Deterministic, 18 MB, trains in ~20 s and predicts 49k rows in ~3 s on a laptop CPU",
        ], "3 · Model")); plt.close("all")

        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.05, 0.92, "Results on the 2023–2025 holdout", fontsize=28, weight="bold", color=INK)
        fig.text(0.05, 0.875, "4 · Results (raw cfs; lower is better except R²)", fontsize=14, color=MUTED)
        rows = [
            ["Monthly climatology baseline", *_fmt(metrics["baseline_monthly_climatology"])],
            ["Our model — known stations", *_fmt(final["known"])],
            ["Our model — unseen stations (CV)", *_fmt(metrics["unseen_log_specific"])],
        ]
        ax_t = fig.add_axes([0.05, 0.58, 0.9, 0.25]); ax_t.axis("off")
        tbl = ax_t.table(cellText=rows, colLabels=["", "RMSE", "RMSE top-10%", "MAE", "R²"], loc="center",
                         cellLoc="center", colWidths=[0.34, 0.16, 0.18, 0.16, 0.12])
        tbl.auto_set_font_size(False); tbl.set_fontsize(14); tbl.scale(1, 2.0)
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

        pdf.savefig(_slide(pdf, "Limitations & responsible use", [
            "Large regulated rivers (Mississippi, Ohio) follow upstream rain and dam releases, not gauge weather",
            "2023–2025 were low-water years on the Mississippi; the model over-predicts there",
            "Dam-controlled rivers (e.g. Colorado at Austin) are poorly captured day-to-day",
            "Unseen-station accuracy depends on correct drainage area; flashy arid basins are under-represented",
            "Weather is a single grid point at the gauge, not a basin average",
            "Research prototype — not an official flood warning; follow local authorities",
        ], "5 · Limitations")); plt.close("all")
    print(f"wrote {REPORTS / 'slides.pdf'}")


if __name__ == "__main__":
    main()
