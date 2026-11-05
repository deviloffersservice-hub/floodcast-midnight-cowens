"""Extended evaluation for the Batch tab: 8 metric families on a scored upload (test + answers).

Families: continuous fit, flood-day detection, alert burden, probabilistic quality, interval quality,
generalization (per-river averages, known vs never-seen rivers), robustness (optional) and efficiency.

Everything here only *reads* predictions and the answer key; the model and its outputs are unchanged.
A "flood day" is a day in a river's top 10% of observed flow in the uploaded answers, the same
definition the brief uses for its top-10% RMSE.
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score

from .metrics import EXTREME_QUANTILE
from .risk import RiskTable
from .schema import DATE_COL, ID_COL

ALERT_BUDGET = 1 - EXTREME_QUANTILE      # alert on the same share of days as there are flood days
Z_90 = float(norm.ppf(0.9))              # the likely range is the 10th-90th percentile band
DAYS_PER_YEAR = 365.25
RELIABILITY_BINS = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CORRUPTION_SEED = 0
GROUPS = ("all", "known", "new")
CLOSE_TOLERANCE = 0.25                   # a prediction within ±25% of the true flow counts as close
MIN_FLOW_CFS = 1.0                       # tolerance floor for (near-)dry days


# ---------------- helpers ----------------
def _nse(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else float("nan")


def accuracy_pct(y: np.ndarray, p: np.ndarray) -> float:
    """Volume-weighted accuracy = 1 - WAPE = 1 - sum|error| / sum(true flow), floored at 0."""
    total = float(np.sum(np.abs(y)))
    return max(0.0, 1 - float(np.sum(np.abs(y - p))) / total) if total > 0 else float("nan")


def within_pct(y: np.ndarray, p: np.ndarray, tol: float = CLOSE_TOLERANCE) -> float:
    """Share of days whose prediction is within ±tol of the true flow."""
    return float(np.mean(np.abs(y - p) <= tol * np.maximum(y, MIN_FLOW_CFS))) if len(y) else float("nan")


def _groups(frame: pd.DataFrame):
    for name in GROUPS:
        part = frame if name == "all" else frame[frame.group == name]
        if not part.empty:
            yield name, part


def peak_memory_mb() -> float:
    """Peak resident memory of this process (MB) on Linux, macOS and Windows; NaN if unknown."""
    try:
        if sys.platform == "win32":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return float("nan")
            return counters.PeakWorkingSetSize / 2**20
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS: bytes, Linux: KiB
    except (OSError, AttributeError, ImportError):
        return float("nan")


def model_size_mb(model_dir: Path) -> float:
    return sum(f.stat().st_size for f in Path(model_dir).iterdir() if f.is_file()) / 1e6


# ---------------- scored frame ----------------
def scored_frame(ids, dates, pred, obs, lo, hi, known: np.ndarray) -> pd.DataFrame:
    """One row per scored day with the per-river flood-day flag and a comparable alert score."""
    frame = pd.DataFrame({ID_COL: np.asarray(ids, dtype=str), DATE_COL: np.asarray(dates, dtype=str),
                          "pred": np.clip(np.asarray(pred, dtype=float), 0, None),
                          "obs": np.asarray(obs, dtype=float), "lo": np.asarray(lo, dtype=float),
                          "hi": np.asarray(hi, dtype=float),
                          "group": np.where(np.asarray(known, dtype=bool), "known", "new")})
    by_river = frame.groupby(ID_COL)
    frame["flood_day"] = frame.obs >= by_river.obs.transform(lambda s: s.quantile(EXTREME_QUANTILE))
    frame["alert_score"] = by_river.pred.rank(pct=True)  # within-river rank: comparable across rivers
    return frame


def per_river(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in frame.groupby(ID_COL):
        y, p = g.obs.to_numpy(), g.pred.to_numpy()
        rows.append({"River": sid, "Group": g.group.iloc[0], "Rows": len(g), "Accuracy": accuracy_pct(y, p),
                     "Days within ±25%": within_pct(y, p), "NSE": _nse(y, p),
                     "log-NSE": _nse(np.log1p(y), np.log1p(p)), "MAE (cfs)": float(np.mean(np.abs(y - p))),
                     "Normalised MAE": float(np.mean(np.abs(y - p)) / y.mean()) if y.mean() > 0 else np.nan,
                     "RMSE (cfs)": float(np.sqrt(np.mean((y - p) ** 2)))})
    return pd.DataFrame(rows)


# ---------------- families ----------------
def continuous_and_generalization(frame: pd.DataFrame, rivers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in _groups(frame):
        r = rivers if name == "all" else rivers[rivers.Group == name]
        y, p = part.obs.to_numpy(), part.pred.to_numpy()
        rows.append({"Rivers": name, "Count": part[ID_COL].nunique(), "Rows": len(part),
                     "Accuracy (all rows)": accuracy_pct(y, p), "Accuracy (avg per river)": r.Accuracy.mean(),
                     "Days within ±25%": within_pct(y, p),
                     "NSE (all rows)": _nse(y, p), "NSE (avg per river)": r.NSE.mean(),
                     "log-NSE (avg per river)": r["log-NSE"].mean(), "MAE (cfs)": float(np.mean(np.abs(y - p))),
                     "Normalised MAE (avg per river)": r["Normalised MAE"].mean()})
    return pd.DataFrame(rows)


def event_detection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in _groups(frame):
        events = part.flood_day.to_numpy()
        alert = part.alert_score.to_numpy() > 1 - ALERT_BUDGET
        hits = int((alert & events).sum())
        auprc = float(average_precision_score(events, part.alert_score)) if 0 < events.sum() < len(events) else np.nan
        rows.append({"Rivers": name, "Flood days": int(events.sum()), "AUPRC": auprc,
                     "AUPRC if random": float(events.mean()),
                     "Precision @10% alert budget": hits / max(int(alert.sum()), 1),
                     "Recall @10% alert budget": hits / max(int(events.sum()), 1)})
    return pd.DataFrame(rows)


def alert_burden(frame: pd.DataFrame, risk: RiskTable) -> pd.DataFrame:
    """The app's own 'elevated' alert (model-climate threshold) vs observed flow above the observed one."""
    model_thr = risk.model_thr.set_index(ID_COL)["elevated"]
    obs_thr = risk.obs_thr.set_index(ID_COL)["elevated"]
    rows = []
    for sid, g in frame.groupby(ID_COL):
        if sid not in model_thr.index or not (np.isfinite(model_thr[sid]) and np.isfinite(obs_thr[sid])):
            continue
        alert = g.pred.to_numpy() >= model_thr[sid]
        event = g.obs.to_numpy() >= obs_thr[sid]
        years = g[DATE_COL].nunique() / DAYS_PER_YEAR
        rows.append({"River": sid, "Alert days": int(alert.sum()), "High-flow days": int(event.sum()),
                     "Hits": int((alert & event).sum()), "False alarms": int((alert & ~event).sum()),
                     "years": years})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    total = table.drop(columns="River").sum().to_dict() | {"River": "total"}
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    table["False alarms / year"] = table["False alarms"] / table.years
    table["POD"] = table.Hits / table["High-flow days"].where(table["High-flow days"] > 0)
    table["FAR"] = table["False alarms"] / table["Alert days"].where(table["Alert days"] > 0)
    return table.drop(columns="years")


def uncertainty(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interval coverage/width, plus Brier/CRPS from a normal distribution fitted to the likely range
    in log space (the model gives a 10-90% band, not probabilities, so these are derived)."""
    log_lo, log_hi = np.log1p(np.maximum(frame.lo, 0)), np.log1p(np.maximum(frame.hi, 0))
    mu = (log_lo + log_hi) / 2
    sigma = np.maximum((log_hi - log_lo) / (2 * Z_90), 1e-6)
    z = (np.log1p(frame.obs) - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    flood_thr = frame.groupby(ID_COL).obs.transform(lambda s: s.quantile(EXTREME_QUANTILE))
    p_flood = 1 - norm.cdf((np.log1p(flood_thr) - mu) / sigma)
    work = frame.assign(inside=(frame.obs >= frame.lo) & (frame.obs <= frame.hi),
                        rel_width=(frame.hi - frame.lo) / np.maximum(frame.pred, 1),
                        crps=crps, p=p_flood, brier=(p_flood - frame.flood_day.astype(float)) ** 2)
    rows = []
    for name, part in _groups(work):
        flood_f = part.flood_day.astype(float)
        ref = float(((flood_f.mean() - flood_f) ** 2).mean())
        flood = part[part.flood_day]
        rows.append({"Rivers": name, "Coverage (target 80%)": part.inside.mean(),
                     "Coverage on flood days": flood.inside.mean() if len(flood) else np.nan,
                     "Median width (× prediction)": part.rel_width.median(),
                     "Brier (flood day)": part.brier.mean(),
                     "Brier skill vs climatology": 1 - part.brier.mean() / ref if ref > 0 else np.nan,
                     "CRPS (log flow)": part.crps.mean()})
    bins = pd.cut(work.p, RELIABILITY_BINS, include_lowest=True)
    reliability = (work.groupby(bins, observed=True)
                   .agg(Days=("p", "size"), **{"Forecast probability": ("p", "mean"),
                                                "Observed frequency": ("flood_day", "mean")})
                   .reset_index().rename(columns={"p": "Probability bin"}))
    reliability["Probability bin"] = reliability["Probability bin"].astype(str)
    return pd.DataFrame(rows), reliability


def corruptions(test: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Controlled input corruptions (fixed seed, so reruns give the same numbers)."""
    rng = np.random.default_rng(CORRUPTION_SEED)
    n = len(test)
    out = []
    t = test.copy(); t["precipitation_mm"] = t.precipitation_mm * rng.lognormal(0, 0.3, n)
    out.append(("Rain ×30% random noise", t))
    t = test.copy(); t.loc[rng.random(n) < 0.10, "precipitation_mm"] = np.nan
    out.append(("10% of rain days missing", t))
    t = test.copy(); t["soil_moisture"] = np.nan
    out.append(("Soil moisture missing", t))
    t = test.copy(); t[["temp_max_c", "temp_min_c"]] += rng.normal(0, 2, (n, 2))
    out.append(("Temperature ±2 °C noise", t))
    t = test.copy(); t[["mean_elevation_ft", "slope_pct", "forest_cover_pct"]] = np.nan
    out.append(("Elevation/slope/forest missing", t))
    t = test.copy(); t["drainage_area_sqmi"] = t.drainage_area_sqmi * 1.10
    out.append(("Drainage area +10%", t))
    return out


def robustness(test: pd.DataFrame, predict: Callable[[pd.DataFrame], np.ndarray], obs: np.ndarray,
               matched: np.ndarray, base_pred: np.ndarray) -> pd.DataFrame:
    """Relative degradation of MAE / RMSE / per-river NSE when inputs are corrupted."""
    ids = test[ID_COL].to_numpy(dtype=str)[matched]

    def stats(p):
        p = np.clip(p[matched], 0, None)
        err = obs - p
        nse = pd.DataFrame({"i": ids, "y": obs, "p": p}).groupby("i").apply(
            lambda g: _nse(g.y.to_numpy(), g.p.to_numpy()), include_groups=False).mean()
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), float(nse)

    mae0, rmse0, nse0 = stats(base_pred)
    rows = []
    for name, corrupted in corruptions(test):
        mae, rmse, nse = stats(np.asarray(predict(corrupted), dtype=float))
        rows.append({"Corruption": name, "MAE change": mae / mae0 - 1, "RMSE change": rmse / rmse0 - 1,
                     "NSE (avg per river) change": nse - nse0})
    return pd.DataFrame(rows)


def efficiency(rows: int, seconds: float, model_dir: Path) -> pd.DataFrame:
    return pd.DataFrame([{"Rows": rows, "Prediction time (s)": seconds,
                          "Rows per second": rows / seconds if seconds > 0 else np.nan,
                          "Peak app memory (MB)": peak_memory_mb(), "Model files (MB)": model_size_mb(model_dir)}])


# ---------------- markdown ----------------
_PERCENT = ("Accuracy", "within", "Coverage", "Precision", "Recall", "POD", "FAR", "change",
            "Observed frequency", "Forecast probability")
_INTEGER = ("Rows", "Count", "days", "Days", "Hits", "alarms", "(cfs)", "per second")


def _fmt(col: str, value) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if "/ year" in col or "(s)" in col or "(MB)" in col:
        return f"{value:,.2f}"
    if "NSE (avg per river) change" in col:
        return f"{value:+.3f}"
    if "change" in col:
        return f"{value:+.1%}"
    if any(k in col for k in _PERCENT):
        return f"{value:.0%}"
    if any(k in col for k in _INTEGER):
        return f"{value:,.0f}"
    return f"{value:.3f}"


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(c, v) for c, v in zip(cols, row)) + " |"
              for row in table.itertuples(index=False)]
    return "\n".join(lines)


SECTIONS = [
    ("continuous", "1 · Continuous forecast + 6 · Generalization",
     "Accuracy = 100% − total error ÷ total true flow. Days within ±25%: share of days the prediction was "
     "within a quarter of the true flow. NSE: 1 is perfect, 0 is no better than the river's own average. "
     "*All rows* is dominated by the largest river; *avg per river* weighs each river equally. *known* = rivers the model trained on, *new* = never seen."),
    ("per_river", "Per river", ""),
    ("events", "2 · Event detection (flood day = a river's top-10% observed days)",
     "AUPRC: how well the forecast ranks flood days first (higher is better; compare with *if random*). "
     "Precision/recall when alerting on each river's top-10% predicted days."),
    ("alerts", "3 · Operational alert burden (the app's “elevated” level)",
     "Only rivers with a flow record have alert levels. POD = share of high-flow days alerted; "
     "FAR = share of alerts that were false."),
    ("uncertainty", "4 · Probabilistic quality + 5 · Interval quality",
     "Coverage: share of true values inside the 10–90% likely range (target 80%). Brier and CRPS are "
     "*derived* from the likely range (normal in log space); lower is better. Brier skill > 0 beats always "
     "forecasting the flood-day base rate."),
    ("reliability", "Reliability (flood-day probability)",
     "For a well-calibrated forecast, observed frequency ≈ forecast probability in each bin."),
    ("robustness", "7 · Robustness (inputs deliberately corrupted, then re-predicted)", "Smaller changes are better."),
    ("efficiency", "8 · Efficiency", "Peak memory is the whole app process, not just this prediction."),
]


def report_markdown(tables: dict[str, pd.DataFrame]) -> str:
    parts = ["## Evaluation metrics"]
    for key, title, note in SECTIONS:
        table = tables.get(key)
        if table is None or table.empty:
            continue
        parts.append(f"### {title}")
        if note:
            parts.append(f"<small>{note}</small>")
        parts.append(to_markdown(table))
    return "\n\n".join(parts)


def evaluate(frame: pd.DataFrame, risk: RiskTable) -> dict[str, pd.DataFrame]:
    """All answer-dependent families for a scored_frame()."""
    rivers = per_river(frame)
    interval, reliability = uncertainty(frame)
    return {"continuous": continuous_and_generalization(frame, rivers), "per_river": rivers,
            "events": event_detection(frame), "alerts": alert_burden(frame, risk),
            "uncertainty": interval, "reliability": reliability}


def timed(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
