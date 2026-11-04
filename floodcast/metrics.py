"""FLOODCAST scoring metrics (raw cfs scale)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTREME_QUANTILE = 0.90


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def score(y: pd.Series, p: np.ndarray, station: pd.Series) -> dict[str, float]:
    """Overall RMSE, RMSE on each station's top-10% actual days, MAE and R²."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    thresh = pd.Series(y).groupby(np.asarray(station)).transform(lambda s: s.quantile(EXTREME_QUANTILE)).to_numpy()
    extreme = y >= thresh
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": rmse(y, p),
        "rmse_extreme": rmse(y[extreme], p[extreme]),
        "mae": float(np.mean(np.abs(y - p))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }
