"""One-command check of a fresh install (e.g. the Windows laptop). Exit code 0 = everything passed.

  1. Re-predicts data/processed/test.csv and compares with the committed predictions.csv.
     Byte-identical is expected on the machine that trained the model. A different CPU
     (Apple arm64 vs Windows x86-64) may move a few values by one rounding step, so the check
     allows a tiny tolerance and reports which case it was.
  2. Predicts the practice kit (judge_test/test.csv) and scores it against judge_test/answers.csv
     with the brief's metrics.

Usage:  python scripts/verify_install.py
"""
from __future__ import annotations

import filecmp
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import solution  # noqa: E402
from floodcast.metrics import score  # noqa: E402
from floodcast.schema import DATE_COL, ID_COL, TARGET  # noqa: E402

COMMITTED = ROOT / "predictions.csv"
HOLDOUT_TEST = ROOT / "data" / "processed" / "test.csv"
KIT = ROOT / "judge_test"
REL_TOL, ABS_TOL = 1e-4, 0.01  # far below any score-relevant change (cfs)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={ID_COL: str, DATE_COL: str})


def compare(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Row-for-row comparison of two prediction tables; keys must match exactly."""
    keys = [ID_COL, DATE_COL]
    if len(expected) != len(actual) or not expected[keys].equals(actual[keys]):
        return {"keys_match": False, "max_abs_diff": float("nan"), "n_outside_tol": len(expected)}
    e, a = expected[TARGET].to_numpy(dtype=float), actual[TARGET].to_numpy(dtype=float)
    diff = np.abs(e - a)
    outside = diff > ABS_TOL + REL_TOL * np.abs(e)
    return {"keys_match": True, "max_abs_diff": float(diff.max(initial=0.0)), "n_outside_tol": int(outside.sum())}


def check_reproduction(tmp: Path) -> bool:
    fresh = tmp / "predictions_check.csv"
    solution.predict(HOLDOUT_TEST, fresh, None)
    if filecmp.cmp(COMMITTED, fresh, shallow=False):
        print("[1/2] PASS  predictions.csv reproduced byte-identically")
        return True
    result = compare(_read(COMMITTED), _read(fresh))
    ok = result["keys_match"] and result["n_outside_tol"] == 0
    verdict = "PASS" if ok else "FAIL"
    print(f"[1/2] {verdict}  predictions.csv reproduced within tolerance (not byte-identical): {result}")
    return ok


def check_practice_kit(tmp: Path) -> bool:
    pred_path = tmp / "kit_predictions.csv"
    solution.predict(KIT / "test.csv", pred_path, None)
    both = _read(pred_path).merge(_read(KIT / "answers.csv"), on=[ID_COL, DATE_COL], suffixes=("_pred", ""))
    if both.empty:
        print("[2/2] FAIL  no practice-kit predictions matched answers.csv")
        return False
    m = score(both[TARGET].to_numpy(), both[f"{TARGET}_pred"].to_numpy(), both[ID_COL].to_numpy())
    print(f"[2/2] PASS  practice kit ({len(both):,} rows, {both[ID_COL].nunique()} rivers): "
          f"RMSE {m['rmse']:,.0f} | RMSE top-10% {m['rmse_extreme']:,.0f} | MAE {m['mae']:,.0f} | R2 {m['r2']:.3f}")
    return True


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        results = [check_reproduction(tmp), check_practice_kit(tmp)]
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
