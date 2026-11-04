"""Score predictions.csv against the held-out labels -> reports/final_metrics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.metrics import score  # noqa: E402


def main(pred_path: Path, labels_path: Path) -> None:
    pred = pd.read_csv(pred_path, dtype={"station_id": str})
    labels = pd.read_csv(labels_path, dtype={"station_id": str})
    both = pred.merge(labels, on=["station_id", "date"], suffixes=("_pred", ""), validate="one_to_one")
    if len(both) != len(labels):
        raise ValueError(f"only {len(both)} of {len(labels)} labelled rows matched")
    result = {"known": score(both.discharge_cfs, both.discharge_cfs_pred, both.station_id)}
    per_station = {
        sid: score(d.discharge_cfs, d.discharge_cfs_pred, d.station_id)
        for sid, d in both.groupby("station_id")
    }
    result["per_station"] = per_station
    out = ROOT / "reports" / "final_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["known"], indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]) if args else ROOT / "predictions.csv",
         Path(args[1]) if len(args) > 1 else ROOT / "data" / "processed" / "test_labels.csv")
