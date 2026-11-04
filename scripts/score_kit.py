"""Score the shipped model on the practice kit (judge_test/) -> a markdown table for the README.

Two inputs, as a judge could run it:  test.csv alone, and test.csv + train_optional.csv (earlier
flow history for the never-seen rivers). Rows are split into rivers the model was trained on
("known") and rivers it has never seen ("new"). Read-only: writes no files.

Usage:  python scripts/score_kit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.metrics import score  # noqa: E402
from floodcast.model import FloodcastModel  # noqa: E402
from floodcast.schema import DATE_COL, ID_COL, TARGET, make_submission, read_table  # noqa: E402

KIT = ROOT / "judge_test"
MODEL_DIR = ROOT / "models"
HEADER = ("| Input | Rivers | Rows | RMSE (cfs) | RMSE top-10% days | MAE | R2 |\n"
          "|---|---|---|---|---|---|---|")


def kit_scores(kit: Path = KIT, model_dir: Path = MODEL_DIR) -> list[dict]:
    """One row per (input, river group): the brief's metrics against kit/answers.csv."""
    model = FloodcastModel.load(model_dir)
    known_ids = set(model.station_stats[ID_COL].astype(str))
    test = read_table(kit / "test.csv")
    answers = pd.read_csv(kit / "answers.csv", dtype={ID_COL: str, DATE_COL: str})
    history_path = kit / "train_optional.csv"
    inputs = {"test.csv": None}
    if history_path.exists():
        inputs["test.csv + train_optional.csv"] = read_table(history_path)
    rows = []
    for name, history in inputs.items():
        pred = make_submission(test, model.predict(test, history))
        both = pred.merge(answers, on=[ID_COL, DATE_COL], suffixes=("_pred", ""), validate="one_to_one")
        is_known = both[ID_COL].isin(known_ids)
        for group, mask in (("all", slice(None)), ("known", is_known), ("new", ~is_known)):
            part = both[mask]
            if part.empty:
                continue
            m = score(part[TARGET].to_numpy(), part[f"{TARGET}_pred"].to_numpy(), part[ID_COL].to_numpy())
            rows.append({"input": name, "group": group, "rivers": part[ID_COL].nunique(), **m})
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [f"| {r['input']} | {r['group']} ({r['rivers']}) | {r['n']:,} | {r['rmse']:,.0f} | "
             f"{r['rmse_extreme']:,.0f} | {r['mae']:,.0f} | {r['r2']:.3f} |" for r in rows]
    return "\n".join([HEADER, *lines])


def main() -> None:
    test = pd.read_csv(KIT / "test.csv", usecols=[DATE_COL], dtype=str)
    print(f"Practice kit {test[DATE_COL].min()} to {test[DATE_COL].max()}\n")
    print(to_markdown(kit_scores()))


if __name__ == "__main__":
    main()
