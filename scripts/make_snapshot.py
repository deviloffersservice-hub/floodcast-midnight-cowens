"""Save today's live forecast for all stations as the offline fallback (data/snapshot/)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodcast.forecast import SNAPSHOT_PATH, live_weather, save_snapshot  # noqa: E402


def main() -> None:
    stations = pd.read_csv(ROOT / "models" / "stations.csv", dtype={"station_id": str})
    frame, source, issued = live_weather(stations)
    if source == "snapshot":
        raise SystemExit("weather service unreachable; existing snapshot kept")
    save_snapshot(frame, issued)
    print(f"saved {len(frame)} rows for {frame.station_id.nunique()} stations (issued {issued}) -> {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
