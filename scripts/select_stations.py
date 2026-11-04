"""Pick a diverse set of USGS stations and build their static catchment attributes.

Static attributes come from USGS GAGES-II (Falcone 2011): the same physical quantities
StreamStats reports (drainage area, mean basin elevation, mean slope, % forest).
Stations missing from GAGES-II fall back to the NWIS site service for drainage area.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
G2 = ROOT / "data" / "raw" / "gages2" / "csv"
OUT = ROOT / "data" / "processed" / "stations.csv"

SQKM_PER_SQMI = 2.58999
FT_PER_M = 3.28084
N_PER_REGION = 3
RANDOM_SEED = 42

# Stations recommended in the FLOODCAST brief, plus a few large rivers so the
# model sees the full discharge range that dominates raw-scale RMSE.
MUST_HAVE = [
    "01646500", "07010000", "08158000", "02087500", "11447650",
    "03294500", "06934500", "07289000", "14105700", "01570500",
    "02489500", "05587450", "03612500", "12340500", "09380000",
]


def load_gages2() -> pd.DataFrame:
    read = lambda name: pd.read_csv(G2 / name, dtype={"STAID": str}, encoding="latin-1")
    ids = read("conterm_basinid.txt")[["STAID", "STANAME", "DRAIN_SQKM", "HUC02", "LAT_GAGE", "LNG_GAGE", "STATE"]]
    topo = read("conterm_topo.txt")[["STAID", "ELEV_MEAN_M_BASIN", "SLOPE_PCT"]]
    lc = read("conterm_lc06_basin.txt")[["STAID", "FORESTNLCD06"]]
    rec = read("conterm_flowrec.txt")[["STAID", "ACTIVE09", "FLOWYRS_1990_2009"]]
    g = ids.merge(topo, on="STAID").merge(lc, on="STAID").merge(rec, on="STAID")
    return g.assign(
        station_id=g.STAID.str.zfill(8),
        drainage_area_sqmi=g.DRAIN_SQKM / SQKM_PER_SQMI,
        mean_elevation_ft=g.ELEV_MEAN_M_BASIN * FT_PER_M,
        slope_pct=g.SLOPE_PCT,
        forest_cover_pct=g.FORESTNLCD06,
        latitude=g.LAT_GAGE,
        longitude=g.LNG_GAGE,
        station_name=g.STANAME,
        state=g.STATE,
    )


def nwis_site_info(site_ids: list[str]) -> pd.DataFrame:
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {"format": "rdb", "sites": ",".join(site_ids), "siteOutput": "expanded", "siteStatus": "all"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    return pd.DataFrame({
        "station_id": df.site_no,
        "station_name": df.station_nm,
        "latitude": pd.to_numeric(df.dec_lat_va, errors="coerce"),
        "longitude": pd.to_numeric(df.dec_long_va, errors="coerce"),
        "state": df.state_cd,
        "drainage_area_sqmi": pd.to_numeric(df.drain_area_va, errors="coerce"),
    })


def main(n_per_region: int = N_PER_REGION) -> None:
    g = load_gages2()
    active = g[(g.ACTIVE09 == "yes") & (g.FLOWYRS_1990_2009 >= 20)].copy()
    active["size_bin"] = pd.qcut(np.log10(active.drainage_area_sqmi), 3, labels=False)
    shuffled = active.sample(frac=1.0, random_state=RANDOM_SEED)
    one_per_size = shuffled.groupby(["HUC02", "size_bin"]).head(1)
    picks = one_per_size.groupby("HUC02").head(n_per_region)
    cols = ["station_id", "station_name", "latitude", "longitude", "state",
            "drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct"]
    chosen = pd.concat([g[g.station_id.isin(MUST_HAVE)][cols], picks[cols]]).drop_duplicates("station_id")

    missing = sorted(set(MUST_HAVE) - set(chosen.station_id))
    if missing:
        chosen = pd.concat([chosen, nwis_site_info(missing)], ignore_index=True)
    chosen = chosen.sort_values("drainage_area_sqmi", na_position="last").drop_duplicates("station_id")
    chosen = chosen.assign(in_gages2=chosen.station_id.isin(g.station_id))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT, index=False)
    print(f"{len(chosen)} stations -> {OUT}")
    print(chosen[["station_id", "station_name", "drainage_area_sqmi", "in_gages2"]].to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_PER_REGION)
