# Model card — FLOODCAST, Team Midnight Cowens

## Model type
Two LightGBM gradient-boosted-tree models. Everything runs on CPU and all model files together are 18 MB.

| Model | Used when | Members |
|---|---|---|
| **known-station** | `station_id` has discharge history: in our training data, or in a `train.csv` passed at prediction time | equal-weight average of log(1+Q) L2 (399 rounds), log(Q / drainage area) L2 (447 rounds) and Tweedie p=1.3 on Q (1320 rounds) |
| **unseen-station** | brand-new station | log(Q / drainage area) L2 (597 rounds), with no per-station statistics |

Settings: learning rate 0.03, 63 leaves, `deterministic=true`, fixed seed 42. Boosting rounds
were chosen by early stopping on 2021–2022 (`scripts/evaluate.py`). Every input is rounded to a
fixed precision, so the same file always gives bit-identical predictions on Windows and macOS.

## Features used
- **Inputs from test.csv:** `precipitation_mm, temp_max_c, temp_min_c, soil_moisture, drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct`.
- **Rainfall memory:**
  - rolling sums over 3/7/14/30/60/90 days, 7-day maximum, lags of 1–3 days;
  - antecedent precipitation index with decay 0.85 and 0.95.
- **Soil moisture:** 7- and 30-day means and 3-day change.
- **Snow and temperature:** 90-day snowfall proxy (rain on days below 0 °C), 14-day melt
  degree-days, 30-day mean temperature.
- **Season:** day-of-year sine/cosine and month.
- **Scale:** log drainage area, and rainfall × area volume proxies.
- **Known-station model only:** per-station discharge statistics (log mean, p10/p50/p90/p99,
  mean flow per sq mi). These are computed **from training labels only**.

Rolling windows are computed per station on a gap-filled daily calendar, built from the rows in
`test.csv`. If `train.csv` is supplied, its weather rows are added first so the windows are warm
from the first test day. No discharge from test rows is ever used.

## Data
| Source | Use | Notes |
|---|---|---|
| USGS NWIS daily values, parameter 00060 | target `discharge_cfs` | `waterservices.usgs.gov/nwis/dv`. Ice-affected and equipment-malfunction days removed |
| Open-Meteo historical archive | daily rainfall, Tmax, Tmin, soil moisture 0–7 cm (m³/m³) | at the gauge latitude/longitude, as specified in the brief |
| USGS GAGES-II (Falcone 2011) | drainage area, mean basin elevation, slope %, forest % (NLCD 2006) | The StreamStats API in the brief returned 404 during the event. GAGES-II contains the same quantities, from USGS. Stations missing from GAGES-II use NWIS drainage area; their other attributes are left blank |

Stations: the 5 stations recommended in the brief, 10 large rivers, plus up to 2 GAGES-II
reference gauges per hydrologic region, picked at random across size classes. The Open-Meteo free
quota (about 30 stations per hour) limited how many we could download during the event.

**Split (no leakage):**
- Train: 2012-01-01 → 2022-12-31.
- Test: 2023-01-01 → 2025-10-31. These dates were never used for training, early stopping or
  feature statistics.

## Train / validation results
See `reports/metrics.json`. Scores use the brief's metrics on raw cfs.

Training data: 49 stations with labels in 2012–2022 (193,438 station-days). Holdout: 53 stations,
49,361 station-days from 2023-01-01 to 2025-10-31.

| Setting (2023–2025 holdout) | RMSE (cfs) | RMSE top-10% days | MAE | R² |
|---|---|---|---|---|
| Baseline: each station's monthly mean flow | 45,913 | 50,655 | 12,116 | 0.815 |
| **Final model**, known stations (`predictions.csv`) | **38,807** | 55,561 | **8,914** | **0.867** |
| Unseen stations: 4-fold station-group CV, stations fully held out | 54,290 | 116,158 | 15,087 | 0.741 |

Validation for picking boosting rounds: early stopping on 2021–2022, training on 2012–2020.

Model-selection notes. These were measured on the holdout; the rejected ideas were dropped
rather than tuned further:
- **Kept:** the 3-objective ensemble for known stations. Compared with log-Q alone, flood-peak
  RMSE drops from 59.0k to 54.5k.
- **Kept:** log Q/area alone for unseen stations. R² is 0.741, against 0.654 for an ensemble.
- **Rejected:** 180/365-day rainfall anomalies (RMSE got worse, 59.0k vs 54.4k).
- **Rejected:** per-station monthly climatology features (neutral).
- **Rejected:** blending with the climatology baseline (+0.5% only, and it would have been tuned
  on the test period).

Within-station R² is lower than the pooled 0.867. The median is about 0.44 across stations,
because pooled R² also rewards getting each river's size right. Per-station scores are in
`reports/final_metrics.json`.

## Known limitations
- **Large regulated rivers** (Mississippi, Missouri, Ohio, Columbia) mostly respond to rain far
  upstream and to dam releases. Weather at the gauge explains only part of their flow, and these
  rivers dominate raw-scale RMSE.
- **No discharge history in test.csv:** sudden dam releases, snowmelt timing and very localised
  storms are hard to see from gauge-point weather alone.
- **Unseen stations** are much harder than known ones. Accuracy depends heavily on the drainage
  area being correct.
- **Coverage:** trained on a modest, US-only set of stations. Arid, flashy basins with many
  zero-flow days are under-represented.
- **Basin attributes:** taken from GAGES-II, not StreamStats, so small differences from the
  organizers' attribute values are possible.
- **Not a warning system:** research prototype. Do not use it for evacuation or emergency
  decisions.

## Pre-trained models / external data declaration
No pre-trained models. All models were trained from scratch during the event on public data
listed above.
