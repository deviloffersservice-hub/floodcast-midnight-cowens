# FLOODCAST — Team Midnight Cowens

Predicts **daily river discharge (cubic feet per second)** at USGS river stations from rainfall,
temperature, soil moisture and catchment attributes, following the FLOODCAST problem statement.

**Team:** Midnight Cowens
**Members:** _TODO: Name 1 (role)_, _TODO: Name 2 (role)_, _TODO: Name 3 (role)_, _TODO: Name 4 (role)_

## Approach

We joined USGS daily discharge (parameter 00060), Open-Meteo daily weather at each gauge, and
catchment attributes for a geographically diverse set of US stations, keeping 2023 onward
completely out of training. Since `test.csv` has no past discharge, the model reads river memory
from weather history: rolling rainfall totals (3–90 days), an antecedent-precipitation index,
soil-moisture trends and a snow-melt proxy. Two CPU-only LightGBM ensembles do the prediction.
One is for stations we have history for, using that station's typical flow. The other is for
never-seen stations and scales flow by drainage area. Each ensemble averages three objectives
(log-discharge, log discharge-per-square-mile, Tweedie) to balance overall accuracy with flood
peaks.

## Results (held-out 2023-01-01 → 2025-10-31, never used for training)

See [`model_card.md`](model_card.md) and `reports/metrics.json` for the full table.

## Run it on Windows (no GPU needed — tested target: i5, 8 GB RAM)

1. Download this repository (green **Code** button → **Download ZIP**, then unzip), or `git clone` it.
2. Double-click **`setup.bat`**. It sets everything up once and needs internet:
   - It installs Python 3.12 (via `winget`) if it's missing.
   - It installs the Microsoft C++ runtime if missing. LightGBM needs it; accept the admin prompt.
   - It creates `.venv`, installs the pinned packages from `requirements.txt`, and runs the self-tests.
3. Then use either:
   - **`run_app.bat`**: opens the web app at <http://127.0.0.1:7860>.
     - Batch tab: upload `test.csv` (and optionally `train.csv`), then download `predictions.csv`.
     - Live station tab: pick a river and dates. It pulls real weather from Open-Meteo and
       plots predicted discharge against observed USGS discharge.
     - Manual tab: type in one day's values and get one prediction.
   - **`predict.bat path\to\test.csv [path\to\train.csv]`** (or drag `test.csv` onto it):
     writes `predictions.csv` next to `test.csv`.

Manual equivalent (any OS, Python 3.12):

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python solution.py predict --test data/processed/test.csv --out predictions.csv
python solution.py predict --test test.csv --train train.csv --out predictions.csv   # organizer files
python app.py                   # web app
pytest -q tests                 # self-tests
```

## Input / output contract

`test.csv` columns:
`station_id, date, precipitation_mm, temp_max_c, temp_min_c, soil_moisture, drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct`

`predictions.csv` has exactly `station_id, date, discharge_cfs`. It has one row per test row, in
the same order, with `station_id` and `date` copied exactly as written in `test.csv`. All values
are non-negative. The file is validated against every rejection rule before it is written.

Files saved by Excel are handled:
- UTF-8 with or without BOM, or Windows-1252 encoding;
- station IDs with leading zeros stripped;
- `dd-mm-yyyy` or `mm/dd/yyyy` dates;
- soil moisture given in % instead of m³/m³;
- blank cells and unknown stations.

## Reproduce training

```bash
python scripts/select_stations.py 2   # needs GAGES-II CSVs in data/raw/gages2/csv (see model_card)
python scripts/fetch_data.py usgs     # USGS NWIS daily discharge
python scripts/fetch_data.py meteo    # Open-Meteo archive (free tier: ~30 stations/hour)
python scripts/build_dataset.py       # -> data/processed/{train,test,test_labels}.csv
python scripts/evaluate.py            # model selection -> reports/metrics.json
python solution.py train              # -> models/
python solution.py predict            # -> predictions.csv
```

## Repository layout

```
solution.py            train / predict entry point
app.py                 local Gradio web app
floodcast/             schema (I/O + validation), features, model, metrics
scripts/               data download, dataset build, evaluation
models/                trained LightGBM ensembles + station statistics
data/processed/        train.csv, test.csv (our held-out split), station metadata
data/sample/           small example test file
tests/                 robustness tests
setup.bat, run_app.bat, predict.bat   Windows one-click scripts
```

_Research prototype — not an official flood warning system._
