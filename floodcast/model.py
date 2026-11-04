"""Two LightGBM regressors: one for stations with discharge history, one for unseen stations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import KNOWN_FEATURES, UNSEEN_FEATURES, build_features, compute_station_stats
from .schema import DATE_COL, ID_COL, TARGET, WEATHER_COLS

SEED = 42
BASE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}
OBJECTIVES = {
    "log_l2": {"objective": "regression"},
    "log_specific": {"objective": "regression"},
    "tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.3},
}
# log_specific models discharge per square mile, log((q + EPS) / area), so predictions
# scale with basin size and extrapolate to rivers larger than any seen in training.
SPECIFIC_EPS = 0.01
FALLBACK_AREA_SQMI = 1000.0
MAX_ROUNDS = 3000
EARLY_STOP = 100


@dataclass(frozen=True)
class FittedModel:
    booster: lgb.Booster
    features: list[str]
    objective: str

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(feats[self.features], num_threads=BASE_PARAMS["num_threads"])
        if self.objective == "log_l2":
            pred = np.expm1(raw)
        elif self.objective == "log_specific":
            pred = np.exp(raw) * _area(feats) - SPECIFIC_EPS
        else:
            pred = raw
        return np.clip(pred, 0.0, None)


def _area(feats: pd.DataFrame) -> np.ndarray:
    area = feats["drainage_area_sqmi"].to_numpy(dtype=float)
    return np.where(np.isfinite(area) & (area > 0), area, FALLBACK_AREA_SQMI)


def _target(feats: pd.DataFrame, objective: str) -> np.ndarray:
    y = feats[TARGET].to_numpy(dtype=float)
    if objective == "log_l2":
        return np.log1p(y)
    if objective == "log_specific":
        return np.log((y + SPECIFIC_EPS) / _area(feats))
    return y


def fit(train_feats: pd.DataFrame, features: list[str], objective: str,
        valid_feats: pd.DataFrame | None = None, rounds: int | None = None) -> FittedModel:
    params = {**BASE_PARAMS, **OBJECTIVES[objective]}
    dtrain = lgb.Dataset(train_feats[features], _target(train_feats, objective), free_raw_data=False)
    if valid_feats is not None:
        dvalid = lgb.Dataset(valid_feats[features], _target(valid_feats, objective), reference=dtrain)
        booster = lgb.train(params, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        booster = lgb.train(params, dtrain, rounds or MAX_ROUNDS)
    return FittedModel(booster, features, objective)


@dataclass(frozen=True)
class Ensemble:
    """Equal-weight average of several fitted models (one per objective)."""
    members: tuple[FittedModel, ...]

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(feats) for m in self.members], axis=0)


@dataclass(frozen=True)
class FloodcastModel:
    known: Ensemble
    unseen: Ensemble
    station_stats: pd.DataFrame

    def predict(self, test: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        """Predict discharge for each test row (order preserved).

        `history` may hold earlier rows (e.g. train.csv) for the same stations. Only their
        weather columns are used, to warm up rolling windows; their discharge is used only
        to compute per-station climatology for stations we have not seen.
        """
        feats, has_stats = self.featurize(test, history)
        return self.predict_features(feats, has_stats)

    def predict_features(self, feats: pd.DataFrame, has_stats: np.ndarray) -> np.ndarray:
        pred = self.unseen.predict(feats)
        if has_stats.any():
            pred[has_stats] = self.known.predict(feats[has_stats])
        return pred

    def featurize(self, test: pd.DataFrame, history: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Model features for the test rows (order preserved) and a 'station has history' mask."""
        stats = self.station_stats
        if history is not None and TARGET in history and history[TARGET].notna().any():
            extra = compute_station_stats(history)
            stats = pd.concat([extra, stats[~stats[ID_COL].isin(extra[ID_COL])]], ignore_index=True)
        frame = test.assign(_is_test=True)
        if history is not None:
            warm = history[history[ID_COL].isin(test[ID_COL])].drop(columns=[TARGET], errors="ignore")
            test_keys = test.set_index([ID_COL, DATE_COL]).index
            warm = warm[~warm.set_index([ID_COL, DATE_COL]).index.isin(test_keys)]
            frame = pd.concat([warm.assign(_is_test=False), frame], ignore_index=True)
        feats = build_features(frame, stats)
        feats = feats[feats["_is_test"].astype(bool)].reset_index(drop=True)
        has_stats = feats["st_log_q_p50"].notna().to_numpy()
        return feats, has_stats

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {}
        for name, ens in (("known", self.known), ("unseen", self.unseen)):
            meta[name] = []
            for m in ens.members:
                fname = f"{name}_{m.objective}.txt"
                m.booster.save_model(str(out_dir / fname))
                meta[name].append({"file": fname, "objective": m.objective, "features": m.features})
        self.station_stats.to_csv(out_dir / "station_stats.csv", index=False, encoding="utf-8")
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "FloodcastModel":
        meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))

        def load_ens(name: str) -> Ensemble:
            return Ensemble(tuple(
                FittedModel(lgb.Booster(model_file=str(out_dir / m["file"])), m["features"], m["objective"])
                for m in meta[name]
            ))

        stats = pd.read_csv(out_dir / "station_stats.csv", dtype={ID_COL: str}, encoding="utf-8")
        return cls(load_ens("known"), load_ens("unseen"), stats)


__all__ = ["FloodcastModel", "FittedModel", "Ensemble", "fit", "KNOWN_FEATURES", "UNSEEN_FEATURES", "WEATHER_COLS"]
