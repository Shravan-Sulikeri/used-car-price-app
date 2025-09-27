# -*- coding: utf-8 -*-
"""
Train LightGBM on cleaned data using the saved preprocessor.

Inputs (env overrides):
- DATA_CLEAN: data/clean_used_cars.csv
- PREPROC_PATH: model/preprocessor.pkl

Outputs:
- model/model_gbm.pkl
- model/metrics_gbm.json
- model/preview_gbm.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError as e:
    raise SystemExit("LightGBM not installed. Run: pip install lightgbm") from e


ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = Path(os.getenv("DATA_CLEAN", ROOT / "data/clean_used_cars.csv"))
PREPROC_PATH = Path(os.getenv("PREPROC_PATH", ROOT / "model/preprocessor.pkl"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model/model_gbm.pkl"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT / "model/metrics_gbm.json"))
PREVIEW_PATH = Path(os.getenv("PREVIEW_PATH", ROOT / "model/preview_gbm.csv"))

RANDOM_STATE = 42


def main():
    assert DATA_CLEAN.exists(), f"Clean CSV not found: {DATA_CLEAN}"
    assert PREPROC_PATH.exists(), f"Preprocessor not found: {PREPROC_PATH}"

    df = pd.read_csv(DATA_CLEAN, low_memory=False)
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in cleaned data.")

    # Features/target
    y = df["price"].astype(float).to_numpy()
    X = df.drop(columns=["price"])

    # Load preprocessor -> numeric matrix
    pre = joblib.load(PREPROC_PATH)
    X_enc = pre.transform(X)

    # Train/val split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_enc, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Log target
    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)

    # Model
    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # NOTE: LightGBM v4.x: use callbacks instead of verbose=
    model.fit(
        X_tr, y_tr_log,
        eval_set=[(X_va, y_va_log)],
        eval_metric="l2",
        callbacks=[
            early_stopping(stopping_rounds=200, verbose=False),
            log_evaluation(period=0),  # silence training logs
        ],
    )

    # Validate in original units
    y_pred_log = model.predict(X_va)
    y_pred = np.expm1(y_pred_log)

    mae = float(mean_absolute_error(y_va, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
    r2 = float(r2_score(y_va, y_pred))

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Preview: a few rows (val set)
    prev = pd.DataFrame({
        "pred_price": y_pred,
        "actual_price": y_va,
        "abs_err": np.abs(y_pred - y_va),
    }).round(2).head(20)
    prev.to_csv(PREVIEW_PATH, index=False)

    print(f"✅ Trained GBM: {MODEL_PATH}")
    print(f"Metrics →  MAE: ${int(mae):,} | RMSE: ${int(rmse):,} | R²: {r2:.3f}")
    print(f"Saved preview → {PREVIEW_PATH}")


if __name__ == "__main__":
    main()