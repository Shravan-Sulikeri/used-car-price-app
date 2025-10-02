#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import joblib

DATA_CANDIDATES = [
    Path("data/clean_listings_clean.parquet"),
    Path("data/clean_listings_clean.csv"),
    Path("data/clean_listings.parquet"),
    Path("data/clean_listings.csv"),
]
ART_DIR = Path("model"); ART_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            print(f"Loaded {p} with {len(df):,} rows")
            return df
    raise SystemExit("No cleaned data found. Run data/ingestion/ingest_any_csvs.py first.")

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("price","year","mileage"):
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["price"].between(1000, 250_000)]
    df = df[df["year"].between(1990, 2026)]
    df = df[(df["mileage"].isna()) | df["mileage"].between(0, 300_000)]
    for c in ("make","model","body_type","drive_type"):
        if c in df.columns: df[c] = df[c].astype(str).str.strip().str.title()
    CURRENT_YEAR = 2025
    df["age"] = (CURRENT_YEAR - df["year"]).clip(lower=0, upper=40)
    df["miles_per_year"] = df["mileage"] / df["age"].replace(0, np.nan)
    df["miles_per_year"] = df["miles_per_year"].fillna(df["mileage"])
    q_hi = df["miles_per_year"].quantile(0.99)
    df["miles_per_year"] = df["miles_per_year"].clip(upper=q_hi)
    return df

def prepare_features(df: pd.DataFrame):
    cat_cols = [c for c in ["make","model","body_type","drive_type"] if c in df.columns]
    for c in cat_cols: df[c] = df[c].astype("category")
    feat_order = [c for c in ["year","mileage","age","miles_per_year"] if c in df.columns] + cat_cols
    X = df[feat_order].copy()
    y = df["price"].astype(float).copy()
    num_defaults = {
        "year": float(df["year"].median()) if "year" in df else 2016.0,
        "mileage": 0.0,
        "age": float(df["age"].median()) if "age" in df else 5.0,
        "miles_per_year": float(df["miles_per_year"].median()) if "miles_per_year" in df else 10000.0,
    }
    cat_levels = {c: [str(x) for x in list(X[c].cat.categories)] for c in cat_cols}
    return X, y, feat_order, num_defaults, cat_levels, cat_cols

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None)))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_pct": mape}

def main():
    df0 = load_data()
    df = engineer(df0)
    df = df.dropna(subset=["price","year","make","model"])
    print(f"After engineering & filter: {len(df):,} rows")

    X, y, feat_order, num_defaults, cat_levels, cat_cols = prepare_features(df)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    params = dict(
        objective="regression_l1",
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    pred_va = model.predict(X_va)
    m = metrics(y_va.values, pred_va)
    print("Validation metrics:", m)

    joblib.dump(model, ART_DIR / "model_gbm.pkl")
    (ART_DIR / "schema_best.json").write_text(json.dumps({
        "feature_order": feat_order,
        "numeric_defaults": num_defaults,
        "generated_at": int(time.time()),
    }, indent=2))
    (ART_DIR / "cat_levels.json").write_text(json.dumps(cat_levels, indent=2))
    (ART_DIR / "metrics_gbm.json").write_text(json.dumps(m, indent=2))

    prev = X_va.copy()
    prev["price_actual"] = y_va.values
    prev["price_pred"] = pred_va
    prev.sort_values("price_actual").head(50).to_csv(ART_DIR / "preview_gbm.csv", index=False)

    print("✅ Saved model/model_gbm.pkl, schema_best.json, cat_levels.json, metrics_gbm.json, preview_gbm.csv")

if __name__ == "__main__":
    main()
