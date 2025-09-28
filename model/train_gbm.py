#!/usr/bin/env python3
import os, json, math, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

DATA_PATH = Path(os.getenv("DATA_CSV", "data/clean_used_cars.csv"))
MODEL_PATH = Path("model/model_gbm.pkl")
METRICS_PATH = Path("model/metrics_gbm.json")
PREVIEW_PATH = Path("model/preview_gbm.csv")
CAT_LEVELS_PATH = Path("model/cat_levels.json")

RANDOM_SEED = 42

NUM_COLS = ["year", "mileage", "age", "mileage_per_year", "high_mileage"]
CAT_COLS = ["make", "model", "body"]
ALL_FEATS = NUM_COLS + CAT_COLS

def compute_features(df: pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if ref_year is None:
        ref_year = pd.Timestamp.today().year
    # age
    out["age"] = (ref_year - out["year"]).clip(lower=0)
    # mileage_per_year
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mileage_per_year"] = out["mileage"] / out["age"].replace({0: np.nan})
    out["mileage_per_year"] = out["mileage_per_year"].fillna(out["mileage"])
    # high_mileage (simple heuristic)
    out["high_mileage"] = (out["mileage"] > 150_000).astype("int8")
    return out

def as_categoricals(df: pd.DataFrame, cat_levels: dict[str, list[str]] | None):
    """Coerce CAT_COLS to pandas Categorical with fixed vocab; unseen -> NaN."""
    out = df.copy()
    for c in CAT_COLS:
        if cat_levels and c in cat_levels:
            dtype = pd.api.types.CategoricalDtype(categories=cat_levels[c])
            # map unseen to NaN before astype
            s = out[c].astype("string").str.strip().where(lambda x: x.isin(dtype.categories), other=pd.NA)
            out[c] = s.astype(dtype)
        else:
            out[c] = out[c].astype("string").str.strip()
            cats = sorted(out[c].dropna().unique().tolist())
            out[c] = out[c].astype(pd.CategoricalDtype(categories=cats))
    return out

def downcast_numeric(df: pd.DataFrame):
    out = df.copy()
    for c in ["price", "mileage", "mileage_per_year"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in ["year", "age", "high_mileage"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("int32", copy=False)
    return out

def main():
    assert DATA_PATH.exists(), f"Data not found: {DATA_PATH}"
    df0 = pd.read_csv(DATA_PATH, low_memory=False)
    # Ensure basic columns exist
    need = {"price","year","mileage","make","model","body"}
    missing = sorted(list(need - set(df0.columns)))
    assert not missing, f"Missing columns in data: {missing}"

    # Filter obvious bad rows (light guardrails)
    df0 = df0[pd.to_numeric(df0["price"], errors="coerce").between(500, 250_000)]
    df0 = df0[pd.to_numeric(df0["mileage"], errors="coerce").between(0, 600_000)]
    df0 = df0[pd.to_numeric(df0["year"], errors="coerce").between(1980, pd.Timestamp.today().year + 1)]
    df0 = df0.dropna(subset=["price","year","mileage","make","model"]).reset_index(drop=True)

    # Feature eng
    df = compute_features(df0)
    df = downcast_numeric(df)

    # Train/valid split (stratify by binned price to stabilize)
    price_bins = pd.qcut(df["price"], q=20, duplicates="drop")
    tr, va = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=price_bins)

    # Build category vocab from TRAIN ONLY
    cat_levels = {}
    for c in CAT_COLS:
        cat_levels[c] = sorted(tr[c].astype("string").str.strip().dropna().unique().tolist())

    # Coerce categoricals with fixed vocab (unseen -> NaN) for both splits
    tr = as_categoricals(tr, cat_levels)
    va = as_categoricals(va, cat_levels)

    # Final feature frames
    X_tr, y_tr = tr[ALL_FEATS].copy(), tr["price"].astype("float32").values
    X_va, y_va = va[ALL_FEATS].copy(), va["price"].astype("float32").values

    # LightGBM model (no one-hot)
    model = LGBMRegressor(
        objective="rmse",          # regression
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # Fit with early stopping; pass categorical feature names
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(100)],
        categorical_feature=CAT_COLS,
    )

    # Evaluate on validation (USD)
    preds = model.predict(X_va)
    mae = float(mean_absolute_error(y_va, preds))
    rmse = float(math.sqrt(mean_squared_error(y_va, preds)))
    r2 = float(r2_score(y_va, preds))

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(CAT_LEVELS_PATH, "w") as f:
        json.dump(cat_levels, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2, "rows": int(len(df)), "features": ALL_FEATS}, f, indent=2)

    # Preview file
    prev = va[["price"] + ALL_FEATS].copy()
    prev["pred_price"] = preds
    prev.sample(min(200, len(prev))).to_csv(PREVIEW_PATH, index=False)

    print(f"✅ Trained GBM on {len(df):,} rows: {MODEL_PATH}")
    print(f"Metrics →  MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | R²: {r2:.3f}")
    print(f"Saved cat levels → {CAT_LEVELS_PATH}")
    print(f"Saved preview → {PREVIEW_PATH}")

if __name__ == "__main__":
    main()