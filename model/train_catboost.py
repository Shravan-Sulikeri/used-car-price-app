# model/train_catboost.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

DATA = Path("data/clean_used_cars.csv")
OUT_MODEL = Path("model/model_gbm.pkl")         # keep dashboard compatibility
OUT_METRICS = Path("model/metrics_gbm.json")    # keep dashboard compatibility

RANDOM_STATE = 42

def main():
    assert DATA.exists(), f"Missing {DATA}. Run your cleaner first."

    df = pd.read_csv(DATA, low_memory=False)

    # --- feature set aligned with your dashboard predictor ---
    # Use whatever exists from this list
    num_cols = [c for c in ["year","mileage","age","mileage_per_year","high_mileage"] if c in df.columns]
    cat_cols = [c for c in ["make","model","body","fuel","transmission","seller_type","state","title_status"] if c in df.columns]

    needed = ["price"] + num_cols + cat_cols
    df = df[needed].dropna(subset=["price"]).copy()

    # train on log-price for stability
    df["price_log"] = np.log1p(df["price"])

    # group by (make, model) to reduce leakage
    if {"make","model"}.issubset(df.columns):
        groups = (df["make"].astype(str) + " | " + df["model"].astype(str)).values
    else:
        groups = df.index.values  # fallback

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(splitter.split(df, groups=groups))

    X_tr, X_va = df.iloc[tr_idx][num_cols + cat_cols], df.iloc[va_idx][num_cols + cat_cols]
    y_tr, y_va = df.iloc[tr_idx]["price_log"].values, df.iloc[va_idx]["price_log"].values

    # CatBoost: pass categorical column indices (within the order of X_tr columns)
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in cat_cols]

    # Build Pools (CatBoost handles NaNs; keep string cats as-is)
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

    # A strong, still-fast starting configuration
    model = CatBoostRegressor(
        loss_function="RMSE",            # optimizing RMSE on log-price
        eval_metric="RMSE",
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=6.0,
        n_estimators=3000,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=RANDOM_STATE,
        od_type="Iter",
        od_wait=200,
        verbose=200
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Evaluate on $ scale
    pred_log = model.predict(valid_pool)
    pred = np.expm1(pred_log)
    y_true = np.expm1(y_va)

    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = float(r2_score(y_true, pred))

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODEL)

    OUT_METRICS.write_text(json.dumps({"mae": mae, "rmse": rmse, "r2": r2}, indent=2))

    print("✅ Trained CatBoost (log-price). Saved →", OUT_MODEL)
    print(f"Metrics →  MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | R²: {r2:.3f}")

if __name__ == "__main__":
    main()