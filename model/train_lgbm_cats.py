# model/train_lgbm_cats.py
import os, json, joblib, pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

DATA = "/workspaces/used-car-price-app/data/clean_used_cars_curated.csv"
ART_DIR = pathlib.Path("/workspaces/used-car-price-app/model")
ART_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1) Load ----------
df = pd.read_csv(DATA)

# ---------- 2) Features + target ----------
REF_YEAR = 2025
y = np.log1p(df["price"])
X = df.drop(columns=["price"]).copy()

# engineered features (cheap wins)
X["age"] = REF_YEAR - X["year"]
X["miles_per_year"] = X["mileage"] / X["age"].clip(lower=1)
X["log_mileage"] = np.log1p(X["mileage"])
X["age_x_mpy"] = X["age"] * X["miles_per_year"]

# set categoricals
cat_cols = X.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

# safety: fill numeric NaNs
num_cols = [c for c in X.columns if c not in cat_cols]
X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

# ---------- 3) Split ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 4) Train ----------
model = LGBMRegressor(
    n_estimators=1800,
    learning_rate=0.03,
    num_leaves=127,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=80,
    random_state=42,
    enable_categorical=True,
)
model.fit(X_tr, y_tr)

# ---------- 5) Evaluate ----------
pred_log = model.predict(X_te)
mae_usd = np.abs(np.expm1(y_te) - np.expm1(pred_log)).mean()

# MAE by price decile (USD)
# MAE by price decile (USD) → JSON-safe keys
eval_df = X_te.copy()
eval_df["price"] = np.expm1(y_te)
eval_df["pred"] = np.expm1(pred_log)
eval_df["abs_err"] = (eval_df["price"] - eval_df["pred"]).abs()
eval_df["price_decile"] = pd.qcut(eval_df["price"], 10, duplicates="drop")

mae_by_decile_series = eval_df.groupby("price_decile", observed=True)["abs_err"].mean().round(2)
mae_by_decile = {str(k): float(v) for k, v in mae_by_decile_series.items()}

print(f"\nOverall MAE (USD): {mae_usd:,.2f}")
print("MAE by price decile:")
for k,v in mae_by_decile.items():
    print(f"  {k}: {v:,.2f}")

# ---------- 6) Persist artifacts ----------
joblib.dump(model, ART_DIR / "best_model.pkl")

metrics = {
    "mae_usd": float(mae_usd),
    "rows_train": int(len(X_tr)),
    "rows_test": int(len(X_te)),
    "feature_count": int(X.shape[1]),
    "mae_by_price_decile": mae_by_decile,
    "model": "LightGBM (native categoricals)"
}
with open(ART_DIR / "metrics_best.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save schema so Streamlit knows how to prep incoming data
schema = {
    "feature_order": X.columns.tolist(),
    "categorical_cols": cat_cols,
    "numeric_fill_median": {c: float(X[c].median()) for c in num_cols},
    "ref_year": REF_YEAR,
    "log_target": True
}
with open(ART_DIR / "schema_best.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\nSaved:")
print("  model/best_model.pkl")
print("  model/metrics_best.json")
print("  model/schema_best.json")
