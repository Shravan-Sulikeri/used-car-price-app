# model/train_gbm.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

CLEAN_PATH = "data/clean_used_cars.csv"
PREPROC_PATH = "model/preprocessor.pkl"
MODEL_PATH = "model/model_gbm.pkl"          # (ignored by .gitignore)
METRICS_PATH = "model/metrics_gbm.json"
PREVIEW_PATH = "model/preview_gbm.csv"

def unlog(x): return np.expm1(x)

def main():
    df = pd.read_csv(CLEAN_PATH)
    pre = joblib.load(PREPROC_PATH)

    X = df.drop(columns=["price"])
    y = np.log1p(df["price"].values)  # train in log space

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("pre", pre),
        ("gbm", LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    pipe.fit(X_tr, y_tr)

    pred_log = pipe.predict(X_val)
    pred = unlog(pred_log)
    y_val_price = unlog(y_val)

    mae = float(mean_absolute_error(y_val_price, pred))
    rmse = float(np.sqrt(mean_squared_error(y_val_price, pred)))
    r2 = float(r2_score(y_val_price, pred))

    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    preview = pd.DataFrame({
        "pred_price": np.round(pred[:10], 2),
        "actual_price": np.round(y_val_price[:10], 2),
        "abs_err": np.round(np.abs(pred[:10] - y_val_price[:10]), 2),
    })
    preview.to_csv(PREVIEW_PATH, index=False)

    print("✅ Trained GBM:", MODEL_PATH)
    print("Metrics →  MAE: ${:,.0f} | RMSE: ${:,.0f} | R²: {:.3f}".format(mae, rmse, r2))
    print("Saved preview →", PREVIEW_PATH)

if __name__ == "__main__":
    main()
