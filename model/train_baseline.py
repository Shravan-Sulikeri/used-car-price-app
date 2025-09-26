# model/train_baseline.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN_PATH = "data/clean_used_cars.csv"
PREPROC_PATH = "model/preprocessor.pkl"
MODEL_PATH = "model/model_baseline.pkl"
METRICS_PATH = "model/metrics_baseline.json"

def unlog(x): return np.expm1(x)

def main():
    # Load data + preprocessor
    df = pd.read_csv(CLEAN_PATH)
    pre = joblib.load(PREPROC_PATH)

    X = df.drop(columns=["price"])
    y = np.log1p(df["price"].values)  # train on log-price

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline: preprocessor + Ridge
    pipe = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0))
    ])

    pipe.fit(X_tr, y_tr)

    # Validate in $ space
    pred_log = pipe.predict(X_val)
    pred = unlog(pred_log)
    y_val_price = unlog(y_val)

    mae = float(mean_absolute_error(y_val_price, pred))
    rmse = float(np.sqrt(mean_squared_error(y_val_price, pred)))
    r2 = float(r2_score(y_val_price, pred))

    # Save model + metrics
    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    # Show a few sample predictions for sanity
    sample = X_val.head(5).copy()
    sample_pred = unlog(pipe.predict(sample))
    sample_actual = y_val_price[:5]
    preview = pd.DataFrame({
        "pred_price": np.round(sample_pred, 2),
        "actual_price": np.round(sample_actual, 2),
        "abs_err": np.round(np.abs(sample_pred - sample_actual), 2)
    })
    print("✅ Trained baseline model:", MODEL_PATH)
    print("Metrics →  MAE: ${:,.0f} | RMSE: ${:,.0f} | R²: {:.3f}".format(mae, rmse, r2))
    print("\nPreview (first 5 val rows):")
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
