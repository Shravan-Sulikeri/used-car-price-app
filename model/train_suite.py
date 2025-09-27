# model/train_suite.py
import json, warnings, joblib, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional imports handled gracefully
try:
    from catboost import CatBoostRegressor, Pool
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import OneHotEncoder
try:
    import category_encoders as ce
    HAVE_CE = True
except Exception:
    HAVE_CE = False

RANDOM_STATE = 42
DATA = Path("data/clean_used_cars.csv")
OUT_BEST = Path("model/model_gbm.pkl")         # dashboard expects this
OUT_METRICS = Path("model/metrics_gbm.json")   # dashboard expects this
MODEL_DIR = Path("model")

# -------------------------- Helpers --------------------------

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def to_dollars(arr):
    """Convert log-price predictions to $ if needed (we detect per-model)."""
    return np.expm1(arr)

def feature_lists(df: pd.DataFrame):
    num_candidates = ["year","mileage","age","mileage_per_year","high_mileage"]
    cat_candidates = ["make","model","body","fuel","transmission","seller_type","state","title_status"]
    num_cols = [c for c in num_candidates if c in df.columns]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    return num_cols, cat_cols

def groups_make_model(df: pd.DataFrame):
    if {"make","model"}.issubset(df.columns):
        return (df["make"].astype(str) + " | " + df["model"].astype(str)).values
    return np.arange(len(df))

@dataclass
class CVResult:
    name: str
    fold_preds: np.ndarray
    fold_true: np.ndarray
    metrics: Dict[str, float]
    final_model: Any
    predict_is_log: bool
    feature_cols: List[str]
    cat_cols: List[str]

# -------------------------- Wrappers for stable predict --------------------------

class CatBoostWrapper:
    def __init__(self, model, feature_cols: List[str], cat_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self._cat_idx = [self.feature_cols.index(c) for c in self.cat_cols if c in self.feature_cols]

    def predict(self, X: pd.DataFrame):
        Xo = X.reindex(columns=self.feature_cols, fill_value=np.nan)
        pool = Pool(Xo, cat_features=self._cat_idx)
        return self.model.predict(pool)

class LGBMWrapper:
    def __init__(self, model, feature_cols: List[str], cat_cols: List[str], cat_categories: Dict[str, List[Any]]):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.cat_categories = cat_categories

    def _cast_categories(self, X):
        X = X.reindex(columns=self.feature_cols, fill_value=np.nan).copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
                # lock categories to those seen in train to avoid unseen-category warnings
                cats = self.cat_categories.get(c, None)
                if cats is not None:
                    X[c] = X[c].cat.set_categories(cats)
        return X

    def predict(self, X: pd.DataFrame):
        Xc = self._cast_categories(X)
        return self.model.predict(Xc)

class SKPipelineWrapper:
    def __init__(self, pipe, feature_cols: List[str]):
        self.pipe = pipe
        self.feature_cols = feature_cols

    def predict(self, X: pd.DataFrame):
        Xo = X.reindex(columns=self.feature_cols, fill_value=np.nan)
        return self.pipe.predict(Xo)

# -------------------------- CV Trainers --------------------------

def cv_catboost(df: pd.DataFrame, y_log, num_cols, cat_cols, groups) -> Optional[CVResult]:
    if not HAVE_CAT:
        return None
    feature_cols = num_cols + cat_cols
    X = df[feature_cols]

    # build folds
    uniq_groups = np.unique(groups)
    n_splits = 5 if len(uniq_groups) >= 5 else max(3, min(5, len(uniq_groups)))
    splitter = GroupKFold(n_splits=n_splits)

    y_true_all, y_pred_all = [], []
    fold = 0
    models = []

    cat_idx = [feature_cols.index(c) for c in cat_cols]

    for tr_idx, va_idx in splitter.split(X, y_log, groups):
        fold += 1
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_log[tr_idx], y_log[va_idx]
        tr_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        va_pool = Pool(Xva, yva, cat_features=cat_idx)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            depth=8,
            learning_rate=0.06,
            l2_leaf_reg=6.0,
            n_estimators=3000,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_state=RANDOM_STATE,
            od_type="Iter",
            od_wait=200,
            verbose=False
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True, verbose=False)
        pred_log = model.predict(va_pool)

        y_true_all.append(yva)
        y_pred_all.append(pred_log)
        models.append(model)

    y_true = np.concatenate(y_true_all)
    y_pred_log = np.concatenate(y_pred_all)
    y_pred = to_dollars(y_pred_log)
    y_true_usd = to_dollars(y_true)

    metrics = {
        "mae": float(mean_absolute_error(y_true_usd, y_pred)),
        "rmse": float(rmse(y_true_usd, y_pred)),
        "r2": float(r2_score(y_true_usd, y_pred)),
    }

    # fit final on full data
    full_pool = Pool(X, y_log, cat_features=cat_idx)
    final = CatBoostRegressor(
        loss_function="RMSE",
        depth=8, learning_rate=0.06, l2_leaf_reg=6.0,
        n_estimators=int(np.median([m.tree_count_ for m in models])),  # use median best-iteration
        subsample=0.8, colsample_bylevel=0.8, random_state=RANDOM_STATE, verbose=False
    )
    final.fit(full_pool)

    wrapped = CatBoostWrapper(final, feature_cols, cat_cols)
    return CVResult("CatBoostRegressor", y_pred, y_true_usd, metrics, wrapped, True, feature_cols, cat_cols)

def cv_lightgbm(df: pd.DataFrame, y_log, num_cols, cat_cols, groups) -> Optional[CVResult]:
    if not HAVE_LGBM:
        return None
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()

    # cast cats
    cat_categories = {}
    for c in cat_cols:
        X[c] = X[c].astype("category")
        cat_categories[c] = list(X[c].cat.categories)

    uniq_groups = np.unique(groups)
    n_splits = 5 if len(uniq_groups) >= 5 else max(3, min(5, len(uniq_groups)))
    splitter = GroupKFold(n_splits=n_splits)

    y_true_all, y_pred_all = [], []
    iters = []

    params = dict(
        n_estimators=5000, learning_rate=0.03, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0,
        objective="regression", random_state=RANDOM_STATE, n_jobs=-1
    )

    for tr_idx, va_idx in splitter.split(X, y_log, groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_log[tr_idx], y_log[va_idx]

        model = LGBMRegressor(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="rmse",
            callbacks=[],
        )
        # LightGBM doesn't expose best_iteration_ reliably without early stopping; use all trees
        pred_log = model.predict(Xva)

        y_true_all.append(yva)
        y_pred_all.append(pred_log)
        iters.append(params["n_estimators"])

    y_true = np.concatenate(y_true_all)
    y_pred_log = np.concatenate(y_pred_all)
    y_pred = to_dollars(y_pred_log)
    y_true_usd = to_dollars(y_true)

    metrics = {
        "mae": float(mean_absolute_error(y_true_usd, y_pred)),
        "rmse": float(rmse(y_true_usd, y_pred)),
        "r2": float(r2_score(y_true_usd, y_pred)),
    }

    final = LGBMRegressor(**params)
    final.fit(X, y_log)

    wrapped = LGBMWrapper(final, feature_cols, cat_cols, cat_categories)
    return CVResult("LightGBM", y_pred, y_true_usd, metrics, wrapped, True, feature_cols, cat_cols)

def cv_elasticnet(df: pd.DataFrame, y_log, num_cols, cat_cols, groups) -> Optional[CVResult]:
    # Hash high-card cats (make, model), one-hot low-card cats
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= 50]
    high_card = [c for c in cat_cols if c not in low_card]

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if low_card:
        transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore"), low_card))

    if high_card and HAVE_CE:
        # Hash each high-card column separately to keep signal per field
        for c in high_card:
            transformers.append((
                f"hash_{c}",
                Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                    ("hash", ce.HashingEncoder(n_components=64, return_df=False)),
                ]),
                [c]
            ))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    model = ElasticNetCV(
        l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95],  # mix of ridge/lasso
        alphas=np.logspace(-3, 1, 20),
        max_iter=3000,
        n_jobs=-1,
        cv=3,
        random_state=RANDOM_STATE
    )
    pipe = Pipeline([("pre", pre), ("m", model)])

    feature_cols = num_cols + cat_cols
    X = df[feature_cols]

    uniq_groups = np.unique(groups)
    n_splits = 5 if len(uniq_groups) >= 5 else max(3, min(5, len(uniq_groups)))
    splitter = GroupKFold(n_splits=n_splits)

    y_true_all, y_pred_all = [], []

    for tr_idx, va_idx in splitter.split(X, y_log, groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_log[tr_idx], y_log[va_idx]
        pipe.fit(Xtr, ytr)
        pred_log = pipe.predict(Xva)
        y_true_all.append(yva)
        y_pred_all.append(pred_log)

    y_true = np.concatenate(y_true_all)
    y_pred_log = np.concatenate(y_pred_all)
    y_pred = to_dollars(y_pred_log)
    y_true_usd = to_dollars(y_true)

    metrics = {
        "mae": float(mean_absolute_error(y_true_usd, y_pred)),
        "rmse": float(rmse(y_true_usd, y_pred)),
        "r2": float(r2_score(y_true_usd, y_pred)),
    }

    # final fit
    pipe.fit(X, y_log)
    wrapped = SKPipelineWrapper(pipe, feature_cols)
    return CVResult("ElasticNet", y_pred, y_true_usd, metrics, wrapped, True, feature_cols, cat_cols)

# -------------------------- Main --------------------------

def main():
    assert DATA.exists(), f"Missing {DATA}. Clean your data first."

    df = pd.read_csv(DATA, low_memory=False)
    assert "price" in df.columns, "Data must include 'price' column."

    # Ensure engineered features exist (safe if already there)
    ref_year = pd.Timestamp.now().year
    if "year" in df and "age" not in df:
        df["age"] = (ref_year - df["year"]).clip(lower=0)
    if {"mileage","age"}.issubset(df.columns) and "mileage_per_year" not in df:
        df["mileage_per_year"] = df["mileage"] / np.where(df["age"] < 1, 1, df["age"])
    if "mileage" in df and "high_mileage" not in df:
        df["high_mileage"] = (df["mileage"] > 150_000).astype("int8")

    num_cols, cat_cols = feature_lists(df)
    feature_cols = num_cols + cat_cols

    # Filter rows with required fields
    need = ["price"] + feature_cols
    df = df.dropna(subset=["price"]).copy()
    # Convert target to log
    y = df["price"].values.astype(float)
    y_log = np.log1p(y)

    groups = groups_make_model(df)

    results: List[CVResult] = []

    if HAVE_CAT and len(cat_cols) > 0:
        print("Training CatBoost…")
        results.append(cv_catboost(df, y_log, num_cols, cat_cols, groups))

    if HAVE_LGBM:
        print("Training LightGBM…")
        results.append(cv_lightgbm(df, y_log, num_cols, cat_cols, groups))

    print("Training ElasticNet…")
    results.append(cv_elasticnet(df, y_log, num_cols, cat_cols, groups))

    # Drop Nones (in case a library missing)
    results = [r for r in results if r is not None]

    # Pick best by MAE
    best = min(results, key=lambda r: r.metrics["mae"])
    print("\n=== CV Results ===")
    for r in results:
        print(f"{r.name}:  MAE ${r.metrics['mae']:,.0f} | RMSE ${r.metrics['rmse']:,.0f} | R² {r.metrics['r2']:.3f}")
    print(f"\n✅ Best: {best.name}")

    # Save best model + metrics (dashboard-compatible)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.final_model, OUT_BEST)
    OUT_METRICS.write_text(json.dumps({
        "model_name": best.name,
        "target": "log_price",
        **best.metrics
    }, indent=2))

    print("\nSaved:")
    print(" -", OUT_BEST)
    print(" -", OUT_METRICS)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()