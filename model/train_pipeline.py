#!/usr/bin/env python3
"""
End-to-end training pipeline to build a stronger used-car price model with:
  • Robust cleaning + canonical model-family mapping (e.g., 328i → 3 Series)
  • Feature engineering (region, drivetrain/transmission simplifications, engine liters, interactions)
  • Outlier filtering per (make, model_family, year)
  • K-Fold CV with early stopping and categorical features
  • Optional Optuna tuning (if installed) to search LightGBM params
  • Conformal prediction intervals (p10–p90) from CV residuals
  • Artifacts: model.pkl, columns.json, cat_cols.json, metrics.json, intervals.json

Usage
-----
python train_pipeline.py \
  --input data/used_cars.csv \
  --output_dir artifacts \
  --target price \
  --folds 5 \
  --seed 42 \
  --remove_outliers 1 \
  --optuna 0

Notes
-----
• Requires: pandas, numpy, scikit-learn, lightgbm. (optuna optional.)
• Categorical columns are passed as pandas category dtypes to LightGBM.
• Target is trained on log(price); metrics are reported on price scale too.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import lightgbm as lgb

# -----------------------------
# Canonicalization utilities
# -----------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _norm_text(s: str) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())


BMW_RULES = [
    (r"\b(3\s*series|e9[0-2]|f3[0-9]|g2[0-1])\b", "3 Series"),
    (r"\b(31[6-9]i|320[d|i]?|323[i]?|325[i]?|328[i|d]?|330[i|e|d]?|335[i|d|is]?|340i|m3|xdrive ?3\d{2}[id]?)\b", "3 Series"),
    (r"\b(5\s*series|e6[0-1]|f1[0-1]|g3[0-1])\b", "5 Series"),
    (r"\b(520[di]?|523i|525[i|d]?|528[i|d]?|530[i|e|d]?|535[i|d]?|540i|m5|xdrive ?5\d{2}[id]?)\b", "5 Series"),
    (r"\b(x1)\b", "X1"), (r"\b(x2)\b", "X2"), (r"\b(x3)\b", "X3"), (r"\b(x4)\b", "X4"), (r"\b(x5)\b", "X5"), (r"\b(x6)\b", "X6"),
]

AUDI_RULES = [
    (r"\b(a[1-9])\b", lambda m: m.group(1).upper()),
    (r"\b(s[1-9])\b", lambda m: m.group(1).upper()),
    (r"\b(rs[1-9])\b", lambda m: m.group(1).upper()),
    (r"\b(q[2-8])\b", lambda m: m.group(1).upper()),
    (r"\b(3\.0t|2\.0t)\b.*\ba4\b", "A4"),
    (r"\b(3\.0t|2\.0t)\b.*\ba6\b", "A6"),
]

MB_RULES = [
    (r"\b(c[-\s]?class|c\d{2,3})\b", "C-Class"),
    (r"\b(e[-\s]?class|e\d{2,3})\b", "E-Class"),
    (r"\b(s[-\s]?class|s\d{2,3})\b", "S-Class"),
    (r"\b(gl[abce]|gle|gls|gla|glc|g-wagen|g\s?class)\b", lambda m: {
        "gla": "GLA", "glb": "GLB", "glc": "GLC", "gle": "GLE", "gls": "GLS",
        "g-wagen": "G-Class", "g class": "G-Class"
    }.get(m.group(1), m.group(1).upper())),
]

TOYOTA_RULES = [
    (r"\b(camry)\b", "Camry"), (r"\b(corolla)\b", "Corolla"), (r"\b(rav4)\b", "RAV4"),
    (r"\b(4runner)\b", "4Runner"), (r"\b(tacoma)\b", "Tacoma"), (r"\b(highlander)\b", "Highlander"),
]

MAKE_RULES: Dict[str, List[Tuple[str, object]]] = {
    "bmw": BMW_RULES,
    "audi": AUDI_RULES,
    "mercedes-benz": MB_RULES,
    "mercedes": MB_RULES,
    "toyota": TOYOTA_RULES,
}


def canonical_model(make: str, raw_model: str) -> str:
    mk = _norm_text(make)
    md = _norm_text(raw_model)
    rules = MAKE_RULES.get(mk, [])
    for pat, repl in rules:
        m = re.search(pat, md)
        if m:
            return (repl(m) if callable(repl) else repl)
    if mk == "bmw":
        m = re.search(r"\b([1-8])\s*series\b", md)
        if m:
            return f"{m.group(1)} Series"
        m = re.search(r"\b(1|2|3|4|5|6|7|8)(\d{2})[a-z]*\b", md)
        if m:
            return f"{m.group(1)} Series"
    return raw_model.strip().title() if isinstance(raw_model, str) else "Unknown"


def apply_canonical_model(df: pd.DataFrame, make_col="make", model_col="model", out_col="model_family") -> pd.DataFrame:
    df[out_col] = df[[make_col, model_col]].apply(lambda r: canonical_model(r[make_col], r[model_col]), axis=1)
    return df


# -----------------------------
# Cleaning, outliers, features
# -----------------------------

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize strings
    for c in ["make", "model", "trim", "fuel", "transmission", "drive", "title_status", "state", "city"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # Standardize unknowns
    unknown_like = ["", "na", "n/a", "none", "unknown", "null", "nan", "—", "-", "unk"]
    for c in ["model", "trim", "fuel", "transmission", "drive", "title_status"]:
        if c in df.columns:
            df[c] = df[c].where(~df[c].str.lower().isin(unknown_like), other=pd.NA)
    # Bounds
    if "year" in df.columns:
        df = df[df["year"].astype(float).between(1990, 2026)]
        df["year"] = df["year"].astype(int)
    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
        df = df[(df["odometer"] >= 0) & (df["odometer"] <= 500_000)]
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[(df["price"] >= 500) & (df["price"] <= 300_000)]
    return df


def remove_outliers_iqr(df: pd.DataFrame, group_cols=("make", "model_family", "year"), target="price", iqr_k=2.0):
    df = df.copy()
    if target not in df.columns:
        return df
    def _clip_grp(g):
        if len(g) < 8:
            return g
        q1 = g[target].quantile(0.25)
        q3 = g[target].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        return g[(g[target] >= lo) & (g[target] <= hi)]
    return df.groupby(list(group_cols), group_keys=False).apply(_clip_grp)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Age and transforms
    if "year" in df.columns:
        df["car_age"] = (pd.Timestamp.now().year - df["year"]).clip(lower=0)
    if "odometer" in df.columns:
        df["log_odometer"] = np.log1p(df["odometer"])
    if "price" in df.columns:
        df["log_price"] = np.log1p(df["price"])
    # Region
    if "state" in df.columns:
        northeast = set("CT MA ME NH RI VT NY NJ PA".split())
        south = set("AL AR DE FL GA KY LA MD MS NC OK SC TN TX VA WV DC".split())
        midwest = set("IL IN IA KS MI MN MO NE ND OH SD WI".split())
        west = set("AK AZ CA CO HI ID MT NM NV OR UT WA WY".split())
        df["region"] = (
            df["state"].str.upper().map(lambda s:
                "Northeast" if s in northeast else
                "South" if s in south else
                "Midwest" if s in midwest else
                "West" if s in west else "Other"
            )
        )
    # Transmission/drivetrain
    if "transmission" in df.columns:
        df["trans_simple"] = df["transmission"].str.lower().str.extract(r"(auto|manual)")[0].map(
            {"auto": "Automatic", "manual": "Manual"}
        ).fillna("Other")
    if "drive" in df.columns:
        df["drive_simple"] = df["drive"].str.upper().replace({
            "AWD": "AWD", "4WD": "4WD", "FWD": "FWD", "RWD": "RWD"
        }).fillna("Other")
    # Engine liters from free text
    text_cols = [c for c in ["trim", "model", "title", "description"] if c in df.columns]
    if text_cols:
        combined = df[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
        disp = pd.to_numeric(combined.str.extract(r"(\d\.\d)\s*l")[0], errors="coerce")
        df["engine_liters"] = disp
    # Luxury brand flag
    luxury = {"bmw", "mercedes-benz", "mercedes", "audi", "lexus", "infiniti", "acura", "porsche", "tesla", "jaguar", "volvo"}
    if "make" in df.columns:
        df["is_luxury"] = df["make"].str.lower().isin(luxury).astype("int8")
    # Miles per year
    if {"car_age", "odometer"}.issubset(df.columns):
        df["miles_per_year"] = (df["odometer"] / df["car_age"].replace(0, np.nan)).fillna(df["odometer"]).clip(upper=60_000)
    # Categorical dtypes
    for c in ["make", "model_family", "region", "trans_simple", "drive_simple", "fuel", "title_status", "state", "city"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


# -----------------------------
# Training and evaluation
# -----------------------------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def price_metrics(y_true_price, y_pred_price) -> Dict[str, float]:
    return {
        "RMSE": float(rmse(y_true_price, y_pred_price)),
        "MAE": float(mean_absolute_error(y_true_price, y_pred_price)),
        "MedAE": float(median_absolute_error(y_true_price, y_pred_price)),
        "R2": float(r2_score(y_true_price, y_pred_price)),
        "MAPE%": float((np.abs(y_true_price - y_pred_price) / np.clip(y_true_price, 1e-6, None)).mean() * 100.0),
    }


def default_lgbm_params() -> Dict:
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=40,
        lambda_l1=0.0,
        lambda_l2=0.0,
        verbosity=-1,
    )


def tune_with_optuna(X, y, cat_cols: List[str], seed: int, n_trials: int = 40):
    try:
        import optuna  # type: ignore
    except Exception as e:
        print("[optuna] not installed; using default params.")
        return default_lgbm_params()

    def objective(trial: "optuna.trial.Trial"):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        rmses = []
        for tr_idx, va_idx in kf.split(X):
            dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx], categorical_feature=cat_cols, free_raw_data=False)
            dvalid = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], categorical_feature=cat_cols, free_raw_data=False)
            model = lgb.train(params, dtrain, num_boost_round=500,
                              valid_sets=[dvalid], valid_names=["valid"],
                              callbacks=[lgb.early_stopping(50, verbose=False)],)
            pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
            rmses.append(rmse(y.iloc[va_idx], pred))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial.params
    best.update({"objective": "regression", "metric": "rmse", "verbosity": -1})
    return best


def conformal_from_cv(residuals: List[np.ndarray], lower_q=0.10, upper_q=0.90) -> Dict[str, float]:
    res = np.concatenate([np.abs(r) for r in residuals])
    return {
        "abs_resid_q_lo": float(np.quantile(res, upper_q)),  # upper quantile for lower bound subtraction
        "abs_resid_q_hi": float(np.quantile(res, upper_q)),  # symmetric for simplicity
        "lower_q": lower_q,
        "upper_q": upper_q,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def run(args):
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Core prep
    df = clean_basic(df)
    df = apply_canonical_model(df, make_col="make", model_col="model", out_col="model_family")
    if args.remove_outliers:
        df = remove_outliers_iqr(df, group_cols=("make", "model_family", "year"), target=args.target, iqr_k=2.0)
    df = add_features(df)

    # Define features/target
    target_col = "log_price" if args.target == "price" else args.target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after preprocessing.")

    drop_cols = {args.target, "log_price", "description", "title"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in feature_cols if str(df[c].dtype) == "category"]

    X = df[feature_cols]
    y = df[target_col]

    # Tune params (optional)
    params = default_lgbm_params()
    if args.optuna:
        params = tune_with_optuna(X, y, cat_cols, seed=args.seed, n_trials=args.optuna_trials)
        print("[optuna] best params:", params)

    # KFold CV
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros(len(X))
    models: List[lgb.Booster] = []
    fold_metrics: List[Dict[str, float]] = []
    residuals_log: List[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols, free_raw_data=False)
        dvalid = lgb.Dataset(Xva, label=yva, categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
        )
        models.append(model)
        pred_log = model.predict(Xva, num_iteration=model.best_iteration)
        oof_pred[va_idx] = pred_log
        residuals_log.append(yva.values - pred_log)

        # Back-transform to price for metrics
        if args.target == "price":
            pred_price = np.expm1(pred_log)
            true_price = np.expm1(yva.values)
        else:
            pred_price = pred_log
            true_price = yva.values
        fold_m = price_metrics(true_price, pred_price)
        fold_metrics.append(fold_m)
        print(f"Fold {fold}: {json.dumps(fold_m)}")

    # Overall OOF metrics
    if args.target == "price":
        oof_price = np.expm1(oof_pred)
        true_price_all = np.expm1(y.values)
    else:
        oof_price = oof_pred
        true_price_all = y.values
    overall = price_metrics(true_price_all, oof_price)
    print("Overall:", overall)

    # Train final model on full data
    dall = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
    final_model = lgb.train(
        params,
        dall,
        num_boost_round=int(np.mean([m.best_iteration for m in models])),
    )

    # Conformal intervals (simple symmetric absolute residual quantile)
    intervals = conformal_from_cv(residuals_log, lower_q=0.10, upper_q=0.90)

    # Save artifacts
    import pickle
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(out_dir / "columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "cat_cols.json", "w") as f:
        json.dump(cat_cols, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"folds": fold_metrics, "overall": overall}, f, indent=2)
    with open(out_dir / "intervals.json", "w") as f:
        json.dump(intervals, f, indent=2)

    print(f"Saved artifacts to: {out_dir.resolve()}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a stronger used-car price model.")
    p.add_argument("--input", type=str, required=True, help="Path to input CSV with raw used-car data")
    p.add_argument("--output_dir", type=str, default="artifacts", help="Directory to save model artifacts")
    p.add_argument("--target", type=str, default="price", help="Target column (default: price → uses log_price)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--remove_outliers", type=int, default=1, help="1/0: apply IQR outlier removal per family")
    p.add_argument("--n_estimators", type=int, default=800)
    p.add_argument("--early_stopping", type=int, default=50)
    p.add_argument("--optuna", type=int, default=0, help="1/0: run optuna hyper-param tuning")
    p.add_argument("--optuna_trials", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
