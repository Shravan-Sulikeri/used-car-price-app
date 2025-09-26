# -*- coding: utf-8 -*-
import os, re, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders.hashing import HashingEncoder
import joblib

RAW_PATH = os.getenv("DATA_CSV") or "data/used_car_sales.csv"
CLEAN_PATH = "data/clean_used_cars.csv"
PREPROC_PATH = "model/preprocessor.pkl"
METRICS_PATH = "model/cleaning_report.json"

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','', s.lower()) if isinstance(s, str) else ""

def find_col(cands, cols):
    norm_cols = {c: norm(c) for c in cols}
    for cand in cands:
        nc = norm(cand)
        for orig, ncol in norm_cols.items():
            if nc and nc in ncol:
                return orig
    return None

def detect_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    mapping = {}
    mapping[find_col(["price","list_price","listprice","selling_price","sellingprice","sale_price","saleprice","askingprice","pricesold"], cols)] = "price"
    mapping[find_col(["year","model_year","yearsold"], cols)] = "year"
    mapping[find_col(["mileage","odometer","miles"], cols)] = "mileage"
    mapping[find_col(["make","brand","manufacturer"], cols)] = "make"
    mapping[find_col(["model"], cols)] = "model"
    mapping[find_col(["state","region","location_state"], cols)] = "state"
    mapping[find_col(["fuel","fuel_type"], cols)] = "fuel"
    mapping[find_col(["transmission","gearbox"], cols)] = "transmission"
    mapping[find_col(["condition","vehicle_condition"], cols)] = "condition"
    mapping[find_col(["body","body_type","bodytype"], cols)] = "body"
    mapping[find_col(["title_status","title","title_status_desc"], cols)] = "title_status"
    mapping[find_col(["seller_type","seller","seller_category"], cols)] = "seller_type"
    mapping[find_col(["listed_at","posting_date","date","created_at","saledate","listingdate"], cols)] = "listing_date"
    mapping = {k:v for k,v in mapping.items() if k is not None and v is not None}
    print("Column mapping ->", mapping)
    return df.rename(columns=mapping)

def main():
    assert Path(RAW_PATH).exists(), f"Raw CSV not found at {RAW_PATH}"
    df0 = pd.read_csv(RAW_PATH, low_memory=False)
    n0 = len(df0)

    df = detect_and_rename(df0)

    if "price" not in df.columns:
        raise KeyError(f"No price-like column found. CSV columns: {list(df0.columns)}")

    # types
    for c in ["price","year","mileage"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["make","model","state","fuel","transmission","condition","body","title_status","seller_type"]:
        if c in df: df[c] = (df[c].astype(str).str.strip().str.lower()
                             .replace({"nan": np.nan,"none": np.nan,"": np.nan}))

    # ref year
    ref_year = datetime.now().year
    if "listing_date" in df:
        dty = pd.to_datetime(df["listing_date"], errors="coerce")
        if dty.notna().any():
            ref_year = int(dty.dt.year.dropna().median())

    # filters
    df = df.drop_duplicates()
    df = df.dropna(subset=["price","year","mileage","make","model"], how="any")
    df = df[df["price"].between(500, 200_000)]
    df = df[df["mileage"].between(0, 400_000)]
    df = df[df["year"].between(1985, ref_year + 1)]
    after = len(df)

    # features
    df["age"] = (ref_year - df["year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage"] / np.where(df["age"] < 1, 1, df["age"])
    df["high_mileage"] = (df["mileage"] > 150_000).astype(int)

    keep_cols = [c for c in [
        "price","year","mileage","make","model","state","fuel","transmission",
        "condition","body","title_status","seller_type","age","mileage_per_year","high_mileage"
    ] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # save
    Path(CLEAN_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    # preprocessor
    numeric_features = [c for c in ["year","mileage","age","mileage_per_year"] if c in df.columns]
    low_card_cats = [c for c in ["state","fuel","transmission","condition","title_status","seller_type","body"] if c in df.columns]
    low_card_cats = [c for c in low_card_cats if df[c].nunique(dropna=True) <= 50]
    high_card_cats = [c for c in ["make","model"] if c in df.columns]

    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if low_card_cats:
        # use 'sparse=False' for broad sklearn compatibility
        transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False), low_card_cats))
    if high_card_cats:
        # impute missing strings, then hash (no unsupported args)
        hash_pipe = Pipeline([
            ("fillna", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("hash", HashingEncoder(n_components=32, return_df=False)),
        ])
        transformers.append(("hash", hash_pipe, high_card_cats))

    preproc = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    X = df.drop(columns=["price"])
    preproc.fit(X)

    Path(PREPROC_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preproc, PREPROC_PATH)

    report = {
        "raw_rows": n0,
        "deduped_and_filtered_rows": after,
        "dropped_rows_percent": round(100 * (1 - after / n0), 2),
        "ref_year_used": ref_year,
        "kept_columns": keep_cols,
        "numeric_features": numeric_features,
        "low_card_cats": low_card_cats,
        "high_card_cats": high_card_cats
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Cleaned data ->", CLEAN_PATH)
    print("✅ Preprocessor ->", PREPROC_PATH)
    print("Summary:", report)

if __name__ == "__main__":
    main()
