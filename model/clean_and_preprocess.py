# model/clean_and_preprocess.py
# End-to-end cleaner + preprocessor builder for used-car listings.
# - Reads CSV from env DATA_CSV or data/used_car_sales.csv
# - Auto-detects column names across common synonyms
# - Converts kilometers -> miles if needed
# - Canonicalizes 'make' if standardize_makes.canonicalize_make is present
# - Feature engineering: age, mileage_per_year, high_mileage
# - Filters extreme/out-of-range values
# - Fits a preprocessing ColumnTransformer and saves it
# - Writes a JSON report with useful stats

from __future__ import annotations
import os, re, json, unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# Optional: category_encoders for hashing high-cardinality cats
try:
    from category_encoders.hashing import HashingEncoder
    HAVE_HASHING = True
except Exception:
    HAVE_HASHING = False

# Optional: make canonicalizer (keeps repo reproducible even if absent)
CANON_MAKE = None
try:
    from standardize_makes import canonicalize_make as _canon_make
    CANON_MAKE = _canon_make
except Exception:
    CANON_MAKE = None

# ---- Paths / Config ----
RAW_PATH = os.getenv("DATA_CSV") or "data/used_car_sales.csv"
CLEAN_PATH = "data/clean_used_cars.csv"
PREPROC_PATH = "model/preprocessor.pkl"
REPORT_PATH = "model/cleaning_report.json"

# ---- Helpers ----
def _safe_to_num(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)

def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_col(candidates, cols):
    """Return the *first* column in `cols` that matches any normalized candidate token."""
    norm_cols = {_norm_name(c): c for c in cols}
    for cand in candidates:
        n = _norm_name(cand)
        # exact normalized match
        if n in norm_cols:
            return norm_cols[n]
        # loose contains (e.g., 'selling price' inside 'final selling price (usd)')
        for nc, orig in norm_cols.items():
            if n and n in nc:
                return orig
    return None

def detect_and_rename(df: pd.DataFrame):
    """Auto-map dataset columns to our canonical names. Returns (renamed_df, mapping, mileage_unit)."""
    cols = df.columns.tolist()
    mapping = {}

    # Synonym lists (expanded for 2025 dataset variants)
    price_col   = find_col(["price","list_price","selling_price","sellingprice","current_price","final_price"], cols)
    year_col    = find_col(["year","model_year","year_built","year_of_manufacture","registration_year"], cols)
    miles_col   = find_col(["mileage","odometer","miles","miles_driven","km","kilometers"], cols)
    make_col    = find_col(["make","brand","manufacturer","make_name"], cols)
    model_col   = find_col(["model","model_name","car_model","trim"], cols)
    state_col   = find_col(["state","region","location_state","province"], cols)
    fuel_col    = find_col(["fuel","fuel_type","fueltype"], cols)
    trans_col   = find_col(["transmission","gearbox","transmission_type"], cols)
    cond_col    = find_col(["condition","vehicle_condition","car_condition"], cols)
    body_col    = find_col(["body","body_type","bodystyle","body_style"], cols)
    title_col   = find_col(["title_status","title","title_status_desc"], cols)
    seller_col  = find_col(["seller_type","seller","seller_category","dealer_type"], cols)
    date_col    = find_col(["listed_at","posting_date","date","created_at","listing_date"], cols)

    mapping.update({
        price_col: "price",
        year_col: "year",
        miles_col: "mileage",
        make_col: "make",
        model_col: "model",
        state_col: "state",
        fuel_col: "fuel",
        trans_col: "transmission",
        cond_col: "condition",
        body_col: "body",
        title_col: "title_status",
        seller_col: "seller_type",
        date_col: "listing_date",
    })
    # drop None keys
    mapping = {k: v for k, v in mapping.items() if k is not None and v is not None}

    df2 = df.rename(columns=mapping)

    # Units: detect if the *source* mileage came from 'km'/'kilometers'
    mileage_unit = "miles"
    if miles_col and _norm_name(miles_col) in ("km", "kilometers"):
        mileage_unit = "kilometers"

    return df2, mapping, mileage_unit

def maybe_canonicalize_make(s: pd.Series) -> pd.Series:
    if CANON_MAKE is None:
        # lightweight normalizer if canonicalizer not available
        return (s.astype(str)
                 .str.strip()
                 .str.lower()
                 .replace({"nan": np.nan, "none": np.nan, "": np.nan})
                 .str.replace(r"\.+", "", regex=True)
                 .str.upper())
    else:
        return s.apply(lambda x: CANON_MAKE(x) if pd.notna(x) else np.nan)

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Feature sets
    numeric_features = [c for c in ["year", "mileage", "age", "mileage_per_year"] if c in df.columns]
    low_card_cats = [c for c in ["state", "fuel", "transmission", "condition", "title_status", "seller_type", "body"]
                     if c in df.columns]
    low_card_cats = [c for c in low_card_cats if df[c].nunique(dropna=True) <= 50]
    high_card_cats = [c for c in ["make", "model"] if c in df.columns]  # hashed

    transformers = []

    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))

    # version-safe OneHotEncoder kwargs
    ohe_kwargs = dict(handle_unknown="ignore")
    try:
        # sklearn >= 1.2
        OneHotEncoder(sparse_output=False)
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        # sklearn < 1.2
        ohe_kwargs["sparse"] = False

    if low_card_cats:
        transformers.append(("onehot", OneHotEncoder(**ohe_kwargs), low_card_cats))

    if high_card_cats and HAVE_HASHING:
        hash_pipe = Pipeline([
            ("fillna", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("hash", HashingEncoder(n_components=32, return_df=False)),
        ])
        transformers.append(("hash", hash_pipe, high_card_cats))
    elif high_card_cats:
        # Fallback: if category_encoders missing, at least one-hot them (might be large)
        transformers.append(("onehot_hicard", OneHotEncoder(**ohe_kwargs), high_card_cats))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3
    )
    return preproc

def main():
    path = Path(RAW_PATH)
    assert path.exists(), f"Raw CSV not found at {path}"

    df0 = pd.read_csv(path, low_memory=False)
    n0 = len(df0)

    # ---- Detect/rename columns
    df, col_map, mileage_unit = detect_and_rename(df0)

    # Ensure required columns exist
    required = ["price", "year", "mileage", "make", "model"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after mapping: {missing}. "
                       f"Mapped columns were: {col_map}. Present columns: {list(df.columns)}")

    # ---- Types
    df["price"] = _safe_to_num(df["price"])
    df["year"] = _safe_to_num(df["year"])
    df["mileage"] = _safe_to_num(df["mileage"])

    # Convert km -> miles if detected
    converted_km = False
    if mileage_unit == "kilometers":
        df["mileage"] = df["mileage"] * 0.621371
        converted_km = True

    # Canonicalize make (if available) or lightweight normalize
    df["make"] = maybe_canonicalize_make(df["make"])

    # Standardize string categoricals
    for c in ["model", "state", "fuel", "transmission", "condition", "body", "title_status", "seller_type"]:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().str.lower()
                        .replace({"nan": np.nan, "none": np.nan, "": np.nan}))

    # ---- Parse listing date and compute reference year
    ref_year = datetime.now().year
    if "listing_date" in df.columns:
        dty = pd.to_datetime(df["listing_date"], errors="coerce")
        if dty.notna().any():
            # use median listing year as reference when present
            ref_year = int(dty.dt.year.dropna().median())

    # ---- Basic filtering and drops
    before = len(df)
    df = df.drop_duplicates()

    # Drop rows with essential nulls
    df = df.dropna(subset=["price", "year", "mileage", "make", "model"], how="any")

    # Reasonable ranges
    df = df[df["price"].between(500, 200_000)]
    df = df[df["mileage"].between(0, 400_000)]
    df = df[df["year"].between(1985, ref_year + 1)]
    after = len(df)

    # ---- Feature engineering
    df["age"] = (ref_year - df["year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage"] / np.where(df["age"] < 1, 1, df["age"])
    df["high_mileage"] = (df["mileage"] > 150_000).astype(int)

    # Keep a compact set of columns (keep only existing)
    keep_cols = [c for c in [
        "price","year","mileage","make","model","state","fuel","transmission","condition",
        "body","title_status","seller_type","age","mileage_per_year","high_mileage"
    ] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # ---- Save cleaned CSV
    Path(CLEAN_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    # ---- Fit preprocessor on features only
    X = df.drop(columns=["price"])
    preproc = build_preprocessor(df)
    preproc.fit(X)

    Path(PREPROC_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preproc, PREPROC_PATH)

    # ---- Build report
    report = {
        "raw_rows": int(n0),
        "rows_after_filters": int(after),
        "dropped_rows_percent": round(100 * (1 - after / n0), 2) if n0 else None,
        "reference_year_used": int(ref_year),
        "input_csv": str(path),
        "mileage_unit_detected": mileage_unit,
        "converted_km_to_miles": bool(converted_km),
        "column_mapping": col_map,
        "kept_columns": keep_cols,
        "unique_makes_after": int(df["make"].nunique()) if "make" in df.columns else None,
        "numeric_features": [c for c in ["year","mileage","age","mileage_per_year"] if c in df.columns],
        "low_card_cats": [c for c in ["state","fuel","transmission","condition","title_status","seller_type","body"] if c in df.columns],
        "high_card_cats": [c for c in ["make","model"] if c in df.columns and HAVE_HASHING],
        "hashing_used": bool(HAVE_HASHING),
    }
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Column mapping ->", col_map)
    print(f"✅ Cleaned data -> {CLEAN_PATH}")
    print(f"✅ Preprocessor -> {PREPROC_PATH}")
    print("Summary:", report)

if __name__ == "__main__":
    main()