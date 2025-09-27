# -*- coding: utf-8 -*-
"""
Clean used-car data + fit preprocessing transformer.

- Robust CSV reader (UTF-8/UTF-16/CP1252; comma/semicolon/tab/pipe)
- Column normalization: price/year/mileage/make/model/body
- Make canonicalization (BMW, Mercedes-Benz, Volkswagen, Chevrolet, ...)
- Feature engineering: age, mileage_per_year, high_mileage
- Preprocessor:
    * Numeric -> median imputer
    * Low-card categorical -> Constant impute + OneHotEncoder
    * High-card categorical (make/model) -> Constant impute + HashingEncoder
      (no pd.NA passed into encoders; dtype kept numeric)

Artifacts:
- data/clean_used_cars.csv (override via OUT_CSV)
- model/preprocessor.pkl
- model/cleaning_report.json
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# category_encoders
try:
    from category_encoders.hashing import HashingEncoder
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'category-encoders'. Install with:\n"
        "  pip install category-encoders"
    ) from e


# ------------------------- paths & runtime options -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Choose input file in this priority if DATA_CSV env not set
POSSIBLE_INPUTS = [
    "data/new/used_car_listings.csv",
    "data/used_car_listings.csv",
    "data/used_car_sales.csv",
    "data/new/cars.csv",
    "data/cars.csv",
]

DATA_CSV = os.getenv("DATA_CSV")
OUT_CSV = os.getenv("OUT_CSV", "data/clean_used_cars.csv")
PREPROC_PATH = os.getenv("PREPROC_PATH", "model/preprocessor.pkl")
REPORT_PATH = os.getenv("REPORT_PATH", "model/cleaning_report.json")


# --------------------------- helpers: file reading --------------------------

def _detect_encoding_and_sep(p: Path) -> Tuple[str, str]:
    """Guess text encoding and field separator from first bytes/line."""
    with open(p, "rb") as f:
        head = f.read(4)

    # BOM checks for UTF-16/UTF-8-SIG
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        enc = "utf-16"
    elif head.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    else:
        enc = "utf-8"

    # crude sep guess from first non-empty text line
    # fallback counts on common separators
    try:
        with open(p, "r", encoding=enc, errors="replace") as f:
            for line in f:
                if line.strip():
                    sample = line
                    break
            else:
                sample = ""
    except Exception:
        sample = ""

    candidates = [",", ";", "\t", "|"]
    counts = {c: sample.count(c) for c in candidates}
    sep = max(counts, key=counts.get) if sample else ","
    return enc, sep


def read_table_smart(path: Path) -> pd.DataFrame:
    """Read CSV/TSV robustly across encodings and separators."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    enc, sep = _detect_encoding_and_sep(path)

    # Try fast engine first
    try:
        return pd.read_csv(path, encoding=enc, sep=sep, engine="c", low_memory=False)
    except Exception as err_c:
        # Fallback to python engine (no low_memory flag here)
        try:
            return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
        except Exception as err_py:
            # Last resort: different encodings
            for e in ("cp1252", "latin1", "utf-8-sig"):
                try:
                    return pd.read_csv(path, encoding=e, sep=sep, engine="python")
                except Exception:
                    continue
            raise RuntimeError(
                f"Could not parse {path}. Last errors:\nC-engine: {err_c}\nPython-engine: {err_py}"
            )


# --------------------------- helpers: columns -------------------------------

# flexible mapping from many raw names -> canonical names
CANON_MAP: Dict[str, str] = {
    # price
    "price": "price", "pricesold": "price", "askingprice": "price", "sale_price": "price",
    # year
    "year": "year", "yearsold": "year", "modelyear": "year",
    # mileage / odometer
    "mileage": "mileage", "odometer": "mileage", "miles": "mileage",
    # make / brand
    "make": "make", "brand": "make", "manufacturer": "make",
    # model
    "model": "model", "car_model": "model",
    # body / body type
    "body": "body", "bodytype": "body", "body_type": "body", "bodystyle": "body",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    c2 = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        # also retain version with underscore for safety
        key2 = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        canon = (
            CANON_MAP.get(key) or
            CANON_MAP.get(key2) or
            CANON_MAP.get(key2.replace(" ", ""))
        )
        c2[c] = canon if canon else c
    df = df.rename(columns=c2)
    return df


# --------------------------- helpers: make cleanup --------------------------

import re
import unicodedata

# synonym/misspelling normalization -> canonical (lowercase key)
MAKE_SYNONYMS: Dict[str, str] = {
    "merc": "mercedes-benz", "mercedes": "mercedes-benz", "mercedesbenz": "mercedes-benz", "mercedes-b": "mercedes-benz",
    "vw": "volkswagen", "volkswagon": "volkswagen", "wolkswagen": "volkswagen",
    "chevy": "chevrolet", "chev": "chevrolet", "cheverolet": "chevrolet",
    "landrover": "land rover", "range rover": "land rover", "rangerover": "land rover",
    "infinity": "infiniti",
    "hyndai": "hyundai", "hyundia": "hyundai",
    "lexsus": "lexus",
    "mitsubushi": "mitsubishi",
    "cadilac": "cadillac",
    "dodge ram": "ram",
    "bwm": "bmw", "b m w": "bmw",
}

# pretty casing
MAKE_CASE: Dict[str, str] = {
    "acura":"Acura","alfa romeo":"Alfa Romeo","aston martin":"Aston Martin","audi":"Audi","bentley":"Bentley",
    "bmw":"BMW","buick":"Buick","cadillac":"Cadillac","chevrolet":"Chevrolet","chrysler":"Chrysler","dodge":"Dodge",
    "ram":"RAM","fiat":"FIAT","ford":"Ford","genesis":"Genesis","gmc":"GMC","honda":"Honda","hyundai":"Hyundai",
    "infiniti":"INFINITI","jaguar":"Jaguar","jeep":"Jeep","kia":"Kia","land rover":"Land Rover","lexus":"Lexus",
    "lincoln":"Lincoln","maserati":"Maserati","mazda":"Mazda","mercedes-benz":"Mercedes-Benz","mini":"MINI",
    "mitsubishi":"Mitsubishi","nissan":"Nissan","porsche":"Porsche","saab":"Saab","saturn":"Saturn","scion":"Scion",
    "smart":"smart","subaru":"Subaru","tesla":"Tesla","toyota":"Toyota","volkswagen":"Volkswagen","volvo":"Volvo",
    "hummer":"HUMMER","pontiac":"Pontiac","oldsmobile":"Oldsmobile","rolls-royce":"Rolls-Royce","ferrari":"Ferrari",
    "mclaren":"McLaren","lotus":"Lotus","bugatti":"Bugatti","polestar":"Polestar","rivian":"Rivian","karma":"Karma",
}

def _norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_make(raw: Optional[str]) -> Optional[str]:
    """Return a brand-like value (title-cased) or None."""
    if pd.isna(raw):
        return None
    n = _norm_text(raw)
    if not n:
        return None

    n = MAKE_SYNONYMS.get(n, n)
    # special join for mercedes benz -> mercedes-benz
    if n.replace(" ", "") == "mercedesbenz":
        n = "mercedes-benz"

    # pretty case if known
    if n in MAKE_CASE:
        return MAKE_CASE[n]

    # If the string looks like a whole brand inside a longer clause
    for k, pretty in MAKE_CASE.items():
        if re.search(rf"\b{k}\b", n):
            return pretty

    # fallback: title-case of first token
    return n.title() if n else None


# --------------------------- main cleaning routine --------------------------

def main():
    # Resolve input path
    if DATA_CSV:
        in_path = Path(DATA_CSV)
    else:
        for cand in POSSIBLE_INPUTS:
            p = PROJECT_ROOT / cand
            if p.exists():
                in_path = p
                break
        else:
            raise FileNotFoundError(
                "No input data found. Set DATA_CSV env or place a CSV at one of:\n"
                + "\n".join(POSSIBLE_INPUTS)
            )

    out_csv = PROJECT_ROOT / OUT_CSV
    preproc_path = PROJECT_ROOT / PREPROC_PATH
    report_path = PROJECT_ROOT / REPORT_PATH

    # Read
    df0 = read_table_smart(in_path)
    n_raw = len(df0)

    # Normalize column names
    df = normalize_columns(df0)

    # Keep only relevant columns if present
    keep = [c for c in ["price", "year", "mileage", "make", "model", "body"] if c in df.columns]
    if not keep:
        raise ValueError(
            f"No expected columns found after normalization in {in_path}.\n"
            f"Columns present: {list(df.columns)[:25]}"
        )
    df = df[keep].copy()

    # Types
    for c in ("price", "year", "mileage"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Canonicalize make
    if "make" in df.columns:
        df["make"] = df["make"].apply(canonicalize_make)

    # Basic filters (robust ranges)
    ref_year = pd.Timestamp.today().year
    if "price" in df.columns:
        df = df[(df["price"] >= 500) & (df["price"] <= 250_000) | df["price"].isna()]
    if "mileage" in df.columns:
        df = df[(df["mileage"] >= 0) & (df["mileage"] <= 500_000) | df["mileage"].isna()]
    if "year" in df.columns:
        df = df[(df["year"] >= 1982) & (df["year"] <= ref_year + 1) | df["year"].isna()]

    # Drop rows with missing target
    df = df.dropna(subset=["price"])

    # Deduplicate approximate rows
    subset_cols = [c for c in ["price", "year", "mileage", "make", "model"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    # Feature engineering
    if "year" in df.columns:
        df["age"] = ref_year - df["year"]
    else:
        df["age"] = np.nan

    if "mileage" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["mileage_per_year"] = df["mileage"] / df["age"].replace(0, np.nan)
    else:
        df["mileage_per_year"] = np.nan

    df["high_mileage"] = (df["mileage_per_year"] > 20_000).astype("int8").fillna(0)

    # Final column order
    cols = ["price", "year", "mileage", "make", "model", "body", "age", "mileage_per_year", "high_mileage"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # ----- Build preprocessor (safe with pd.NA) -----
    # Identify feature groups
    numeric_features: List[str] = [c for c in ["year", "mileage", "age", "mileage_per_year"] if c in df.columns]
    low_card_cats: List[str] = []
    if "body" in df.columns:
        # treat body as low-card if present
        low_card_cats.append("body")

    high_card_cats: List[str] = [c for c in ["make", "model"] if c in df.columns]

    # Ensure categorical columns are plain 'object' (avoid pandas NAType in encoders)
    for group in (low_card_cats, high_card_cats):
        if group:
            df[group] = df[group].astype("object")

    # Pipelines
    num_pipe = SimpleImputer(strategy="median")

    low_cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])

    hash_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("hash", HashingEncoder(n_components=32, return_df=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", num_pipe, numeric_features))
    if low_card_cats:
        transformers.append(("low", low_cat_pipe, low_card_cats))
    if high_card_cats:
        transformers.append(("hash", hash_pipe, high_card_cats))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
    )

    # Fit preprocessor (X only)
    X = df.drop(columns=["price"])
    preproc.fit(X)

    # Save outputs
    out_csv_path = out_csv
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    preproc_file = preproc_path
    preproc_file.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(preproc, preproc_file)

    report = {
        "input_file": str(in_path),
        "output_file": str(out_csv_path),
        "preprocessor_file": str(preproc_file),
        "raw_rows": int(n_raw),
        "final_rows": int(len(df)),
        "dropped_rows_percent": round(100.0 * (1 - len(df) / max(n_raw, 1)), 2),
        "ref_year_used": int(ref_year),
        "kept_columns": cols,
        "numeric_features": numeric_features,
        "low_card_cats": low_card_cats,
        "high_card_cats": high_card_cats,
        "unique_makes": int(df["make"].nunique()) if "make" in df.columns else None,
        "unique_models": int(df["model"].nunique()) if "model" in df.columns else None,
        "median_price": int(np.nanmedian(df["price"])) if "price" in df.columns else None,
        "median_mileage": int(np.nanmedian(df["mileage"])) if "mileage" in df.columns else None,
    }
    report_path = report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Cleaned data  -> {out_csv_path}")
    print(f"✅ Preprocessor  -> {preproc_file}")
    print("Summary:", report)


if __name__ == "__main__":
    main()