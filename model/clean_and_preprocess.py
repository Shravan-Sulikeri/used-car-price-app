# model/clean_and_preprocess.py
"""
Robust cleaner & preprocessor for used-car listings.

- Reads source from env DATA_CSV (fallback: data/used_car_sales.csv)
- Smart CSV/XLSX reader (encodings + separators)
- Flexible schema detection (price/year/mileage/make/model/…)
- Price parsing ($, commas, "12.3k"), km→miles conversion
- Make canonicalization (BMW, Mercedes-Benz, Volkswagen, …)
- Feature engineering: age, mileage_per_year, high_mileage
- Filters extremes; drops rare models (< MIN_MODEL_SUPPORT, default 15)
- Saves: data/clean_used_cars.csv, model/preprocessor.pkl, model/cleaning_report.json
"""

from __future__ import annotations
import os, re, json, unicodedata, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional: high-card hashing
try:
    from category_encoders.hashing import HashingEncoder
    HAVE_HASHING = True
except Exception:
    HAVE_HASHING = False

# Optional: your previous canonicalizer (if present in repo)
try:
    from standardize_makes import canonicalize_make as _canon_make
    CANON_MAKE_FUNC = _canon_make
except Exception:
    CANON_MAKE_FUNC = None

# -------------------- Config / Paths --------------------
RAW_PATH = os.getenv("DATA_CSV") or "data/used_car_sales.csv"
CLEAN_PATH = "data/clean_used_cars.csv"
PREPROC_PATH = "model/preprocessor.pkl"
REPORT_PATH = "model/cleaning_report.json"
MIN_MODEL_SUPPORT = int(os.getenv("MIN_MODEL_SUPPORT", "15"))  # drop models with fewer rows than this

# -------------------- Smart reader ----------------------
def read_table_smart(path: Path) -> pd.DataFrame:
    """
    Robust CSV/XLSX reader:
    - Detects BOM (utf-16-le/be, utf-8-sig)
    - Sniffs delimiter (comma/semicolon/tab/pipe)
    - Falls back to a grid of encodings × seps
    """
    import csv

    sfx = path.suffix.lower()
    if sfx in (".xlsx", ".xls"):
        return pd.read_excel(path)

    # --- Peek header bytes
    with open(path, "rb") as fb:
        head = fb.read(256 * 1024)

    # --- Encoding guess from BOM
    enc_guess = None
    if head.startswith(b"\xff\xfe"):
        enc_guess = "utf-16-le"
    elif head.startswith(b"\xfe\xff"):
        enc_guess = "utf-16-be"
    elif head.startswith(b"\xef\xbb\xbf"):
        enc_guess = "utf-8-sig"

    # --- Delimiter sniff
    try:
        sample_txt = head.decode(enc_guess or "utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(sample_txt, delimiters=[",", ";", "\t", "|"])
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = None

    # --- Try guessed combo first, then a grid
    tries = []
    if enc_guess or sep_guess:
        tries.append((enc_guess or "utf-8", sep_guess or ","))

    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin1"]
    seps = [",", ";", "\t", "|"]
    for e in encodings:
        for s in seps:
            if (e, s) not in tries:
                tries.append((e, s))

    last_err = None
    for e, s in tries:
        try:
            df = pd.read_csv(path, encoding=e, sep=s)  # no engine arg
            if df.shape[1] > 1:
                return df
        except Exception as err:
            last_err = err
            continue

    raise RuntimeError(f"Failed to parse {path}. Last error: {last_err}")

# -------------------- Utilities -------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def find_col(candidates, cols):
    norm_cols = {_norm(c): c for c in cols}
    for cand in candidates:
        n = _norm(cand)
        if n in norm_cols:
            return norm_cols[n]
        # loose contains (selling price inside 'final selling price (usd)')
        for nc, orig in norm_cols.items():
            if n and n in nc:
                return orig
    return None

def parse_price_series(s: pd.Series) -> pd.Series:
    """Turn strings like '$12,300', '12.3k', '€ 5.000' into floats (USD-ish)."""
    # normalize thousands & currency symbols
    x = (s.astype(str)
           .str.strip()
           .str.replace(r"[^\d\.\,\-kK]", "", regex=True)
           .str.replace(",", "", regex=False))
    # '12.3k' -> 12300
    k_mask = x.str.contains(r"[kK]$", na=False)
    out = pd.to_numeric(x.str.replace(r"[kK]$", "", regex=True), errors="coerce")
    out[k_mask] = out[k_mask] * 1000.0
    return out.replace([np.inf, -np.inf], np.nan)

# Canonical brand set + aliases
CANON_BRANDS = [
    "Acura","Alfa Romeo","Aston Martin","Audi","BMW","Bentley","Buick","Cadillac","Chevrolet","Chrysler",
    "Dodge","RAM","FIAT","Ford","Genesis","GMC","Honda","Hyundai","INFINITI","Jaguar","Jeep","Kia",
    "Land Rover","Lexus","Lincoln","Maserati","Mazda","Mercedes-Benz","MINI","Mitsubishi","Nissan",
    "Porsche","Saab","Saturn","Scion","smart","Subaru","Suzuki","Tesla","Toyota","Volkswagen","Volvo",
    "HUMMER","Pontiac","Oldsmobile","Rolls-Royce","Ferrari","McLaren","Lotus","Bugatti"
]
ALIASES = {
    "merc":"Mercedes-Benz","mercedes":"Mercedes-Benz","mercedesbenz":"Mercedes-Benz","benz":"Mercedes-Benz",
    "vw":"Volkswagen","wolkswagen":"Volkswagen","volkswagon":"Volkswagen",
    "chevy":"Chevrolet","chev":"Chevrolet","cheverolet":"Chevrolet","cheverlet":"Chevrolet",
    "landrover":"Land Rover","range rover":"Land Rover","rangerover":"Land Rover",
    "infinity":"INFINITI","porche":"Porsche","hyndai":"Hyundai","hyundia":"Hyundai",
    "lexsus":"Lexus","mitsubushi":"Mitsubishi","cadilac":"Cadillac","bwm":"BMW","b m w":"BMW",
    "ram":"RAM"
}
CANON_LOWER = [b.lower() for b in CANON_BRANDS]

def canonicalize_make_text(make: str, *extra_texts: str) -> str | float:
    """Best-effort brand canonicalization without external module."""
    if make is None or (isinstance(make, float) and math.isnan(make)):
        make = ""
    txt = " ".join([str(make), *map(str, extra_texts)]).lower()

    # direct alias hit
    for k, v in ALIASES.items():
        if k in txt:
            return v

    # brand token contained in text
    for brand in CANON_BRANDS:
        if brand.lower() in txt:
            return brand

    # dumb upper-case fallback for simple strings like "bmw.."
    m = re.sub(r"\.+", "", str(make)).strip().upper()
    if len(m) <= 12 and any(b in m for b in ["BMW","AUDI","FORD","JEEP","GMC","VW"]):
        # map some obvious cases
        if m == "VW": return "Volkswagen"
        if m == "BMW": return "BMW"
    return pd.NA

def maybe_canon_make(s_make: pd.Series, s_model: pd.Series | None = None) -> pd.Series:
    if CANON_MAKE_FUNC:
        return s_make.apply(lambda x: CANON_MAKE_FUNC(x) if pd.notna(x) else pd.NA)
    # internal detector
    extra = s_model if s_model is not None else pd.Series([""] * len(s_make))
    return [canonicalize_make_text(a, b) for a, b in zip(s_make, extra)]

# -------------------- Schema detection ------------------
def detect_and_rename(df: pd.DataFrame):
    """Return (df_renamed, mapping, mileage_unit, price_source)."""
    cols = df.columns.tolist()
    mapping = {}

    price = find_col(["price","list_price","selling_price","sellingprice","price_usd","current_price","final_price","pricesold"], cols)
    year = find_col(["year","model_year","registration_year","year_built","year_of_manufacture"], cols)
    miles = find_col(["mileage","odometer","odometer_value","miles","miles_driven","km","kilometers"], cols)
    make = find_col(["make","brand","manufacturer","make_name"], cols)
    model = find_col(["model","model_name","car_model","trim","variant"], cols)
    state = find_col(["state","state_code","region","province"], cols)
    fuel = find_col(["fuel","fuel_type","fueltype"], cols)
    trans = find_col(["transmission","gearbox","transmission_type"], cols)
    cond  = find_col(["condition","vehicle_condition","car_condition"], cols)
    body  = find_col(["body","body_type","bodystyle","body_style"], cols)
    title = find_col(["title_status","title","title_status_desc"], cols)
    seller= find_col(["seller_type","seller","seller_category","dealer_type"], cols)
    date  = find_col(["listed_at","posting_date","date","created_at","listing_date"], cols)

    mapping.update({
        price:"price", year:"year", miles:"mileage", make:"make", model:"model",
        state:"state", fuel:"fuel", trans:"transmission", cond:"condition",
        body:"body", title:"title_status", seller:"seller_type", date:"listing_date"
    })
    mapping = {k:v for k,v in mapping.items() if k is not None}

    df2 = df.rename(columns=mapping)

    # mileage unit
    unit = "miles"
    if miles and _norm(miles) in ("km","kilometers"):
        unit = "kilometers"

    return df2, mapping, unit, price

# -------------------- Preprocessor ----------------------
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in ["year","mileage","age","mileage_per_year"] if c in df.columns]
    lowcard = [c for c in ["state","fuel","transmission","condition","title_status","seller_type","body"] if c in df.columns]
    lowcard = [c for c in lowcard if df[c].nunique(dropna=True) <= 50]
    highcard = [c for c in ["make","model"] if c in df.columns]

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))

    # version-safe OHE
    ohe_kwargs = dict(handle_unknown="ignore")
    try:
        OneHotEncoder(sparse_output=False)  # sklearn >= 1.2
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False       # sklearn < 1.2

    if lowcard:
        transformers.append(("onehot", OneHotEncoder(**ohe_kwargs), lowcard))

    if highcard and HAVE_HASHING:
        transformers.append(("hash", Pipeline([
            ("fillna", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("hash", HashingEncoder(n_components=48, return_df=False)),
        ]), highcard))
    elif highcard:
        # Fallback if no category_encoders
        transformers.append(("onehot_hi", OneHotEncoder(**ohe_kwargs), highcard))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

# -------------------- Main ------------------------------
def main():
    path = Path(RAW_PATH)
    assert path.exists(), f"Input not found: {path}"

    df0 = read_table_smart(path)
    n0 = len(df0)

    # Detect schema & rename
    df, col_map, mileage_unit, price_source = detect_and_rename(df0)

    # ---- Ensure minimum fields exist
    needed = ["price","year","mileage","make","model"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Mapped columns: {col_map}. All columns: {list(df.columns)}")

    # ---- Coerce types
    df["price"] = parse_price_series(df["price"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

    # km → miles (by name) OR heuristic (very large medians)
    converted_km = False
    if mileage_unit == "kilometers":
        df["mileage"] = df["mileage"] * 0.621371
        converted_km = True
    else:
        med = df["mileage"].median(skipna=True)
        if pd.notna(med) and med > 350_000:   # likely kilometers
            df["mileage"] = df["mileage"] * 0.621371
            converted_km = True
            mileage_unit = "kilometers*heuristic"

    # ---- Canonicalize make
    df["make"] = maybe_canon_make(df["make"], df.get("model"))

    # ---- Normalize other categoricals
    for c in ["model","state","fuel","transmission","condition","body","title_status","seller_type"]:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().str.lower()
                        .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA}))

    # ---- Listing date → reference year
    ref_year = datetime.now().year
    if "listing_date" in df.columns:
        d = pd.to_datetime(df["listing_date"], errors="coerce")
        if d.notna().any():
            ref_year = int(d.dt.year.dropna().median())

    # ---- Clean filters
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["price","year","mileage","make","model"], how="any")

    df = df[df["price"].between(500, 250_000)]
    df = df[df["mileage"].between(0, 500_000)]
    df = df[df["year"].between(1980, ref_year + 1)]
    after_basic = len(df)

    # ---- Feature engineering
    df["age"] = (ref_year - df["year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage"] / np.where(df["age"] < 1, 1, df["age"])
    df["high_mileage"] = (df["mileage"] > 150_000).astype("int8")

    # ---- Drop rare models to stabilize training
    dropped_rare = 0
    if MIN_MODEL_SUPPORT > 1 and "model" in df.columns:
        vc = df["model"].value_counts(dropna=True)
        keep_models = vc[vc >= MIN_MODEL_SUPPORT].index
        dropped_rare = int((~df["model"].isin(keep_models)).sum())
        df = df[df["model"].isin(keep_models)]

    # ---- Column order / subset
    keep_cols = [c for c in [
        "price","year","mileage","make","model","state","fuel","transmission",
        "condition","body","title_status","seller_type","age","mileage_per_year","high_mileage"
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

    # ---- Report
    report = {
        "input_csv": str(path),
        "raw_rows": int(n0),
        "rows_after_basic_filters": int(after_basic),
        "rows_final": int(len(df)),
        "dropped_rows_percent": round(100 * (1 - len(df) / n0), 2) if n0 else None,
        "reference_year_used": int(ref_year),
        "mileage_unit_detected": mileage_unit,
        "converted_km_to_miles": bool(converted_km),
        "column_mapping": col_map,
        "price_source_column": price_source,
        "kept_columns": keep_cols,
        "unique_makes_after": int(df["make"].nunique()) if "make" in df.columns else None,
        "min_model_support": MIN_MODEL_SUPPORT,
        "dropped_rare_models_rows": dropped_rare,
        "numeric_features": [c for c in ["year","mileage","age","mileage_per_year"] if c in df.columns],
        "low_card_cats": [c for c in ["state","fuel","transmission","condition","title_status","seller_type","body"] if c in df.columns],
        "high_card_cats_hashing": [c for c in ["make","model"] if c in df.columns and HAVE_HASHING],
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