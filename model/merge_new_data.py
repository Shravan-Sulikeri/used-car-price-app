# -*- coding: utf-8 -*-
"""
Merge a new, normalized car CSV into data/clean_used_cars.csv

Usage:
  # Preview only (writes data/clean_merged_preview.csv + model/merge_preview_report.json)
  python model/merge_new_data.py --new data/new/msnbehdani_norm.csv --preview

  # Finalize (backs up current clean_used_cars.csv to .bak and writes merged file)
  python model/merge_new_data.py --new data/new/msnbehdani_norm.csv --finalize
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

BASE_PATH = Path("data/clean_used_cars.csv")
PREVIEW_OUT = Path("data/clean_merged_preview.csv")
REPORT_OUT = Path("model/merge_preview_report.json")

# Try to reuse your existing make canonicalizer if present
def _fallback_canon_make(s: str) -> str:
    if pd.isna(s): return np.nan
    t = re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    syn = {
        "vw": "Volkswagen", "volkswagon": "Volkswagen",
        "chevy": "Chevrolet", "chev": "Chevrolet",
        "mercedes": "Mercedes-Benz", "mercedes benz": "Mercedes-Benz",
        "land rover": "Land Rover", "rangerover": "Land Rover", "range rover": "Land Rover",
        "infinity": "INFINITI",
    }
    if t in syn: return syn[t]
    return t.title()

try:
    from standardize_makes import canonicalize_make as CANON_MAKE  # if your earlier helper exists
except Exception:
    CANON_MAKE = _fallback_canon_make

def coerce_numeric(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace(",", "")
    s = s.replace("$", "").replace(" ", "")
    if s.endswith("k"):
        try: return float(s[:-1]) * 1000
        except: return np.nan
    try: return float(s)
    except: return np.nan

def coerce_year(x):
    try:
        y = int(float(str(x).strip()))
        this_year = pd.Timestamp.today().year
        if 1980 <= y <= this_year + 1:
            return y
        return np.nan
    except:
        return np.nan

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    # If the new file is already normalized, this is a no-op; else it’s tolerant.
    rename = {}
    for col in df.columns:
        k = re.sub(r"[^a-z0-9]", "", col.lower())
        if k in {"manufacturer","brand","make","carbrand"}:         rename[col] = "make"
        elif k in {"model","carmodel","cartype"}:                   rename[col] = "model"
        elif k in {"year","yearofmanufacture","manufactureyear"}:   rename[col] = "year"
        elif k in {"mileage","miles","odometer","odometerreading"}: rename[col] = "mileage"
        elif k in {"price","listingprice","saleprice","amount","usd"}: rename[col] = "price"
        elif k in {"body","bodytype","vehicletype"}:                rename[col] = "body"
    df = df.rename(columns=rename)
    for need in ["price","year","mileage","make","model","body"]:
        if need not in df.columns:
            df[need] = np.nan
    return df[["price","year","mileage","make","model","body"]].copy()

def clean_new(df_new: pd.DataFrame) -> pd.DataFrame:
    df = ensure_schema(df_new)
    df["price"] = df["price"].map(coerce_numeric)
    df["mileage"] = df["mileage"].map(coerce_numeric)
    df["year"] = df["year"].map(coerce_year)
    # Basic filters to avoid junk
    this_year = pd.Timestamp.today().year
    ok = (
        df["price"].between(500, 250_000, inclusive="both") &
        df["mileage"].between(0, 500_000, inclusive="both") &
        df["year"].between(1980, this_year+1, inclusive="both") &
        df["make"].notna() & df["model"].notna()
    )
    df = df.loc[ok].copy()
    # Canonicalize make
    df["make"] = df["make"].map(CANON_MAKE)
    # Features
    ref_year = this_year
    df["age"] = (ref_year - df["year"]).clip(lower=0)
    df["mileage_per_year"] = (df["mileage"] / df["age"].replace(0, 1)).round(2)
    df["high_mileage"] = (df["mileage_per_year"] > 15000).astype(int)
    return df

def align_to_base(df_new: pd.DataFrame, base_cols) -> pd.DataFrame:
    out = df_new.copy()
    for c in base_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[base_cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="Path to normalized CSV (e.g., data/new/msnbehdani_norm.csv)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true", help="Create preview CSV and report (no overwrite)")
    g.add_argument("--finalize", action="store_true", help="Overwrite data/clean_used_cars.csv (backs up .bak)")
    args = ap.parse_args()

    new_path = Path(args.new)
    assert new_path.exists(), f"New CSV not found: {new_path}"
    assert BASE_PATH.exists(), f"Base CSV not found: {BASE_PATH}"

    base = pd.read_csv(BASE_PATH, low_memory=False)
    new_raw = pd.read_csv(new_path, low_memory=False)

    new_clean = clean_new(new_raw)

    # Align to base columns (your base has engineered features already)
    base_cols = list(base.columns)
    new_aligned = align_to_base(new_clean, base_cols)

    # Merge and drop exact duplicates (all columns)
    merged = pd.concat([base, new_aligned], ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates()
    after_dedup = len(merged)

    # Report
    report = {
        "base_rows": int(len(base)),
        "new_rows_raw": int(len(new_raw)),
        "new_rows_after_clean": int(len(new_aligned)),
        "merged_rows": int(len(merged)),
        "dedup_removed": int(before_dedup - after_dedup),
        "ref_year_used": int(pd.Timestamp.today().year),
        "top_makes_new": new_aligned["make"].value_counts().head(15).to_dict() if "make" in new_aligned else {},
    }

    if args.preview:
        PREVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
        REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(PREVIEW_OUT, index=False)
        REPORT_OUT.write_text(json.dumps(report, indent=2))
        print(f"✅ Preview written -> {PREVIEW_OUT} (rows={len(merged):,})")
        print("Summary:", json.dumps(report, indent=2))
        return

    # finalize
    backup = BASE_PATH.with_suffix(".bak")
    if BASE_PATH.exists():
        BASE_PATH.replace(backup)
        print(f"💾 Backed up base -> {backup}")
    merged.to_csv(BASE_PATH, index=False)
    print(f"✅ Finalized merge -> {BASE_PATH} (rows={len(merged):,})")
    print("Summary:", json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
