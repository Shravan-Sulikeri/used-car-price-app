# -*- coding: utf-8 -*-
"""
Non-destructive merge preview:
- Reads base cleaned data: data/clean_used_cars.csv
- Reads the first CSV under: data/new/msnbehdani/
- Auto-maps columns to canonical schema: price, year, mileage, make, model, body
- Cleans, canonicalizes makes, adds features (age, mileage_per_year, high_mileage)
- Filters obvious outliers
- Writes preview: data/clean_merged_preview.csv (does NOT overwrite base)
"""

from __future__ import annotations
import re, unicodedata, json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

BASE_PATH = Path("data/clean_used_cars.csv")
NEW_DIR   = Path("data/new/msnbehdani")
OUT_PREVIEW = Path("data/clean_merged_preview.csv")
REPORT     = Path("model/merge_preview_report.json")

# ----------------------------
# Helpers: reading robustly
# ----------------------------
def _discover_first_csv(folder: Path) -> Path:
    cands = sorted(folder.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return cands[0]

def _sniff_table(path: Path) -> Tuple[str, str]:
    """
    Return (encoding, sep) that works. Avoids low_memory with python engine.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                pd.read_csv(path, encoding=enc, sep=sep, engine="python", nrows=200)
                return enc, sep
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not parse {path}. Last error: {last_err}")

def read_table_smart(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    enc, sep = _sniff_table(path)
    # use python engine to honor sep, avoid low_memory here
    return pd.read_csv(path, encoding=enc, sep=sep, engine="python")

# ----------------------------
# Canonicalize makes (inline fallback)
# ----------------------------
def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

CANON_SYNONYMS = {
    "benz": "mercedes-benz", "mercedes": "mercedes-benz", "mercedesbenz": "mercedes-benz", "merc": "mercedes-benz",
    "vw": "volkswagen", "v w": "volkswagen", "volkswagon": "volkswagen",
    "chevy": "chevrolet", "chev": "chevrolet", "cheverolet": "chevrolet",
    "landrover": "land rover", "range rover": "land rover", "rangerover": "land rover",
    "infinity": "infiniti", "porche": "porsche",
    "hyndai": "hyundai", "hyundia": "hyundai",
    "lexsus": "lexus",
    "mitsubushi": "mitsubishi",
    "cadilac": "cadillac",
    "dodge ram": "ram",
    "b m w": "bmw", "bwm": "bmw",
}
PROPER_CASE = {
    "acura":"Acura","alfa romeo":"Alfa Romeo","aston martin":"Aston Martin","audi":"Audi",
    "bmw":"BMW","bentley":"Bentley","buick":"Buick","cadillac":"Cadillac","chevrolet":"Chevrolet",
    "chrysler":"Chrysler","dodge":"Dodge","ram":"RAM","fiat":"FIAT","ford":"Ford","genesis":"Genesis",
    "gmc":"GMC","honda":"Honda","hyundai":"Hyundai","infiniti":"INFINITI","jaguar":"Jaguar","jeep":"Jeep",
    "kia":"Kia","land rover":"Land Rover","lexus":"Lexus","lincoln":"Lincoln","maserati":"Maserati",
    "mazda":"Mazda","mercedes-benz":"Mercedes-Benz","mini":"MINI","mitsubishi":"Mitsubishi","nissan":"Nissan",
    "porsche":"Porsche","saab":"Saab","saturn":"Saturn","scion":"Scion","smart":"smart","subaru":"Subaru",
    "suzuki":"Suzuki","tesla":"Tesla","toyota":"Toyota","volkswagen":"Volkswagen","volvo":"Volvo",
    "hummer":"HUMMER","pontiac":"Pontiac","oldsmobile":"Oldsmobile","rolls-royce":"Rolls-Royce",
    "ferrari":"Ferrari","mclaren":"McLaren","lotus":"Lotus","bugatti":"Bugatti",
}

def canonicalize_make(raw) -> Optional[str]:
    if pd.isna(raw): return np.nan
    n = _norm_text(raw)
    if not n: return np.nan
    n = CANON_SYNONYMS.get(n, n)
    if n.replace(" ", "") == "mercedesbenz":
        n = "mercedes-benz"
    return PROPER_CASE.get(n, n.title())

# ----------------------------
# Column auto-mapper
# ----------------------------
CANON_COLS = ["price", "year", "mileage", "make", "model", "body"]

def _lower_cols(df: pd.DataFrame) -> Dict[str, str]:
    return {c: c.lower().strip() for c in df.columns}

def _guess_mapping(df: pd.DataFrame) -> Dict[str, str]:
    lc = _lower_cols(df)
    mapping = {}
    for orig, col in lc.items():
        if col in {"price","price_usd","saleprice","sale_price","list_price","sellingprice","selling_price"}:
            mapping[orig] = "price"
        elif col in {"year","yr","modelyear","model_year"}:
            mapping[orig] = "year"
        elif any(k in col for k in ["mileage","odometer","miles","km","kilometer"]):
            mapping[orig] = "mileage"
        elif col in {"make","brand","manufacturer","marque"}:
            mapping[orig] = "make"
        elif col in {"model","variant","trim"}:
            mapping[orig] = "model"
        elif col in {"body","bodytype","body_type","type","segment"}:
            mapping[orig] = "body"
    # de-dup targets (keep first)
    inv: Dict[str, str] = {}
    for k,v in mapping.items():
        if v not in inv.values():
            inv[k]=v
    return inv

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# ----------------------------
# Cleaning & features
# ----------------------------
def _feature_eng(df: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    if "year" in df:
        df["age"] = ref_year - df["year"]
        df.loc[df["age"] < 0, "age"] = np.nan
    else:
        df["age"] = np.nan

    if {"mileage", "age"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["mileage_per_year"] = (df["mileage"] / df["age"]).replace([np.inf, -np.inf], np.nan)
    else:
        df["mileage_per_year"] = np.nan

    if "mileage" in df:
        df["high_mileage"] = (df["mileage"] > 150_000).astype("Int64")
    else:
        df["high_mileage"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df

def _filter_reasonable(df: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    if "price" in df:
        df = df[(df["price"] >= 500) & (df["price"] <= 250_000)]
    if "mileage" in df:
        df = df[(df["mileage"] >= 0) & (df["mileage"] <= 500_000)]
    if "year" in df:
        df = df[(df["year"] >= 1980) & (df["year"] <= ref_year + 1)]
    return df

# ----------------------------
# Main
# ----------------------------
def main():
    assert BASE_PATH.exists(), f"Missing base cleaned data: {BASE_PATH}"
    new_path = _discover_first_csv(NEW_DIR)

    base = pd.read_csv(BASE_PATH, low_memory=False)
    print(f"[base] rows={len(base):,} cols={list(base.columns)}")

    raw = read_table_smart(new_path)
    print(f"[new] file={new_path.name} rows={len(raw):,} cols={list(raw.columns)}")

    # Map columns
    mapping = _guess_mapping(raw)
    print("Auto mapping ->", mapping)

    new = raw.rename(columns=mapping)[list(set(mapping.values()))].copy()
    # Ensure canonical columns exist (create if missing)
    for c in CANON_COLS:
        if c not in new.columns:
            new[c] = pd.NA

    # Types
    for c in ("price","year","mileage"):
        if c in new.columns:
            new[c] = _coerce_numeric(new[c])

    # Canonicalize make
    if "make" in new.columns:
        new["make"] = new["make"].apply(canonicalize_make)

    # Title-case model softly (don’t break known all-caps like BMW)
    if "model" in new.columns:
        new["model"] = new["model"].astype(str).str.strip()
        new.loc[new["model"].str.len()==0, "model"] = pd.NA

    # Body normalize
    if "body" in new.columns:
        new["body"] = new["body"].astype(str).str.strip().replace({"": pd.NA})

    # Reference year: prefer max year observed vs. current calendar year
    ref_year = int(pd.Timestamp.today().year)
    if "year" in new and new["year"].notna().any():
        ref_year = int(pd.Series([ref_year, int(np.nanmax(new["year"]))]).max())

    # Features & filters
    new = _feature_eng(new, ref_year)
    new = _filter_reasonable(new, ref_year)

    # Keep the same columns as base where possible
    keep_cols = ["price","year","mileage","make","model","body","age","mileage_per_year","high_mileage"]
    new = new[keep_cols]

    # Concatenate
    base_cols_ok = [c for c in keep_cols if c in base.columns]
    base2 = base[base_cols_ok].copy()
    # reindex new to base columns order
    new = new.reindex(columns=base_cols_ok)
    merged = pd.concat([base2, new], axis=0, ignore_index=True)

    # Drop obvious duplicates on a light key if available
    dedup_keys = [c for c in ["make","model","year","mileage","price"] if c in merged.columns]
    if dedup_keys:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_keys, keep="first")
        print(f"Dedup by {dedup_keys}: {before:,} -> {len(merged):,}")

    # Save preview ONLY
    OUT_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PREVIEW, index=False)
    print(f"✅ Preview written -> {OUT_PREVIEW} (rows={len(merged):,})")

    # Report
    rep = {
        "base_rows": int(len(base)),
        "new_rows_raw": int(len(raw)),
        "new_rows_after_clean": int(len(new)),
        "merged_rows_preview": int(len(merged)),
        "auto_mapping": mapping,
        "ref_year_used": ref_year,
        "top_makes_new": new["make"].value_counts(dropna=True).head(10).to_dict() if "make" in new else {},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2))
    print("Summary:", json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
    