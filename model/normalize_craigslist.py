# model/normalize_craigslist.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

CANON = ["price", "year", "mileage", "make", "model", "body"]

def num_from_str(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"[,$ ]", "", s)
    if s.endswith("k"):
        try: return float(s[:-1]) * 1000
        except: return np.nan
    try: return float(s)
    except: return np.nan

def to_year(x):
    try:
        y = int(float(str(x).strip()))
        yr = pd.Timestamp.today().year
        return y if 1980 <= y <= yr + 1 else np.nan
    except:
        return np.nan

def clean_chunk(df, min_price, max_price, max_mileage) -> pd.DataFrame:
    # Rename Craigslist fields to our canonical schema
    # vehicles.csv has: price, year, odometer, manufacturer, model, type (body)
    df = df.rename(columns={
        "odometer": "mileage",
        "manufacturer": "make",
        "type": "body",
    })

    # Ensure the required columns exist
    for c in CANON:
        if c not in df: df[c] = np.nan
    df = df[CANON].copy()

    # Coerce types
    df["price"]   = df["price"].map(num_from_str)
    df["mileage"] = df["mileage"].map(num_from_str)
    df["year"]    = df["year"].map(to_year)

    # Strings
    for c in ["make","model","body"]:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # Filters
    keep = (
        df["price"].between(min_price, max_price, inclusive="both") &
        df["mileage"].between(0, max_mileage, inclusive="both") &
        df["year"].notna() &
        df["make"].notna() & df["make"].ne("") &
        df["model"].notna() & df["model"].ne("")
    )
    df = df.loc[keep]

    # Light de-dupe within chunk
    df = df.drop_duplicates(subset=CANON)

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/new/vehicles.csv")
    ap.add_argument("--out", default="data/new/craigslist_norm.csv")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--min-price", type=float, default=500)
    ap.add_argument("--max-price", type=float, default=250_000)
    ap.add_argument("--max-mileage", type=float, default=500_000)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    usecols = ["price","year","odometer","manufacturer","model","type"]
    total_in = total_kept = 0
    wrote_header = False
    chunks = pd.read_csv(src, usecols=usecols, chunksize=args.chunksize, low_memory=False)

    for i, chunk in enumerate(chunks, 1):
        total_in += len(chunk)
        cleaned = clean_chunk(chunk, args.min_price, args.max_price, args.max_mileage)
        total_kept += len(cleaned)

        # Append to output incrementally
        cleaned.to_csv(out, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        print(f"[{i}] kept {len(cleaned):,} / {len(chunk):,} (cum {total_kept:,} / {total_in:,})")

    # Final de-dupe across all appended rows (optional; fast on disk read)
    df_all = pd.read_csv(out, low_memory=False)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=CANON).reset_index(drop=True)
    if len(df_all) != before:
        df_all.to_csv(out, index=False)

    summary = {
        "src": str(src),
        "out": str(out),
        "rows_in": int(total_in),
        "rows_kept_after_filters": int(total_kept),
        "rows_after_global_dedupe": int(len(df_all)),
        "columns": CANON
    }
    print("✅ Craigslist normalized:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()