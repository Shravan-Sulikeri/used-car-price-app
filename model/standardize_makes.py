# -*- coding: utf-8 -*-
import json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/clean_used_cars.csv")
REPORT_PATH = Path("model/make_cleaning_report.json")

# --- Synonyms & misspellings mapped to canonical (lowercase) ---
CANON_SYNONYMS = {
    # Mercedes / VW / Chevy
    "benz": "mercedes-benz", "mercedes": "mercedes-benz", "mercedesbenz": "mercedes-benz", "merc": "mercedes-benz",
    "vw": "volkswagen", "v w": "volkswagen", "volkswagon": "volkswagen", "wolkswagen": "volkswagen",
    "chevy": "chevrolet", "chev": "chevrolet", "cheverolet": "chevrolet", "cheverlet": "chevrolet",
    # Land Rover / Range Rover
    "landrover": "land rover", "range rover": "land rover", "rangerover": "land rover",
    # Common misspellings
    "infinity": "infiniti", "porche": "porsche",
    "hyndai": "hyundai", "hyundia": "hyundai", "hyuandai": "hyundai",
    "lexsus": "lexus",
    "mitsubushi": "mitsubishi", "mitsubish": "mitsubishi",
    "cadilac": "cadillac",
    # RAM / Dodge RAM
    "dodge ram": "ram",
    # BMW variants
    "b m w": "bmw", "bwm": "bmw",
}

# --- Pretty casing for final display ---
PROPER_CASE = {
    "acura":"Acura", "alfa romeo":"Alfa Romeo", "aston martin":"Aston Martin",
    "audi":"Audi", "bmw":"BMW", "bentley":"Bentley", "buick":"Buick",
    "cadillac":"Cadillac", "chevrolet":"Chevrolet", "chrysler":"Chrysler",
    "dodge":"Dodge", "ram":"RAM", "fiat":"FIAT", "ford":"Ford", "genesis":"Genesis",
    "gmc":"GMC", "honda":"Honda", "hyundai":"Hyundai", "infiniti":"INFINITI",
    "jaguar":"Jaguar", "jeep":"Jeep", "kia":"Kia", "land rover":"Land Rover",
    "lexus":"Lexus", "lincoln":"Lincoln", "maserati":"Maserati", "mazda":"Mazda",
    "mercedes-benz":"Mercedes-Benz", "mini":"MINI", "mitsubishi":"Mitsubishi",
    "nissan":"Nissan", "porsche":"Porsche", "saab":"Saab", "saturn":"Saturn",
    "scion":"Scion", "smart":"smart", "subaru":"Subaru", "suzuki":"Suzuki",
    "tesla":"Tesla", "toyota":"Toyota", "volkswagen":"Volkswagen", "volvo":"Volvo",
    "hummer":"HUMMER", "pontiac":"Pontiac", "oldsmobile":"Oldsmobile",
    "rolls-royce":"Rolls-Royce", "ferrari":"Ferrari", "mclaren":"McLaren", "lotus":"Lotus", "bugatti":"Bugatti",
}

# --- Two-word brand detection (normalized) ---
TWO_WORD_BRANDS = {
    "alfa romeo": "alfa romeo",
    "aston martin": "aston martin",
    "land rover": "land rover",
    "mercedes benz": "mercedes-benz",
    "rolls royce": "rolls-royce",
}

# --- Single-token brand mapping (normalized token -> canonical brand key) ---
SINGLE_TOKEN_BRAND = {
    # direct tokens
    "acura":"acura","audi":"audi","bmw":"bmw","bentley":"bentley","buick":"buick","cadillac":"cadillac",
    "chevrolet":"chevrolet","chrysler":"chrysler","dodge":"dodge","ram":"ram","fiat":"fiat","ford":"ford",
    "genesis":"genesis","gmc":"gmc","honda":"honda","hyundai":"hyundai","infiniti":"infiniti","jaguar":"jaguar",
    "jeep":"jeep","kia":"kia","lexus":"lexus","lincoln":"lincoln","maserati":"maserati","mazda":"mazda",
    "mini":"mini","mitsubishi":"mitsubishi","nissan":"nissan","porsche":"porsche","saab":"saab","saturn":"saturn",
    "scion":"scion","smart":"smart","subaru":"subaru","suzuki":"suzuki","tesla":"tesla","toyota":"toyota",
    "volkswagen":"volkswagen","volvo":"volvo","hummer":"hummer","pontiac":"pontiac","oldsmobile":"oldsmobile",
    "ferrari":"ferrari","mclaren":"mclaren","lotus":"lotus","bugatti":"bugatti","bwm":"bmw",
    # helpful synonyms as tokens
    "chevy":"chevrolet","chev":"chevrolet","benz":"mercedes-benz","mercedes":"mercedes-benz","vw":"volkswagen",
}

def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)   # drop punctuation like "..", "-", etc.
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _detect_two_word(n: str):
    # check for any two-word brand anywhere in the string (token-boundary)
    for pair, canon in TWO_WORD_BRANDS.items():
        if re.search(rf"\b{re.escape(pair)}\b", n):
            return canon
    return None

def _detect_single_token(n: str):
    toks = n.split()
    if not toks:
        return None
    # prefer first token brand (e.g., "bmw 335i" -> "bmw")
    if toks[0] in SINGLE_TOKEN_BRAND:
        return SINGLE_TOKEN_BRAND[toks[0]]
    # otherwise, any token brand present
    for t in toks:
        if t in SINGLE_TOKEN_BRAND:
            return SINGLE_TOKEN_BRAND[t]
    return None

def canonicalize_make(raw) -> str:
    # empty
    if pd.isna(raw):
        return np.nan
    n = _norm(raw)
    if not n:
        return np.nan

    # exact synonyms
    n = CANON_SYNONYMS.get(n, n)

    # two-word brands anywhere
    two = _detect_two_word(n)
    if two:
        key = two
        return PROPER_CASE.get(key, key.title())

    # single-token brand (first token or any token)
    tok = _detect_single_token(n)
    if tok:
        key = tok
        return PROPER_CASE.get(key, key.title())

    # special case: "mercedesbenz"
    if n.replace(" ", "") == "mercedesbenz":
        return PROPER_CASE["mercedes-benz"]

    # fallback: title-case whatever is left (rare)
    return PROPER_CASE.get(n, n.title())
    # Canonicalize car make values (BMW/Mercedes-Benz/etc.)
    if "make" in df.columns:
        df["make"] = df["make"].apply(canonicalize_make)
        df = df.dropna(subset=["make"])

def main():
    assert CLEAN_PATH.exists(), f"Clean CSV not found: {CLEAN_PATH}"
    df_before = pd.read_csv(CLEAN_PATH, low_memory=False)
    if "make" not in df_before.columns:
        raise KeyError("Column 'make' not found in cleaned CSV")

    full_before_unique = int(df_before["make"].astype(str).nunique())
    top20_before = df_before["make"].astype(str).value_counts().head(20)

    df = df_before.copy()
    df["make"] = df["make"].apply(canonicalize_make)
    df = df.dropna(subset=["make"])

    full_after_unique = int(df["make"].nunique())
    top20_after = df["make"].value_counts().head(20)

    # write back
    df.to_csv(CLEAN_PATH, index=False)

    report = {
        "unique_makes_before": full_before_unique,
        "unique_makes_after": full_after_unique,
        "top20_before": top20_before.to_dict(),
        "top20_after": top20_after.to_dict(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Canonicalized 'make' values in:", CLEAN_PATH)
    print(f"Unique makes: {full_before_unique} → {full_after_unique}")
    print("Top-20 after:")
    print(top20_after.to_string())

if __name__ == "__main__":
    main()
