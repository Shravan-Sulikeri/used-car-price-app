# model/standardize_models.py
# v1: conservative model cleaner (keeps families, removes ad-hoc noise)
import re, json
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/clean_used_cars.csv")
REPORT_PATH = Path("model/model_cleaning_report.json")

# tokens/phrases that clearly are NOT model names (trim-levels/features/ads)
NOISE_PATTERNS = [
    r"\b(one\s*owner|only\s*\d+\s*k?|carfax|accident\s*free|clean|immaculate)\b",
    r"\b(nav(igation)?|sunroof|moonroof|leather|sport\s*pkg?)\b",
    r"\b(awd|4x4|4wd|2wd|xdrive)\b",
    r"\b(manual|automatic|diesel|gas|hybrid|electric)\b",
    r"\b(cargo\s*van|passenger\s*van)\b",
    r"\b(sedan|coupe|convertible|wagon|hatchback|truck|suv)\b",
    r"\b(only\s*\d+\s*(miles|mi|k))\b",
]

NOISE_RX = re.compile("|".join(NOISE_PATTERNS), re.I)

def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    # normalize separators
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^\w\s\-]+", " ", s)    # keep hyphen and word chars
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _squash_spaces_and_hyphens(s: str) -> str:
    # unify patterns like "f 150" -> "f-150", "f150" -> "f-150"
    s = re.sub(r"\bf[\s\-]?(\d{3})\b", r"f-\1", s)  # F-series
    s = re.sub(r"\bsilverado\s*(\d{4})\b", r"silverado \1", s)  # keep 1500/2500 etc.
    s = re.sub(r"\b(\d)-series\b", r"\1-series", s)  # ensure hyphen
    return s

def _bmw_series(model: str) -> str | None:
    # x1/x3/x5 keep as-is
    if re.match(r"^x[1-7]\b", model): return model
    # 1xx/2xx/3xx -> n-series
    m = re.match(r"^([1-8])\d{2}[a-z]*\b", model)
    if m:
        return f"{m.group(1)}-series"
    # already like "3 series"
    m = re.match(r"^([1-8])\s*series\b", model)
    if m:
        return f"{m.group(1)}-series"
    return None

def _mb_class(model: str) -> str | None:
    # Mercedes classes: c-class/e-class/s-class
    if re.match(r"^c[\s\-]?(class|\d{2,3})\b", model): return "c-class"
    if re.match(r"^e[\s\-]?(class|\d{2,3})\b", model): return "e-class"
    if re.match(r"^s[\s\-]?(class|\d{2,3})\b", model): return "s-class"
    return None

def _title_proper(make: str, model: str) -> str:
    special_caps = {
        "bmw": {"-series": True, "x1": True, "x3": True, "x5": True},
        "mercedes-benz": {"-class": True, "glc": True, "gle": True, "gls": True, "gla": True, "cla": True, "cls": True, "c-class": True, "e-class": True, "s-class": True},
        "ram": {"1500": True, "2500": True, "3500": True},
        "ford": {"f-150": True, "f-250": True, "f-350": True},
        "mini": {"cooper": True},
    }
    mk = (make or "").lower()
    s = model
    # Uppercase families we recognize
    if mk in special_caps:
        for token in special_caps[mk]:
            if s == token:  # exact family
                return token.upper() if token.startswith("f-") else token.title().replace("-Class", "-Class").replace("-Series", "-Series")
    # Default Title Case but keep hyphen casing like "F-150"
    s = re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), s)
    s = s.replace("-Class", "-Class").replace("-Series", "-Series")
    # Ensure F-150/F-250… uppercased
    s = re.sub(r"\bF-(\d{3})\b", lambda m: f"F-{m.group(1)}", s, flags=re.I)
    return s

def canonicalize_model(make: str, model: str) -> str | None:
    if model is None or str(model).strip() == "":
        return None
    mk = (make or "").strip().lower()
    m = _norm(model)

    # drop known noise
    m = NOISE_RX.sub(" ", m)
    m = re.sub(r"\s+", " ", m).strip()

    # brand-specific light rules
    if mk == "bmw":
        b = _bmw_series(m)
        if b: m = b
    elif mk == "mercedes-benz":
        b = _mb_class(m)
        if b: m = b

    # normalize f-150 pattern, series/class hyphens
    m = _squash_spaces_and_hyphens(m)

    # very short/empty after cleaning -> None
    if not m or len(m) < 2:
        return None

    return _title_proper(mk, m)

def main():
    assert CLEAN_PATH.exists(), f"Clean CSV not found: {CLEAN_PATH}"
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    if "model" not in df.columns:
        print("No 'model' column found; nothing to do.")
        return

    before_unique = int(df["model"].astype(str).nunique())

    # preview top noisy examples (optional)
    top_before = df["model"].value_counts().head(20).to_dict()

    # apply
    df["model"] = [canonicalize_model(df.at[i, "make"] if "make" in df.columns else None, v)
                   for i, v in enumerate(df["model"])]

    # drop rows where model became None? (keep for now)
    # df = df.dropna(subset=["model"])

    after_unique = int(df["model"].astype(str).nunique())

    # write back
    df.to_csv(CLEAN_PATH, index=False)

    # report
    report = {
        "unique_models_before": before_unique,
        "unique_models_after": after_unique,
        "top20_models_before": top_before,
        "examples": {
            "bmw 335i ->": canonicalize_model("BMW", "335i"),
            "mercedes c300 ->": canonicalize_model("Mercedes-Benz", "c300"),
            "f150 ->": canonicalize_model("Ford", "f150"),
            "f 150 ->": canonicalize_model("Ford", "f 150"),
            "silverado1500 ->": canonicalize_model("Chevrolet", "silverado1500"),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Canonicalized 'model' values in:", CLEAN_PATH)
    print(f"Unique models: {before_unique} → {after_unique}")
    print("Sample examples:", report["examples"])

if __name__ == "__main__":
    main()