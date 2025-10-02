#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, re, hashlib

RAW_DIRS = [Path("data/raw"), Path("data/raw/sources")]
OUT_CSV, OUT_PARQ = Path("data/clean_listings_clean.csv"), Path("data/clean_listings_clean.parquet")

KNOWN_MAKES = {"acura","alfa romeo","audi","bmw","buick","cadillac","chevrolet","chrysler","dodge","fiat","ford","gmc",
"honda","hyundai","infiniti","jaguar","jeep","kia","land rover","lexus","lincoln","mazda","mercedes-benz",
"mini","mitsubishi","nissan","porsche","ram","scion","subaru","tesla","toyota","volkswagen","volvo"}
ALIASES = {"chevy":"Chevrolet","mercedes":"Mercedes-Benz","vw":"Volkswagen"}
TESLA_MAP = {"3":"Model 3","model3":"Model 3","model 3":"Model 3","m3":"Model 3",
"y":"Model Y","modely":"Model Y","model y":"Model Y","x":"Model X","modelx":"Model X","model x":"Model X",
"s":"Model S","models":"Model S","model s":"Model S"}

def _norm(s:str)->str:
    if not isinstance(s,str): return ""
    s=s.strip().lower()
    return re.sub(r"[^a-z0-9\s\-]+"," ",s)

def clean_make_model(make:str|None, model:str|None)->tuple[str,str]:
    mk_raw, md_raw = (make or "").strip(), (model or "").strip()
    nmk, nmd = _norm(mk_raw), _norm(md_raw)
    mk, md = mk_raw.title(), " ".join(md_raw.split()).title()
    toks=nmk.split()
    if toks and toks[0] in KNOWN_MAKES and len(toks)>1:
        mk=toks[0].title(); md=" ".join(toks[1:]).title()
    if (not mk or _norm(mk) not in KNOWN_MAKES) and nmd:
        first=nmd.split()[0]
        if first in KNOWN_MAKES:
            mk=first.title(); md=" ".join(nmd.split()[1:]).title() or md
    ak=_norm(mk)
    if ak in ALIASES: mk=ALIASES[ak]
    if _norm(mk)=="tesla":
        t=nmd.replace(" ","")
        for k,v in TESLA_MAP.items():
            if k.replace(" ","")==t or k in nmd: return ("Tesla",v)
        if "model3" in nmd or nmd in ("3","m3"): return ("Tesla","Model 3")
        if "modely" in nmd or nmd=="y":          return ("Tesla","Model Y")
        if "modelx" in nmd or nmd=="x":          return ("Tesla","Model X")
        if "models" in nmd or nmd=="s":          return ("Tesla","Model S")
    mk=mk.title()
    md=re.sub(r"[^A-Za-z0-9\-\s]","",md).strip()
    md=re.sub(r"\s+"," ",md).title()
    return mk, md

def read_any(p:Path)->pd.DataFrame:
    return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)

def collect_sources()->list[Path]:
    files=[]
    for root in RAW_DIRS:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".csv",".parquet"): files.append(p)
    return files

def standardize_columns(df:pd.DataFrame)->pd.DataFrame:
    rename={"Manufacturer":"make","Make":"make","Model":"model","Year":"year","Mileage":"mileage","odometer":"mileage",
            "pricesold":"price","Price":"price","Price_in_thousands":"price_thousands","priceUSD":"price",
            "BodyType":"body_type","body":"body_type","DriveType":"drive_type","drivetrain":"drive_type",
            "Trim":"trim","Engine":"engine","Cylinders":"num_cylinders","NumCylinders":"num_cylinders",
            "zipcode":"zipcode","Zip":"zipcode"}
    df=df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()
    if "price" not in df and "price_thousands" in df: df["price"]=pd.to_numeric(df["price_thousands"],errors="coerce")*1000.0
    for c in ("price","year","mileage"):
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in ("make","model","body_type","drive_type","trim","engine"):
        if c in df: df[c]=df[c].astype(str).str.strip()
    if "make" in df and "model" in df:
        df[["make","model"]]=df.apply(lambda r: clean_make_model(r["make"],r["model"]),axis=1,result_type="expand")
    elif "make" in df:
        df["make"],df["model"]=zip(*df["make"].map(lambda s: clean_make_model(s,None)))
    elif "model" in df:
        df["make"],df["model"]=zip(*df["model"].map(lambda s: clean_make_model(None,s)))
    if "year" in df:    df=df[df["year"].between(1990,2026)]
    if "mileage" in df: df=df[(df["mileage"].isna()) | df["mileage"].between(0,300_000)]
    if "price" in df:   df=df[df["price"].between(1000,250_000)]
    keep=[c for c in ("price","year","mileage","make","model","body_type","drive_type","zipcode") if c in df.columns]
    return df[keep]

def dedupe(df:pd.DataFrame)->pd.DataFrame:
    def key(r):
        parts=[str(r.get("make","")).lower(),
               str(r.get("model","")).lower(),
               str(int(r["year"])) if not pd.isna(r.get("year")) else "",
               str(int(round((r.get("mileage") or 0)/1000.0,0))*1000),
               str(int(round((r.get("price") or 0)/500.0,0))*500)]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()
    k=df.apply(key,axis=1)
    return df.loc[~k.duplicated()].reset_index(drop=True)

def main():
    files=collect_sources()
    if not files: raise SystemExit("No raw files found. Put CSV/Parquet under data/raw/ then rerun.")
    frames=[]
    for p in files:
        try:
            df=read_any(p); frames.append(standardize_columns(df))
            print(f"Loaded {p} -> {len(frames[-1])} rows")
        except Exception as e:
            print(f"Skip {p}: {e}")
    if not frames: raise SystemExit("No usable files after standardization.")
    df=pd.concat(frames,ignore_index=True)
    df=df[df["make"].astype(str).str.len()>0]
    df["make_norm"]=df["make"].str.lower()
    df=df[df["make_norm"].isin(KNOWN_MAKES)].drop(columns=["make_norm"])
    MIN_PER_MODEL=3
    if "model" in df.columns:
        freq=df.groupby(["make","model"])["model"].transform("count")
        df=df[(freq>=MIN_PER_MODEL) | (df["make"].map(df["make"].value_counts())<MIN_PER_MODEL)]
    df["make"]=df["make"].str.title()
    df["model"]=df["model"].str.replace(r"[^A-Za-z0-9\-\s]","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip().str.title()
    df=dedupe(df)
    OUT_PARQ.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(OUT_PARQ,index=False); df.to_csv(OUT_CSV,index=False)
    print(f"✅ Wrote {OUT_PARQ} and {OUT_CSV} | rows: {len(df)}")

if __name__=="__main__":
    main()
