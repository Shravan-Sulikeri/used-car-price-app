# api/main.py
import os
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------- Config --------
DATA_PATH = os.getenv("DATA_CSV", "data/clean_used_cars.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model/model_gbm.pkl")

# -------- Robust CSV/Excel loader --------
def read_table_smart(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    # Excel?
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)

    # Try python engine first (can infer delimiter). DO NOT pass low_memory here.
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(p, engine="python", sep=None, encoding=enc)
        except Exception as e:
            last_err = e

    # Fall back to C engine with explicit seps (low_memory OK here)
    for enc in encodings:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(p, engine="c", sep=sep, encoding=enc, low_memory=False)
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Could not parse {p}. Last error: {last_err}")

# -------- Load data/model once --------
df = read_table_smart(DATA_PATH)

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL = None
    print("[WARN] Model not loaded:", e)

# -------- FastAPI app --------
app = FastAPI(title="Used Car Price API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------- Helpers --------
def apply_filters(
    base: pd.DataFrame,
    make: Optional[str], model: Optional[str], body: Optional[str],
    y0: Optional[int], y1: Optional[int]
) -> pd.DataFrame:
    d = base
    if make and "make" in d:
        d = d[d["make"].astype(str) == make]
    if model and "model" in d:
        d = d[d["model"].astype(str) == model]
    if body and "body" in d:
        d = d[d["body"].astype(str) == body]
    if y0 is not None and y1 is not None and "year" in d:
        d = d[d["year"].between(int(y0), int(y1))]
    return d

# -------- Routes --------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
        "model_loaded": MODEL is not None,
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
    }

@app.get("/options")
def options(make: Optional[str] = None) -> Dict[str, Any]:
    d = df
    makes = sorted(d["make"].dropna().astype(str).unique().tolist()) if "make" in d else []
    if make and "model" in d:
        models = sorted(d.loc[d["make"].astype(str) == make, "model"].dropna().astype(str).unique().tolist())
    else:
        models = sorted(d["model"].dropna().astype(str).unique().tolist()) if "model" in d else []
    bodies = sorted(d["body"].dropna().astype(str).unique().tolist()) if "body" in d else []
    years = (int(d["year"].min()), int(d["year"].max())) if "year" in d else (None, None)
    return {"makes": makes, "models": models, "bodies": bodies, "year_min": years[0], "year_max": years[1]}

@app.get("/summary")
def summary(
    make: Optional[str] = None, model: Optional[str] = None, body: Optional[str] = None,
    y0: Optional[int] = None, y1: Optional[int] = None
) -> Dict[str, Any]:
    d = apply_filters(df, make, model, body, y0, y1)
    return {
        "rows": int(len(d)),
        "median_price": float(d["price"].median()) if "price" in d else None,
        "median_mileage": float(d["mileage"].median()) if "mileage" in d else None,
        "unique_makes": int(d["make"].nunique()) if "make" in d else None,
        "unique_models": int(d["model"].nunique()) if "model" in d else None,
    }

@app.get("/charts")
def charts(
    make: Optional[str] = None, model: Optional[str] = None, body: Optional[str] = None,
    y0: Optional[int] = None, y1: Optional[int] = None
) -> Dict[str, Any]:
    d = apply_filters(df, make, model, body, y0, y1)

    # Price histogram (cap at 99th pct for display)
    if "price" in d:
        hi = min(200_000, float(d["price"].quantile(0.99)))
        bins = np.linspace(0, max(1000, hi), 30)
        hist, edges = np.histogram(d["price"], bins=bins)
        price_hist = {"bins": edges[:-1].round(0).tolist(), "counts": hist.tolist()}
    else:
        price_hist = {"bins": [], "counts": []}

    # Median price by year
    if {"year", "price"}.issubset(d.columns):
        by_year = d.groupby("year")["price"].median().reset_index()
        price_by_year = {"year": by_year["year"].astype(int).tolist(), "price": by_year["price"].round(0).tolist()}
    else:
        price_by_year = {"year": [], "price": []}

    # Top models by median price
    if {"model", "price"}.issubset(d.columns):
        top = d.groupby("model")["price"].median().sort_values(ascending=False).head(10).sort_values()
        top_models = {"model": top.index.astype(str).tolist(), "price": top.round(0).tolist()}
    else:
        top_models = {"model": [], "price": []}

    # Donuts
    make_share = {"make": [], "count": []}
    model_share = {"model": [], "count": []}
    if "make" in d:
        ms = d["make"].value_counts().head(10)
        make_share = {"make": ms.index.astype(str).tolist(), "count": ms.values.tolist()}
    if "model" in d:
        mo = d["model"].value_counts().head(10)
        model_share = {"model": mo.index.astype(str).tolist(), "count": mo.values.tolist()}

    return {
        "price_hist": price_hist,
        "price_by_year": price_by_year,
        "top_models": top_models,
        "make_share": make_share,
        "model_share": model_share,
    }
from fastapi.responses import RedirectResponse, Response

@app.get("/", include_in_schema=False)
def root():
    # redirect people who hit '/' to the docs
    return RedirectResponse("/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # avoid noisy 404s for the browser's favicon request
    return Response(status_code=204)

# ---------- Prediction ----------
class PredictIn(BaseModel):
    year: Optional[int] = None
    mileage: Optional[float] = None
    make: Optional[str] = None
    model: Optional[str] = None
    body: Optional[str] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    seller_type: Optional[str] = None
    state: Optional[str] = None

@app.post("/predict")
def predict(p: PredictIn) -> Dict[str, Any]:
    if MODEL is None:
        return {"ok": False, "error": "Model not loaded"}

    ref_year = int(df["year"].median()) if "year" in df else 2020
    age = max(0, (ref_year - (p.year or ref_year)))
    mpyear = (p.mileage or 0) / (age if age >= 1 else 1)
    high_m = 1 if (p.mileage or 0) > 150_000 else 0

    row = {
        "year": p.year, "mileage": p.mileage, "make": p.make, "model": p.model,
        "body": p.body, "fuel": p.fuel, "transmission": p.transmission,
        "seller_type": p.seller_type, "state": p.state,
        "age": age, "mileage_per_year": mpyear, "high_mileage": high_m,
    }
    X = pd.DataFrame([row])
    yhat = MODEL.predict(X)
    yhat = float(np.expm1(yhat[0] if isinstance(yhat, (list, np.ndarray)) else yhat))
    return {"ok": True, "pred_price": round(yhat, 2)}