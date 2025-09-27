# api/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# ----------------------------
# Paths / config
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_CSV", ROOT / "data" / "clean_used_cars.csv"))
PREPROC_PATH = Path(os.getenv("PREPROCESSOR_PKL", ROOT / "model" / "preprocessor.pkl"))
MODEL_PATH = Path(os.getenv("MODEL_PKL", ROOT / "model" / "model_gbm.pkl"))

# If you trained on log(price) set to "1"
USE_LOG_TARGET = bool(int(os.getenv("PRICE_MODEL_LOG", "1")))

# Features expected by the trained preprocessor/model
BASE_FEATS = ["year", "mileage", "make", "model", "body"]
ENG_FEATS  = ["age", "mileage_per_year", "high_mileage"]
MODEL_FEATS = ["year","mileage","age","mileage_per_year","high_mileage","make","model","body"]

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Used Car Price API", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ----------------------------
# Load data + artifacts
# ----------------------------
def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, low_memory=False)

try:
    df: pd.DataFrame = _read_csv_safe(DATA_PATH)
except Exception as e:
    df = pd.DataFrame(columns=["price"] + MODEL_FEATS)
    print(f"[WARN] Could not load data: {e}")

def _safe_load_joblib(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load artifact {path}: {e}")
        return None

PREPROCESSOR = _safe_load_joblib(PREPROC_PATH)
MODEL = _safe_load_joblib(MODEL_PATH)

def _infer_ref_year(frame: pd.DataFrame) -> int:
    now_y = datetime.utcnow().year
    if "year" in frame.columns and frame["year"].dropna().size:
        return int(max(now_y, frame["year"].max()))
    return now_y

REF_YEAR = _infer_ref_year(df)

# ----------------------------
# Schemas
# ----------------------------
class PredictIn(BaseModel):
    year: int = Field(..., ge=1980, le=2100)
    mileage: int = Field(..., ge=0, le=1_000_000)
    make: str
    model: str
    body: Optional[str] = None

class PredictOut(BaseModel):
    ok: bool
    price_usd: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ----------------------------
# Helpers
# ----------------------------
def check_ready() -> None:
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

def _engineer_features(year: int, mileage: float) -> Dict[str, Any]:
    age = max(0, int(REF_YEAR) - int(year))
    # avoid div by zero; if age==0, use mileage as per-year proxy
    mpy = float(mileage) / max(age, 1)
    high = int((mileage > 150_000) or (mpy > 20_000))
    return {"age": age, "mileage_per_year": mpy, "high_mileage": high}

def _make_feature_row(req: PredictIn) -> pd.DataFrame:
    eng = _engineer_features(req.year, req.mileage)
    row = {
        "year": int(req.year),
        "mileage": int(req.mileage),
        "make": (req.make or "").strip(),
        "model": (req.model or "").strip(),
        "body": (req.body if req.body not in ("", "null", "None") else None),
        **eng,
    }
    # Ensure all model features exist, in a stable order
    for c in MODEL_FEATS:
        row.setdefault(c, None)
    return pd.DataFrame([row], columns=MODEL_FEATS)

def _filter_frame(
    make: Optional[str] = None,
    model: Optional[str] = None,
    y0: Optional[int] = None,
    y1: Optional[int] = None,
) -> pd.DataFrame:
    m = pd.Series([True] * len(df))
    if y0 is not None and y1 is not None and "year" in df.columns:
        m &= df["year"].between(y0, y1)
    if make:
        m &= df["make"].astype(str).str.casefold().eq(make.casefold())
    if model and "model" in df.columns:
        m &= df["model"].astype(str).str.casefold().eq(model.casefold())
    return df.loc[m].copy()

def _hist(series: pd.Series, bins: int = 30) -> Dict[str, List[float]]:
    x = series.dropna().astype(float).values
    if x.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(x, bins=bins)
    return {"bins": edges[:-1].round(0).astype(int).tolist(), "counts": counts.tolist()}

# ----------------------------
# Routes
# ----------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    info = {
        "status": "ok",
        "rows": int(len(df)),
        "cols": list(df.columns),
        "ref_year": int(REF_YEAR),
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "data_path": str(DATA_PATH.resolve()),
        "model_path": str(MODEL_PATH.resolve()),
    }
    return info

@app.get("/options")
def options():
    makes = sorted(map(str, df["make"].dropna().unique())) if "make" in df.columns else []
    models = sorted(map(str, df["model"].dropna().unique())) if "model" in df.columns else []
    bodies = sorted(map(str, df["body"].dropna().unique())) if "body" in df.columns else []

    models_by_make: Dict[str, List[str]] = {}
    if "make" in df.columns and "model" in df.columns:
        for mk, g in df.groupby("make"):
            models_by_make[str(mk)] = sorted(map(str, g["model"].dropna().unique()))

    y0 = int(df["year"].min()) if "year" in df.columns and len(df) else None
    y1 = int(df["year"].max()) if "year" in df.columns and len(df) else None

    return {
        "makes": makes,
        "models": models,
        "models_by_make": models_by_make,
        "bodies": bodies,
        "year_min": y0,
        "year_max": y1,
    }

@app.get("/summary")
def summary(
    make: Optional[str] = None,
    model: Optional[str] = None,
    y0: Optional[int] = Query(None, ge=1900, le=2100),
    y1: Optional[int] = Query(None, ge=1900, le=2100),
):
    dff = _filter_frame(make, model, y0, y1)
    resp = {
        "rows": int(len(dff)),
        "median_price": int(dff["price"].median()) if "price" in dff.columns and len(dff) else None,
        "median_mileage": int(dff["mileage"].median()) if "mileage" in dff.columns and len(dff) else None,
        "unique_makes": int(dff["make"].nunique()) if "make" in dff.columns else None,
        "unique_models": int(dff["model"].nunique()) if "model" in dff.columns else None,
    }
    return resp

@app.get("/charts")
def charts(
    make: Optional[str] = None,
    model: Optional[str] = None,
    y0: Optional[int] = Query(None, ge=1900, le=2100),
    y1: Optional[int] = Query(None, ge=1900, le=2100),
):
    dff = _filter_frame(make, model, y0, y1)

    price_hist = _hist(dff["price"], bins=30) if "price" in dff.columns else {"bins": [], "counts": []}

    price_by_year = {"year": [], "price": []}
    if "price" in dff.columns and "year" in dff.columns:
        tmp = dff.groupby("year")["price"].median().sort_index()
        price_by_year = {"year": tmp.index.astype(int).tolist(), "price": tmp.round(0).astype(int).tolist()}

    top_models = {"model": [], "price": []}
    if "model" in dff.columns and "price" in dff.columns:
        med = dff.groupby("model")["price"].median().sort_values(ascending=False).head(10).sort_values()
        top_models = {"model": med.index.tolist(), "price": med.round(0).astype(int).tolist()}

    make_share = {"make": [], "count": []}
    if "make" in dff.columns:
        vc = dff["make"].value_counts().head(10)
        make_share = {"make": vc.index.tolist(), "count": vc.tolist()}

    model_share = {"model": [], "count": []}
    if "model" in dff.columns:
        vc2 = dff["model"].value_counts().head(10)
        model_share = {"model": vc2.index.tolist(), "count": vc2.tolist()}

    return {
        "price_hist": price_hist,
        "price_by_year": price_by_year,
        "top_models": top_models,
        "make_share": make_share,
        "model_share": model_share,
    }

@app.post("/predict", response_model=PredictOut)
def predict(req: PredictIn):
    check_ready()
    try:
        X = _make_feature_row(req)             # includes engineered cols
        Z = PREPROCESSOR.transform(X)          # transform with fitted ColumnTransformer
        yhat = MODEL.predict(Z)

        price = float(np.exp(yhat[0])) if USE_LOG_TARGET else float(yhat[0])
        return PredictOut(ok=True, price_usd=price, details={"inputs": X.iloc[0].to_dict()})
    except Exception as e:
        return PredictOut(ok=False, error=f"inference_error: {e}")