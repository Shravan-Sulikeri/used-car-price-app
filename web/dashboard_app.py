# web/dashboard_app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme: Breaking Bad palette
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#E9D71B",  # bright yellow
    "#C7C51B",  # olive yellow
    "#A49828",  # mustard/olive
    "#31553B",  # deep green
    "#12140F",  # near-black
    "#9FA12A",  # desaturated olive
    "#4B6E56",  # mid green
    "#D8C85E",  # light olive
]
px.defaults.color_discrete_sequence = PALETTE
px.defaults.template = None

def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        font=dict(color="#F2F2F2"),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Brand knowledge / cleaners
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_MAKES = {
    "acura","alfa romeo","audi","bmw","buick","cadillac","chevrolet","chevy",
    "chrysler","dodge","fiat","ford","gmc","honda","hyundai","infiniti","jaguar",
    "jeep","kia","land rover","lexus","lincoln","mazda","mercedes-benz","mercedes",
    "mini","mitsubishi","nissan","porsche","ram","scion","subaru","tesla",
    "toyota","volkswagen","vw","volvo"
}
MAKE_ALIASES = {
    "chevy": "Chevrolet",
    "mercedes": "Mercedes-Benz",
    "vw": "Volkswagen",
}
TESLA_MAP = {
    "3": "Model 3", "model3": "Model 3", "model 3": "Model 3", "m3": "Model 3",
    "y": "Model Y", "modely": "Model Y", "model y": "Model Y",
    "x": "Model X", "modelx": "Model X", "model x": "Model X",
    "s": "Model S", "models": "Model S", "model s": "Model S",
}

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)

def clean_make_model(make: Optional[str], model: Optional[str]) -> Tuple[str, str]:
    """
    Robustly derive canonical (make, model).
    - Splits cases like 'Bmw 3 Series' in the make field.
    - If make is missing but model starts with brand, split.
    - Applies common aliases and Tesla canonicalization.
    """
    mk_raw, md_raw = (make or "").strip(), (model or "").strip()
    nmk, nmd = _norm(mk_raw), _norm(md_raw)
    mk, md = mk_raw.title(), " ".join(md_raw.split()).title()

    # If make contains both brand and model (e.g., 'Bmw 3 Series')
    toks = nmk.split()
    if toks:
        first = toks[0]
        if first in KNOWN_MAKES and len(toks) > 1:
            mk = first.title()
            md = " ".join(toks[1:]).title()

    # If make missing but model starts with a brand
    if (not mk or _norm(mk) not in KNOWN_MAKES) and nmd:
        first = nmd.split()[0]
        if first in KNOWN_MAKES:
            mk = first.title()
            md = " ".join(nmd.split()[1:]).title() or md

    # Aliases
    alias_key = _norm(mk)
    if alias_key in MAKE_ALIASES:
        mk = MAKE_ALIASES[alias_key]

    # Tesla canonicalization
    if _norm(mk) == "tesla":
        t = nmd.replace(" ", "")
        for k, v in TESLA_MAP.items():
            if k.replace(" ", "") == t or k in nmd:
                return ("Tesla", v)
        if "model3" in nmd or nmd in ("3", "m3"):  return ("Tesla", "Model 3")
        if "modely" in nmd or nmd == "y":          return ("Tesla", "Model Y")
        if "modelx" in nmd or nmd == "x":          return ("Tesla", "Model X")
        if "models" in nmd or nmd == "s":          return ("Tesla", "Model S")

    return (mk.title(), md.title())

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (uploader in sidebar; no widgets inside cached funcs)
# ──────────────────────────────────────────────────────────────────────────────
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load CSV/Parquet from upload or defaults under /data, then clean.
    Prefers cleaned files to avoid 'Other' in charts.
    """
    if uploaded_file is not None:
        df = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file)
        return _post_load_clean(df)

    for p in [
        Path("data/clean_listings_clean.parquet"),
        Path("data/clean_listings_clean.csv"),
        Path("data/clean_listings.parquet"),
        Path("data/clean_listings.csv"),
    ]:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            return _post_load_clean(df)

    return pd.DataFrame()

def _post_load_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize names
    rename = {
        "Manufacturer": "make",
        "Make": "make",
        "Model": "model",
        "Year": "year",
        "Mileage": "mileage",
        "pricesold": "price",
        "Price": "price",
        "Price_in_thousands": "price_thousands",
        "BodyType": "body_type",
        "DriveType": "drive_type",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    if "price" not in df.columns and "price_thousands" in df.columns:
        df["price"] = pd.to_numeric(df["price_thousands"], errors="coerce") * 1000.0

    for c in ("price", "year", "mileage"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("make", "model"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Clean / split make-model
    if "make" in df.columns and "model" in df.columns:
        df[["make", "model"]] = df.apply(lambda r: clean_make_model(r["make"], r["model"]), axis=1, result_type="expand")
    elif "make" in df.columns:
        df["make"], df["model"] = zip(*df["make"].map(lambda s: clean_make_model(s, None)))
    elif "model" in df.columns:
        df["make"], df["model"] = zip(*df["model"].map(lambda s: clean_make_model(None, s)))

    # sensible ranges
    if "year" in df:    df = df[(df["year"].between(1990, 2026)) | df["year"].isna()]
    if "mileage" in df: df = df[(df["mileage"].between(0, 300_000)) | df["mileage"].isna()]
    if "price" in df:   df = df[(df["price"].between(1000, 250_000)) | df["price"].isna()]

    # Final tidy
    if "make" in df:  df["make"]  = df["make"].str.title()
    if "model" in df: df["model"] = df["model"].str.replace(r"[^A-Za-z0-9\-\s]", "", regex=True)\
                                               .str.replace(r"\s+", " ", regex=True)\
                                               .str.strip().str.title()

    return df.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Make → Model cascades
# ──────────────────────────────────────────────────────────────────────────────
def build_cascades(df: pd.DataFrame):
    makes = sorted(df["make"].dropna().unique().tolist()) if "make" in df else []
    models_by_make = {mk: sorted(df.loc[df["make"] == mk, "model"].dropna().unique().tolist()) for mk in makes}
    return makes, models_by_make

# ──────────────────────────────────────────────────────────────────────────────
# Legacy model loading & STRICT, model-driven prediction
# ──────────────────────────────────────────────────────────────────────────────
def load_legacy_model():
    try:
        import joblib
        mp, sp, cp = Path("model/model_gbm.pkl"), Path("model/schema_best.json"), Path("model/cat_levels.json")
        if not (mp.exists() and sp.exists()):
            return None, None, None
        model = joblib.load(mp)
        schema = json.loads(sp.read_text())
        cats = json.loads(cp.read_text()) if cp.exists() else {}
        return model, schema, cats
    except Exception as e:
        st.warning(f"Model load failed: {e}")
        return None, None, None

def _coerce_to_category(val, levels: List[str]) -> str:
    """Return a value that exists in levels, preferring 'Other' if available."""
    if pd.isna(val) or val is None:
        return "Other" if "Other" in levels else (levels[0] if levels else "")
    sval = str(val)
    if sval in levels:
        return sval
    return "Other" if "Other" in levels else (levels[0] if levels else "")

def _get_model_feature_names(model) -> List[str]:
    """Pull exact feature names the trained LightGBM model expects."""
    try:
        if hasattr(model, "feature_name_") and model.feature_name_:
            return list(model.feature_name_)
    except Exception:
        pass
    try:
        booster = getattr(model, "booster_", None) or getattr(model, "booster", None)
        if booster is not None:
            names = booster.feature_name()
            if names:
                return list(names)
    except Exception:
        pass
    return []

def build_design_matrix(schema: dict, cats: dict, row: dict, model_feature_names: List[str]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame that EXACTLY matches the training matrix:
    - column set and order come from the MODEL (authoritative)
    - numeric defaults from schema (fallback 0)
    - categoricals cast to fixed levels from cats
    """
    if not model_feature_names:
        model_feature_names = schema.get("feature_order", schema.get("columns", [])) or []

    X = pd.DataFrame([{c: np.nan for c in model_feature_names}], columns=model_feature_names)

    # Fill provided row values (only those the model uses)
    for k, v in (row or {}).items():
        if k in X.columns:
            X.at[0, k] = v

    # Numeric defaults (schema) or 0
    num_defaults: dict = schema.get("numeric_defaults", {}) if schema else {}
    for c in X.columns:
        if c in (cats or {}):  # categorical handled later
            continue
        if c in num_defaults:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(num_defaults[c])
        else:
            X[c] = pd.to_numeric(X[c], errors="ignore")
            if X[c].dtype.kind in "fcbiu":
                X[c] = X[c].fillna(0)

    # Categorical casting
    for c, levels in (cats or {}).items():
        if c in X.columns:
            coerced = _coerce_to_category(X.at[0, c], levels)
            X[c] = pd.Categorical([coerced], categories=levels)

    # Ensure order
    X = X[model_feature_names]
    return X

def predict_with_legacy_gbm(model, schema: dict, cats: dict, row: dict) -> float:
    feat_names = _get_model_feature_names(model)
    X = build_design_matrix(schema or {}, cats or {}, row, feat_names)
    return float(model.predict(X)[0])

def comps_estimate(df: pd.DataFrame, row: dict) -> Tuple[float, Dict[str, float]]:
    base = df.copy()
    if "make" in base and row.get("make"):
        base = base[base["make"] == row["make"]]
    if "model" in base and row.get("model"):
        base = base[base["model"] == row["model"]]
    if "year" in base and pd.notna(row.get("year")):
        base = base[base["year"].between(row["year"] - 2, row["year"] + 2)]
    if "mileage" in base and pd.notna(row.get("mileage")):
        lo, hi = row["mileage"] * 0.75, row["mileage"] * 1.25
        base = base[(base["mileage"] >= lo) & (base["mileage"] <= hi)]

    prices = base["price"].dropna()
    if len(prices) >= 3:
        return float(prices.median()), {
            "p25": float(prices.quantile(0.25)),
            "p75": float(prices.quantile(0.75)),
            "n_comp": int(len(prices)),
        }
    g = df["price"].dropna()
    if len(g) > 0:
        return float(g.median()), {"p25": float(g.quantile(0.25)), "p75": float(g.quantile(0.75)), "n_comp": 0}
    return float("nan"), {"p25": float("nan"), "p75": float("nan"), "n_comp": 0}

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: navigation + upload
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Used Car Dashboard")
    page = st.radio("Navigate", ["Graphs", "Distribution (Pie)", "Price Prediction"], label_visibility="collapsed")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    st.caption("If no file is uploaded, the app looks for data under `data/`.")

# Load data (no widgets inside)
df = load_data(uploaded)
if df.empty:
    st.info("No data found. Upload a CSV/Parquet or place `clean_listings_clean.{csv,parquet}` under `/data`.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────────────────────────
def kpis(d: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Cars", f"{len(d):,}")
    with c2:
        if "price" in d: st.metric("Avg Price", f"${d['price'].mean():,.0f}")
    with c3:
        if "mileage" in d and d["mileage"].notna().any():
            st.metric("Median Mileage", f"{d['mileage'].median():,.0f}")
        else:
            st.metric("Median Mileage", "—")
    with c4:
        if "make" in d:
            vc = d["make"].value_counts()
            st.metric("Top Make", vc.index[0] if len(vc) else "—")

# ──────────────────────────────────────────────────────────────────────────────
# Page: Graphs
# ──────────────────────────────────────────────────────────────────────────────
if page == "Graphs":
    st.markdown("## 📈 KPIs & Interactive Graphs")
    kpis(df); st.markdown("")

    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            makes_all = ["(All)"] + sorted(df["make"].dropna().unique().tolist())
            f_make = st.selectbox("Make", makes_all, index=0)
        with c2:
            years = df["year"].dropna().astype(int).sort_values().unique().tolist() if "year" in df else []
            yr_min, yr_max = (min(years), max(years)) if years else (1990, 2026)
            f_year = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
        with c3:
            f_miles = st.slider("Mileage (max)", 0, 300_000, 300_000)

    dff = df.copy()
    if f_make != "(All)":
        dff = dff[dff["make"] == f_make]
    if "year" in dff:
        dff = dff[dff["year"].between(f_year[0], f_year[1])]
    if "mileage" in dff:
        dff = dff[(dff["mileage"].isna()) | (dff["mileage"] <= f_miles)]

    colA, colB = st.columns(2)
    if {"year", "price"}.issubset(dff.columns):
        fig = px.scatter(dff, x="year", y="price", color="make",
                         opacity=0.7, labels={"year": "Year", "price": "Price ($)"})
        colA.plotly_chart(style_fig(fig), use_container_width=True)
    if {"mileage", "price"}.issubset(dff.columns):
        fig = px.scatter(dff, x="mileage", y="price", color="make",
                         opacity=0.7, labels={"mileage": "Mileage", "price": "Price ($)"})
        colB.plotly_chart(style_fig(fig), use_container_width=True)

    if {"make", "price"}.issubset(dff.columns):
        st.markdown("#### Price by Make")
        top = dff["make"].value_counts().head(12).index.tolist()
        dd = dff[dff["make"].isin(top)]
        fig = px.box(dd, x="make", y="price", color="make",
                     points="outliers", labels={"make": "Make", "price": "Price ($)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Page: Distribution (Pie)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Distribution (Pie)":
    st.markdown("## 🥧 Market Distribution")
    kpis(df); st.markdown("")
    dims = [c for c in ["make", "body_type", "drive_type", "year"] if c in df.columns]
    if not dims:
        st.warning("No categorical columns available.")
        st.stop()

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        dim = st.selectbox("Distribution by", options=dims, index=0)
    with colB:
        top_n = st.slider("Show top N", 5, 25, 12, step=1)
    with colC:
        min_share = st.slider("Min % share", 0.0, 5.0, 1.0, step=0.5)
    include_other = st.toggle("Include 'Other' slice", value=False)

    dff = df.copy()
    if dim == "year":
        dff["year"] = dff["year"].astype("Int64")

    vc = dff[dim].dropna().astype(str).str.strip().value_counts()
    total = vc.sum()

    # filter by min_share first, then keep top_n
    vc = vc[(vc / total * 100) >= min_share].head(top_n)

    if include_other:
        remaining = total - vc.sum()
        if remaining > 0:
            vc.loc["Other"] = remaining

    pie = vc.reset_index()
    pie.columns = [dim, "count"]
    pie["share"] = (pie["count"] / pie["count"].sum() * 100).round(1)

    fig = px.pie(pie, names=dim, values="count", hole=0.45, color=dim)
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent:.1%}",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}",
    )
    st.plotly_chart(style_fig(fig, height=520), use_container_width=True)
    st.caption("Modern donut chart in the Breaking Bad palette.")

# ──────────────────────────────────────────────────────────────────────────────
# Page: Price Prediction
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("## 💰 Price Prediction (Schema-Safe)")
    model, schema, cats = load_legacy_model()
    MAKES, MODELS_BY_MAKE = build_cascades(df)

    with st.expander("Feature Schema (detected)", expanded=False):
        st.json({"year": "numeric", "mileage": "numeric", "make": "string", "model": "string"})

    # Inputs
    c1, c2 = st.columns(2)
    with c1:
        y_default = int(df["year"].median()) if "year" in df and df["year"].notna().any() else 2016
        year = st.number_input("year", min_value=1990, max_value=2026, value=y_default, step=1, format="%d")
    with c2:
        m_default = int(df["mileage"].median()) if "mileage" in df and df["mileage"].notna().any() else 60000
        mileage = st.number_input("mileage", min_value=0, max_value=300_000, value=m_default, step=1000, format="%d")

    c3, c4 = st.columns(2)
    with c3:
        if not MAKES:
            st.error("No makes found in the dataset.")
            st.stop()
        make = st.selectbox("make", options=MAKES, index=MAKES.index("Toyota") if "Toyota" in MAKES else 0)
    with c4:
        models = MODELS_BY_MAKE.get(make, [])
        model_name = st.selectbox("model", options=models, index=0 if models else None, disabled=(len(models) == 0))

    ask_price = st.number_input("Optional: Asking/Listing Price (to rate the deal)", min_value=0.0, step=500.0, value=0.0)

    def show_estimate(price: float, p25: float, p75: float, n_comp: int, note: str):
        a, b, c = st.columns([2, 2, 1])
        with a:
            st.success(f"Estimated Price: ${price:,.0f}" if np.isfinite(price) else "Estimated Price: n/a")
        with b:
            st.markdown(
                f"**Fair Market Range (P25–P75)**  \n${p25:,.0f} - ${p75:,.0f}"
                if np.isfinite(p25) and np.isfinite(p75) else "**Fair Market Range (P25–P75)**  \n—"
            )
        with c:
            st.markdown(f"**Comparable Listings Used**  \n{int(n_comp)}")
        st.caption(note)

    if st.button("Predict Price", type="primary", use_container_width=True):
        # Only pass features you trained with (trim removed)
        feat = {"year": int(year), "mileage": int(mileage), "make": str(make), "model": str(model_name)}

        try:
            if model is not None and schema is not None:
                est = predict_with_legacy_gbm(model, schema, cats or {}, feat)
                p50, extra = comps_estimate(df, feat)  # range for context
                show_estimate(est, extra.get("p25", np.nan), extra.get("p75", np.nan), extra.get("n_comp", 0),
                              "Predicted with Legacy GBM — exact model feature order matched.")
            else:
                p50, extra = comps_estimate(df, feat)
                show_estimate(p50, extra.get("p25", np.nan), extra.get("p75", np.nan), extra.get("n_comp", 0),
                              "Predicted from comparable listings (no model available).")
        except Exception as e:
            st.error(f"Prediction failed: {e}")