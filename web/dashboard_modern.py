# web/dashboard_modern.py
import os, json, sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Used Car Price — Dashboard", page_icon="🚗", layout="wide")

# ---------- Theme (black + neon green) ----------
PRIMARY_BG = "#000000"   # black
ACCENT     = "#7CFC00"   # neon green
CARD_BG    = "rgba(255,255,255,0.06)"  # subtle glass

st.markdown(f"""
<style>
:root {{
  --bg: {PRIMARY_BG};
  --accent: {ACCENT};
  --text: #ECECEC;
  --muted: #A7A7A7;
}}
/* App background & header */
[data-testid="stAppViewContainer"] {{ background-color: var(--bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
/* Typography */
h1, h2, h3, h4, h5, h6 {{ color: var(--text); }}
.caption {{ color: var(--muted); font-size:.9rem; }}
/* Layout */
.block-container {{ padding-top: 1.1rem; padding-bottom: 1.8rem; }}
/* KPI Cards */
.kpi {{
  background: {CARD_BG};
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 18px 18px;
}}
.kpi h3 {{ margin: 0; font-size: 0.95rem; font-weight: 600; color: var(--muted); }}
.kpi .num {{ margin-top: 6px; font-size: 2rem; font-weight: 800; color: var(--text); letter-spacing: .3px; }}
/* Inputs: borders & focus */
div[data-baseweb="select"] > div {{ border-color: rgba(255,255,255,0.18); }}
div[data-baseweb="select"] > div:hover {{ border-color: rgba(255,255,255,0.30); }}
.stSlider > div [role='slider'] {{ background: var(--accent) !important; }}
.stSlider > div [data-baseweb='slider'] > div > div {{ background: rgba(124,252,0,.25) !important; }}
/* Buttons */
.stButton>button {{ background: var(--accent); color:#000; border-radius: 10px; border: 0; font-weight: 700; }}
.stButton>button:hover {{ filter: brightness(0.9); }}
</style>
""", unsafe_allow_html=True)

# ---------- Canonicalizer for 'make' (runtime safety) ----------
sys.path.append("model")
from standardize_makes import canonicalize_make

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_data_and_metrics():
    df = pd.read_csv("data/clean_used_cars.csv", low_memory=False)

    # Enforce clean 'make' for UI (even if CSV is stale)
    if "make" in df.columns:
        df["make"] = df["make"].apply(canonicalize_make)
        df = df.dropna(subset=["make"])

    metrics = {}
    if os.path.exists("model/metrics_gbm.json"):
        with open("model/metrics_gbm.json") as f:
            metrics = json.load(f)

    # Make report (KPI/info)
    unique_makes_before = None
    unique_makes_after = int(df["make"].nunique()) if "make" in df.columns else 0
    make_report = {}
    if os.path.exists("model/make_cleaning_report.json"):
        with open("model/make_cleaning_report.json") as f:
            make_report = json.load(f)
            unique_makes_before = make_report.get("unique_makes_before")
            unique_makes_after = make_report.get("unique_makes_after", unique_makes_after)

    return df, metrics, make_report, unique_makes_before, unique_makes_after

df, metrics, make_report, unique_makes_before, unique_makes_after = load_data_and_metrics()

# ---------- Header ----------
left, right = st.columns([1.2, 1])
with left:
    st.markdown("<h1 style='margin-bottom:0'>Used Car Price — Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Modern dark UI • interactive filters • LightGBM metrics</div>", unsafe_allow_html=True)
with right:
    gbm_mae = metrics.get("mae")
    gbm_rmse = metrics.get("rmse")
    st.markdown(
        "<div style='text-align:right'>"
        "<span class='caption'>Model: <b>LightGBM</b></span><br/>"
        f"<span class='caption'>MAE: <b>{f'${int(gbm_mae):,}' if gbm_mae else 'N/A'}</b>"
        f" • RMSE: <b>{f'${int(gbm_rmse):,}' if gbm_rmse else 'N/A'}</b></span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------- KPI Cards (global) ----------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='kpi'><h3>Total Listings</h3><div class='num'>{len(df):,}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><h3>Median Price</h3><div class='num'>${int(df['price'].median()):,}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><h3>Median Mileage</h3><div class='num'>{int(df['mileage'].median()):,} mi</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><h3>Unique Makes</h3><div class='num'>{unique_makes_after:,}</div></div>", unsafe_allow_html=True)

# Data quality callout
if unique_makes_before and unique_makes_before >= unique_makes_after:
    drop_pct = 100.0 * (unique_makes_before - unique_makes_after) / max(unique_makes_before, 1)
    st.info(
        f"**Make hygiene:** {unique_makes_before} → **{unique_makes_after}** unique makes "
        f"(**-{drop_pct:.1f}%** variants) after canonicalization."
    )

st.markdown("")

# ---------- Filters ----------
flt_left, flt_right = st.columns([1, 3])
with flt_left:
    # Make
    make_opts = ["(all)"] + sorted(df["make"].dropna().unique().tolist())
    make = st.selectbox("Make", make_opts, index=0, key="make")

    # Model (depends on Make)
    if make != "(all)":
        models_in_make = (
            df.loc[df["make"].eq(make), "model"]
              .dropna()
              .astype(str)
              .unique()
              .tolist()
        )
        model_opts = ["(all)"] + sorted(models_in_make)
        model = st.selectbox("Model", model_opts, index=0, key="model")
    else:
        st.selectbox("Model", ["(pick a make)"], index=0, key="model_disabled", disabled=True)
        model = "(all)"

    # Year + Body
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    years = st.slider("Year range", yr_min, yr_max, (max(yr_min, 2005), yr_max))
    body_opts = ["(all)"] + (sorted(df["body"].dropna().unique().tolist()) if "body" in df.columns else [])
    body = st.selectbox("Body type", body_opts, index=0)

# Build filtered view
mask = df["year"].between(*years)
if make != "(all)":
    mask &= df["make"].eq(make)
    if model != "(all)" and "model" in df.columns:
        mask &= df["model"].astype(str).eq(model)
if body != "(all)" and "body" in df.columns:
    mask &= df["body"].eq(body)

dff = df.loc[mask].copy()

# Show filter summary
flt_right.caption(f"Showing **{len(dff):,}** rows after filters")

st.markdown("")

# ---------- Sales KPIs (current view) ----------
k1, k2 = st.columns(2)
k1.markdown(f"<div class='kpi'><h3>Units (current view)</h3><div class='num'>{len(dff):,}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><h3>Revenue (current view)</h3><div class='num'>${int(dff['price'].sum()):,}</div></div>", unsafe_allow_html=True)
st.markdown("")

# ---------- Plotly style helper ----------
def fig_style(f):
    f.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#ECECEC"),
        coloraxis_colorbar=None,
    )
    # accent markers for applicable traces
    for tr in f.data:
        t = getattr(tr, "type", "")
        if t in ("bar", "histogram", "scatter"):
            tr.update(marker=dict(color=ACCENT))
        elif t == "pie":
            tr.update(marker=dict(line=dict(color="rgba(255,255,255,0.12)", width=1)))
    return f

# ---------- Charts ----------
colA, colB = st.columns(2)

# Price distribution (trim heavy tail for view)
with colA:
    st.markdown("<h2 class='section'>Price distribution</h2>", unsafe_allow_html=True)
    f = px.histogram(dff.assign(price=dff["price"].clip(upper=120_000)), x="price", nbins=30)
    st.plotly_chart(fig_style(f), use_container_width=True)

# Top 10 models by median price (filtered)
with colB:
    st.markdown("<h2 class='section'>Top 10 models (median price)</h2>", unsafe_allow_html=True)
    top_models = (
        dff.groupby("model", dropna=False)["price"]
        .median()
        .dropna()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    f = px.bar(top_models, orientation="h", labels={"value": "Median price", "index": "Model"})
    st.plotly_chart(fig_style(f), use_container_width=True)

colC, colD = st.columns(2)

# Median price by year (filtered)
with colC:
    st.markdown("<h2 class='section'>Median price by year</h2>", unsafe_allow_html=True)
    by_year = (
        dff.groupby("year", dropna=False)["price"]
        .median()
        .dropna()
        .reset_index()
    )
    f = px.bar(by_year, x="year", y="price")
    st.plotly_chart(fig_style(f), use_container_width=True)

# Make share (Top 10, filtered) — donut
with colD:
    st.markdown("<h2 class='section'>Make share (Top 10)</h2>", unsafe_allow_html=True)
    make_share = dff["make"].value_counts().head(10).reset_index()
    make_share.columns = ["make", "count"]
    f = px.pie(
        make_share, names="make", values="count", hole=0.55,
        color_discrete_sequence=px.colors.sequential.Greens  # green family to match theme
    )
    f.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_style(f), use_container_width=True)

st.markdown("<h2 class='section'>Sales (units)</h2>", unsafe_allow_html=True)
sA, sB = st.columns(2)

# Sales by make (units) — Top 15
with sA:
    make_counts = (
        dff["make"].dropna()
        .value_counts()
        .head(15)
        .sort_values()
        .reset_index()
    )
    make_counts.columns = ["make", "units"]
    f = px.bar(
        make_counts, x="units", y="make", orientation="h",
        labels={"units": "Units", "make": "Make"},
    )
    st.plotly_chart(fig_style(f), use_container_width=True)

# Sales by model (units) — Top 15 (respects Make filter via dff)
with sB:
    model_counts = (
        dff["model"].dropna()
        .value_counts()
        .head(15)
        .sort_values()
        .reset_index()
    )
    model_counts.columns = ["model", "units"]
    f = px.bar(
        model_counts, x="units", y="model", orientation="h",
        labels={"units": "Units", "model": "Model"},
    )
    st.plotly_chart(fig_style(f), use_container_width=True)

st.markdown("---")
st.caption("Tip: Use Make / Year / Body filters. All brand spellings are canonicalized at load (e.g., 'bmw 335i' → 'BMW').")