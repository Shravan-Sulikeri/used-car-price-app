# web/dashboard_modern.py
import os, json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page & style ----------
st.set_page_config(page_title="Used Car Price — Dashboard", page_icon="🚗", layout="wide")

ACCENT = "#A855F7"  # purple accent
CARD_BG = "rgba(255,255,255,0.04)"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.kpi {background: %s; border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 18px 18px;}
.kpi h3 {margin: 0; font-size: 0.95rem; font-weight: 600; color: #9CA3AF;}
.kpi .num {margin-top: 6px; font-size: 2rem; font-weight: 800; color: #E5E7EB; letter-spacing: .3px;}
h2.section {margin: .5rem 0 0.25rem 0; font-weight: 800;}
.caption {color:#9CA3AF; font-size:.9rem;}
</style>
""" % CARD_BG, unsafe_allow_html=True)

# ---------- Data ----------
df = pd.read_csv("data/clean_used_cars.csv")
metrics = {}
if os.path.exists("model/metrics_gbm.json"):
    with open("model/metrics_gbm.json") as f:
        metrics = json.load(f)

make_report = {}
unique_makes_after = int(df["make"].nunique()) if "make" in df.columns else 0
unique_makes_before = None
if os.path.exists("model/make_cleaning_report.json"):
    with open("model/make_cleaning_report.json") as f:
        make_report = json.load(f)
        unique_makes_before = make_report.get("unique_makes_before")
        unique_makes_after = make_report.get("unique_makes_after", unique_makes_after)

# ---------- Header ----------
left, right = st.columns([1.2, 1])
with left:
    st.markdown("<h1 style='margin-bottom:0'>Used Car Price — Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Modern dark UI • interactive filters • GBM model metrics</div>", unsafe_allow_html=True)
with right:
    gbm_mae = metrics.get("mae", None)
    gbm_rmse = metrics.get("rmse", None)
    st.markdown(
        "<div style='text-align:right'>"
        "<span class='caption'>Model: <b>LightGBM</b></span><br/>"
        f"<span class='caption'>MAE: <b>{f'${int(gbm_mae):,}' if gbm_mae else 'N/A'}</b>"
        f" • RMSE: <b>{f'${int(gbm_rmse):,}' if gbm_rmse else 'N/A'}</b></span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------- KPI Cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='kpi'><h3>Listings</h3><div class='num'>{len(df):,}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><h3>Median Price</h3><div class='num'>${int(df['price'].median()):,}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><h3>Median Mileage</h3><div class='num'>{int(df['mileage'].median()):,} mi</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><h3>Unique Makes</h3><div class='num'>{unique_makes_after:,}</div></div>", unsafe_allow_html=True)

# Data quality callout
if unique_makes_before and unique_makes_before >= unique_makes_after:
    drop_pct = 100.0 * (unique_makes_before - unique_makes_after) / max(unique_makes_before, 1)
    st.info(f"**Make hygiene:** {unique_makes_before} → **{unique_makes_after}** unique makes "
            f"(**-{drop_pct:.1f}%** variants) after canonicalization.")

st.markdown("")

# ---------- Filters ----------
flt_left, flt_right = st.columns([1, 3])
with flt_left:
    make_opts = ["(all)"] + sorted(df["make"].dropna().unique().tolist())
    make = st.selectbox("Make", make_opts, index=0)
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    years = st.slider("Year range", yr_min, yr_max, (max(yr_min, 2005), yr_max))
    body_opts = ["(all)"] + sorted(df["body"].dropna().unique().tolist()) if "body" in df.columns else ["(all)"]
    body = st.selectbox("Body type", body_opts, index=0)
mask = df["year"].between(*years)
if make != "(all)":
    mask &= df["make"].eq(make)
if body != "(all)" and "body" in df.columns:
    mask &= df["body"].eq(body)
dff = df.loc[mask].copy()

flt_right.caption(f"Showing **{len(dff):,}** rows after filters")

st.markdown("")

# ---------- Plotly style helper ----------
def fig_style(f):
    f.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        coloraxis_colorbar=None,
    )
    # Only apply accent to trace types that accept a single marker.color
    for tr in f.data:
        t = getattr(tr, "type", "")
        if t in ("bar", "histogram", "scatter"):
            tr.update(marker=dict(color=ACCENT))
        elif t == "pie":
            tr.update(marker=dict(line=dict(color="rgba(255,255,255,0.08)", width=1)))
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
        dff.groupby("model")["price"]
        .median()
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
    by_year = dff.groupby("year")["price"].median().reset_index()
    f = px.bar(by_year, x="year", y="price")
    st.plotly_chart(fig_style(f), use_container_width=True)

# Make share (Top 10, filtered) — donut
with colD:
    st.markdown("<h2 class='section'>Make share (Top 10)</h2>", unsafe_allow_html=True)
    make_share = dff["make"].value_counts().head(10).reset_index()
    make_share.columns = ["make", "count"]
    f = px.pie(
        make_share, names="make", values="count", hole=0.55,
        color_discrete_sequence=px.colors.sequential.Purples
    )
    f.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_style(f), use_container_width=True)

st.markdown("---")
st.caption("Tip: Use the Make/Year/Body filters to explore. Canonicalization reduces duplicate brand spellings (e.g., 'bmw..' → 'BMW').")