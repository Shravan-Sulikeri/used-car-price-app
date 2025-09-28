# web/dashboard_app.py
from __future__ import annotations
import os, json, math, hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ====================== Page & Theme ======================
st.set_page_config(page_title="Used Car Price — Pro Dashboard", page_icon="🚗", layout="wide")

DATA_PATH   = os.getenv("DATA_PATH", "data/clean_used_cars.csv")
MODEL_PATH  = os.getenv("MODEL_PATH", "model/model_gbm.pkl")
PREPROC_PATH= os.getenv("PREPROC_PATH", "model/preprocessor.pkl")
API_URL     = os.getenv("API_URL", "http://localhost:8000")

NEON  = "#39FF14"
BG    = "#0b0f0f"
FG    = "#EAEAEA"
MUTED = "#A6A6A6"

st.markdown(f"""
<style>
:root {{
  --bg: {BG};
  --fg: {FG};
  --muted: {MUTED};
  --neon: {NEON};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--fg); }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }}

.neon-title {{ color: var(--fg); font-size: 1.6rem; font-weight: 800; letter-spacing: .2px; margin: .25rem 0 .75rem 0; }}
.neon-badge {{ display:inline-block; padding:.2rem .6rem; border:1px solid #1f2937; background:#101414; border-radius:10px; color:var(--muted); font-size:.85rem; }}

.kpi {{ border-radius:14px; padding:12px 16px; background:#0f1313; border:1px solid #1f2937;
       box-shadow: inset 0 0 0 1px rgba(57,255,20,0.06); min-height:92px; }}
.kpi h4 {{ margin:0 0 .35rem 0; font-size:.95rem; color:var(--muted); font-weight:600; }}
.kpi .v {{ font-size:1.6rem; font-weight:800; color:var(--fg); }}
.kpi .accent {{ color:var(--neon); text-shadow:0 0 6px rgba(57,255,20,.35); }}

.stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
.stTabs [data-baseweb="tab"] {{ background:#0f1414; border:1px solid #1f2937; border-radius:12px 12px 0 0; padding:.5rem .9rem; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:var(--neon) !important; border-bottom:2px solid var(--neon) !important; }}
.stButton>button, .stDownloadButton>button {{ background: linear-gradient(90deg, #1b4, #19d75e); border:0; color:black; font-weight:800; border-radius:10px; }}
svg text {{ fill: var(--fg) !important; }}
</style>
""", unsafe_allow_html=True)

# ====================== Helpers ======================
@st.cache_data(show_spinner=False)
def load_data(path: str):
    p = Path(path)
    if not p.exists():
        st.error(f"Data not found at **{p}**")
        st.stop()
    df = pd.read_csv(p, low_memory=False)

    # Coerce dtypes
    for c in ("price","year","mileage"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("make","model","body"):
        if c in df:
            s = df[c].astype(str).str.strip()
            s = s.replace({"nan":"Unknown","None":"Unknown","":"Unknown"})
            df[c] = s.fillna("Unknown")

    ref_year = int(df["year"].max()) if "year" in df else 2025
    if "age" not in df and "year" in df:
        df["age"] = ref_year - df["year"]
    if "mileage_per_year" not in df and {"mileage","age"} <= set(df.columns):
        df["mileage_per_year"] = df["mileage"] / df["age"].clip(lower=0.5)
    if "high_mileage" not in df and "mileage" in df:
        df["high_mileage"] = (df["mileage"] > df["mileage"].median()).astype(int)

    # Keep sensible rows
    if {"price","mileage","year"} <= set(df.columns):
        df = df[(df["price"].between(500, 250_000)) &
                (df["mileage"].between(0, 600_000)) &
                (df["year"].between(1980, ref_year+1))].copy()
    return df, ref_year

def neon_fig(fig: go.Figure, title: str|None=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor="#0f1414",
        font=dict(color=FG, size=13),
        legend=dict(bgcolor="#0f1414", bordercolor="#1f2937", borderwidth=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=18, color=NEON)))
    return fig

def agg_top_with_other(s: pd.Series, n=10):
    s = s.fillna("Unknown").replace({"nan":"Unknown","None":"Unknown","":"Unknown"})
    vc = s.value_counts()
    top = vc.head(n)
    other = int(vc.iloc[n:].sum()) if len(vc) > n else 0
    data = top.to_dict()
    if other > 0:
        data["Other"] = other
    return data

@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, f_make, f_model, f_body, year_min, year_max, trim_99=True):
    dff = df.copy()
    if f_make:  dff = dff[dff["make"].isin(f_make)]
    if f_model: dff = dff[dff["model"].isin(f_model)]
    if f_body:  dff = dff[dff["body"].isin(f_body)]
    dff = dff[dff["year"].between(year_min, year_max)]
    if trim_99:
        for c in ("price","mileage"):
            if c in dff:
                hi = dff[c].quantile(0.99)
                dff = dff[dff[c] <= hi]
    return dff

# ====================== Data ======================
df, REF_YEAR = load_data(DATA_PATH)

# ====================== Filters ======================
st.markdown("<div class='neon-badge'>Filters</div>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([2,2,2,3])

with f1:
    f_make = st.multiselect("Make", sorted(df["make"].unique()), placeholder="All")
with f2:
    base_models = df.loc[df["make"].isin(f_make), "model"] if f_make else df["model"]
    f_model = st.multiselect("Model", sorted(base_models.unique()), placeholder="All")
with f3:
    f_body = st.multiselect("Body", sorted(df["body"].unique()), placeholder="All")
with f4:
    y0, y1 = int(df["year"].min()), int(df["year"].max())
    f_year = st.slider("Year range", min_value=y0, max_value=y1, value=(max(y0, 2010), y1))

exp = st.expander("More filters", expanded=False)
with exp:
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        trim_99 = st.checkbox("Trim price & mileage at 99th pct", value=True)
    with c2:
        perf_mode = st.radio("Chart detail", ["Fast (aggregate)", "Detailed (sample)"], index=0)
    with c3:
        dl = st.checkbox("Enable download of filtered CSV", value=False)

dff = apply_filters(df, f_make, f_model, f_body, f_year[0], f_year[1], trim_99)
st.caption(f"Showing **{len(dff):,}** rows after filters")

# ====================== KPIs ======================
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(box, title: str, value: str, accent: bool=False):
    content = f"<span class='accent'>{value}</span>" if accent else value
    html = f"<div class='kpi'><h4>{title}</h4><div class='v'>{content}</div></div>"
    with box:
        st.markdown(html, unsafe_allow_html=True)

kpi(k1, "Listings", f"{len(dff):,}")
kpi(k2, "Avg Price", f"${dff['price'].mean():,.0f}" if "price" in dff else "—")
kpi(k3, "Top Make",  dff["make"].mode().iat[0] if not dff.empty else "—", accent=True)
kpi(k4, "Top Model", dff["model"].mode().iat[0] if not dff.empty else "—")
kpi(k5, "Top Body",  dff["body"].mode().iat[0] if not dff.empty else "—")

# ====================== Tabs ======================
tab_market, tab_compare, tab_top, tab_predict = st.tabs(["Market share", "Comparisons", "Top lists", "Predict"])

# ----- Market share -----
with tab_market:
    st.markdown("<h3 class='neon-title'>Market share</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        make_counts = agg_top_with_other(dff["make"], 10)
        fig = go.Figure(go.Pie(
            labels=list(make_counts.keys()),
            values=list(make_counts.values()),
            hole=0.55, textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} listings<extra></extra>",
        ))
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)))
        st.plotly_chart(neon_fig(fig, "Make share (Top 10 + Other)"), use_container_width=True)

    with c2:
        body_counts = agg_top_with_other(dff["body"], 10)
        fig = go.Figure(go.Pie(
            labels=list(body_counts.keys()),
            values=list(body_counts.values()),
            hole=0.55, textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} listings<extra></extra>",
        ))
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)))
        st.plotly_chart(neon_fig(fig, "Body share (Top 10 + Other)"), use_container_width=True)

# ----- Comparisons -----
with tab_compare:
    st.markdown("<h3 class='neon-title'>Comparisons</h3>", unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)

    with cc1:
        if dff.empty:
            st.info("No data for current filters.")
        else:
            grp = dff.groupby("year")["price"].agg(
                med="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75)
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grp["year"], y=grp["p25"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=grp["year"], y=grp["p75"], mode="lines",
                                     fill="tonexty", fillcolor="rgba(57,255,20,.12)",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=grp["year"], y=grp["med"], mode="lines+markers",
                                     line=dict(color=NEON, width=3),
                                     marker=dict(size=6),
                                     name="Median price",
                                     hovertemplate="Year %{x}<br>Median: $%{y:,.0f}<extra></extra>"))
            st.plotly_chart(neon_fig(fig, "Price vs Year (median + IQR)"), use_container_width=True)

    with cc2:
        if dff.empty:
            st.info("No data for current filters.")
        else:
            if perf_mode.startswith("Fast"):
                d2 = dff[["mileage","price"]].dropna().copy()
                d2["bin"] = pd.cut(d2["mileage"], bins=60)
                mids = d2["bin"].apply(lambda b: b.mid)
                med = d2.groupby(mids)["price"].median().reset_index().rename(columns={"bin":"mileage","price":"price"})
                med.columns = ["mileage","price"]
                fig = px.line(med, x="mileage", y="price")
                fig.update_traces(line=dict(color=NEON, width=3))
                st.plotly_chart(neon_fig(fig, "Price vs Mileage (median by bins)"), use_container_width=True)
            else:
                sam = dff.sample(min(80_000, len(dff)), random_state=42)
                fig = px.scatter(sam, x="mileage", y="price", opacity=0.35)
                fig.update_traces(marker=dict(size=5))
                st.plotly_chart(neon_fig(fig, "Price vs Mileage (sample)"), use_container_width=True)

# ----- Top lists -----
with tab_top:
    st.markdown("<h3 class='neon-title'>Top lists</h3>", unsafe_allow_html=True)
    tt1, tt2 = st.columns(2)

    with tt1:
        tm = dff["model"].value_counts().head(10).reset_index()
        tm.columns = ["model","count"]
        fig = px.bar(tm, x="model", y="count", text="count")
        fig.update_traces(marker_color=NEON)
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(neon_fig(fig, "Top 10 models (by units)"), use_container_width=True)

    with tt2:
        tb = dff["body"].value_counts().head(10).reset_index()
        tb.columns = ["body","count"]
        fig = px.bar(tb, x="body", y="count", text="count")
        fig.update_traces(marker_color=NEON)
        fig.update_layout(xaxis_tickangle=-10)
        st.plotly_chart(neon_fig(fig, "Top 10 body types (by units)"), use_container_width=True)

    st.markdown("<div class='neon-title' style='margin-top:1rem'>Median price by model (Top 15 by units)</div>", unsafe_allow_html=True)
    top15_idx = dff["model"].value_counts().head(15).index
    g = dff[dff["model"].isin(top15_idx)].groupby("model")["price"].median().sort_values(ascending=False)
    fig = px.bar(g.reset_index(), x="model", y="price", text="price")
    fig.update_traces(marker_color=NEON, texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, yaxis_title="Median price (USD)")
    st.plotly_chart(neon_fig(fig, "Median price — top models"), use_container_width=True)

    if dl:
        st.download_button("Download filtered CSV",
            data=dff.to_csv(index=False).encode("utf-8"),
            file_name="used_cars_filtered.csv",
            mime="text/csv")

# ----- Predictor -----
with tab_predict:
    st.markdown("<h3 class='neon-title'>Price Predictor</h3>", unsafe_allow_html=True)
    pc1, pc2, pc3, pc4, pc5 = st.columns([1.4,1.4,1.2,1,1])

    with pc1:
        p_make = st.selectbox("Make", sorted(df["make"].unique()))
    with pc2:
        models_for_make = sorted(df.loc[df["make"].eq(p_make), "model"].unique())
        p_model = st.selectbox("Model", models_for_make)
    with pc3:
        p_body = st.selectbox("Body", sorted(df["body"].unique()))
    with pc4:
        p_year = st.number_input("Year",
                                 min_value=int(df["year"].min()),
                                 max_value=int(df["year"].max()),
                                 value=int(np.clip(REF_YEAR-4, df["year"].min(), df["year"].max())),
                                 step=1)
    with pc5:
        p_miles = st.number_input("Mileage",
                                  min_value=0,
                                  max_value=int(df["mileage"].max()),
                                  value=int(df["mileage"].median()),
                                  step=500)

    def local_or_api_predict(year, mileage, make, model, body):
        age = REF_YEAR - int(year)
        mpgyr = mileage / (age if age > 0 else 0.5)
        high_m = int(mileage > df["mileage"].median())
        X = pd.DataFrame([{
            "year": int(year),
            "mileage": float(mileage),
            "age": int(age),
            "mileage_per_year": float(mpgyr),
            "high_mileage": high_m,
            "make": str(make),
            "model": str(model),
            "body": str(body),
        }])

        # Local model first (with/without preprocessor)
        try:
            mdl = joblib.load(MODEL_PATH)
            try:
                pre = joblib.load(PREPROC_PATH)
                Xt = pre.transform(X)
            except Exception:
                Xt = X
            y = mdl.predict(Xt)
            price = float(y[0])
            if math.isfinite(price):
                return True, price, "local"
        except Exception:
            pass

        # API fallback
        try:
            r = requests.post(f"{API_URL}/predict",
                              headers={"content-type":"application/json"},
                              data=json.dumps({
                                  "year": int(year),
                                  "mileage": int(mileage),
                                  "make": make,
                                  "model": model,
                                  "body": body
                              }), timeout=4.0)
            if r.ok and r.json().get("ok"):
                return True, float(r.json()["price_usd"]), "api"
        except Exception:
            pass

        return False, None, None

    if st.button("Predict price", type="primary"):
        ok, price, mode = local_or_api_predict(p_year, p_miles, p_make, p_model, p_body)
        if ok:
            st.success(f"Estimated price: **${price:,.0f}**  \n*({mode})*")
        else:
            st.error("Could not predict. Ensure model pickle or API is available.")

st.caption("Neon theme • Fast aggregates for big slices • Detailed scatter via sampling • Local/HTTP prediction.")