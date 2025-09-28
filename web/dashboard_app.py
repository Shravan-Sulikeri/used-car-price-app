# web/dashboard_app.py
import os
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Used Car Price — Dashboard", page_icon="🚗", layout="wide")

CSV_PATH = os.getenv("DATA_CSV", "/workspaces/used-car-price-app/data/clean_used_cars.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspaces/used-car-price-app/model/model_gbm.pkl")
PREPROC_PATH = os.getenv("PREPROC_PATH", "/workspaces/used-car-price-app/model/preprocessor.pkl")

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Basic typing
    for col in ("price", "year", "mileage"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Minimal feature engineering (defensive)
    if "year" in df and "age" not in df:
        ref_year = pd.Timestamp.today().year
        df["age"] = ref_year - df["year"]
    if {"mileage", "age"}.issubset(df.columns) and "mileage_per_year" not in df:
        df["mileage_per_year"] = (df["mileage"] / df["age"].replace(0, np.nan)).clip(upper=500_000)
    if "mileage_per_year" in df and "high_mileage" not in df:
        df["high_mileage"] = (df["mileage_per_year"] > 20_000).astype(int)

    # Canonicalize strings
    for c in ("make", "model", "body"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

ref_year = int(df["year"].max()) if "year" in df else pd.Timestamp.today().year

# ---------- Artifacts (predictor) ----------
try:
    import joblib
    @st.cache_resource(show_spinner=False)
    def load_artifacts():
        if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH)):
            return None, None
        pre = joblib.load(PREPROC_PATH)
        model = joblib.load(MODEL_PATH)
        return model, pre
    model, preproc = load_artifacts()
except Exception as e:
    model, preproc = None, None

# ---------- Header / KPIs ----------
left, right = st.columns([1.2, 1], vertical_alignment="center")
with left:
    st.markdown("## Used Car Price — Dashboard")
    st.caption(f"Loaded: `{CSV_PATH}` • Rows: **{len(df):,}**")
with right:
    k1, k2, k3 = st.columns(3)
    k1.metric("Median price", f"${int(df['price'].median()):,}" if "price" in df else "—")
    k2.metric("Median mileage", f"{int(df['mileage'].median()):,} mi" if "mileage" in df else "—")
    k3.metric("Unique makes", f"{df['make'].nunique():,}" if "make" in df else "—")

st.divider()

# ---------- Filters ----------
flt1, flt2, flt3, flt4 = st.columns(4)

all_makes = ["(all)"] + (sorted(df["make"].dropna().unique().tolist()) if "make" in df else [])
make = flt1.selectbox("Make", all_makes, index=0)

if make != "(all)" and "model" in df:
    model_list = ["(all)"] + sorted(df.loc[df["make"].eq(make), "model"].dropna().unique().tolist())
else:
    model_list = ["(all)"] + (sorted(df["model"].dropna().unique().tolist()) if "model" in df else [])
model_sel = flt2.selectbox("Model", model_list, index=0)

if "body" in df:
    bodies = ["(all)"] + sorted(df["body"].dropna().unique().tolist())
    body = flt3.selectbox("Body", bodies, index=0)
else:
    body = "(all)"

if "year" in df:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    y0, y1 = flt4.slider("Year range", min_value=yr_min, max_value=yr_max, value=(max(yr_min, 2005), yr_max))
else:
    y0, y1 = None, None

mask = pd.Series(True, index=df.index)
if "year" in df and y0 is not None:
    mask &= df["year"].between(y0, y1)
if make != "(all)":
    mask &= df["make"].eq(make)
if model_sel != "(all)":
    mask &= df["model"].eq(model_sel)
if body != "(all)" and "body" in df:
    mask &= df["body"].eq(body)

dff = df.loc[mask].copy()
st.caption(f"Showing **{len(dff):,}** rows after filters")

st.divider()

# ---------- Charts ----------
def style(fig):
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig

c1, c2 = st.columns(2)

# Price vs Mileage
if {"price", "mileage"}.issubset(dff.columns):
    f = px.scatter(
        dff.sample(min(len(dff), 800), random_state=42),
        x="mileage", y="price",
        color="make" if make == "(all)" else None,
        hover_data=[c for c in ["make", "model", "year"] if c in dff.columns],
        title="Price vs. Mileage"
    )
    c1.plotly_chart(style(f), use_container_width=True)
else:
    c1.info("Need `price` and `mileage` for the scatter chart.")

# Median price by year
if {"price", "year"}.issubset(dff.columns):
    by_year = dff.groupby("year", as_index=False)["price"].median()
    f = px.line(by_year, x="year", y="price", markers=True, title="Median Price by Year")
    c2.plotly_chart(style(f), use_container_width=True)
else:
    c2.info("Need `price` and `year` for the trend chart.")

c3, c4 = st.columns(2)

# Make share
if "make" in dff:
    make_share = dff["make"].value_counts().head(10).to_frame("count").reset_index()
    make_share.columns = ["make", "count"]
    f = px.pie(make_share, names="make", values="count", hole=0.55, title="Make Share (Top 10)")
    c3.plotly_chart(style(f), use_container_width=True)

# Model share
if "model" in dff:
    model_share = dff["model"].value_counts().head(10).to_frame("count").reset_index()
    model_share.columns = ["model", "count"]
    f = px.pie(model_share, names="model", values="count", hole=0.55, title="Model Share (Top 10)")
    c4.plotly_chart(style(f), use_container_width=True)

st.divider()

# ---------- Predictor ----------
st.markdown("### 💡 Price Predictor")

if (model is None) or (preproc is None):
    st.info(
        "Prediction artifacts not found. Drop these files in place and rerun:\n\n"
        f"- `{PREPROC_PATH}`\n- `{MODEL_PATH}`\n\n"
        "Or set `MODEL_PATH` / `PREPROC_PATH` env vars."
    )
else:
    p1, p2, p3 = st.columns(3)

    # Make → Model for inputs (exclude '(all)')
    makes_in = sorted(df["make"].dropna().unique().tolist())
    make_in = p1.selectbox("Make (predict)", makes_in, index=makes_in.index(make) if make in makes_in else 0)

    if "model" in df:
        models_for_make = sorted(df.loc[df["make"].eq(make_in), "model"].dropna().unique().tolist())
    else:
        models_for_make = sorted(df["model"].dropna().unique().tolist())
    model_in = p1.selectbox("Model (predict)", models_for_make, index=0)

    # Body optional
    if "body" in df:
        bodies_in = ["(none)"] + sorted(df["body"].dropna().unique().tolist())
        body_in = p1.selectbox("Body (optional)", bodies_in, index=0)
        body_in = None if body_in == "(none)" else body_in
    else:
        body_in = None

    # Year / mileage
    yr_min, yr_max = (int(df["year"].min()), int(df["year"].max())) if "year" in df else (2005, ref_year)
    year_in = p2.slider("Year", min_value=yr_min, max_value=yr_max, value=min(ref_year, max(yr_min, ref_year - 4)))
    mileage_in = p3.number_input("Mileage (mi)", min_value=0, max_value=500_000, value=45_000, step=1_000)

    # Compute engineered features for the single row
    age_in = max(0, ref_year - int(year_in))
    miles_per_year = float(mileage_in) / (age_in if age_in > 0 else 1)
    high_mi = 1 if miles_per_year > 20_000 else 0

    row = {
        "year": int(year_in),
        "mileage": float(mileage_in),
        "age": int(age_in),
        "mileage_per_year": float(miles_per_year),
        "high_mileage": int(high_mi),
        "make": make_in,
        "model": model_in,
        "body": body_in,
    }
    x = pd.DataFrame([row])

    if st.button("Estimate price", type="primary"):
        try:
            Xt = preproc.transform(x)  # to model space
            pred = model.predict(Xt)
            # Heuristic: if model was trained on log-price, values are usually small (< 1e3)
            price = float(np.expm1(pred[0])) if pred[0] < 1000 else float(pred[0])
            st.success(f"Estimated price: **${price:,.0f}**")
            with st.expander("Show inputs"):
                st.json(row)
        except Exception as e:
            st.error(f"Failed to predict: {e}")

st.divider()

# ---------- Data preview ----------
st.markdown("#### Preview")
st.dataframe(dff.head(500), use_container_width=True)