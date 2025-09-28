# web/dashboard_app.py
# Streamlit dashboard for Used Car Price — ML & Analytics

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from typing import Dict, List

# ======= CONFIG =======
st.set_page_config(
    page_title="Used Car Price — ML & Analytics",
    page_icon="🚗",
    layout="wide",
)

DATA_CSV = "/workspaces/used-car-price-app/data/clean_used_cars_curated.csv"
MODEL_PKL = "/workspaces/used-car-price-app/model/best_model.pkl"
SCHEMA_JSON = "/workspaces/used-car-price-app/model/schema_best.json"
METRICS_JSON = "/workspaces/used-car-price-app/model/metrics_best.json"

# ======= CACHE LAYERS =======
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # basic dtype hygiene (helps filters)
    if "year" in df: df["year"] = df["year"].astype(int)
    if "mileage" in df: df["mileage"] = df["mileage"].astype(int)
    for col in ["make","model","body"]:
        if col in df:
            df[col] = df[col].astype("string")
    return df

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PKL) if os.path.exists(MODEL_PKL) else None
    schema = None
    if os.path.exists(SCHEMA_JSON):
        with open(SCHEMA_JSON) as f:
            schema = json.load(f)
    metrics = None
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            metrics = json.load(f)
    return model, schema, metrics

@st.cache_data(show_spinner=False)
def cached_options(df: pd.DataFrame):
    opts = {
        "makes": sorted(df["make"].dropna().unique().tolist()) if "make" in df else [],
        "bodies": sorted(df["body"].dropna().unique().tolist()) if "body" in df else [],
        "years": sorted(df["year"].dropna().unique().astype(int).tolist()) if "year" in df else [],
        "models_by_make": {},
    }
    if "make" in df and "model" in df:
        g = df.groupby("make")["model"].apply(lambda s: sorted(s.dropna().unique().tolist()))
        opts["models_by_make"] = g.to_dict()
    return opts

# ======= FEATURE ENGINEERING TO MATCH TRAINING =======
def _engineer(df: pd.DataFrame, ref_year: int):
    df = df.copy()
    # engineered features used in training script
    if "year" in df and "age" not in df:
        df["age"] = ref_year - df["year"]
    if {"mileage","age"}.issubset(df.columns) and "miles_per_year" not in df:
        df["miles_per_year"] = df["mileage"] / df["age"].clip(lower=1)
    if "mileage" in df and "log_mileage" not in df:
        df["log_mileage"] = np.log1p(df["mileage"])
    if {"age","miles_per_year"}.issubset(df.columns) and "age_x_mpy" not in df:
        df["age_x_mpy"] = df["age"] * df["miles_per_year"]
    return df

def prepare_features(df_in: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """Ensure columns/dtypes/order match training schema."""
    df = _engineer(df_in, schema.get("ref_year", 2025)).copy()

    # ensure every training column exists
    for c in schema["feature_order"]:
        if c not in df.columns:
            df[c] = np.nan

    # order columns
    df = df[schema["feature_order"]]

    # set dtypes & fill medians for numerics
    for c in schema.get("categorical_cols", []):
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c, med in schema.get("numeric_fill_median", {}).items():
        if c in df.columns:
            df[c] = df[c].fillna(med)

    return df

def predict_prices(df_in: pd.DataFrame) -> np.ndarray:
    model, schema, _ = load_artifacts()
    if model is None or schema is None:
        raise RuntimeError("Model or schema not found. Train and save artifacts first.")
    X = prepare_features(df_in, schema)
    pred_log = model.predict(X)
    return np.expm1(pred_log)  # back to USD

# ======= SMALL UTILS =======
def format_currency(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def k_formatter(x: float) -> str:
    return f"{x/1000:.1f}k"

def describe_block(df: pd.DataFrame) -> pd.DataFrame:
    sel = ["price", "year", "mileage"]
    out = {}
    for c in sel:
        if c in df:
            s = df[c].dropna()
            out[c] = {
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
    return pd.DataFrame(out)

# ======= SIDEBAR =======
df = load_data()
model, schema, metrics = load_artifacts()
opts = cached_options(df)

with st.sidebar:
    st.header("Filters")
    make = st.selectbox("Make", ["All"] + opts["makes"])
    models = opts["models_by_make"].get(make, []) if make != "All" else sorted(df["model"].dropna().unique()) if "model" in df else []
    model_sel = st.selectbox("Model", ["All"] + models) if "model" in df else "All"

    year_min, year_max = (min(opts["years"]), max(opts["years"])) if opts["years"] else (2000, 2025)
    year_range = st.slider("Year range", min_value=int(year_min), max_value=int(year_max),
                           value=(int(year_min), int(year_max)), step=1)

    mileage_max = int(df["mileage"].max()) if "mileage" in df else 300_000
    mileage_sel = st.slider("Mileage (max)", min_value=0, max_value=mileage_max, value=min(120_000, mileage_max), step=1000)

    body_sel = st.multiselect("Body type", options=opts["bodies"], default=opts["bodies"][:5] if len(opts["bodies"])>0 else [])

    st.divider()
    st.caption("Model artifacts")
    st.write("Model:", "✅ found" if model is not None else "❌ missing")
    st.write("Schema:", "✅ found" if schema is not None else "❌ missing")
    if metrics:
        st.write("MAE (USD):", format_currency(metrics.get("mae_usd", float("nan"))))

# apply filters
q = df.copy()
if make != "All":
    q = q[q["make"] == make]
if "model" in q and model_sel != "All":
    q = q[q["model"] == model_sel]
if "year" in q:
    q = q[(q["year"] >= year_range[0]) & (q["year"] <= year_range[1])]
if "mileage" in q:
    q = q[q["mileage"] <= mileage_sel]
if body_sel and "body" in q:
    q = q[q["body"].isin(body_sel)]

# ======= HEADER =======
st.title("Used Car Price — ML & Analytics")
st.caption("Explore the market, filter by make/model/year, and generate predictions. 🚗📈")

# ======= TABS =======
tab_overview, tab_explore, tab_predict, tab_explain, tab_data, tab_about = st.tabs(
    ["Overview", "Explore", "Predictions", "Explain", "Data", "About"]
)

# ======= OVERVIEW =======
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{q.shape[0]:,}")
    c2.metric("Median Price", format_currency(q["price"].median() if "price" in q else float("nan")))
    c3.metric("Median Year", f"{int(q['year'].median())}" if "year" in q and q.shape[0] else "—")
    c4.metric("Median Mileage", k_formatter(q["mileage"].median()) if "mileage" in q and q.shape[0] else "—")

    st.subheader("Distributions")
    cA, cB, cC = st.columns(3)
    if "price" in q:
        chart_price = alt.Chart(q).mark_bar().encode(
            alt.X("price:Q", bin=alt.Bin(maxbins=50), title="Price (USD)"),
            alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Cars")]
        ).properties(height=250)
        cA.altair_chart(chart_price, use_container_width=True)
    if "year" in q:
        chart_year = alt.Chart(q).mark_bar().encode(
            alt.X("year:Q", bin=alt.Bin(maxbins=40), title="Year"),
            alt.Y("count()", title="Count"),
        ).properties(height=250)
        cB.altair_chart(chart_year, use_container_width=True)
    if "mileage" in q:
        chart_miles = alt.Chart(q).mark_bar().encode(
            alt.X("mileage:Q", bin=alt.Bin(maxbins=50), title="Mileage"),
            alt.Y("count()", title="Count"),
        ).properties(height=250)
        cC.altair_chart(chart_miles, use_container_width=True)

    st.subheader("Top Makes / Models")
    cM, cMo = st.columns(2)
    if "make" in q:
        top_make = (q["make"].value_counts().head(15).reset_index())
        top_make.columns = ["make", "count"]
        cM.altair_chart(
            alt.Chart(top_make).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("make:N", sort="-x", title="Make"),
                tooltip=["make","count"]
            ).properties(height=350),
            use_container_width=True
        )
    if "model" in q:
        top_model = (q["model"].value_counts().head(15).reset_index())
        top_model.columns = ["model", "count"]
        cMo.altair_chart(
            alt.Chart(top_model).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("model:N", sort="-x", title="Model"),
                tooltip=["model","count"]
            ).properties(height=350),
            use_container_width=True
        )

# ======= EXPLORE =======
with tab_explore:
    st.subheader("Price vs Year / Mileage")
    c1, c2 = st.columns(2)
    if {"price","year"}.issubset(q.columns):
        c1.altair_chart(
            alt.Chart(q.sample(min(5000, len(q)), random_state=42)).mark_circle(size=30, opacity=0.5).encode(
                x=alt.X("year:Q", title="Year"),
                y=alt.Y("price:Q", title="Price (USD)"),
                color=alt.Color("make:N", legend=None),
                tooltip=["make","model","year","mileage","price"]
            ).interactive().properties(height=400),
            use_container_width=True
        )
    if {"price","mileage"}.issubset(q.columns):
        c2.altair_chart(
            alt.Chart(q.sample(min(5000, len(q)), random_state=42)).mark_circle(size=30, opacity=0.5).encode(
                x=alt.X("mileage:Q", title="Mileage"),
                y=alt.Y("price:Q", title="Price (USD)"),
                color=alt.Color("make:N", legend=None),
                tooltip=["make","model","year","mileage","price"]
            ).interactive().properties(height=400),
            use_container_width=True
        )

# ======= PREDICTIONS =======
with tab_predict:
    st.subheader("🔮 Predict Price")

    if (model is None) or (schema is None):
        st.warning("Model or schema not found. Train first (see model/).")
    else:
        left, right = st.columns([1,1])

        with left:
            st.caption("Single car")
            # dynamic options from dataset to avoid bad categories
            makes = ["(choose)"] + opts["makes"]
            make_in = st.selectbox("Make", makes, index=0)
            models_for_make = opts["models_by_make"].get(make_in, []) if make_in != "(choose)" else []
            model_in = st.selectbox("Model", ["(choose)"] + models_for_make, index=0)
            bodies = ["(choose)"] + opts["bodies"]
            body_in = st.selectbox("Body", bodies, index=0)

            years = opts["years"]
            year_in = st.number_input("Year", min_value=min(years) if years else 1985,
                                      max_value=max(years) if years else 2025,
                                      value=2017, step=1)
            mileage_in = st.number_input("Mileage", min_value=0, max_value=300_000, value=80_000, step=500)

            row = pd.DataFrame([{
                "make": None if make_in == "(choose)" else make_in,
                "model": None if model_in == "(choose)" else model_in,
                "body": None if body_in == "(choose)" else body_in,
                "year": int(year_in),
                "mileage": int(mileage_in),
            }])

            if st.button("Predict price (USD)"):
                try:
                    pred = predict_prices(row)
                    st.success(format_currency(pred[0]))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with right:
            st.caption("Batch scoring")
            up = st.file_uploader("Upload CSV (columns: make, model, body, year, mileage)", type=["csv"])
            if up:
                try:
                    df_up = pd.read_csv(up)
                    preds = predict_prices(df_up)
                    out = df_up.copy()
                    out["pred_price"] = preds
                    st.dataframe(out.head(25), use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions.csv",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# ======= EXPLAIN =======
with tab_explain:
    st.subheader("🧠 Explain (quick)")
    if model is None:
        st.info("Train a model to view feature importance.")
    else:
        # Try to grab feature importances if available
        if hasattr(model, "feature_importances_"):
            # Feature names are the training columns
            feat_names = schema["feature_order"] if schema and "feature_order" in schema else [f"f{i}" for i in range(len(model.feature_importances_))]
            imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_.astype(float)})
            imp = imp.sort_values("importance", ascending=False).head(20)

            chart_imp = alt.Chart(imp).mark_bar().encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]
            ).properties(height=500)
            st.altair_chart(chart_imp, use_container_width=True)
        else:
            st.info("This model doesn't expose feature_importances_. Try CatBoost or a tree-based model.")

# ======= DATA =======
with tab_data:
    st.subheader("Data preview")
    st.dataframe(q.head(100), use_container_width=True)
    st.caption("Quick stats on filtered subset")
    st.dataframe(describe_block(q).style.format({
        "mean": "{:,.2f}", "median": "{:,.2f}", "min": "{:,.2f}",
        "p25": "{:,.2f}", "p75": "{:,.2f}", "max": "{:,.2f}"
    }), use_container_width=True)

# ======= ABOUT =======
with tab_about:
    st.markdown("""
**Used Car Price — ML & Analytics**  
- Curated dataset: `data/clean_used_cars_curated.csv`  
- Model artifacts: `model/best_model.pkl`, `model/schema_best.json`, `model/metrics_best.json`  
- Training target in log-price; predictions are returned in USD.

**Tips**
- Filters above affect all charts.
- Use the *Predictions* tab for single & batch scoring.
- If you update the model, reload the app to refresh cached artifacts.
    """)

