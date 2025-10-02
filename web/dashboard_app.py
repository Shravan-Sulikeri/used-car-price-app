# web/dashboard_app.py
# Used Car Price Dashboard — KBB-style + schema-safe prediction with robust LightGBM alignment

import os
import io
import json
import math
import pickle
import textwrap
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# ---- CONFIG & FLAGS -----
# =========================
PAGE_TITLE = "Used Car Price Dashboard"
DEFAULT_DATA_GLOB = [
    "data/clean_listings.parquet",
    "data/clean_listings.csv",
    "data/clean_used_cars_curated.csv",
    "data/clean_used_cars.csv",
    "data/listings.parquet",
    "data/listings.csv",
]

USE_LEGACY_GBM_DEFAULT = True
LEGACY_MODEL_PATH = "model/model_gbm.pkl"
LEGACY_CAT_LEVELS_PATH = "model/cat_levels.json"
NEW_PIPELINE_PATH = "model/new_pipeline.joblib"  # optional

COMPS_YEAR_WINDOW = 1
COMPS_MILES_WINDOW = 10000
SEED = 42

np.random.seed(SEED)
alt.data_transformers.disable_max_rows()

# =========================
# ---- UTIL FUNCTIONS -----
# =========================
def _first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".feather", ".ft"]:
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

@st.cache_data(show_spinner=False)
def load_dataset(user_file: Optional[bytes] = None, user_filename: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if user_file is not None and user_filename is not None:
        ext = os.path.splitext(user_filename)[1].lower()
        bio = io.BytesIO(user_file)
        if ext in [".csv", ".txt"]:
            df = pd.read_csv(bio)
        elif ext in [".parquet", ".pq"]:
            df = pd.read_parquet(bio)
        elif ext in [".feather", ".ft"]:
            df = pd.read_feather(bio)
        else:
            raise ValueError("Unsupported upload format. Use CSV, Parquet, or Feather.")
        return df, f"Uploaded: {user_filename}"

    path = _first_existing_path(DEFAULT_DATA_GLOB)
    if path is None:
        return pd.DataFrame(), "No default data found. Upload a file to proceed."
    return _read_any(path), f"Loaded: {path}"

@st.cache_resource(show_spinner=False)
def load_legacy_gbm(model_path: str = LEGACY_MODEL_PATH, cat_levels_path: str = LEGACY_CAT_LEVELS_PATH):
    if not os.path.exists(model_path) or not os.path.exists(cat_levels_path):
        return None, None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(cat_levels_path, "r") as f:
            cat_levels = json.load(f)
        cat_levels = {str(k): [str(v) for v in vals] for k, vals in cat_levels.items()}
        # ensure "Other" exists for every cat col
        for k, vals in list(cat_levels.items()):
            if "Other" not in vals:
                cat_levels[k] = vals + ["Other"]
        return model, cat_levels
    except Exception as e:
        st.warning(f"Could not load legacy model: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_new_pipeline(path: str = NEW_PIPELINE_PATH):
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load new pipeline: {e}")
        return None

CATEGORICAL_NAME_HINTS = {
    "make","manufacturer","brand","model","variant","trim",
    "fuel","transmission","state","body_type","engine","drivetrain","condition"
}

def infer_feature_schema(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, str]:
    schema = {}
    for c in feature_cols:
        if c not in df.columns:
            schema[c] = "numeric"; continue
        if c.lower() in CATEGORICAL_NAME_HINTS:
            schema[c] = "categorical"; continue
        if pd.api.types.is_numeric_dtype(df[c]):
            schema[c] = "numeric"
        elif pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            schema[c] = "categorical"
        else:
            nunq = df[c].astype(str).nunique(dropna=False)
            schema[c] = "categorical" if nunq <= 200 else "string"
    return schema

def _align_categories(s: pd.Series, allowed_levels: List[str], other_label: str = "Other") -> pd.Series:
    s = s.astype(str).fillna(other_label)
    s_aligned = s.where(s.isin(allowed_levels), other_label)
    return pd.Categorical(s_aligned, categories=list(dict.fromkeys(allowed_levels)), ordered=False)

def schema_guard_and_prepare(
    X: pd.DataFrame,
    expected_cols: List[str],
    schema: Dict[str, str],
    legacy_cat_levels: Optional[Dict[str, List[str]]] = None,
    fillna_num: float = 0.0,
    other_label: str = "Other",
) -> pd.DataFrame:
    X = X.copy()
    for c in expected_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[expected_cols]
    for c in expected_cols:
        role = schema.get(c, "numeric")
        if role == "numeric":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(fillna_num).astype(float)
        elif role == "categorical":
            if legacy_cat_levels and c in legacy_cat_levels:
                allowed = [str(v) for v in legacy_cat_levels[c]]
                X[c] = _align_categories(X[c], allowed, other_label=other_label)
            else:
                X[c] = X[c].astype(str).fillna(other_label)
        else:
            X[c] = X[c].astype(str).fillna("")
    return X

def value_counts_top_n(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    vc = df[col].astype(str).value_counts(dropna=False).reset_index()
    vc.columns = [col, "count"]
    vc["pct"] = vc["count"] / max(1, vc["count"].sum())
    return vc.head(n)

# ---- LightGBM helpers ----
def lgbm_feature_names(model) -> Optional[List[str]]:
    try:
        if hasattr(model, "booster_") and model.booster_ is not None:
            return list(model.booster_.feature_name())
        if hasattr(model, "feature_name"):
            return list(model.feature_name() if callable(model.feature_name) else model.feature_name)
        if hasattr(model, "feature_name_"):
            return list(model.feature_name_)
    except Exception:
        pass
    return None

def align_df_to_feature_names(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def looks_one_hot(feature_names: List[str]) -> bool:
    # crude: if any feature contains '__', assume one-hot (make__Honda)
    return any("__" in f for f in feature_names)

def encode_cats_to_codes(df_in: pd.DataFrame, cat_levels: Dict[str, List[str]]) -> pd.DataFrame:
    """Replace categorical columns with stable integer codes based on cat_levels order (unseen -> index of 'Other')."""
    df = df_in.copy()
    for c in df.columns:
        if c in cat_levels:
            levels = cat_levels[c]
            idx_map = {lvl: i for i, lvl in enumerate(levels)}
            df[c] = df[c].astype(str).map(lambda v: idx_map.get(v, idx_map.get("Other", len(levels)-1))).astype(float)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# =========================
# ---- PAGE LAYOUT --------
# =========================
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV/Parquet/Feather", type=["csv", "parquet", "pq", "feather", "ft"])
    st.caption("If you don’t upload, the app will try default files under `/data`.")

df, source_str = load_dataset(
    user_file=(uploaded.read() if uploaded is not None else None),
    user_filename=(uploaded.name if uploaded is not None else None),
)
st.caption(source_str)

if df.empty:
    st.info("No data loaded. Upload a dataset or add one at `data/clean_listings.parquet` or `data/clean_listings.csv`.")
    st.stop()

# ---- Column mapping helpers ----
def _find_col(cands: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in lower_cols:
            return lower_cols[c]
    return None

col_price   = _find_col(["price", "list_price", "sellingprice", "msrp"])
col_year    = _find_col(["year", "model_year"])
col_mileage = _find_col(["mileage", "odometer"])
col_make    = _find_col(["make", "manufacturer", "brand"])
col_model   = _find_col(["model", "car_model", "variant"])

with st.expander("Column Mapping (optional)"):
    c1, c2, c3, c4, c5 = st.columns(5)
    col_price   = c1.selectbox("Price column",   ["<none>"] + list(df.columns), index=(["<none>"] + list(df.columns)).index(col_price)   if col_price   else 0)
    col_year    = c2.selectbox("Year column",    ["<none>"] + list(df.columns), index=(["<none>"] + list(df.columns)).index(col_year)    if col_year    else 0)
    col_mileage = c3.selectbox("Mileage column", ["<none>"] + list(df.columns), index=(["<none>"] + list(df.columns)).index(col_mileage) if col_mileage else 0)
    col_make    = c4.selectbox("Make column",    ["<none>"] + list(df.columns), index=(["<none>"] + list(df.columns)).index(col_make)    if col_make    else 0)
    col_model   = c5.selectbox("Model column",   ["<none>"] + list(df.columns), index=(["<none>"] + list(df.columns)).index(col_model)   if col_model   else 0)

col_price   = None if col_price   == "<none>" else col_price
col_year    = None if col_year    == "<none>" else col_year
col_mileage = None if col_mileage == "<none>" else col_mileage
col_make    = None if col_make    == "<none>" else col_make
col_model   = None if col_model   == "<none>" else col_model

# Clean fields
df_plot = df.copy()
if col_year:
    df_plot[col_year] = pd.to_numeric(df_plot[col_year], errors="coerce")
if col_mileage:
    df_plot[col_mileage] = pd.to_numeric(df_plot[col_mileage], errors="coerce")
if col_price:
    df_plot[col_price] = pd.to_numeric(df_plot[col_price], errors="coerce")

# =========================
# ---- KPI TILES ----------
# =========================
k1, k2, k3, k4, k5 = st.columns(5)
if col_price:
    avg_price  = float(pd.to_numeric(df_plot[col_price], errors="coerce").mean())
    med_price  = float(pd.to_numeric(df_plot[col_price], errors="coerce").median())
    total_cars = int(df_plot[col_price].notna().sum())
    k1.metric("Average Price", f"${avg_price:,.0f}")
    k2.metric("Median Price",  f"${med_price:,.0f}")
    k3.metric("Total Cars",    f"{total_cars:,}")
if col_mileage:
    avg_miles = float(pd.to_numeric(df_plot[col_mileage], errors="coerce").mean())
    k4.metric("Average Mileage", f"{avg_miles:,.0f} mi")
if col_year:
    yr_min, yr_max = (int(df_plot[col_year].min()), int(df_plot[col_year].max()))
    k5.metric("Year Range", f"{yr_min} – {yr_max}")

# ============ FILTERS ============
with st.sidebar:
    st.header("Filters")
    subset = df_plot.copy()

    if col_year and subset[col_year].notna().sum() > 0:
        min_year, max_year = int(subset[col_year].dropna().min()), int(subset[col_year].dropna().max())
        yr_range = st.slider("Year range", min_year, max_year, (min_year, max_year), step=1)
        subset = subset[(subset[col_year] >= yr_range[0]) & (subset[col_year] <= yr_range[1])]

    if col_price and subset[col_price].notna().sum() > 0:
        pmin, pmax = float(subset[col_price].min()), float(subset[col_price].max())
        pr = st.slider("Price range", float(math.floor(pmin)), float(math.ceil(pmax)), (float(math.floor(pmin)), float(math.ceil(pmax))))
        subset = subset[(subset[col_price] >= pr[0]) & (subset[col_price] <= pr[1])]

    if col_mileage and subset[col_mileage].notna().sum() > 0:
        mmin, mmax = float(subset[col_mileage].min()), float(subset[col_mileage].max())
        mr = st.slider("Mileage range", float(math.floor(mmin)), float(math.ceil(mmax)), (float(math.floor(mmin)), float(math.ceil(mmax))))
        subset = subset[(subset[col_mileage] >= mr[0]) & (subset[col_mileage] <= mr[1])]

    if col_make:
        top_makes = value_counts_top_n(subset, col_make, n=30)[col_make].tolist()
        make_sel = st.multiselect("Make", options=top_makes, default=[])
        if make_sel:
            subset = subset[subset[col_make].astype(str).isin(make_sel)]

    if col_model:
        top_models = value_counts_top_n(subset, col_model, n=30)[col_model].tolist()
        model_sel = st.multiselect("Model", options=top_models, default=[])
        if model_sel:
            subset = subset[subset[col_model].astype(str).isin(model_sel)]

st.subheader("Exploratory Charts")

def _year_axis(title: str = "Year") -> alt.Axis:
    return alt.Axis(title=title, labelAngle=0, labelOverlap=True)

# Year vs Price with P10/P50/P90
if col_year and col_price:
    df_yp = subset[[col_year, col_price]].dropna().copy()
    if not df_yp.empty:
        df_yp["YearInt"] = df_yp[col_year].astype(int)
        bands = (
            df_yp.groupby("YearInt")[col_price]
            .quantile([0.10, 0.50, 0.90]).unstack().reset_index()
            .rename(columns={0.10: "p10", 0.50: "p50", 0.90: "p90"})
        )
        bands["YearLabel"] = bands["YearInt"].astype(str)
        base = alt.Chart(bands).encode(x=alt.X("YearLabel:N", axis=_year_axis()))
        area = base.mark_area(opacity=0.2).encode(y="p10:Q", y2="p90:Q")
        line = base.mark_line().encode(y=alt.Y("p50:Q", title="Price (P50)"))
        st.altair_chart((area + line).interactive().properties(height=320), width='stretch')

# Mileage vs Price
if col_mileage and col_price:
    df_mp = subset[[col_mileage, col_price]].dropna().copy()
    if not df_mp.empty:
        chart2 = (
            alt.Chart(df_mp).mark_circle(size=30, opacity=0.35)
            .encode(
                x=alt.X(f"{col_mileage}:Q", title="Mileage"),
                y=alt.Y(f"{col_price}:Q", title="Price"),
                tooltip=[alt.Tooltip(f"{col_mileage}:Q", title="Mileage", format=",.0f"),
                         alt.Tooltip(f"{col_price}:Q", title="Price", format=",.0f")],
            ).properties(height=320)
        )
        st.altair_chart(chart2.interactive(), width='stretch')

# Make vs Price (box)
if col_make and col_price:
    top_m = value_counts_top_n(subset, col_make, n=15)[col_make].tolist()
    df_m = subset[subset[col_make].astype(str).isin(top_m)][[col_make, col_price]].dropna()
    if not df_m.empty:
        st.altair_chart(
            alt.Chart(df_m).mark_boxplot()
            .encode(x=alt.X(f"{col_make}:N", sort="-y", title="Make"),
                    y=alt.Y(f"{col_price}:Q", title="Price")).properties(height=360),
            width='stretch'
        )

# Model vs Price (box)
if col_model and col_price:
    top_mod = value_counts_top_n(subset, col_model, n=15)[col_model].tolist()
    df_mod = subset[subset[col_model].astype(str).isin(top_mod)][[col_model, col_price]].dropna()
    if not df_mod.empty:
        st.altair_chart(
            alt.Chart(df_mod).mark_boxplot()
            .encode(x=alt.X(f"{col_model}:N", sort="-y", title="Model"),
                    y=alt.Y(f"{col_price}:Q", title="Price")).properties(height=360),
            width='stretch'
        )

# Pie
pie_dim = col_make or col_model
if pie_dim:
    st.markdown(f"#### Distribution by {pie_dim}")
    vc = value_counts_top_n(subset, pie_dim, n=12)
    if not vc.empty:
        st.altair_chart(
            alt.Chart(vc).mark_arc(outerRadius=120, innerRadius=40)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(f"{pie_dim}:N", legend=alt.Legend(title=pie_dim)),
                tooltip=[alt.Tooltip(f"{pie_dim}:N", title=pie_dim),
                         alt.Tooltip("count:Q", title="Count", format=",.0f"),
                         alt.Tooltip("pct:Q", title="Share", format=".1%")],
            ).properties(height=380),
            width='stretch'
        )

st.divider()

# =========================
# ---- PREDICTION (KBB) ---
# =========================
st.subheader("Price Prediction (Schema-Safe, KBB-style Range)")

legacy_model, legacy_cat_levels = load_legacy_gbm()
new_pipeline = load_new_pipeline()

available_models = []
if legacy_model is not None: available_models.append("Legacy GBM")
if new_pipeline is not None: available_models.append("New Pipeline")
if not available_models: available_models = ["(No model found)"]

default_index = 0
if "Legacy GBM" in available_models and USE_LEGACY_GBM_DEFAULT:
    default_index = available_models.index("Legacy GBM")
elif "New Pipeline" in available_models:
    default_index = available_models.index("New Pipeline")

cA, cB = st.columns([1.2, 1])
with cB:
    model_choice = st.selectbox("Model", options=available_models, index=default_index)

# Heuristic feature selection
common_feature_candidates = ["year","mileage","make","model","fuel","transmission",
                             "condition","state","trim","body_type","engine","drivetrain"]
feature_cols = []
for cand in common_feature_candidates:
    hit = _find_col([cand])
    if hit: feature_cols.append(hit)
if len(feature_cols) < 3:
    feature_cols = [c for c in df.columns if c != col_price][:8]

schema = infer_feature_schema(df, feature_cols)

# ----- Dependent dropdown for model by make -----
# Build map: make -> models (top 50)
make_to_models = {}
if col_make and col_model:
    tmp = df[[col_make, col_model]].dropna()
    if not tmp.empty:
        make_to_models = (tmp.astype(str)
                          .groupby(col_make)[col_model]
                          .apply(lambda s: s.value_counts().head(50).index.tolist())
                          .to_dict())

with st.expander("Feature Schema (detected)", expanded=False):
    st.json(schema)

st.markdown("#### Enter Features")
form_cols = st.columns(3)
user_row = {}
selected_make_value = None

for i, fcol in enumerate(feature_cols):
    role = schema.get(fcol, "numeric")
    slot = form_cols[i % 3]

    # --- YEAR: force integer (no decimals) ---
    if fcol == col_year and role == "numeric":
        yr_min = int(df_plot[col_year].dropna().min()) if col_year else 1990
        yr_max = int(df_plot[col_year].dropna().max()) if col_year else 2025
        user_row[fcol] = slot.number_input(fcol, min_value=yr_min, max_value=yr_max,
                                           value=int(np.nan_to_num(df_plot[col_year].median())) if col_year else 2018,
                                           step=1, format="%d")
        continue

    # --- MILEAGE: nicer defaults ---
    if fcol == col_mileage and role == "numeric":
        default_val = float(pd.to_numeric(df[fcol], errors="coerce").median()) if fcol in df.columns else 0.0
        user_row[fcol] = slot.number_input(fcol, value=float(np.nan_to_num(default_val)), step=500.0)
        continue

    if role == "numeric":
        default_val = float(pd.to_numeric(df[fcol], errors="coerce").median()) if fcol in df.columns else 0.0
        user_row[fcol] = slot.number_input(fcol, value=float(np.nan_to_num(default_val)))
    elif role == "categorical":
        # Special case: model options depend on make
        if fcol == col_make:
            opts = value_counts_top_n(df, fcol, n=50)[fcol].astype(str).tolist() or [""]
            selected_make_value = slot.selectbox(fcol, options=opts, index=0)
            user_row[fcol] = selected_make_value
        elif fcol == col_model and selected_make_value and selected_make_value in make_to_models:
            opts = make_to_models[selected_make_value] or [""]
            user_row[fcol] = slot.selectbox(fcol, options=opts, index=0)
        else:
            opts = value_counts_top_n(df, fcol, n=25)[fcol].astype(str).tolist() or [""]
            user_row[fcol] = slot.selectbox(fcol, options=opts, index=0)
    else:
        user_row[fcol] = slot.text_input(fcol, value="")

asking_price = st.number_input("Optional: Asking/Listing Price (to rate the deal)", value=0.0, min_value=0.0, step=500.0)
user_df = pd.DataFrame([user_row], columns=feature_cols)

# ---- comps & fair range ----
def compute_comps_and_range(df_base: pd.DataFrame, row: Dict[str, any]) -> Tuple[pd.DataFrame, Optional[float], Optional[float], int]:
    if not (col_price and col_year and col_mileage):
        return pd.DataFrame(), None, None, 0
    dfC = df_base[[col_price, col_year, col_mileage] + [c for c in [col_make, col_model] if c is not None]].dropna()
    if dfC.empty:
        return pd.DataFrame(), None, None, 0
    mask = pd.Series(True, index=dfC.index)
    if col_make and row.get(col_make):
        mask &= (dfC[col_make].astype(str) == str(row.get(col_make)))
    if col_model and row.get(col_model):
        mask &= (dfC[col_model].astype(str) == str(row.get(col_model)))
    year_val = pd.to_numeric(row.get(col_year), errors="coerce")
    miles_val = pd.to_numeric(row.get(col_mileage), errors="coerce")
    if pd.notna(year_val):
        mask &= (dfC[col_year] >= year_val - COMPS_YEAR_WINDOW) & (dfC[col_year] <= year_val + COMPS_YEAR_WINDOW)
    if pd.notna(miles_val):
        mask &= (dfC[col_mileage] >= miles_val - COMPS_MILES_WINDOW) & (dfC[col_mileage] <= miles_val + COMPS_MILES_WINDOW)
    comps = dfC[mask].copy()
    if comps.empty:
        return pd.DataFrame(), None, None, 0
    p25 = float(comps[col_price].quantile(0.25))
    p75 = float(comps[col_price].quantile(0.75))
    return comps, p25, p75, len(comps)

pred = None
explain = ""
if st.button("Predict Price", width='stretch'):
    try:
        # Decide model
        use_legacy = (model_choice == "Legacy GBM" and legacy_model is not None)
        use_new = (model_choice == "New Pipeline" and new_pipeline is not None)

        if not (use_legacy or use_new):
            st.warning("No model available. Please add a model or switch selection.")
        elif use_new:
            pred_val = float(new_pipeline.predict(user_df.copy())[0])
            pred = pred_val
            explain = "Predicted using New Pipeline."
        else:
            # ===== Legacy LightGBM path =====
            expected_cats = list(legacy_cat_levels.keys()) if legacy_cat_levels else []
            expected_nums = [c for c, r in schema.items() if r == "numeric"]
            expected_cols = expected_nums + expected_cats

            X_prep = schema_guard_and_prepare(
                user_df,
                expected_cols=expected_cols,
                schema=infer_feature_schema(user_df, expected_cols),
                legacy_cat_levels=legacy_cat_levels,
                fillna_num=0.0,
                other_label="Other",
            )

            model_feats = lgbm_feature_names(legacy_model)
            one_hot_train = looks_one_hot(model_feats) if model_feats else False

            if one_hot_train:
                # build one-hot matrix with exact training levels
                X_enc = pd.DataFrame(index=X_prep.index)
                for c in expected_cols:
                    if c in legacy_cat_levels:
                        lvls = legacy_cat_levels[c]
                        s = X_prep[c].astype(str)
                        for lvl in lvls:
                            X_enc[f"{c}__{lvl}"] = (s == lvl).astype(int)
                    else:
                        X_enc[c] = pd.to_numeric(X_prep[c], errors="coerce").fillna(0.0)
                X_final = align_df_to_feature_names(X_enc, model_feats) if model_feats else X_enc
            else:
                # raw columns: replace categoricals with integer codes based on cat_levels
                X_codes = encode_cats_to_codes(X_prep, legacy_cat_levels or {})
                # ensure only the columns the model knows about (if we have names)
                if model_feats:
                    # If model was trained on raw names, reorder; add missing as 0
                    X_final = align_df_to_feature_names(X_codes, model_feats)
                else:
                    X_final = X_codes
                # Ensure pure numeric ndarray (no category dtype leakage)
                X_final = X_final.astype(np.float32)

            # predict with numpy matrix to avoid dtype quirks
            X_mat = X_final.values if hasattr(X_final, "values") else np.asarray(X_final, dtype=np.float32)
            pred_val = float(legacy_model.predict(X_mat)[0])
            pred = pred_val
            explain = "Predicted using Legacy GBM with strict numeric schema alignment."
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if pred is not None:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.success(f"Estimated Price: ${pred:,.0f}")
    c1.caption(explain)

    comps, p25, p75, n_comps = compute_comps_and_range(df_plot, user_row)
    if p25 is not None and p75 is not None:
        c2.metric("Fair Market Range (P25–P75)", f"${p25:,.0f} - ${p75:,.0f}")
        c3.metric("Comparable Listings Used", f"{n_comps:,}")
        if asking_price and asking_price > 0:
            if asking_price < p25:
                st.success(f"**Deal Rating:** Great (asking ${asking_price:,.0f} is below the fair range)")
            elif p25 <= asking_price <= p75:
                st.info(f"**Deal Rating:** Fair (asking ${asking_price:,.0f} is within the fair range)")
            else:
                st.warning(f"**Deal Rating:** High (asking ${asking_price:,.0f} is above the fair range)")
    else:
        st.caption("Not enough similar comps to compute a fair market range. Consider widening the filters.")

# =========================
# ---- DATA PREVIEW -------
# =========================
st.divider()
st.subheader("Data Preview")
st.dataframe(subset.head(100), width='stretch')

# =========================
# ---- FOOTER -------------
# =========================
with st.expander("Notes & Tips"):
    st.markdown(
        textwrap.dedent(
            f"""
            - **Year input** is integer-only; mileage uses a sensible step.
            - **Model list depends on selected Make** (based on your dataset).
            - **Legacy GBM alignment**:
              - If your model was trained with one-hot columns like `make__Honda`, we rebuild those *exact* columns and order them to the model’s stored feature names.
              - If it was trained on raw columns (e.g., `make`), we map categories → stable integer codes using `cat_levels.json` (unseen → "Other") and pass a pure **NumPy float matrix**, avoiding LightGBM categorical mismatches.
            - **Fair Market Range** uses comps within ±{COMPS_YEAR_WINDOW} year and ±{COMPS_MILES_WINDOW:,} miles for the same Make/Model.
            """
        )
    )