<h1 align="center">Used Car Price — ML & Analytics</h1>

<p align="center">
  Predict used-car prices and explore market trends with a modern data pipeline, ML models, and an interactive multi-page Streamlit dashboard.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Web-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-3FB950">
  <img alt="scikit-learn" src="https://img.shields.io/badge/sklearn-Pipelines-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-EDA-150458?logo=pandas&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---
## Live repo metrics

<p align="center">
  <a href="https://github.com/Shravan-Sulikeri/used-car-price-app/commits/main">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/Shravan-Sulikeri/used-car-price-app?logo=github">
  </a>
  <img alt="Commit Activity" src="https://img.shields.io/github/commit-activity/m/Shravan-Sulikeri/used-car-price-app">
  <a href="https://github.com/Shravan-Sulikeri/used-car-price-app/issues">
    <img alt="Open Issues" src="https://img.shields.io/github/issues/Shravan-Sulikeri/used-car-price-app">
  </a>
  <a href="https://github.com/Shravan-Sulikeri/used-car-price-app/pulls">
    <img alt="Open PRs" src="https://img.shields.io/github/issues-pr/Shravan-Sulikeri/used-car-price-app">
  </a>
  <a href="https://github.com/Shravan-Sulikeri/used-car-price-app/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/Shravan-Sulikeri/used-car-price-app">
  </a>
  <a href="https://github.com/Shravan-Sulikeri/used-car-price-app/network/members">
    <img alt="Forks" src="https://img.shields.io/github/forks/Shravan-Sulikeri/used-car-price-app">
  </a>
  <img alt="Top Language" src="https://img.shields.io/github/languages/top/Shravan-Sulikeri/used-car-price-app">
  <img alt="Repo Size" src="https://img.shields.io/github/repo-size/Shravan-Sulikeri/used-car-price-app">
  <img alt="Contributors" src="https://img.shields.io/github/contributors/Shravan-Sulikeri/used-car-price-app?logo=github">
</p>

---

## Overview

A portfolio-ready machine learning project for **used car price estimation and analytics**:

- **Cleaning**: Outlier removal, canonical make/model normalization, dropping “Other/Unknown”  
- **Dataset**: Taustin’s **US Used Car Sales Data** (`used_car_sales.csv`) cleaned to ~65k records  
- **Features**: Engineered `age`, `miles_per_year`, mileage bands  
- **Models**: LightGBM baseline (MAE-optimized) with sklearn pipelines  
- **Dashboard**: Multi-page Streamlit app:
  - 📊 Graphs (year vs price, mileage vs price, make/model vs price)  
  - 🥧 Market share pie chart (cleaned brands only)  
  - 💰 Prediction page with linked make → model selectors  
- **Artifacts**: Model `.pkl`, schema JSON, cleaning reports

> Full pipeline: raw → cleaned parquet/CSV → training → dashboard.

---

## Tech Stack (with logos)

**Data & ML**  
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ndarray-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-3FB950)

**Web UI**  
![Streamlit](https://img.shields.io/badge/Streamlit-Multi--page-FF4B4B?logo=streamlit&logoColor=white)

**Dev & Ops**  
![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-181717?logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-EDA-F37626?logo=jupyter&logoColor=white)

---

## Repository Structure

| Path | What’s in here |
|---|---|
| 📊 `notebooks/` | Cleaning notebooks (canonical make/model, Taustin dataset audit) |
| 🗂️ `data/` | Raw, cleaned, and parquet CSVs |
| 🧠 `model/` | Training pipeline, saved LightGBM model, metrics JSON |
| 🖥️ `web/` | Streamlit multi-page dashboard (`dashboard_app.py`) |
| ⚙️ `.gitignore` | ignores raw dumps, large CSVs |
| 📄 `README.md` | this doc |

---

## AI/ML Approach

- **Target**: Price (log-transformed for training → back to USD)  
- **Features**: Year, mileage, make, model, body type, + engineered fields  
- **Cleaning**:
  - Outlier removal with IQR + caps  
  - Canonicalization of makes/models (e.g., Tesla Model 3/Y/S/X)  
  - Drop “Other/Unknown” categories from charts  
- **Model**:
  - LightGBM tuned for regression with early stopping  
  - sklearn pipelines for consistent preprocessing  
- **Outputs**:
  - `clean_listings_clean.parquet/csv`  
  - Model: `model_gbm.pkl`  
  - Metrics + schema JSON

---

## Dashboard

Interactive **Streamlit dashboard**:

- Graphs page: Year vs Price, Mileage vs Price, Make vs Price, Model vs Price  
- Pie Chart page: Brand market share (clean palette, no “Other”)  
- Prediction page: Make → Model linked dropdowns, instant price prediction  

<p align="center">
  <img src="assets/dashboard.png" width="960" alt="Streamlit Dashboard Preview">
</p>

---

## Roadmap

| Feature / Task                          | Status       |
|-----------------------------------------|--------------|
| Data ingestion + cleaning (Taustin)     | ✅ Completed |
| Streamlit 3-page dashboard              | ✅ Completed |
| LightGBM baseline model                 | ✅ Completed |
| Canonical make/model mapping            | ✅ Completed |
| Advanced explainability (SHAP)          | ⏳ Planned   |
| Multi-dataset integration               | ⏳ Planned   |
| Automated retraining pipeline           | ⏳ Planned   |
| Public deployment (Render/Streamlit)    | ⏳ Planned   |
