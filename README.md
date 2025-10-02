<h1 align="center">🚗 Used Car Price — ML & Analytics</h1>

<p align="center">
  Predict used-car prices and explore market trends with a modern data pipeline, ML models, and an interactive multi-page Streamlit dashboard.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Web-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-3FB950">
  <img alt="CatBoost" src="https://img.shields.io/badge/Model-CatBoost-FFCA28">
  <img alt="scikit-learn" src="https://img.shields.io/badge/sklearn-Pipelines-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-EDA-150458?logo=pandas&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 📊 Live repo metrics

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

## 📝 Overview

A portfolio-ready machine learning project for **used car price estimation and market analytics**:

- **Cleaning**: Outlier removal, canonical make/model normalization, dropping `Other/Unknown`  
- **Dataset**: Taustin’s **US Used Car Sales Data** (`used_car_sales.csv`) cleaned to ~65k records  
- **Features**: Engineered `age`, `miles_per_year`, mileage bands  
- **Models**: LightGBM (MAE-optimized), CatBoost, and sklearn pipelines  
- **Dashboard**: Interactive multi-page Streamlit dashboard for EDA + prediction  
- **Reports**: JSON cleaning reports for reproducibility  

> Data pipeline: **Raw → Cleaned → Curated → Model-ready** (fully reproducible)

---

## ⚙️ Tech Stack

**Data & ML**  
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ndarray-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-3FB950)
![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-FFCA28)

**Web UI**  
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

**Dev & Ops**  
![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-181717?logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![CI/CD](https://img.shields.io/badge/GitHub-Actions-2088FF?logo=githubactions&logoColor=white)

---
## 📂 Repository Structure

| Path | What’s in here |
|---|---|
| 📊 `notebooks/` | Data audits (`01_data_audit.ipynb`), cleaning (`02_clean_makes.ipynb`, `clean_taustin.ipynb`) |
| 🗂️ `data/` | Raw CSVs + curated datasets (`clean_listings.csv`, `clean_listings_clean.parquet`) |
| 🧠 `model/` | Training pipelines, metrics, LightGBM artifacts (`model_gbm.pkl`) |
| 🖥️ `web/` | Streamlit dashboard (`dashboard_app.py`) |
| ⚙️ `.github/workflows/` | CI/CD pipelines (lint, test, build, deploy) |
| 📄 `README.md` | This doc |

---

## 🤖 AI/ML Approach

- **Target**: Price in log-space → inverse-transformed to USD  
- **Features**: `year`, `mileage`, `make`, `model`, engineered signals (`age`, `miles_per_year`)  
- **Cleaning**:
  - Quantile-based outlier removal  
  - Rare model pruning  
  - Drop unknown/other categories  
- **Models**:
  - **LightGBM** — robust GBDT, tuned with early stopping  
  - **CatBoost** — categorical boosting for high-cardinality features  
  - **sklearn Pipelines** — reproducible feature engineering + training  
- **Artifacts**:
  - `model_gbm.pkl`, `cat_levels.json`, `schema_best.json`, metrics JSON  

---

## 📊 Dashboard (Streamlit)

- Interactive **EDA visualizations** (year vs price, mileage vs price, pie chart by make)  
- **KPI Cards**: Average price, mileage, most common make  
- **Price Prediction**: Enter year, mileage, make, model → estimated price + fair value band  

<p align="center">
  <img src="assets/dashboard.png" width="960" alt="Streamlit Dashboard Preview">
</p>

---

## 🚀 CI/CD Pipeline

This repo includes **GitHub Actions** workflows for automation:  

- ✅ **Lint & Test**: Runs flake8, black, and pytest on every push  
- ✅ **Model Training**: Runs LightGBM pipeline with curated dataset  
- ✅ **Build Dashboard**: Ensures Streamlit app runs without errors  
- ✅ **Deploy**: Optionally deploys to **Streamlit Cloud / Render**  

Example workflow snippet (`.github/workflows/ci.yml`):

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint & Test
        run: |
          flake8 .
          pytest -q


---

---


