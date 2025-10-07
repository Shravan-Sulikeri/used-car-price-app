<h1 align="center">Used Car Price — Machine Learning & Analytics Platform</h1>

<p align="center">
  Predict used-car prices and explore market trends using a modern data pipeline, advanced ML models, and an interactive Streamlit analytics dashboard.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white&style=flat-square">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit&logoColor=white&style=flat-square">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-3FB950?logo=lightgbm&logoColor=white&style=flat-square">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-Gradient_Boosting-FFCA28?style=flat-square">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white&style=flat-square">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas&logoColor=white&style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## Overview

This project provides an end-to-end, portfolio-ready machine learning solution for **used-car price estimation** and **market trend analytics**.  
It demonstrates data engineering, feature design, model training, and deployment of a real-time web dashboard.

### Key Highlights
- **End-to-End Pipeline:** From raw CSV data to deployed Streamlit dashboard.  
- **Data Cleaning:** Outlier removal, canonical make/model normalization, and filtering of incomplete records.  
- **Feature Engineering:** Derived variables such as `age`, `miles_per_year`, and segmented mileage bands.  
- **Models:** Tuned LightGBM and CatBoost regressors built via reproducible scikit-learn pipelines.  
- **Deployment:** Multi-page Streamlit app for exploration, visualization, and live price prediction.

---

## Architecture

Raw Data → Cleaning & Normalization → Feature Engineering → Model Training (GBDTs)
↓ ↓
Curated Dataset Serialized Models (.pkl)
↓ ↓
→ Interactive Streamlit Dashboard ←

---

## Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Programming Language** | Python 3.11 |
| **ML Frameworks** | scikit-learn, LightGBM, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Streamlit Components |
| **Experiment Tracking** | JSON/CSV metrics logs, joblib artifacts |
| **Version Control & CI/CD** | Git, GitHub Actions, Codespaces |
| **Deployment** | Streamlit Cloud / Render (container-ready) |

---

## Repository Structure

| Directory | Description |
|------------|-------------|
| `data/` | Raw and curated datasets (`used_car_sales.csv`, `clean_listings.csv`) |
| `notebooks/` | Jupyter notebooks for data audits, cleaning, and feature analysis |
| `model/` | Model training scripts, serialized `.pkl` files, metrics, and schema JSON |
| `web/` | Streamlit application (`dashboard_app.py`) |
| `.github/workflows/` | CI/CD pipelines for linting, testing, and build verification |
| `README.md` | Project documentation |

---

## Machine Learning Methodology

| Stage | Description |
|--------|--------------|
| **Target Variable** | Vehicle price (log-transformed for stability, inverse-converted for output) |
| **Cleaning** | Quantile-based outlier removal and category normalization |
| **Feature Engineering** | Continuous + categorical encodings, age and mileage ratios |
| **Modeling** | Gradient-boosted decision trees (LightGBM, CatBoost) optimized for MAE |
| **Validation** | Stratified K-Fold cross-validation with early stopping |
| **Artifacts** | `model_gbm.pkl`, `cat_levels.json`, `schema_best.json`, metrics reports |

---

## Streamlit Dashboard

The Streamlit web interface provides:

- **Data Exploration:** Dynamic filtering and visual analysis of price distributions.  
- **Model Insights:** Feature impact and model comparison charts.  
- **Price Prediction:** User input form for instant car-price estimation.  
- **KPI Summary:** Average price, top brands, and mileage statistics.


---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Shravan-Sulikeri/used-car-price-app.git
cd used-car-price-app
###2. Create a Virtual Environmentpython -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
###3. Install Dependencies
pip install -U pip
pip install -r requirements.txt
###4. Run the Application
streamlit run web/dashboard_app.py
###The app will launch in your default browser at http://localhost:8501
```
---

Continuous Integration
Automated workflows ensure code quality and reproducibility.
Linting & Testing: flake8, black, pytest
Build Verification: Streamlit app startup validation
Artifact Management: Serialize models and metrics to model/
Optional Deployment: Streamlit Cloud / Render integration
Example CI configuration (.github/workflows/ci.yml):

```bash
name: CI Pipeline
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint and test
        run: |
          flake8 .
          pytest -q
```
Future Enhancements
Model explainability (SHAP and feature importance visualization)
API endpoint for external predictions
Integration with cloud data sources (AWS S3, GCP BigQuery)
Automated model retraining pipeline

---

Author: Shravan Sulikeri
Repository: Shravan-Sulikeri/used-car-price-app


