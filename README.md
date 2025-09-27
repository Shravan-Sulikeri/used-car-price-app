<h1 align="center">Used Car Price — ML & Analytics</h1>

<p align="center">
  Predict used-car prices and explore market trends with a modern data pipeline, API, and web UI.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Next.js" src="https://img.shields.io/badge/Web-Next.js-000000?logo=nextdotjs&logoColor=white">
  <img alt="Tailwind" src="https://img.shields.io/badge/Tailwind-CSS-38BDF8?logo=tailwindcss&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-3FB950">
  <img alt="CatBoost" src="https://img.shields.io/badge/Model-CatBoost-FFCA28">
  <img alt="scikit-learn" src="https://img.shields.io/badge/sklearn-Pipelines-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---
<p align="center">
  <!-- Repo activity -->
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/Shravan-Sulikeri/used-car-price-app">
  <img alt="Commit Activity" src="https://img.shields.io/github/commit-activity/m/Shravan-Sulikeri/used-car-price-app">
  <img alt="Open Issues" src="https://img.shields.io/github/issues/Shravan-Sulikeri/used-car-price-app">
  <img alt="Top Language" src="https://img.shields.io/github/languages/top/Shravan-Sulikeri/used-car-price-app">
  <img alt="Repo Size" src="https://img.shields.io/github/repo-size/Shravan-Sulikeri/used-car-price-app">
</p>


## Overview

A production-friendly portfolio project for used-car price estimation:
- **Data** cleansing & canonicalization (e.g., messy `make/model` fixed to brand standards)
- **Features** like `age`, `mileage_per_year`, and quality filters for outliers/rare models
- **Models** trained in **log-price space**, exported as portable artifacts
- **API** (FastAPI) serving options/summary/charts/prediction
- **Web** (Next.js + Tailwind + charts) for an interactive dashboard

> Datasets live under `data/`. Please respect the underlying Kaggle licenses.

---

## Tech Stack (with logos)

**Data & ML**  
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ndarray-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-3FB950)
![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-FFCA28)
![joblib](https://img.shields.io/badge/joblib-Model_Artifacts-4B8BBE)

**API & Web**  
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=nextdotjs&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-Design-38BDF8?logo=tailwindcss&logoColor=white)

**Dev & Ops**  
![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-181717?logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

---

## Repository Structure

## Repository Structure

| Path | What’s in here |
|---|---|
| 📦 `api/` | FastAPI app (`main.py`) exposing `/health`, `/options`, `/summary`, `/charts`, `/predict` |
| 🧠 `model/` | Cleaning (`clean_and_preprocess.py`), training (`train_suite.py`, `train_gbm.py`), metrics/artifacts |
| 📓 `notebooks/` | `01_data_audit.ipynb`, `02_clean_makes.ipynb` for EDA and canonicalization |
| 🖥️ `frontend/` | Next.js + Tailwind dashboard (`src/app/...`) |
| 🗂️ `data/` | `clean_used_cars.csv` (working), `new/` (raw inputs; git-ignored) |
| 🖼️ `assets/` | `dashboard.png` for README |
| ⚙️ `.gitignore` | ignores large data, artifacts, node modules |
| 📄 `README.md` | documentation |

## AI/ML Approach (brief)

- **Target**: `price` modeled in **log space** to stabilize variance; predictions returned in USD  
- **Signals**: `year`, `mileage`, `make`, `model`, `body`, engineered **`age`**, **`mileage_per_year`**, **`high_mileage`**  
- **Cleaning**:
  - Smart CSV reader (encoding + delimiter detection)
  - Fuzzy column mapping (`pricesold→price`, `yearsold→year`, etc.)
  - Brand normalization (e.g., *mercedes benz → Mercedes-Benz*, *vw → Volkswagen*, *infinity → INFINITI*)
  - Drop extreme outliers and very-rare models (support threshold)  
- **Models**:
  - **LightGBM** and **CatBoost** for non-linear tabular patterns
  - **ElasticNet** baseline for stability
  - Best model and metrics exported to JSON for the UI/API

---

## Dashboard

A modern, **Power BI–style** UI powered by **Next.js + Tailwind** with donuts/lines/bars and a **predictor** panel:
- Landing shows **all cars** → charts react to filters
- **Cascading Make → Model** selector
- Price vs **Year** and **Mileage**, market share donuts, and top models

<p align="center">
  <img src="assets/dashboard.png" width="960" alt="Dashboard preview">
</p>

*(If you haven’t captured a screenshot yet, keep this placeholder and update later.)*

---

## Roadmap

- [x] Data audit notebooks (schema, missingness, distributions)  
- [x] Robust cleaner + brand/model canonicalization  
- [x] Training suite (CatBoost/LightGBM/ElasticNet) + metrics artifacts  
- [x] FastAPI with `/health`, `/options`, `/summary`, `/charts`, `/predict`  
- [x] Next.js dashboard scaffold (App Router + Tailwind + charts)  
- [ ] Model explainability (SHAP), per-feature attributions  
- [ ] Geo insights (state/region price & volume)  
- [ ] Scheduled retraining & experiment tracking  
- [ ] CI/CD: API + Web build checks, smoke tests  
- [ ] Public demo (Vercel + Render) + Lighthouse perf pass

---

## Notes & License

- Datasets reside under `data/` and may originate from Kaggle; **follow the source license rules** if sharing.
- Source code is MIT-licensed (see `LICENSE`).

