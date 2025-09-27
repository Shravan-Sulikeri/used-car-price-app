<h1 align="center">Used Car Price — ML & Analytics</h1>

<p align="center">
  Predict used-car listing prices and explore market trends with a modern, reproducible pipeline.
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-Work_in_progress-yellow">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Codespaces" src="https://img.shields.io/badge/Codespaces-Ready-181717?logo=github">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-GBDT-3FB950">
</p>


> **Dataset:** Kaggle — `tsaustin/us-used-car-sales-data`. Review the dataset’s license on Kaggle before redistribution.

---

## Tech Stack

## Tech Stack

**Data & Modeling**
- ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)  
  pandas • numpy • scikit-learn (pipelines, ColumnTransformer) • **LightGBM** • category-encoders • joblib

**Dev & Tooling**
- ![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-181717?logo=github)  
  Dev Containers • Jupyter

**Planned Delivery**
- ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi) • ![Next.js](https://img.shields.io/badge/Next.js-UI-000000?logo=nextdotjs) • ![Tailwind](https://img.shields.io/badge/Tailwind-CSS-38BDF8?logo=tailwindcss)  
  Hosting: Vercel (web), Render (API)


---

## Features

- **Reproducible environment** (`.devcontainer/`) — one-click Codespaces
- **Audit notebook** for schema/missingness sanity checks
- **Robust cleaning** with fuzzy column mapping (e.g., `pricesold → price`, `yearsold → year`)
- **Feature engineering:** `age`, `mileage_per_year`, `high_mileage`
- **Two models:** Ridge (baseline) and **LightGBM** (stronger)
- **Artifacted metrics & previews** (`metrics_*.json`, `preview_gbm.csv`)

---

---
## Quick Links
- ▶️ [Quickstart](#quickstart-codespaces)
- 🧼 [Cleaning & Features](#cleaning--feature-engineering)
- 🤖 [Modeling](#modeling)
- 📈 [Results](#results-current-run)
- 🧪 [Predict Example](#example-predict-with-the-trained-pipeline)
- 🗺️ [Repository Structure](#repository-structure)
- 🖥️ [Dashboard](#dashboard)

---


## Repository Structure

| Path | Description |
|---|---|
| 📁 `.devcontainer/` | Codespaces environment (Python/Node): `devcontainer.json`, `post-create.sh` |
| 📁 `data/` | Raw & processed CSVs *(_gitignored_)* |
| 📁 `model/` | ML assets and scripts |
| ├── `clean_and_preprocess.py` | Cleaning + feature engineering pipeline |
| ├── `preprocessor.pkl` | Fitted `ColumnTransformer` (committed for reuse) |
| ├── `train_baseline.py` | Baseline Ridge trainer (log-price) |
| ├── `train_gbm.py` | LightGBM trainer (stronger model) |
| ├── `metrics_baseline.json` | Baseline metrics (MAE/RMSE/R²) |
| ├── `metrics_gbm.json` | GBM metrics (MAE/RMSE/R²) |
| └── `preview_gbm.csv` | Sample GBM predictions vs. actuals |
| 📁 `notebooks/` | Exploratory notebooks |
| └── `00_quick_audit.ipynb` | Fast schema/missingness audit |
| 📁 `api/` | *(planned)* FastAPI `/predict` service |
| 📁 `web/` | *(planned)* Next.js + Tailwind dashboard |
| 📄 `README.md` | Project documentation |


---

## Results (current run)

| Model     | MAE ($) | RMSE ($) | R²   |
|-----------|---------|----------|------|
| Ridge     | 6,660   | 12,538   | 0.125 |
| LightGBM  | **5,826** | **10,898** | **0.339** |

---
## Data Quality — Make Canonicalization

Cleaning raw **make** values dramatically improves consistency and model signal.  
This project consolidates messy variants (e.g., `bmw 335i`, `mercedes benz`, `volkswagon`, `chevy`, `studabaker`) to clean, branded forms.

**How it works**
- Normalization (case, punctuation, spacing)
- Synonym + misspelling rules (e.g., `vw → Volkswagen`, `chevy → Chevrolet`, `infinity → INFINITI`)
- Two-word brand detection (e.g., `mercedes benz → Mercedes-Benz`, `land rover → Land Rover`)
- Token fallback for title-like strings (e.g., `cadillac only 80k → Cadillac`)

**Artifacts**
- Report: [`model/make_cleaning_report.json`](model/make_cleaning_report.json)  
- Summary table: [`reports/makes_canonical_summary.csv`](reports/makes_canonical_summary.csv)

**Examples (before → after)**

| Raw                         | Canonical        |
|----------------------------|------------------|
| `bmw 335i`                 | **BMW**          |
| `mercedes benz`            | **Mercedes-Benz**|
| `volkswagon`, `vw`         | **Volkswagen**   |
| `chevy`                    | **Chevrolet**    |
| `landrover`, `range rover` | **Land Rover**   |
| `infinity`                 | **INFINITI**     |
| `chrylser`, `crhysler`     | **Chrysler**     |
| `studabaker`               | **Studebaker**   |


---


## Dashboard

<p align="center">
  <!-- Replace YOUR_DASHBOARD_URL after you deploy -->
  <a href="https://YOUR_DASHBOARD_URL" target="_blank" rel="noopener">
    <img alt="Live Dashboard" src="https://img.shields.io/badge/Live_Dashboard-Open_App-6D28D9?logo=streamlit&logoColor=white">
  </a>
</p>

<p align="center">
  <img src="assets/dashboard.png" width="980" alt="Used Car Price — Dashboard (dark purple)">
</p>

---

## Model Card (brief)
**Intended use:** Educational/portfolio demo for US used-car price estimation.  
**Data:** Kaggle “US Used Car Sales Data”; columns normalized (e.g., `pricesold→price`, `yearsold→year`).  
**Target:** Price (trained in log-space; predictions returned in USD).  
**Main features:** `year`, `mileage`, `make`, `model`, `body`, `age`, `mileage_per_year`, `high_mileage`.  
**Limitations:** No trim/options, regional demand, or vehicle history → larger error on rare configs.  
**Ethics:** Not for credit/insurance decisions; may reflect historical market biases.

---

## Contributing
PRs welcomed for: API `/predict`, Next.js dashboard, CI, and ONNX export.

## Contact
**Shravan Sulikeri** — feel free to open an issue or reach out via GitHub.

## License
MIT — see `LICENSE`. Verify the Kaggle dataset license before redistributing data/models.

---
