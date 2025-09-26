# Used Car Price — ML & Analytics

Predict used-car listing prices and explore market trends with a modern, reproducible pipeline (pandas + scikit-learn + LightGBM) and a production-ready GitHub Codespaces setup.

> **Dataset:** Kaggle — `tsaustin/us-used-car-sales-data`. Review the dataset’s license on Kaggle before redistribution.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started (GitHub Codespaces)](#getting-started-github-codespaces)
- [Data Intake & Audit](#data-intake--audit)
- [Cleaning & Feature Engineering](#cleaning--feature-engineering)
- [Modeling](#modeling)
- [Quick Prediction Example](#quick-prediction-example)
- [Results (Current Run)](#results-current-run)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Data & Security Notes](#data--security-notes)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview
This project builds a price-prediction model for used cars and lays the groundwork for a modern dashboard (Next.js/Tailwind) and a simple prediction API (FastAPI). It is designed to be:
- **Reproducible** (devcontainer + scripted steps)
- **Auditable** (notebook for early checks, JSON metrics)
- **Deployable** (API/UI planned; free-tier friendly)

---

## Features
- **One-click environment** via GitHub Codespaces (`.devcontainer/`)
- **Data audit notebook** to discover schema & quality
- **Robust cleaning** with automatic column mapping (e.g., `pricesold → price`)
- **Reusable preprocessor** (`ColumnTransformer`) for training & serving
- **Models:** Baseline Ridge and LightGBM, trained on log-price
- **Metrics artifacts** (`metrics_*.json`) and sample predictions

---

## Tech Stack
- **Python:** pandas, numpy, scikit-learn, LightGBM, category-encoders, joblib  
- **Dev:** GitHub Codespaces, Jupyter  
- **Planned:** FastAPI (API), Next.js + Tailwind (web dashboard), Vercel/Render (hosting)

---

## Repository Structure

# Used Car Price — ML & Analytics

Predict used-car listing prices and explore market trends with a modern, reproducible pipeline (pandas + scikit-learn + LightGBM) and a production-ready GitHub Codespaces setup.

> **Dataset:** Kaggle — `tsaustin/us-used-car-sales-data`. Review the dataset’s license on Kaggle before redistribution.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started (GitHub Codespaces)](#getting-started-github-codespaces)
- [Data Intake & Audit](#data-intake--audit)
- [Cleaning & Feature Engineering](#cleaning--feature-engineering)
- [Modeling](#modeling)
- [Quick Prediction Example](#quick-prediction-example)
- [Results (Current Run)](#results-current-run)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Data & Security Notes](#data--security-notes)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview
This project builds a price-prediction model for used cars and lays the groundwork for a modern dashboard (Next.js/Tailwind) and a simple prediction API (FastAPI). It is designed to be:
- **Reproducible** (devcontainer + scripted steps)
- **Auditable** (notebook for early checks, JSON metrics)
- **Deployable** (API/UI planned; free-tier friendly)

---

## Features
- **One-click environment** via GitHub Codespaces (`.devcontainer/`)
- **Data audit notebook** to discover schema & quality
- **Robust cleaning** with automatic column mapping (e.g., `pricesold → price`)
- **Reusable preprocessor** (`ColumnTransformer`) for training & serving
- **Models:** Baseline Ridge and LightGBM, trained on log-price
- **Metrics artifacts** (`metrics_*.json`) and sample predictions

---

## Tech Stack
- **Python:** pandas, numpy, scikit-learn, LightGBM, category-encoders, joblib  
- **Dev:** GitHub Codespaces, Jupyter  
- **Planned:** FastAPI (API), Next.js + Tailwind (web dashboard), Vercel/Render (hosting)

---

## Repository Structure
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



