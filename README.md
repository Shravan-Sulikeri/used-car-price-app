<h1 align="center">Used Car Price — ML & Analytics</h1>

<p align="center">
  Predict used-car prices and explore market trends with a modern data pipeline, ML models, and an interactive Streamlit dashboard.
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

A portfolio-ready machine learning project for **used car price estimation and market analytics**:

- **Cleaning**: Outlier removal, rare-model pruning, dropping unknowns  
- **Curated Dataset**: `clean_used_cars_curated.csv` with 188k+ cars  
- **Features**: Engineered fields like `age`, `mileage_per_year`, high-mileage flags  
- **Models**: LightGBM, CatBoost, and sklearn pipelines trained in log-price space  
- **Dashboard**: Streamlit app for interactive data exploration and predictions  
- **Reports**: JSON cleaning reports for reproducibility

> Raw → curated → model pipeline is fully reproducible. Datasets live in `/data/`.

---

## Tech Stack (with logos)

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

---

## Repository Structure

| Path | What’s in here |
|---|---|
| 📊 `notebooks/` | Data audit (`01_data_audit.ipynb`), cleaning playbook (`02_cleaning_playbook.ipynb`) |
| 🗂️ `data/` | Raw and curated CSVs (`clean_used_cars.csv`, `clean_used_cars_curated.csv`) |
| 🧠 `model/` | Cleaning report JSON, training scripts, saved model artifacts |
| 🖥️ `web/` | Streamlit app (`dashboard_app.py`) |
| ⚙️ `.gitignore` | ignores large raw dumps, backups, preview CSVs |
| 📄 `README.md` | this doc |

---

## AI/ML Approach

- **Target**: Price in log-space → transformed back to USD  
- **Features**: Year, mileage, make, model, body type, engineered signals (`age`, `mileage_per_year`, `is_high_mileage`)  
- **Cleaning**:
  - Outlier removal with quantiles + z-score per (make, model, year)  
  - Rare model pruning with threshold  
  - Unknown value drop  
- **Models**:
  - LightGBM + CatBoost tuned for non-linear tabular data  
  - sklearn pipelines for reproducibility  
- **Outputs**:
  - `clean_used_cars_curated.csv` (cleaned dataset)  
  - `cleaning_report.json` (removal counts + params)  
  - Model artifacts (`best_model.pkl`, metrics JSON)

---

## Dashboard

An interactive **Streamlit dashboard** lets you:
- Explore distributions (price, year, mileage)  
- Filter by make/model/year and update charts dynamically  
- Run quick predictions with the trained ML model  

<p align="center">
  <img src="assets/dashboard.png" width="960" alt="Streamlit Dashboard Preview">
</p>
## Roadmap

| Feature / Task                          | Status       |
|-----------------------------------------|--------------|
| Data audit & cleaning notebooks         | ✅ Completed |
| Curated baseline dataset (versioned)    | ✅ Completed |
| Streamlit dashboard (EDA + predictions) | ✅ Completed |
| Model training suite (LightGBM/CatBoost)| ✅ Completed |
| SHAP explainability                     | ⏳ Planned   |
| Regional insights (geo pricing trends)  | ⏳ Planned   |
| Scheduled retraining jobs               | ⏳ Planned   |
| Public deployment (Render/Streamlit)    | ⏳ Planned   |

