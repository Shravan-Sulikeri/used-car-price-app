#!/usr/bin/env bash
set -e

# Ensure pip exists and is current
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

# Core Python deps
python -m pip install \
  pandas numpy scikit-learn lightgbm category-encoders \
  fastapi uvicorn joblib jupyter ipykernel onnx skl2onnx kaggle

# Jupyter kernel for this workspace
python -m ipykernel install --user --name usedcar --display-name "Python (usedcar)" || true

echo "Devcontainer ready."

