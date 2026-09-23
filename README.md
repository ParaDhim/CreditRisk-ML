# Credit Default Risk Model

This repository builds an end-to-end credit default risk classifier based on the Kaggle "Home Credit Default Risk" competitive dataset. It demonstrates fully reproducible feature engineering at the applicant level over multiple relational tables (bureau, past applications, installment payments), deals with class imbalance using techniques like SMOTE and weighted loss, and uses tree-based ML (LightGBM). A core deliverable of the project is the translation of model-predicted probabilities into a business-facing threshold analysis.

## Setup and Reproduction

This project uses `uv` for modern, fast Python package management. 

1. **Install uv** (if missing):
   ```bash
   pip install uv
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```
   This will install all dependencies defined in `pyproject.toml` into a `.venv`.

3. **Download Data**:
   Ensure you have a Kaggle API key configured globally or locally (`~/.kaggle/kaggle.json`).
   ```bash
   source .venv/bin/activate
   kaggle competitions download -c home-credit-default-risk -p data
   unzip data/home-credit-default-risk.zip -d data/
   ```

4. **Run the Pipeline**:
   The full pipeline processes data, evaluates baseline models, engineers 130+ features, compares imbalance strategies (CV), and generates business metrics.
   ```bash
   uv run python src/pipeline.py
   ```

## Results

Detailed business impacts, optimal thresholds, and SHAP analyses are located in [reports/results.md](reports/results.md).

| Model Pipeline | Folds | CV AUC |
|----------------|-------|--------|
| Baseline       | 5     | 0.7568 |
| Final + SMOTE  | 5     | 0.7483 |
| Final + Weight | 5     | 0.7718 |
