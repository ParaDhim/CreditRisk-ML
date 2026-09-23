# Credit Default Risk Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

An end-to-end, reproducible credit default risk classifier based on the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk). This project demonstrates the full lifecycle of an applied Machine Learning project in the Credit Risk domain: from multi-table data aggregation and robust feature engineering, to handling extreme class imbalances, and finally translating statistical outputs into business-actionable threshold strategies.

---

## Executive Summary
Banks and consumer finance providers rely on predictive models to evaluate an applicant's likelihood of defaulting on a loan. A slight increase in predictive accuracy significantly trims bad debt and boosts overall margin. 

In this project, I:
1. **Engineered 139 domain-specific features** across 300,000+ applicants using historical credit bureau, installment, and POS-cash records.
2. **Battled an ~8% class imbalance** by pitting SMOTE oversampling against dynamically weighted loss schemas.
3. **Achieved a robust 0.77 ROC-AUC** utilizing 5-fold stratified cross-validation on a LightGBM backend.
4. **Translated the model output** into a business-facing approval/decline cost-analysis table, optimizing expected approval volumes against default capture rates.

---

## Impact & Results

To bridge the gap between machine learning and business operations, the model's raw probabilities were mapped out into a decision matrix:

| Threshold | Approval Rate (%) | Recall (Defaults Caught %) | Precision (%) |
|-----------|-------------------|----------------------------|---------------|
| **0.30**  | 45.47             | 86.73                      | 12.84         |
| **0.50**  | 72.33             | 64.25                      | 18.74         |
| **0.70**  | 90.85             | 31.72                      | 28.00         |

* **The Business Trade-Off**: Operating at a strict **0.30** classification threshold allows the business to mathematically avoid ~87% of all potential eventual defaults, at the steep cost of declining over half the applicant pool. Shifting the risk appetite to **0.50** optimizes revenue flow, approving ~72% of applicants while still efficiently intercepting ~64% of bad debt.

**Key Driving Variables (SHAP)**:
The model heavily relied on explicit historical behavioral ratios: 
* `EXT_SOURCES_MEAN`: An arithmetic average of normalized external bureau risk scores.
* `CREDIT_TO_ANNUITY`: A custom-engineered ratio measuring extreme borrower over-leverage relative to their structured annuity commitments.
* `INST_LATE_RATE`: A behavioral tracker counting historic frequency of slipping payment schedules.

*(For an in-depth breakdown, see the [results write-up](reports/results.md))*

---

## Technical Architecture & Pipeline

### 1. Data Aggregation & Feature Engineering
- **Multi-Table Joins**: Rolled up granular, time-series historical data (bureau records, previous applications, daily installments) into single applicant-level (`SK_ID_CURR`) profiles.
- **Domain Ratios**: Hand-crafted variables like *Employment-to-Age Ratio*, *Debt-to-Income*, and *Payment Shortfalls* rather than relying purely on blunt statistical aggregates. 
- **Dimensionality Control**: Aggressively pruned sparsely populated domains (filtering columns with >80% missingness).

### 2. Experimental Modeling Methods
Tree-based boosted frameworks (LightGBM) serve as the backbone due to their efficiency with tabular missing/categorical schemas. 
Evaluated imbalance strategies head-to-head out-of-fold:
- **Baseline Model**: Raw applicant data without relational joins (`0.7568 +/- 0.0042` AUC).
- **SMOTE Resampling**: Applied strictly within the training folds (to avoid synthetic data leakage into validation).
- **Weighted Loss (Winner)**: Utilized native algorithmic cost-weighting (`scale_pos_weight`), capturing complex minority structures naturally and achieving the peak **0.7718 AUC**.

---

## Setup & Reproduction Guide

This pipeline utilizes `uv` to ensure blazingly fast, locked Python environments exactly as executed.

### Prerequisites:
1. Ensure `uv` is installed globally: `pip install uv`
2. Configure your Kaggle API key (either dynamically exported in `.env` as `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`, or saved securely in `~/.kaggle/kaggle.json`). Note: *You must explicitly click "Accept Rules" on the Kaggle competition page before downloading.*

### Execution:

```bash
# 1. Clone Repo & Enter Directory
git clone https://github.com/ParaDhim/CreditRisk-ML.git
cd CreditRisk-ML

# 2. Sync identical python environment (.venv is built instantly)
uv sync

# 3. Download Kaggle Dataset organically 
source .venv/bin/activate
kaggle competitions download -c home-credit-default-risk -p data
unzip data/home-credit-default-risk.zip -d data/

# 4. Spin up the ML Training & Generation Pipeline
uv run python src/pipeline.py
```

---

## Repository Structure
```
CreditRisk-ML/
|
|-- src/
|   \-- pipeline.py           # Core modeling logic, CV, & Feature Engineering
|-- data/
|   \-- README.md             # Dataset instructions & raw CSV dump 
|-- reports/
|   \-- results.md            # Deep-dive on thresholds, SHAP, and findings
|-- pyproject.toml            # Strict uv dependencies map
\-- resume_bullet.md          # 1-liner ATS-optimized project resume chunk
```

---

## License
This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.
