# Risk Analysis: Scorecard, Calibration, Cost-Based Threshold, Monitoring

Everything here is computed by `src/run_upgrade.py` on a stratified random split of the Home Credit training file (55% train / 15% calibration / 30% holdout = 92,254 applicants, 8.07% default rate). All numbers below are holdout numbers. Raw tables are in `reports/artifacts/`, charts in `reports/figures/`.

> **Caveats that apply to everything**
> * The split is **random, not out-of-time**: the dataset has no application dates.
> * Home Credit's file only contains applicants who were **approved**, so "declined" applicants' outcomes are never observed (no reject inference). Profit numbers describe re-ranking *the people who were approved*, not a full through-the-door population.
> * Revenue and loss are **not in the data**. Margin and LGD below are **assumptions**.
> * `CODE_GENDER` is in the scorecard because the original model used it. A production lender would need a fair-lending review before using it.

## 1. Scorecard vs LightGBM

WoE = ln(%good / %bad) per bin; numeric features binned by decision-tree cuts and merged until the bad rate is monotone; missing values get their own bin. Features kept: IV between 0.02 and ~0.6, pairwise WoE correlation < 0.6, and a negative LR coefficient (a positive one means the feature contradicts its own binning). 139 features binned, 61 had IV >= 0.02, **18 make the final scorecard**. Points use 600 at 50:1 good:bad odds, 20 points to double the odds (`reports/artifacts/scorecard_points.csv`).

`EXT_SOURCES_MEAN` has IV 0.61, above the usual 0.5 "check for leakage" line. It is the mean of three external bureau scores available at application time, so it was reviewed and kept.

| Model | AUC (95% CI) | GINI | KS | Brier | ECE |
|---|---|---|---|---|---|
| Scorecard (WoE + LR, 18 features) | 0.745 (0.740–0.750) | 0.490 | 0.362 | 0.0688 | 0.001 |
| LightGBM, `scale_pos_weight=10`, raw | 0.770 (0.764–0.774) | 0.541 | 0.410 | 0.1627 | **0.274** |
| LightGBM, calibrated (Platt) | 0.770 (0.764–0.774) | 0.541 | 0.410 | 0.0671 | 0.003 |
| LightGBM, unweighted | 0.771 (0.765–0.775) | 0.542 | 0.408 | 0.0670 | 0.003 |

The gradient-boosted model buys about 2.5 AUC points (0.045 GINI) over a transparent 18-variable scorecard. That is the price of using a scorecard where explainability or regulation demands one.

## 2. Decile table (LightGBM, decile 1 = riskiest)

| Decile | Default rate | Lift | Cumulative defaults caught |
|---|---|---|---|
| 1 | 28.2% | 3.50 | 35.0% |
| 2 | 15.3% | 1.90 | 54.0% |
| 3 | 10.5% | 1.30 | 66.9% |
| 4 | 8.0% | 0.99 | 76.8% |
| 5 | 5.7% | 0.70 | 83.9% |
| 6 | 4.2% | 0.52 | 89.1% |
| 7 | 3.3% | 0.40 | 93.1% |
| 8 | 2.7% | 0.33 | 96.4% |
| 9 | 1.8% | 0.22 | 98.6% |
| 10 | 1.2% | 0.14 | 100% |

The riskiest decile defaults at 28% versus 1.2% for the safest, a 24x spread; the riskiest 20% of applicants hold 54% of defaults. Figures: `gain_lift.png`, `ks.png`.

## 3. Calibration

Pass criteria, fixed before looking at results: ECE <= 0.01, calibration slope in [0.8, 1.2], observed/expected default rate in [0.9, 1.1].

* **The original weighted model fails.** `scale_pos_weight=10` ranks well (AUC 0.770) but inflates probabilities: it predicts an average 35.5% default probability against an actual 8.07% (observed/expected 0.23, ECE 0.27, Brier worse than predicting the base rate for everyone). Its "0.50 threshold" is not a 50% chance of default.
* A Platt rescaling fitted on the separate calibration set fixes this without changing the ranking (same AUC): ECE 0.003, slope 1.01, observed/expected 1.00. **Passes.**
* The scorecard is naturally calibrated (logistic regression).

Cost-based thresholds below use only the calibrated probabilities. Figure: `calibration.png`.

## 4. Cost-based threshold

**ASSUMPTIONS** (labelled as such everywhere they appear): a loan that repays earns a margin of **10% of AMT_CREDIT**; a default loses **60% of AMT_CREDIT** (loss given default); declined applicants earn and lose nothing. With calibrated PD, approving is worthwhile exactly when PD < margin / (margin + LGD) = **0.143**.

| Policy (holdout, currency units) | Approval | Defaults caught | Profit |
|---|---|---|---|
| Approve everyone | 100% | 0% | 2,624M |
| Old rule: raw weighted score <= 0.50 | 73.5% | 62.8% | 3,028M |
| Calibrated PD <= 0.143 (breakeven, no tuning) | 82.6% | 49.7% | **3,115M** (+18.7% vs approve-all) |
| Calibrated PD <= 0.140 (best on this holdout) | 82.1% | 50.6% | 3,119M |

The best cutoff found by searching the holdout (0.140) is almost exactly the theoretical breakeven (0.143). That agreement is the practical payoff of calibration. Prefer the breakeven figure when quoting impact, since the "best" cutoff was chosen on the same data it is scored on.

**Sensitivity** (`threshold_sensitivity.png`, `threshold_sensitivity.csv`): over margin 4–20% and LGD 30–90%, the best cutoff ranges from 0.049 (approve 48%) to 0.340 (approve 98%). It rises with margin and falls with LGD. The cutoff is driven by the margin-to-loss ratio, so these two assumptions deserve more scrutiny than any model tuning. At a 0.5 cutoff on calibrated PD the policy would be nearly "approve everyone".

## 5. Monitoring (PSI) and the retraining rule

**The split is simulated.** Home Credit has no dates, so there is no genuine later population. A random holdout is used for plumbing and tilted resamples of the holdout (younger applicants, weaker external scores) mimic a shifting applicant mix. These test the monitoring code and thresholds; they are not evidence of real drift.

PSI = sum over bins of (actual% - expected%) x ln(actual% / expected%), with 10 training-quantile bins and a separate bin for missing. Bands: < 0.10 stable, 0.10–0.25 watch, >= 0.25 shifted.

| Scenario | Score PSI | Max top-10 feature PSI | Default rate | Obs/Exp | Rule outcome |
|---|---|---|---|---|---|
| A. random holdout (no drift) | 0.0005 | 0.0002 | 8.07% | 1.00 | OK |
| B. mild: slightly younger | 0.019 | 0.101 | 9.14% | 1.00 | OK |
| C. moderate: younger + weaker ext. scores | 0.212 | 0.310 | 12.4% | 1.01 | INVESTIGATE |
| D. large: same, stronger | 0.584 | 0.858 | 15.6% | 1.01 | RETRAIN |

**Retraining rule** (implemented in `src/monitoring.py::retrain_decision`):

```
RETRAIN      if score PSI >= 0.25
             OR >= 3 of the top-10 model features have PSI >= 0.25
             OR (>= 500 matured defaults in the window AND observed/expected default
                 ratio outside [0.80, 1.25] for two consecutive months)
INVESTIGATE  if score PSI in [0.10, 0.25) OR any top-10 feature PSI >= 0.25
OK           otherwise
```

Score and feature PSI are the early warning (available at application time); observed/expected defaults is the confirmation once outcomes mature. Note that in scenarios C and D the model stays calibrated (obs/exp ~1.0), because only the applicant mix moved and the relationship between features and default did not. PSI flags the change in population; it would not tell you if that relationship had broken, which is why the O/E trigger exists.

## 6. SQL layer

`sql/01_raw.sql` (views over the CSVs), `sql/02_cleaned.sql` (typed tables, row-level fixes and flags), `sql/03_features.sql` (one row per applicant). `python src/sql_features.py --parity` rebuilds in DuckDB and checks the result against the pandas pipeline column by column: **136 columns compared, 0 mismatches**. Five columns are intentionally excluded from that check because cleaning changes them on purpose: the `DAYS_EMPLOYED = 365243` placeholder becomes NULL (plus a `DAYS_EMPLOYED_ANOM` flag, and the ratio `EMP_TO_AGE_RATIO` derived from it), and `'XNA'` becomes NULL in `CODE_GENDER` and `ORGANIZATION_TYPE`. Division by zero yields NULL in SQL where pandas yields `inf`. The model in this document is trained on the SQL-built features.
