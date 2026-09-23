# Business Framing Analysis

## Threshold Analysis

| Threshold | Approval Rate (%) | Recall (Defaults Caught %) | Precision (%) |
|-----------|-------------------|----------------------------|---------------|
| 0.10      | 7.52              | 99.23                      | 8.66          |
| 0.30      | 45.47             | 86.73                      | 12.84         |
| 0.50      | 72.33             | 64.25                      | 18.74         |
| 0.70      | 90.85             | 31.72                      | 28.00         |
| 0.90      | 99.96             | 0.28                       | 58.33         |

**Decision Impact:**
At a threshold of 0.50, we approve 72.33% of applicants while catching 64.25% of eventual defaults, trading off approval volume against expected loss. Utilizing a stricter 0.30 threshold scales our default capture rate to nearly 87%, but introduces significant friction by bottlenecking our approval rate to 45.47%, demanding a structural analysis of lifetime-value per customer versus default attrition costs.

## Feature Importance

Top 5 Features modeled (by SHAP Value):
1. **EXT_SOURCES_MEAN**: The arithmetic mean of the normalized external scores; aggregating historical third-party bureau metrics is universally the strongest baseline predictor of default in this dataset.
2. **ORGANIZATION_TYPE**: Categorical capture of the applicant's employer structure (e.g., Business Entity vs Self-Employed), directly segmenting high-volatility income structures from stable ones.
3. **CREDIT_TO_ANNUITY**: An engineered ratio linking total credit requested to the annuity structure; highly over-leveraged borrowers consistently showcase degraded repayment integrity.
4. **INST_LATE_RATE**: An aggregated feature counting the historical frequency where payment dates exceeded expected installments, accurately signaling creeping cash flow strains.
5. **OWN_CAR_AGE**: Explains risk variance tied to asset depreciation; an older or missing vehicle metric acts as an implicit proxy for liquid wealth and historical credit capacity.
