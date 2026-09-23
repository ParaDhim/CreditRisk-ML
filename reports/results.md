# Business Framing Analysis

> [!WARNING]
> **Data Access Blocker**: The Kaggle "Home Credit Default Risk" data could not be downloaded due to a missing Kaggle API credential (`kaggle.json`) in the environment. Per instructions not to fabricate results, the below analyses are empty/placeholder. The code to generate these tables is implemented in `src/pipeline.py` and will populate once the data is present.

## Threshold Analysis

| Threshold | Approval Rate (%) | Recall (Defaults Caught %) | Precision (%) |
|-----------|-------------------|----------------------------|---------------|
| 0.1       | [Blocked]         | [Blocked]                  | [Blocked]     |
| 0.3       | [Blocked]         | [Blocked]                  | [Blocked]     |
| 0.5       | [Blocked]         | [Blocked]                  | [Blocked]     |
| 0.7       | [Blocked]         | [Blocked]                  | [Blocked]     |
| 0.9       | [Blocked]         | [Blocked]                  | [Blocked]     |

**Decision Impact:**
At a threshold of [X], we approve [Y]% of applicants while catching [Z]% of eventual defaults, trading off approval volume against expected loss. (Actual calculation blocked by missing data).

## Feature Importance (SHAP)

Top 5 Features modeled:
1. **[Blocked by Data Access]**: Would reflect the most influential credit feature, commonly an external score like `EXT_SOURCE_1`. Its presence typically confirms consistency with third-party assessments.
2. **[Blocked by Data Access]**: Usually relates to debt-to-income or another engineered ratio, capturing ability to pay directly.
3. **[Blocked by Data Access]**: Represents secondary external data or historical payment discipline.
4. **[Blocked by Data Access]**: E.g., employment length relative to age – captures career stability.
5. **[Blocked by Data Access]**: Indicates credit utilization or recent credit enquiries, which gauge recent credit hunger.
