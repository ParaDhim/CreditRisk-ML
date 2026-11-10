-- LAYER 2: CLEANED
-- Materialised tables. This is the only layer that fixes data problems or derives
-- row-level fields. Nothing is aggregated here; grain is unchanged from raw.
CREATE SCHEMA IF NOT EXISTS cleaned;

-- application: DAYS_EMPLOYED = 365243 is a placeholder (pensioners / unemployed), not
-- ~1000 years of employment. 'XNA' is the source's spelling of "unknown".
CREATE OR REPLACE TABLE cleaned.application AS
SELECT
    * REPLACE (
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        NULLIF(CODE_GENDER, 'XNA')       AS CODE_GENDER,
        NULLIF(ORGANIZATION_TYPE, 'XNA') AS ORGANIZATION_TYPE
    ),
    (DAYS_EMPLOYED = 365243)::INTEGER AS DAYS_EMPLOYED_ANOM
FROM raw.application;

-- bureau_balance rolled to one row per bureau credit, then attached to bureau
CREATE OR REPLACE TABLE cleaned.bureau AS
SELECT
    b.SK_ID_CURR, b.SK_ID_BUREAU, b.DAYS_CREDIT, b.CREDIT_DAY_OVERDUE,
    b.AMT_CREDIT_SUM, b.AMT_CREDIT_SUM_DEBT,
    bb.BB_COUNT
FROM raw.bureau b
LEFT JOIN (
    SELECT SK_ID_BUREAU, COUNT(MONTHS_BALANCE) AS BB_COUNT
    FROM raw.bureau_balance GROUP BY SK_ID_BUREAU
) bb USING (SK_ID_BUREAU);

CREATE OR REPLACE TABLE cleaned.previous_application AS
SELECT SK_ID_CURR, SK_ID_PREV, AMT_APPLICATION, AMT_CREDIT,
       (NAME_CONTRACT_STATUS = 'Approved')::INTEGER AS IS_APPROVED
FROM raw.previous_application;
-- NB: a NULL status yields NULL here; features layer treats it as "not approved",
-- matching the pandas pipeline, where (x == 'Approved') is False for NaN.

-- installments: row-level payment behaviour flags
CREATE OR REPLACE TABLE cleaned.installments AS
SELECT
    SK_ID_CURR,
    CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END AS LATE_PAYMENT,
    AMT_INSTALMENT - AMT_PAYMENT AS PAYMENT_SHORTFALL
FROM raw.installments;
