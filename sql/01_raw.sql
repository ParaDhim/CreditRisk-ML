-- LAYER 1: RAW
-- Views straight over the Kaggle CSVs. No casting rules, no fixes, no filtering:
-- whatever is in the file is what you see. sample_size=-1 makes DuckDB scan the whole
-- file for type inference so late-appearing values cannot break a column type;
-- auto_type_candidates stops "Yes"/"No" columns being silently turned into BOOLEAN.
CREATE SCHEMA IF NOT EXISTS raw;

CREATE OR REPLACE VIEW raw.application      AS SELECT * FROM read_csv('data/application_train.csv',     header=true, sample_size=-1, auto_type_candidates=['BIGINT','DOUBLE','VARCHAR']);
CREATE OR REPLACE VIEW raw.bureau           AS SELECT * FROM read_csv('data/bureau.csv',                header=true, sample_size=-1, auto_type_candidates=['BIGINT','DOUBLE','VARCHAR']);
CREATE OR REPLACE VIEW raw.bureau_balance   AS SELECT * FROM read_csv('data/bureau_balance.csv',        header=true, sample_size=-1, auto_type_candidates=['BIGINT','DOUBLE','VARCHAR']);
CREATE OR REPLACE VIEW raw.previous_application AS SELECT * FROM read_csv('data/previous_application.csv', header=true, sample_size=-1, auto_type_candidates=['BIGINT','DOUBLE','VARCHAR']);
CREATE OR REPLACE VIEW raw.installments     AS SELECT * FROM read_csv('data/installments_payments.csv', header=true, sample_size=-1, auto_type_candidates=['BIGINT','DOUBLE','VARCHAR']);
