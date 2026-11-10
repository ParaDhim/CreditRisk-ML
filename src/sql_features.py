"""Build the raw -> cleaned -> features layers in DuckDB and export the model table.

    python src/sql_features.py            # build + export data/features.parquet
    python src/sql_features.py --parity   # also check the output against the pandas pipeline

Run from the repository root (SQL files reference data/*.csv relatively).
"""
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "creditrisk.duckdb"
OUT_PATH = ROOT / "data" / "features.parquet"
LAYERS = ["01_raw.sql", "02_cleaned.sql", "03_features.sql"]

# Columns intentionally different from the pandas pipeline (see sql/02_cleaned.sql)
EXPECTED_DIFF = {"DAYS_EMPLOYED", "EMP_TO_AGE_RATIO", "CODE_GENDER", "ORGANIZATION_TYPE", "DAYS_EMPLOYED_ANOM"}


def build():
    con = duckdb.connect(str(DB_PATH))
    for f in LAYERS:
        t = time.time()
        con.execute((ROOT / "sql" / f).read_text())
        print(f"  {f}: {time.time() - t:.1f}s")
    con.execute(f"COPY features.model_table TO '{OUT_PATH}' (FORMAT PARQUET)")
    n, k = con.execute("SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM features.model_table").fetchone()
    assert n == k, "model_table must have one row per applicant"
    print(f"  model_table: {n:,} rows, one per SK_ID_CURR -> {OUT_PATH.relative_to(ROOT)}")
    con.close()


def parity_check():
    """Compare SQL features with src/pipeline.py:feature_engineering, column by column."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import load_data, feature_engineering

    print("Parity check vs pandas pipeline (loads all CSVs in pandas; slow)...")
    app, bureau, bb, prev, inst = load_data()
    ref = feature_engineering(app, bureau, bb, prev, inst).set_index("SK_ID_CURR").sort_index()
    sql = pd.read_parquet(OUT_PATH).set_index("SK_ID_CURR").sort_index()
    assert ref.index.equals(sql.index), "applicant sets differ"

    bad, checked = [], 0
    for col in ref.columns:           # ref already had >80%-missing columns dropped
        if col in EXPECTED_DIFF:
            continue
        a, b = ref[col], sql[col]
        if a.dtype == object or str(a.dtype) == "category":
            ok = (a.astype(object).where(a.notna(), '<NA>').astype(str).values
                  == b.astype(object).where(b.notna(), '<NA>').astype(str).values).all()
        else:
            a = a.replace([np.inf, -np.inf], np.nan).astype(float)
            b = b.replace([np.inf, -np.inf], np.nan).astype(float)
            ok = (a.isna() == b.isna()).all() and np.allclose(a.fillna(0), b.fillna(0), rtol=1e-6, atol=1e-6)
        checked += 1
        if not ok:
            bad.append(col)
    print(f"  checked {checked} columns, {len(bad)} mismatches {bad}")
    return bad


if __name__ == "__main__":
    if "--skip-build" not in sys.argv:
        print("Building DuckDB layers...")
        build()
    if "--parity" in sys.argv:
        mismatches = parity_check()
        sys.exit(1 if mismatches else 0)
