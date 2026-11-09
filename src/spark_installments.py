"""PySpark port of the installments feature-engineering step (pipeline.py, step 3).

Generates a synthetic installments_payments table with the same schema as the
Home Credit data, runs the pandas and PySpark versions of the aggregation, checks
they agree, and benchmarks both. Runs in Spark local mode.

Usage: python src/spark_installments.py [rows ...]   (default: 1M 5M 10M)
"""
import sys
import time
import shutil
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

DATA_DIR = "data/synthetic"
ROWS_PER_CLIENT = 40


def get_spark():
    return (SparkSession.builder.master("local[*]").appName("installments-fe")
            .config("spark.driver.memory", "6g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def generate(spark, n_rows, path):
    """Synthetic installments with realistic nulls. Seeded, so reproducible."""
    n_clients = max(n_rows // ROWS_PER_CLIENT, 1)
    df = (spark.range(n_rows)
          .withColumn("SK_ID_CURR", (F.rand(1) * n_clients).cast("int") + 100000)
          .withColumn("DAYS_INSTALMENT", -(F.rand(2) * 2800).cast("int"))
          .withColumn("AMT_INSTALMENT", F.round(F.rand(3) * 20000 + 500, 2))
          # payment lands 0-30 days early to 40 days late; 0.5% never paid (null)
          .withColumn("DAYS_ENTRY_PAYMENT",
                      F.when(F.rand(4) < 0.005, F.lit(None).cast("double"))
                       .otherwise(F.col("DAYS_INSTALMENT") + (F.rand(5) * 70 - 30).cast("int")))
          .withColumn("AMT_PAYMENT",
                      F.when(F.col("DAYS_ENTRY_PAYMENT").isNull(), F.lit(None).cast("double"))
                       .otherwise(F.round(F.col("AMT_INSTALMENT") * (0.7 + F.rand(6) * 0.4), 2)))
          .drop("id"))
    df.write.mode("overwrite").parquet(path)


def pandas_features(installments):
    # Identical to pipeline.py feature_engineering step 3
    installments = installments.copy()
    installments['LATE_PAYMENT'] = (installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']).astype(int)
    installments['PAYMENT_SHORTFALL'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    inst_agg = installments.groupby('SK_ID_CURR').agg({
        'LATE_PAYMENT': 'mean',
        'PAYMENT_SHORTFALL': 'mean'
    })
    inst_agg.columns = ['INST_LATE_RATE', 'INST_SHORTFALL_MEAN']
    return inst_agg


def spark_features(installments):
    # pandas: NaN > x is False -> 0, so null comparisons must map to 0, not null
    late = F.when(F.col("DAYS_ENTRY_PAYMENT") > F.col("DAYS_INSTALMENT"), 1).otherwise(0)
    return (installments
            .withColumn("LATE_PAYMENT", late)
            .withColumn("PAYMENT_SHORTFALL", F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"))
            .groupBy("SK_ID_CURR")
            .agg(F.avg("LATE_PAYMENT").alias("INST_LATE_RATE"),
                 F.avg("PAYMENT_SHORTFALL").alias("INST_SHORTFALL_MEAN")))


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def run(spark, n_rows):
    path = f"{DATA_DIR}/installments_{n_rows}.parquet"
    generate(spark, n_rows, path)

    # Both timings are end-to-end: read Parquet + features + materialised result
    pd_res, t_pd = timed(lambda: pandas_features(pd.read_parquet(path)))

    def spark_job():
        res = spark_features(spark.read.parquet(path))
        return res.toPandas().set_index("SK_ID_CURR")
    sp_res, t_sp = timed(spark_job)

    pd.testing.assert_frame_equal(pd_res.sort_index(), sp_res.sort_index(),
                                  check_dtype=False, rtol=1e-6)
    return len(pd_res), t_pd, t_sp


if __name__ == "__main__":
    sizes = [int(float(x)) for x in sys.argv[1:]] or [1_000_000, 5_000_000, 10_000_000]
    spark = get_spark()
    spark_features(spark.range(1).selectExpr(  # warm up JVM so timings exclude startup
        "0 as SK_ID_CURR", "0 as DAYS_INSTALMENT", "0 as DAYS_ENTRY_PAYMENT",
        "1.0 as AMT_INSTALMENT", "1.0 as AMT_PAYMENT")).collect()

    print(f"{'rows':>12} | {'clients':>9} | {'pandas (s)':>10} | {'pyspark (s)':>11} | match")
    for n in sizes:
        clients, t_pd, t_sp = run(spark, n)
        print(f"{n:>12,} | {clients:>9,} | {t_pd:>10.2f} | {t_sp:>11.2f} | OK")
    spark.stop()
    shutil.rmtree(DATA_DIR, ignore_errors=True)
