"""Standard credit-risk model metrics: GINI, KS, gain/lift, decile table, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks_stat(y, p) -> float:
    """Max separation between cumulative bad and cumulative good distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def decile_table(y, risk, n_bins: int = 10) -> pd.DataFrame:
    """Rank by predicted risk, decile 1 = riskiest. Gain = cumulative share of all defaults captured."""
    d = pd.DataFrame({"y": np.asarray(y), "risk": np.asarray(risk)})
    d = d.sort_values("risk", ascending=False, kind="mergesort").reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * n_bins // len(d)) + 1
    g = d.groupby("decile").agg(applicants=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean"),
                                avg_pd=("risk", "mean"), min_risk=("risk", "min"), max_risk=("risk", "max"))
    base = d["y"].mean()
    g["lift"] = g["default_rate"] / base
    g["cum_defaults_captured"] = g["defaults"].cumsum() / d["y"].sum()
    g["cum_lift"] = g["cum_defaults_captured"] / (g["applicants"].cumsum() / len(d))
    return g.reset_index()


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability vs observed default rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()


def calibration_report(y, p, n_bins: int = 10) -> dict:
    """Brier, Brier skill vs base rate, ECE, calibration slope/intercept, observed/expected."""
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    t = calibration_table(y, p, n_bins)
    ece = float((t["n"] * (t["mean_pred"] - t["observed"]).abs()).sum() / t["n"].sum())
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, y.mean()))
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(logit, y)   # ~unpenalised
    return dict(brier=brier, brier_skill=1 - brier / brier_ref, ece=ece,
                cal_slope=float(lr.coef_[0][0]), cal_intercept=float(lr.intercept_[0]),
                mean_pred=float(p.mean()), observed=float(y.mean()), obs_to_exp=float(y.mean() / p.mean()))


# Explicit pass criteria, used before any cost-based threshold is trusted.
CAL_PASS = dict(ece_max=0.01, slope_range=(0.8, 1.2), obs_exp_range=(0.9, 1.1))


def calibration_passes(rep: dict) -> bool:
    return (rep["ece"] <= CAL_PASS["ece_max"]
            and CAL_PASS["slope_range"][0] <= rep["cal_slope"] <= CAL_PASS["slope_range"][1]
            and CAL_PASS["obs_exp_range"][0] <= rep["obs_to_exp"] <= CAL_PASS["obs_exp_range"][1])


def bootstrap_auc_ci(y, p, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))
