"""Population Stability Index (PSI) and the retraining rule.

IMPORTANT: Home Credit has no application dates, so there is no real "later" population.
Everything here uses a SIMULATED split: a random holdout (which should show ~0 PSI and only
proves the plumbing) and deliberately tilted holdouts that mimic a shifting applicant mix.
They test the monitoring code and thresholds; they are not evidence of real-world drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WATCH, PSI_ACTION = 0.10, 0.25   # industry rule-of-thumb bands

RETRAIN_RULE = """\
Monitored monthly on the live applicant population against the training population:
  RETRAIN      if score PSI >= 0.25
               OR >= 3 of the top-10 model features have PSI >= 0.25
               OR (>= 500 matured defaults in the window AND observed/expected default
                   ratio outside [0.80, 1.25] for two consecutive months)
  INVESTIGATE  if score PSI in [0.10, 0.25) OR any top-10 feature PSI >= 0.25
  OK           otherwise
Outcomes take months to mature, so score/feature PSI is the early warning and
observed/expected defaults is the confirmation."""


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of `actual` vs `expected`, using quantile bins of `expected`; NaN is its own bin."""
    e, a = pd.Series(expected), pd.Series(actual)
    e_nn = e.dropna()
    cuts = np.unique(np.quantile(e_nn, np.linspace(0, 1, bins + 1)[1:-1]))
    def dist(s):
        idx = np.where(s.isna(), len(cuts) + 1, np.searchsorted(cuts, s.fillna(0).to_numpy(), side="left"))
        return np.bincount(idx, minlength=len(cuts) + 2) / len(s)
    pe, pa = np.clip(dist(e), eps, None), np.clip(dist(a), eps, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def psi_label(v: float) -> str:
    return "stable" if v < PSI_WATCH else ("watch" if v < PSI_ACTION else "shifted")


def retrain_decision(score_psi: float, feature_psis: dict, obs_to_exp: float = None,
                     n_defaults: int = 0, oe_breach_months: int = 0) -> tuple:
    """Return (status, reasons). Implements RETRAIN_RULE for one monitoring window."""
    shifted = [f for f, v in feature_psis.items() if v >= PSI_ACTION]
    reasons = []
    retrain = False
    if score_psi >= PSI_ACTION:
        retrain = True; reasons.append(f"score PSI {score_psi:.3f} >= {PSI_ACTION}")
    if len(shifted) >= 3:
        retrain = True; reasons.append(f"{len(shifted)} top features shifted: {', '.join(shifted)}")
    if (obs_to_exp is not None and n_defaults >= 500 and not 0.80 <= obs_to_exp <= 1.25
            and oe_breach_months >= 2):
        retrain = True; reasons.append(f"observed/expected defaults {obs_to_exp:.2f} for {oe_breach_months} months")
    if retrain:
        return "RETRAIN", reasons
    if PSI_WATCH <= score_psi < PSI_ACTION:
        reasons.append(f"score PSI {score_psi:.3f} in watch band")
    if shifted:
        reasons.append(f"feature(s) shifted: {', '.join(shifted)}")
    return ("INVESTIGATE", reasons) if reasons else ("OK", ["all indicators within bands"])
