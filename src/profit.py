"""Cost-based approval threshold.

ASSUMPTIONS (not facts about Home Credit's economics; the dataset has no revenue or loss data):
  MARGIN_PCT : net profit on a loan that repays, as a share of AMT_CREDIT      -> ASSUMPTION
  LGD        : share of AMT_CREDIT lost when a loan defaults (net of recovery) -> ASSUMPTION
  Declined applicants earn and lose nothing.
Profit per approved applicant = (1 - default) * MARGIN_PCT * credit - default * LGD * credit.
If PD is calibrated, approving is worthwhile iff PD < MARGIN_PCT / (MARGIN_PCT + LGD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN_PCT = 0.10   # ASSUMPTION
LGD = 0.60          # ASSUMPTION
MARGIN_GRID = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]   # ASSUMPTION range
LGD_GRID = [0.30, 0.45, 0.60, 0.75, 0.90]                   # ASSUMPTION range


def breakeven_pd(margin: float, lgd: float) -> float:
    return margin / (margin + lgd)


def profit_at(y, pd_hat, credit, threshold, margin=MARGIN_PCT, lgd=LGD) -> dict:
    """Realised profit on labelled data if everyone with pd_hat <= threshold is approved."""
    y, pd_hat, credit = map(np.asarray, (y, pd_hat, credit))
    ok = pd_hat <= threshold
    good_rev = margin * credit[ok & (y == 0)].sum()
    loss = lgd * credit[ok & (y == 1)].sum()
    return dict(threshold=threshold, approval_rate=ok.mean(), defaults_caught=1 - y[ok].sum() / y.sum(),
                bad_rate_approved=y[ok].mean() if ok.any() else np.nan,
                revenue=good_rev, expected_loss=loss, profit=good_rev - loss)


def profit_curve(y, pd_hat, credit, thresholds, margin=MARGIN_PCT, lgd=LGD) -> pd.DataFrame:
    return pd.DataFrame([profit_at(y, pd_hat, credit, t, margin, lgd) for t in thresholds])


def optimal_threshold(y, pd_hat, credit, margin=MARGIN_PCT, lgd=LGD) -> dict:
    """Exact optimum over all cutoffs via cumulative sums (applicants sorted by PD)."""
    y, pd_hat, credit = map(np.asarray, (y, pd_hat, credit))
    o = np.argsort(pd_hat, kind="mergesort")
    a = np.cumsum(((1 - y) * credit)[o])      # credit of good loans approved
    b = np.cumsum((y * credit)[o])            # credit of defaulted loans approved
    prof = margin * a - lgd * b
    k = int(np.argmax(prof))
    if prof[k] <= 0:
        return dict(threshold=0.0, approval_rate=0.0, profit=0.0, approve_all_profit=float(prof[-1]))
    return dict(threshold=float(pd_hat[o][k]), approval_rate=(k + 1) / len(y), profit=float(prof[k]),
                approve_all_profit=float(prof[-1]))


def sensitivity_grid(y, pd_hat, credit, margins=MARGIN_GRID, lgds=LGD_GRID) -> pd.DataFrame:
    rows = []
    for m in margins:
        for l in lgds:
            r = optimal_threshold(y, pd_hat, credit, m, l)
            rows.append(dict(margin=m, lgd=l, breakeven_pd=breakeven_pd(m, l), best_threshold=r["threshold"],
                             approval_rate=r["approval_rate"], profit=r["profit"],
                             approve_all_profit=r["approve_all_profit"],
                             uplift_vs_approve_all=r["profit"] - r["approve_all_profit"]))
    return pd.DataFrame(rows)
