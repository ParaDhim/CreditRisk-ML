"""Weight-of-evidence (WoE) binning, information value (IV), logistic regression scorecard.

Convention: WoE = ln(%good / %bad) per bin, so a higher WoE means a safer bin. A logistic
regression on default therefore has *negative* coefficients; a positive one means a feature
is behaving against its own binning and is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

SMOOTH = 0.5   # added to good/bad counts per bin so empty cells don't give infinite WoE


def _woe_iv(good: np.ndarray, bad: np.ndarray):
    G, B = good.sum(), bad.sum()
    dg, db = (good + SMOOTH) / G, (bad + SMOOTH) / B
    woe = np.log(dg / db)
    return woe, float(((dg - db) * woe).sum())


class WoEBinner:
    """Bins one feature on the training data and maps values to WoE.

    Numeric: decision-tree cut points (<= max_bins, each bin >= min_bin_frac of non-null rows),
    then adjacent bins are merged until bad rate is monotone in the feature. Missing values
    get their own bin. Categorical: one bin per level, rare levels pooled into 'OTHER'.
    """

    def __init__(self, max_bins: int = 6, min_bin_frac: float = 0.05, rare_frac: float = 0.01):
        self.max_bins, self.min_bin_frac, self.rare_frac = max_bins, min_bin_frac, rare_frac

    # ---- fit -------------------------------------------------------------------------
    def fit(self, x: pd.Series, y: pd.Series):
        y = np.asarray(y)
        self.numeric = pd.api.types.is_numeric_dtype(x) and not isinstance(x.dtype, pd.CategoricalDtype)
        if self.numeric:
            self._fit_numeric(x.astype(float).to_numpy(), y)
        else:
            self._fit_categorical(x.astype(object).where(x.notna(), None).to_numpy(), y)
        return self

    def _fit_numeric(self, x, y):
        ok = ~np.isnan(x)
        xv, yv = x[ok], y[ok]
        self.cuts = []
        if len(np.unique(xv)) > 1:
            tree = DecisionTreeClassifier(max_leaf_nodes=self.max_bins,
                                          min_samples_leaf=max(int(self.min_bin_frac * len(xv)), 50),
                                          random_state=0).fit(xv.reshape(-1, 1), yv)
            self.cuts = sorted(tree.tree_.threshold[tree.tree_.feature >= 0].tolist())
            direction = np.sign(spearmanr(xv, yv)[0]) if len(self.cuts) else 0
            self.cuts = self._merge_to_monotone(xv, yv, self.cuts, direction)
        idx = np.where(ok, np.searchsorted(self.cuts, x, side="left"), len(self.cuts) + 1)
        n_bins = len(self.cuts) + 2   # last index = missing
        self._finish(idx, y, n_bins)
        edges = [-np.inf] + self.cuts + [np.inf]
        self.labels = [f"({edges[i]:.4g}, {edges[i+1]:.4g}]" for i in range(len(self.cuts) + 1)] + ["MISSING"]

    @staticmethod
    def _merge_to_monotone(x, y, cuts, direction):
        if direction == 0:
            return cuts
        while len(cuts) > 0:
            idx = np.searchsorted(cuts, x, side="left")
            rates = np.array([y[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(cuts) + 1)])
            diffs = np.diff(rates)
            viol = np.where(diffs * direction < 0)[0]   # direction>0 => bad rate should rise with x
            if len(viol) == 0:
                break
            cuts = cuts[:viol[0]] + cuts[viol[0] + 1:]    # merge bin viol[0] with its right neighbour
        return cuts

    def _fit_categorical(self, x, y):
        s = pd.Series(x).fillna("MISSING")
        freq = s.value_counts(normalize=True)
        self.keep = set(freq[freq >= self.rare_frac].index) | {"MISSING"}
        s = s.where(s.isin(self.keep), "OTHER")
        codes = pd.Categorical(s)
        self.labels = list(codes.categories)
        self._finish(codes.codes, y, len(self.labels))

    def _finish(self, idx, y, n_bins):
        good = np.bincount(idx[y == 0], minlength=n_bins).astype(float)
        bad = np.bincount(idx[y == 1], minlength=n_bins).astype(float)
        self.woe, self.iv = _woe_iv(good, bad)
        self.counts, self.bad_rate = good + bad, bad / np.maximum(good + bad, 1)

    # ---- transform ---------------------------------------------------------------------
    def bin_index(self, x: pd.Series) -> np.ndarray:
        if self.numeric:
            v = x.astype(float).to_numpy()
            return np.where(np.isnan(v), len(self.cuts) + 1, np.searchsorted(self.cuts, v, side="left"))
        s = x.astype(object).where(x.notna(), "MISSING")
        s = s.where(s.isin(self.keep), "OTHER")
        return s.map({l: i for i, l in enumerate(self.labels)}).fillna(-1).astype(int).to_numpy()

    def transform(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        return np.where(idx >= 0, self.woe[np.clip(idx, 0, None)], 0.0)   # unseen level -> neutral


def fit_binners(X: pd.DataFrame, y: pd.Series, **kw) -> dict:
    out = {}
    for c in X.columns:
        if X[c].nunique(dropna=True) < 2:
            continue
        out[c] = WoEBinner(**kw).fit(X[c], y)
    return out


def iv_table(binners: dict) -> pd.DataFrame:
    t = pd.DataFrame({"feature": list(binners), "iv": [b.iv for b in binners.values()],
                      "n_bins": [len(b.woe) for b in binners.values()]})
    t["strength"] = pd.cut(t["iv"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                           labels=["useless (<0.02)", "weak", "medium", "strong", "suspicious (>0.5)"])
    return t.sort_values("iv", ascending=False).reset_index(drop=True)


def select_features(binners, Wtrain: pd.DataFrame, iv_min=0.02, iv_max=0.5, corr_max=0.6, max_features=20):
    """IV filter -> greedy correlation prune on WoE columns (highest IV wins)."""
    cand = [c for c, b in sorted(binners.items(), key=lambda kv: -kv[1].iv) if iv_min <= b.iv <= iv_max]
    kept = []
    corr = Wtrain[cand].corr().abs()
    for c in cand:
        if all(corr.loc[c, k] < corr_max for k in kept):
            kept.append(c)
        if len(kept) == max_features:
            break
    return kept


class Scorecard:
    """Logistic regression on WoE features, converted to points (PDO scaling)."""

    def __init__(self, base_score=600, base_odds=50, pdo=20, C=1.0):
        self.base_score, self.base_odds, self.pdo, self.C = base_score, base_odds, pdo, C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, binners: dict, W: pd.DataFrame, y, features: list):
        feats = list(features)
        while True:
            lr = LogisticRegression(C=self.C, max_iter=1000).fit(W[feats], y)
            wrong = [f for f, c in zip(feats, lr.coef_[0]) if c > 0]
            if not wrong:
                break
            worst = max(wrong, key=lambda f: lr.coef_[0][feats.index(f)])
            feats.remove(worst)
        self.features, self.lr, self.binners = feats, lr, binners
        return self

    def _W(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.binners[f].transform(X[f]) for f in self.features}, index=X.index)

    def predict_pd(self, X):
        return self.lr.predict_proba(self._W(X))[:, 1]

    def score(self, X):
        """Points: higher = safer. Equals offset + factor * ln(odds good:bad) of the model."""
        z = self.lr.decision_function(self._W(X))
        return self.offset - self.factor * z

    def points_table(self) -> pd.DataFrame:
        n, b0 = len(self.features), self.lr.intercept_[0]
        rows = []
        for f, coef in zip(self.features, self.lr.coef_[0]):
            bn = self.binners[f]
            for i, lab in enumerate(bn.labels):
                pts = -(coef * bn.woe[i] + b0 / n) * self.factor + self.offset / n
                rows.append(dict(feature=f, bin=lab, count=int(bn.counts[i]), bad_rate=bn.bad_rate[i],
                                 woe=bn.woe[i], coef=coef, points=round(pts)))
        return pd.DataFrame(rows)
