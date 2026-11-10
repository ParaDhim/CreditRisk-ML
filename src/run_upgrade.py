"""Run the full risk-analyst analysis on data/features.parquet (built by src/sql_features.py).

    python src/run_upgrade.py

Writes figures to reports/figures/ and tables + holdout scores to reports/artifacts/.
Split (stratified random, seed 42): 55% train / 15% calibration / 30% holdout.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitoring as mon
import profit as pf
import risk_metrics as rm
import scorecard as sc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG, ART = ROOT / "reports" / "figures", ROOT / "reports" / "artifacts"
BLUE, ORANGE, GREY, RED = "#2a6fbb", "#d9822b", "#8a8f98", "#c0392b"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------- data + split
def load():
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    miss = df.isnull().mean()
    df = df.drop(columns=[c for c in miss[miss > 0.8].index if c != "TARGET"])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category")
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    idx_tr, idx_ho = train_test_split(df.index, test_size=0.30, stratify=y, random_state=42)
    idx_tr, idx_ca = train_test_split(idx_tr, test_size=0.15 / 0.70, stratify=y[idx_tr], random_state=42)
    return X, y, idx_tr, idx_ca, idx_ho


def train_lgb(X, y, tr, ca, spw):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, colsample_bytree=0.8,
                  subsample=0.8, subsample_freq=1, scale_pos_weight=spw, metric="auc",
                  verbosity=-1, seed=42)
    dtr = lgb.Dataset(X.loc[tr], y[tr])
    dca = lgb.Dataset(X.loc[ca], y[ca], reference=dtr)
    return lgb.train(params, dtr, 2000, valid_sets=[dca],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


# ------------------------------------------------------------------------------ figures
def fig_iv(iv):
    top = iv.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["feature"], top["iv"], color=BLUE)
    for x in (0.02, 0.1, 0.3):
        ax.axvline(x, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Information value (training set)")
    ax.set_title("Top 15 features by information value")
    save(fig, "iv_top15.png")


def fig_gain_lift(tabs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for (name, t), col in zip(tabs.items(), (BLUE, ORANGE)):
        x = np.r_[0, t["applicants"].cumsum() / t["applicants"].sum()]
        a1.plot(x, np.r_[0, t["cum_defaults_captured"]], color=col, lw=2, label=name)
        a2.plot(t["decile"], t["lift"], color=col, lw=2, marker="o", ms=5, label=name)
    a1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="random")
    a1.set(xlabel="Share of applicants declined (riskiest first)", ylabel="Share of all defaults caught",
           title="Cumulative gain")
    a2.axhline(1, color=GREY, ls="--", lw=1)
    a2.set(xlabel="Risk decile (1 = riskiest)", ylabel="Default rate / base rate", title="Lift by decile",
           xticks=range(1, 11))
    a1.legend(frameon=False); a2.legend(frameon=False)
    save(fig, "gain_lift.png")


def fig_ks(y, p, name):
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    cb = (d["y"] == 1).cumsum() / (d["y"] == 1).sum()
    cg = (d["y"] == 0).cumsum() / (d["y"] == 0).sum()
    x = np.arange(len(d)) / len(d)
    k = (cg - cb).to_numpy(); i = int(np.argmax(k))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, cg, color=BLUE, lw=2, label="good loans"); ax.plot(x, cb, color=ORANGE, lw=2, label="defaults")
    ax.vlines(x[i], cb.iloc[i], cg.iloc[i], color=RED, lw=2)
    ax.annotate(f"KS = {k[i]:.3f}", (x[i], (cg.iloc[i] + cb.iloc[i]) / 2), xytext=(8, 0),
                textcoords="offset points", color=RED, fontweight="bold")
    ax.set(xlabel="Applicants sorted from safest to riskiest", ylabel="Cumulative share",
           title=f"KS separation: {name}")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "ks.png")


def fig_calibration(cals):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cols = {"LightGBM raw (scale_pos_weight=10)": RED, "LightGBM calibrated (Platt)": BLUE, "Scorecard": ORANGE}
    for name, t in cals.items():
        ax.plot(t["mean_pred"], t["observed"], marker="o", ms=5, lw=1.8, color=cols[name], label=name)
    lim = max(max(t["mean_pred"].max(), t["observed"].max()) for t in cals.values()) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1, label="perfect")
    ax.set(xlabel="Mean predicted default probability (decile bin)", ylabel="Observed default rate",
           title="Calibration on holdout", xlim=(0, lim), ylim=(0, lim))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "calibration.png")


def fig_profit(curve, opt, be, old_rule):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve["threshold"], curve["profit"] / 1e6, color=BLUE, lw=2)
    ax.axvline(opt["threshold"], color=ORANGE, lw=1.5, ls="--")
    ax.annotate(f"best cutoff PD <= {opt['threshold']:.3f}\napproves {opt['approval_rate']:.0%}",
                (opt["threshold"], opt["profit"] / 1e6), xytext=(12, -34), textcoords="offset points", color=ORANGE)
    ax.axvline(be, color=GREY, lw=1, ls=":")
    ax.annotate(f"theoretical breakeven {be:.3f}", (be, ax.get_ylim()[0]), xytext=(4, 12),
                textcoords="offset points", color=GREY, fontsize=8)
    ax.axhline(opt["approve_all_profit"] / 1e6, color=GREY, lw=1)
    ax.annotate("approve everyone", (curve["threshold"].max(), opt["approve_all_profit"] / 1e6), ha="right",
                xytext=(0, -12), textcoords="offset points", color=GREY, fontsize=8)
    ax.set(xlabel="Approval cutoff on calibrated PD (approve if PD <= cutoff)",
           ylabel="Realised profit on holdout (millions, currency units)",
           title=f"Profit vs cutoff (ASSUMED margin {pf.MARGIN_PCT:.0%}, LGD {pf.LGD:.0%})")
    save(fig, "profit_curve.png")


def fig_sensitivity(sens):
    piv = sens.pivot(index="lgd", columns="margin", values="best_threshold").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:.0%}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:.0%}" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > np.nanmean(piv.values) else "#222")
    ax.grid(False)
    ax.set(xlabel="ASSUMED margin on good loan (% of credit)", ylabel="ASSUMED loss given default",
           title="Profit-maximising PD cutoff under different assumptions")
    save(fig, "threshold_sensitivity.png")


def fig_psi(psi_tab):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = psi_tab["scenario"].tolist()
    ax.bar(order, psi_tab["score_psi"], color=[BLUE if v < .1 else ORANGE if v < .25 else RED
                                               for v in psi_tab["score_psi"]], width=0.6)
    for y_, c in ((mon.PSI_WATCH, GREY), (mon.PSI_ACTION, GREY)):
        ax.axhline(y_, color=c, ls=":", lw=1)
    ax.set(ylabel="Score PSI vs training", title="Score PSI by (simulated) holdout scenario")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    save(fig, "psi_scenarios.png")


# -------------------------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    X, y, tr, ca, ho = load()
    print(f"train {len(tr):,} | calibration {len(ca):,} | holdout {len(ho):,} | default rate {y.mean():.2%}")
    credit = X.loc[ho, "AMT_CREDIT"].to_numpy()
    yh = y[ho].to_numpy()
    summary = {"n_train": len(tr), "n_calib": len(ca), "n_holdout": len(ho), "base_rate": float(y.mean())}

    # ---- 1. scorecard -------------------------------------------------------------------
    print("\n[1] Scorecard (WoE / IV)")
    Xtr = X.loc[tr]
    binners = sc.fit_binners(Xtr, y[tr])
    iv = sc.iv_table(binners)
    iv.to_csv(ART / "iv_table.csv", index=False)
    Wtr = pd.DataFrame({c: b.transform(Xtr[c]) for c, b in binners.items()}, index=Xtr.index)
    # IV > 0.5 is conventionally a leakage red flag. EXT_SOURCES_MEAN (0.61) was reviewed: it is the mean of three
    # external bureau scores known at application time, so it is kept rather than dropped.
    chosen = sc.select_features(binners, Wtr, iv_max=1.0)
    card = sc.Scorecard().fit(binners, Wtr, y[tr], chosen)
    pts = card.points_table(); pts.to_csv(ART / "scorecard_points.csv", index=False)
    print(f"  {len(binners)} features binned; {int((iv.iv >= .02).sum())} with IV>=0.02; "
          f"{len(card.features)} in final scorecard")
    print(iv.head(8).to_string(index=False))
    fig_iv(iv)
    p_card = card.predict_pd(X.loc[ho]); score_card = card.score(X.loc[ho])
    summary["scorecard_features"] = card.features

    # ---- LightGBM: weighted (existing recipe), calibrated, unweighted ---------------------
    print("\n[LightGBM]")
    m_w = train_lgb(X, y, tr, ca, 10)
    raw_w_ca, raw_w_ho = m_w.predict(X.loc[ca]), m_w.predict(X.loc[ho])
    platt = LogisticRegression(C=1e9, max_iter=1000).fit(logit(raw_w_ca).reshape(-1, 1), y[ca])
    p_lgb = platt.predict_proba(logit(raw_w_ho).reshape(-1, 1))[:, 1]
    m_u = train_lgb(X, y, tr, ca, 1)
    p_unw = m_u.predict(X.loc[ho])
    print(f"  weighted: {m_w.best_iteration} rounds | unweighted: {m_u.best_iteration} rounds")

    # ---- 2+3. metrics, calibration ------------------------------------------------------------
    print("\n[2-3] Holdout metrics")
    models = {"Scorecard (WoE + LR)": p_card, "LightGBM raw (scale_pos_weight=10)": raw_w_ho,
              "LightGBM calibrated (Platt)": p_lgb, "LightGBM unweighted": p_unw}
    rows = []
    for name, p in models.items():
        rep = rm.calibration_report(yh, p)
        lo, hi = rm.bootstrap_auc_ci(yh, p, 100)
        rows.append(dict(model=name, auc=roc_auc_score(yh, p), auc_lo=lo, auc_hi=hi, gini=rm.gini(yh, p),
                         ks=rm.ks_stat(yh, p), **rep, calibration_pass=rm.calibration_passes(rep)))
    cmp_ = pd.DataFrame(rows); cmp_.to_csv(ART / "model_comparison.csv", index=False)
    print(cmp_[["model", "auc", "gini", "ks", "brier", "brier_skill", "ece", "cal_slope", "obs_to_exp",
                "calibration_pass"]].round(4).to_string(index=False))
    summary["model_comparison"] = cmp_.round(5).to_dict("records")

    dec_l = rm.decile_table(yh, p_lgb); dec_s = rm.decile_table(yh, p_card)
    dec_l.to_csv(ART / "decile_table_lightgbm.csv", index=False)
    dec_s.to_csv(ART / "decile_table_scorecard.csv", index=False)
    print("\nDecile table (LightGBM, decile 1 = riskiest):")
    print(dec_l.round(4).to_string(index=False))
    fig_gain_lift({"LightGBM": dec_l, "Scorecard": dec_s})
    fig_ks(yh, p_lgb, "LightGBM")
    cal_tabs = {"LightGBM raw (scale_pos_weight=10)": rm.calibration_table(yh, raw_w_ho),
                "LightGBM calibrated (Platt)": rm.calibration_table(yh, p_lgb),
                "Scorecard": rm.calibration_table(yh, p_card)}
    pd.concat({k: v for k, v in cal_tabs.items()}).reset_index(level=0).rename(columns={"level_0": "model"}) \
        .to_csv(ART / "calibration_table.csv", index=False)
    fig_calibration(cal_tabs)
    cal_ok = bool(cmp_.loc[cmp_.model == "LightGBM calibrated (Platt)", "calibration_pass"].iloc[0])
    print(f"\nCalibration gate for cost-based thresholds: {'PASS' if cal_ok else 'FAIL'}")
    if not cal_ok:
        print("  !! calibrated model failed the gate; cost-based thresholds below are NOT trustworthy")

    # ---- 4. cost-based threshold ------------------------------------------------------------------
    print("\n[4] Cost-based threshold (assumed margin/LGD)")
    grid = np.round(np.r_[np.arange(0.01, 0.40, 0.005)], 3)
    curve = pf.profit_curve(yh, p_lgb, credit, grid)
    opt = pf.optimal_threshold(yh, p_lgb, credit)
    be = pf.breakeven_pd(pf.MARGIN_PCT, pf.LGD)
    old = pf.profit_at(yh, raw_w_ho, credit, 0.5)                       # existing README rule
    at_be = pf.profit_at(yh, p_lgb, credit, be)
    curve.to_csv(ART / "profit_curve.csv", index=False)
    sens = pf.sensitivity_grid(yh, p_lgb, credit); sens.to_csv(ART / "threshold_sensitivity.csv", index=False)
    fig_profit(curve, opt, be, old); fig_sensitivity(sens)
    print(f"  breakeven PD {be:.3f} | profit-max cutoff {opt['threshold']:.3f} approves {opt['approval_rate']:.1%}")
    print(f"  profit: approve-all {opt['approve_all_profit']/1e6:,.0f}M | optimal {opt['profit']/1e6:,.0f}M | "
          f"old 0.5-on-raw rule {old['profit']/1e6:,.0f}M | at breakeven cutoff {at_be['profit']/1e6:,.0f}M")
    summary["profit"] = dict(margin_assumed=pf.MARGIN_PCT, lgd_assumed=pf.LGD, breakeven_pd=be, optimal=opt,
                             old_rule_0_5_raw=old, at_breakeven=at_be,
                             opt_detail=pf.profit_at(yh, p_lgb, credit, opt["threshold"]))
    print(sens.assign(best_threshold=sens.best_threshold.round(3), approval_rate=sens.approval_rate.round(3))
          [["margin", "lgd", "breakeven_pd", "best_threshold", "approval_rate"]].round(3).to_string(index=False))

    # ---- 5. monitoring ----------------------------------------------------------------------------
    print("\n[5] Monitoring (SIMULATED split: dataset has no dates)")
    imp = pd.Series(m_w.feature_importance("gain"), index=m_w.feature_name()).sort_values(ascending=False)
    top_feats = [f for f in imp.index if pd.api.types.is_numeric_dtype(X[f])][:10]
    p_tr = platt.predict_proba(logit(m_w.predict(Xtr)).reshape(-1, 1))[:, 1]
    Xho = X.loc[ho]
    rng = np.random.default_rng(7)
    age = -Xho["DAYS_BIRTH"] / 365.25
    ext = Xho["EXT_SOURCES_MEAN"].fillna(Xho["EXT_SOURCES_MEAN"].median())
    z = lambda s: (s - s.mean()) / s.std()
    scenarios = {"A. random holdout (no drift)": np.zeros(len(Xho)),
                 "B. mild shift: slightly younger": 0.5 * -z(age).to_numpy(),
                 "C. moderate: younger + weaker ext. scores": 0.6 * (-z(age) - z(ext)).to_numpy() / 1.4,
                 "D. large: younger + weaker ext. scores": 1.0 * (-z(age) - z(ext)).to_numpy() / 1.4}
    rows = []
    for name, tilt in scenarios.items():
        w = np.exp(tilt); w /= w.sum()
        sel = rng.choice(len(Xho), size=len(Xho), replace=True, p=w) if tilt.any() else np.arange(len(Xho))
        Xs, ys, ps = Xho.iloc[sel], yh[sel], p_lgb[sel]
        s_psi = mon.psi(p_tr, ps)
        f_psi = {f: mon.psi(Xtr[f], Xs[f]) for f in top_feats}
        oe = ys.mean() / ps.mean()
        status, why = mon.retrain_decision(s_psi, f_psi, oe, int(ys.sum()))
        rows.append(dict(scenario=name, score_psi=s_psi, max_feature_psi=max(f_psi.values()),
                         n_features_shifted=sum(v >= .25 for v in f_psi.values()), default_rate=ys.mean(),
                         avg_pred_pd=ps.mean(), obs_to_exp=oe, status=status, reasons="; ".join(why)))
        if name.startswith("D"):
            pd.Series(f_psi, name="psi").to_csv(ART / "psi_features_scenario_D.csv")
        if name.startswith("A"):
            pd.Series(f_psi, name="psi").to_csv(ART / "psi_features_scenario_A.csv")
    psi_tab = pd.DataFrame(rows); psi_tab.to_csv(ART / "psi_scenarios.csv", index=False)
    print(psi_tab.drop(columns="reasons").round(4).to_string(index=False))
    fig_psi(psi_tab)
    summary["psi_scenarios"] = psi_tab.round(5).to_dict("records")
    summary["top_features_monitored"] = top_feats

    # ---- artefacts for the dashboard -------------------------------------------------------------
    pd.DataFrame({"default": yh, "pd": p_lgb, "pd_raw_weighted": raw_w_ho, "credit": credit,
                  "score_scorecard": score_card}).to_parquet(ART / "holdout_scores.parquet", index=False)
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nWrote reports/artifacts/* and reports/figures/*")


if __name__ == "__main__":
    main()
