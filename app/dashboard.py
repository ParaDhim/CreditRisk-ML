"""Threshold dashboard.   streamlit run app/dashboard.py

Reads reports/artifacts/holdout_scores.parquet (written by src/run_upgrade.py), so it runs
without the raw Kaggle data. All money figures depend on the two ASSUMPTIONS in the sidebar.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parent.parent / "reports" / "artifacts"
st.set_page_config(page_title="Credit approval threshold", layout="wide")


@st.cache_data
def load():
    d = pd.read_parquet(ART / "holdout_scores.parquet").sort_values("pd", kind="mergesort").reset_index(drop=True)
    d["cum_good"] = ((1 - d["default"]) * d["credit"]).cumsum()
    d["cum_bad"] = (d["default"] * d["credit"]).cumsum()
    d["cum_defaults"] = d["default"].cumsum()
    d["cum_pd_credit"] = (d["pd"] * d["credit"]).cumsum()
    return d


d = load()
n, total_def = len(d), int(d["default"].sum())

st.title("Credit approval threshold explorer")
st.caption(f"Scored on a {n:,}-applicant holdout of the Home Credit data (stratified random split; the dataset has "
           "no dates, so this is not an out-of-time test). Model: LightGBM with Platt-calibrated probabilities.")

with st.sidebar:
    st.header("Assumptions")
    st.caption("ASSUMPTIONS, not Home Credit facts. The dataset has no revenue or loss figures.")
    margin = st.slider("Margin on a loan that repays (% of credit)", 2.0, 25.0, 10.0, 0.5) / 100
    lgd = st.slider("Loss given default (% of credit)", 20.0, 100.0, 60.0, 5.0) / 100
    breakeven = margin / (margin + lgd)
    st.markdown(f"**Breakeven PD** = margin / (margin + LGD) = **{breakeven:.3f}**")

# whole profit curve by cumulative sums (applicants sorted by PD; approve the k lowest-risk)
profit = margin * d["cum_good"] - lgd * d["cum_bad"]
best_k = int(profit.idxmax())
best_t = float(d.loc[best_k, "pd"])

t = st.slider("Approval cutoff: approve applicants whose predicted default probability is at or below",
              0.01, 0.60, float(np.clip(round(breakeven, 3), 0.01, 0.60)), 0.005, format="%.3f")
k = int(np.searchsorted(d["pd"].to_numpy(), t, side="right"))   # number approved
if k == 0:
    st.warning("Nothing is approved at this cutoff.")
    st.stop()
i = k - 1

approval = k / n
caught = 1 - d.loc[i, "cum_defaults"] / total_def
loss = lgd * d.loc[i, "cum_bad"]
revenue = margin * d.loc[i, "cum_good"]
prof = revenue - loss
all_profit = float(profit.iloc[-1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Approval rate", f"{approval:.1%}")
c2.metric("Defaults caught (declined)", f"{caught:.1%}", help="Share of all eventual defaults that fall in the declined group")
c3.metric("Loss on approved defaults", f"{loss/1e6:,.0f}M", help="LGD x credit of approved applicants who defaulted (realised on holdout)")
c4.metric("Profit", f"{prof/1e6:,.0f}M", f"{(prof - all_profit)/1e6:+,.0f}M vs approving everyone")

c5, c6, c7 = st.columns(3)
c5.metric("Default rate among approved", f"{d.loc[i, 'cum_defaults'] / k:.2%}", f"base rate {d['default'].mean():.2%}", delta_color="off")
c6.metric("Model-expected loss on approved", f"{lgd * d.loc[i, 'cum_pd_credit']/1e6:,.0f}M",
          help="Sum of PD x LGD x credit. Close to realised loss only because the model is calibrated.")
c7.metric("Profit-maximising cutoff", f"{best_t:.3f}", f"approves {(best_k+1)/n:.1%}", delta_color="off")

curve = pd.DataFrame({"cutoff": d["pd"].iloc[::200], "profit (M)": profit.iloc[::200] / 1e6})
curve = curve[curve["cutoff"] <= 0.6]
line = alt.Chart(curve).mark_line(color="#2a6fbb").encode(
    x=alt.X("cutoff:Q", title="Approval cutoff (predicted PD)"), y=alt.Y("profit (M):Q", title="Profit on holdout (millions)"))
now = alt.Chart(pd.DataFrame({"cutoff": [t]})).mark_rule(color="#d9822b", strokeWidth=2).encode(x="cutoff:Q")
be = alt.Chart(pd.DataFrame({"cutoff": [breakeven]})).mark_rule(color="#8a8f98", strokeDash=[4, 4]).encode(x="cutoff:Q")
st.altair_chart(line + be + now, use_container_width=True)
st.caption("Orange = your cutoff. Grey dashed = theoretical breakeven PD. "
           "The best realised cutoff sits near breakeven because the probabilities are calibrated.")
