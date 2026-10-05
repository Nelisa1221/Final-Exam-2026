import json, os
from pathlib import Path
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="eThekwini Election Analytics 2026", layout="wide")
# Find outputs beside app.py
OUT = Path(__file__).resolve().parent / "outputs"


    
@st.cache_data
def load(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
hist, proj, wards = load("metro_shares_hist.csv"), load("metro_projection.csv"), load("ward_predictions.csv")
if proj.empty or "party" not in proj.columns:
    st.error(
        f"Missing or invalid file: {OUT / 'metro_projection.csv'}. "
        "Export the notebook results and place them in the outputs folder."
    )
    st.stop()
turn, align, conf = load("turnout_summary.csv"), load("alignment.csv"), load("confusion.csv")
meta = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}

st.title(f"{meta.get('metro', 'Metro')} - Local Government Election Analytics 2026")
st.warning("Historical figures are IEC counts. Everything labelled 'Model estimate' is a prediction for 04 Nov 2026 with. "
           "This dashboard reports model output only; it is not a political opinion or a prediction of who will govern.")
tabs = st.tabs(["Overview", "Party shares", "Wards", "Turnout", "Model performance", "Data & limitations"])

with tabs[0]:
    c = st.columns(4)
    if not proj.empty:
        r = proj.iloc[0]
        c[0].metric("Largest projected share (model estimate)", f"{r.party}  {r.share_point:.1%}")
        c[1].metric("90% sampling band", f"{r.share_p05:.1%} - {r.share_p95:.1%}")
        c[2].metric("Party above 50% in model?", "Yes" if meta.get("majority") else "No")
    if not turn.empty and (turn.record_type == "predicted_2026").any():
        t = turn[turn.record_type == "predicted_2026"].iloc[0]; c[3].metric("Projected turnout (model estimate)", f"{t.turnout_rate:.1%}")
    st.caption(f"Ward calls flagged uncertain: {meta.get('uncertain_wards','?')} of {meta.get('n_wards','?')}. Boundary basis: {meta.get('boundary_basis','?')}.")

with tabs[1]:
    st.subheader("Historical vs model-estimated vote share")
    parties = st.multiselect("Parties", sorted(proj.party.unique()), default=list(proj.party.head(6)))
    yrs = st.multiselect("Historical years", sorted(hist.year.unique()), default=sorted(hist.year.unique()))
    h = hist[hist.party.isin(parties) & hist.year.isin(yrs)].assign(label=lambda d: "Historical " + d.year.astype(str))
    p = proj[proj.party.isin(parties)]
    fig = px.bar(h, x="party", y="share", color="label", barmode="group")
    fig.add_trace(go.Bar(x=p.party, y=p.share_point, name="Model estimate 2026", marker_color="black",
                         error_y=dict(type="data", symmetric=False, array=p.share_p95 - p.share_point, arrayminus=p.share_point - p.share_p05)))
    fig.update_yaxes(tickformat=".0%", title="share of valid votes"); st.plotly_chart(fig, use_container_width=True)
    show = proj[proj.party.isin(parties)].copy()
    st.dataframe(show.round(4), use_container_width=True)
    st.caption("Bars with whiskers: 90% bootstrap band (sampling uncertainty only). scen_lo/scen_hi in the table give the wider scenario range based on the last observed swing.")

with tabs[2]:
    st.subheader("Ward-level model output")
    only_sel = st.checkbox("Selected wards only", value=True)
    thr = st.slider("Flag as uncertain if probability gap is below", 0.0, 0.5, 0.15, 0.01)
    w = wards.copy(); w["uncertain"] = w.margin_prob < thr
    if only_sel: w = w[w.selected]
    ward = st.selectbox("Ward", w.ward.astype(str).tolist()) if len(w) else None
    if ward:
        row = w[w.ward.astype(str) == ward].iloc[0]
        pc = [c for c in w.columns if c.startswith("p_")]
        pf = pd.DataFrame({"party": [c[2:] for c in pc], "probability": [row[c] for c in pc]}).sort_values("probability", ascending=False)
        st.plotly_chart(px.bar(pf, x="party", y="probability", title=f"Ward {ward}: model probability of leading (estimate)"), use_container_width=True)
        lc = [c for c in w.columns if c.startswith("leader_")][0]
        st.write(f"**Model estimate:** {row.pred_leader} ({row.top_prob:.0%}). Last observed leader (historical): {row[lc]}. "
                 f"{'UNCERTAIN: close call.' if row.uncertain else 'Clear model lead.'}")
    st.dataframe(w[["ward", "pred_leader", "top_prob", "margin_prob", "uncertain", "selected"]].round(3), use_container_width=True)

with tabs[3]:
    st.subheader("Turnout")
    if turn.empty: st.info("No turnout data available.")
    else:
        fig = px.bar(turn, x="year", y="turnout_rate", color="record_type", color_discrete_map={"historical": "steelblue", "predicted_2026": "black"})
        fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig, use_container_width=True); st.dataframe(turn.round(4))
        st.caption("Registered voters for 2026 are assumed equal to the last cycle unless the IEC figure was supplied; the true 2026 roll will differ.")

with tabs[4]:
    st.subheader("Validation evidence")
    m = meta.get("metrics", {})
    if m:
        st.dataframe(pd.DataFrame({k: v for k, v in m.items() if isinstance(v, dict) and "accuracy" in v}).T.round(3))
        if "turnout_models" in m: st.dataframe(pd.DataFrame(m["turnout_models"]).T.round(4))
    if not conf.empty:
        c0 = conf.set_index(conf.columns[0]); st.plotly_chart(px.imshow(c0, text_auto=True, title="Out-of-fold confusion matrix (winner at voting-district level)"), use_container_width=True)

with tabs[5]:
   if not align.empty:
    st.dataframe(align.head(25))
    st.table(pd.DataFrame(meta.get("sources", [])))
    st.markdown("- Only three municipal cycles exist, so a simple damped-swing projection is used instead of a forced time-series model.\n"
                "- Parties that did not contest the historical municipal elections cannot be modelled from these data.\n"
                "- Ward boundaries change between elections; ward history is only used where voting-district overlap is at least 90%.\n"
                "- Bootstrap bands show sampling noise only; true uncertainty is larger (see scenario range).\n"
                "- Municipal turnout and party support can differ sharply from national elections.")
    if not align.empty: st.dataframe(align.sort_values("jaccard_same_label").head(25))
