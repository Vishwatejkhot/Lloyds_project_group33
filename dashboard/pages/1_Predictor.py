"""
Company Predictor — browse dataset, run ML predictions, SHAP waterfall, AI explanation, full report.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_loader import load_all_models, best_model
from utils.shap_utils import compute_shap, top_features, waterfall_figure
from agents.report_agent import quick_explain, generate_full_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "feature_engineered_dataset.csv")

st.set_page_config(page_title="Predictor · Lloyds SME", page_icon="🔮", layout="wide")
st.title("🔮 Company Predictor")
st.caption("Browse 50k sampled UK SMEs · ML predictions · SHAP explainability · AI report")


# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading company dataset…")
def load_data():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH, low_memory=False, nrows=500_000)
    df = df.sample(n=min(50_000, len(df)), random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Scoring companies…")
def score_sample(_df, _models):
    rows = []
    for label in ["growth", "risk", "lending"]:
        model, scaler, features, _ = best_model(_models, label)
        X = _df[features].fillna(0)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        for i, p in enumerate(probs):
            rows.append({"idx": i, "label": label, "prob": round(float(p), 4)})
    wide = pd.DataFrame(rows).pivot(index="idx", columns="label", values="prob").reset_index()
    wide.columns.name = None
    return wide


def prob_color(p):
    if p >= 0.7: return "🟢"
    if p >= 0.4: return "🟡"
    return "🔴"


# ── Sidebar filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    models = load_all_models()
    if not models:
        st.error("No models found in models/ folder.")
        st.stop()

    df = load_data()
    if df is None:
        st.warning("feature_engineered_dataset.csv not found locally.")
        st.stop()

    sectors = ["All"] + sorted(df["bcb_sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("BCB Sector", sectors)
    sort_by = st.selectbox("Sort companies by", ["growth", "risk", "lending"])
    n_show = st.slider("Companies to display", 10, 200, 50)

# ── Score & filter ─────────────────────────────────────────────────────────
scores = score_sample(df, models)
df_scored = df.copy()
df_scored["growth_prob"]  = scores["growth"].values
df_scored["risk_prob"]    = scores["risk"].values
df_scored["lending_prob"] = scores["lending"].values

if sel_sector != "All":
    df_scored = df_scored[df_scored["bcb_sector"] == sel_sector]

df_display = (
    df_scored
    .sort_values(f"{sort_by}_prob", ascending=False)
    .head(n_show)
    .reset_index(drop=True)
)

# ── Company table ──────────────────────────────────────────────────────────
st.subheader(f"Top {n_show} companies by {sort_by} probability")

show_cols = [c for c in ["bcb_sector", "sic_code_1", "company_size_score",
                          "has_mortgage_history", "num_mortgages_total",
                          "growth_prob", "risk_prob", "lending_prob"] if c in df_display.columns]
st.dataframe(
    df_display[show_cols].style.background_gradient(
        subset=["growth_prob", "risk_prob", "lending_prob"], cmap="RdYlGn"
    ),
    use_container_width=True,
    height=300,
)

# ── Select a company ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Select a company for detailed analysis")
row_idx = st.number_input("Row index (from table above)", min_value=0,
                           max_value=len(df_display) - 1, value=0, step=1)
company = df_display.iloc[int(row_idx)]
sector = str(company.get("bcb_sector", "Unknown"))

c1, c2, c3 = st.columns(3)
g, r, l = company["growth_prob"], company["risk_prob"], company["lending_prob"]
c1.metric(f"{prob_color(g)} Growth Opportunity",   f"{g:.1%}")
c2.metric(f"{prob_color(r)} Risk Signal",          f"{r:.1%}")
c3.metric(f"{prob_color(l)} Lending Need",         f"{l:.1%}")

st.caption(f"Sector: **{sector}** · SIC: {company.get('sic_code_1','?')} · "
           f"Size score: {company.get('company_size_score','?')}")

# ── SHAP ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("SHAP Explainability")
shap_label = st.selectbox("Explain label", ["growth", "risk", "lending"])

with st.spinner("Computing SHAP values…"):
    model, scaler, features, mname = best_model(models, shap_label)
    X_orig   = company[features].fillna(0).values.reshape(1, -1)
    X_scaled = scaler.transform(X_orig)
    sv, ev   = compute_shap(model, X_scaled)
    top_feats = top_features(sv, features)

col_shap, col_info = st.columns([3, 1])
with col_shap:
    fig = waterfall_figure(sv, ev, features, X_orig[0])
    st.pyplot(fig, use_container_width=True)
    plt.close("all")

with col_info:
    st.markdown(f"**Model:** {mname}")
    st.markdown("**Top drivers:**")
    for fname, fval in top_feats[:6]:
        arrow = "↑" if fval > 0 else "↓"
        st.markdown(f"- {arrow} `{fname}` ({fval:+.3f})")

# ── AI Explanation ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("AI Explanation")
if st.button("⚡ Quick Explain (Groq)", use_container_width=False):
    with st.spinner("Generating explanation via Groq…"):
        prob_val = float(company[f"{shap_label}_prob"])
        try:
            text, model_used = quick_explain(shap_label, prob_val, top_feats)
            st.success(text)
            st.markdown(f'<span class="llm-tag">via {model_used}</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"LLM error: {e}")

# ── Full Report ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Full BI Report")
if st.button("📄 Generate Full Report (OpenAI)", use_container_width=False):
    with st.spinner("Generating report…"):
        preds = {"growth": float(g), "risk": float(r), "lending": float(l)}
        shap_all = {}
        for lbl in ["growth", "risk", "lending"]:
            m2, s2, f2, _ = best_model(models, lbl)
            X2 = company[f2].fillna(0).values.reshape(1, -1)
            sv2, ev2 = compute_shap(m2, s2.transform(X2))
            shap_all[lbl] = top_features(sv2, f2, n=5)
        try:
            report, model_used = generate_full_report(preds, shap_all, sector)
            st.markdown(report)
            st.markdown(f'<span class="llm-tag">via {model_used}</span>', unsafe_allow_html=True)
            # Store context for AI Agent page
            st.session_state["last_report_context"] = {
                "sector": sector, "growth": float(g),
                "risk": float(r), "lending": float(l),
                "top_features": {k: [(f, v) for f, v in vals] for k, vals in shap_all.items()},
            }
        except Exception as e:
            st.error(f"LLM error: {e}")
