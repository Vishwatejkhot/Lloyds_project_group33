"""
Model Performance — metrics tables, ROC/PR/confusion charts, statistical tests.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")

st.set_page_config(page_title="Performance · Lloyds SME", page_icon="📊", layout="wide")
st.title("📊 Model Performance")
st.caption("Evaluation across 3 labels × 3 models · Statistical significance · SHAP importance")


def load_csv(name):
    path = os.path.join(OUTPUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def show_img(name, caption=""):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{name} not found.")


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Baseline Metrics", "📈 ROC & PR Curves",
    "🔲 Confusion Matrices", "🧪 Statistical Tests", "🔍 SHAP Importance"
])

# ── Tab 1: Baseline metrics ────────────────────────────────────────────────
with tab1:
    df = load_csv("baseline_results_summary.csv")
    if df is not None:
        st.subheader("All Models · All Labels")
        col_order = [c for c in ["Label","Model","f1","roc_auc","precision","recall",
                                   "accuracy","avg_precision","brier_score"] if c in df.columns]
        styled = (
            df[col_order]
            .style
            .background_gradient(subset=[c for c in ["f1","roc_auc","avg_precision"] if c in col_order],
                                  cmap="RdYlGn")
            .format({c: "{:.4f}" for c in col_order if c not in ["Label","Model"]})
        )
        st.dataframe(styled, use_container_width=True)

        st.markdown("---")
        st.subheader("Recommended Models")
        rec = load_csv("recommended_models.csv")
        if rec is not None:
            st.dataframe(rec, use_container_width=True)

    st.markdown("---")
    show_img("baseline_performance_chart.png", "F1, Precision, Recall, Accuracy — all models")

# ── Tab 2: ROC & PR ────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves")
        show_img("roc_curves.png")
    with c2:
        st.subheader("Precision-Recall Curves")
        show_img("pr_curves.png")
        st.caption("PR curves are more informative than ROC for imbalanced labels (Risk 14:1, Lending 7:1).")

    st.markdown("---")
    st.subheader("Average Precision Scores")
    pr = load_csv("pr_auc_results.csv")
    if pr is not None:
        st.dataframe(
            pr.style.background_gradient(subset=["Average_Precision"], cmap="RdYlGn")
              .format({"Average_Precision": "{:.4f}"}),
            use_container_width=True
        )

# ── Tab 3: Confusion matrices ──────────────────────────────────────────────
with tab3:
    show_img("confusion_matrices.png", "Sample (100k) models")

# ── Tab 4: Statistical tests ───────────────────────────────────────────────
with tab4:
    st.subheader("Bootstrap 95% Confidence Intervals (n=1000)")
    boot = load_csv("bootstrap_ci.csv")
    if boot is not None:
        st.dataframe(
            boot.style.format({c: "{:.4f}" for c in boot.columns
                                if c not in ["Label","Model","Training"]}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("McNemar's Test — Pairwise Classifier Significance")
    st.caption("H₀: both classifiers make the same errors · p < 0.05 = significantly different")
    mc = load_csv("mcnemar_results.csv")
    if mc is not None:
        def sig_color(val):
            if val in ["***","**"]: return "color: green; font-weight:bold"
            if val == "*": return "color: orange"
            return "color: grey"
        styled_mc = mc.style.applymap(sig_color, subset=["Significant"]) \
                            .format({"chi2": "{:.2f}", "p_value": "{:.4f}"})
        st.dataframe(styled_mc, use_container_width=True)

    st.markdown("---")
    st.subheader("Bayesian Model Comparison")
    st.caption("P(Model A better than B) via Beta posterior sampling (100k samples)")
    bay = load_csv("bayesian_comparison.csv")
    if bay is not None:
        st.dataframe(
            bay.style.format({"P_A_better": "{:.3f}", "P_B_better": "{:.3f}"}),
            use_container_width=True
        )

# ── Tab 5: SHAP ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("XGBoost Feature Importance")
    show_img("feature_importance.png")

    st.markdown("---")
    st.subheader("SHAP Beeswarm Plots (Pratik)")
    c1, c2, c3 = st.columns(3)
    with c1: show_img("shap_beeswarm_growth.png", "Growth")
    with c2: show_img("shap_beeswarm_risk.png", "Risk")
    with c3: show_img("shap_beeswarm_lending.png", "Lending")

    st.markdown("---")
    st.subheader("SHAP Bar Plots")
    c1, c2, c3 = st.columns(3)
    with c1: show_img("shap_bar_growth.png", "Growth")
    with c2: show_img("shap_bar_risk.png", "Risk")
    with c3: show_img("shap_bar_lending.png", "Lending")

    st.markdown("---")
    st.subheader("SHAP Waterfall — Individual Companies")
    c1, c2 = st.columns(2)
    with c1:
        show_img("shap_waterfall_growth_top.png", "Growth — top company")
        show_img("shap_waterfall_risk_top.png", "Risk — top company")
    with c2:
        show_img("shap_waterfall_growth_borderline.png", "Growth — borderline company")
        show_img("shap_waterfall_lending_top.png", "Lending — top company")

    st.markdown("---")
    st.subheader("Cross-Label Feature Importance (SHAP)")
    show_img("shap_cross_label_comparison.png")
    cross = load_csv("shap_cross_label_importance.csv")
    if cross is not None:
        st.dataframe(cross, use_container_width=True)
