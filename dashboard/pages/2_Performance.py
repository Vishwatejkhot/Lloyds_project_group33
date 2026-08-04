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

st.markdown("""
<style>
[data-testid="stHeader"] { background: transparent !important; }
.stApp {
    background:
        radial-gradient(circle at top right, rgba(0, 106, 77, 0.10), transparent 28%),
        linear-gradient(180deg, #f7faf9 0%, #eef4f1 100%);
    color: #17362d;
}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp div { color: #17362d; }
.stApp h1, .stApp h2, .stApp h3 { color: #10231d !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #003f2f 0%, #006a4d 100%); }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stAlert"] p { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Performance")
st.caption("Evaluation across 3 labels × 3 models · Statistical significance · SHAP importance")
st.info(
    """
    **Executive Summary**

    Three machine-learning models were evaluated across Growth Opportunity,
    Risk Signal and Lending Need objectives. XGBoost achieved the strongest
    overall performance, while statistical testing and SHAP analysis support
    model reliability, transparency and business use.
    """
)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Models Evaluated", "9")

with kpi2:
    st.metric("Prediction Labels", "3")

with kpi3:
    st.metric("Bootstrap Samples", "1,000")

with kpi4:
    st.metric("Confidence Level", "95%")
st.markdown("### Executive Model Highlights")

highlight1, highlight2, highlight3 = st.columns(3)

with highlight1:
    st.success(
        """
        **Growth Opportunity**

        Identifies SMEs with strong potential for business growth and expansion.
        """
    )

with highlight2:
    st.warning(
        """
        **Risk Signal**

        Detects companies showing elevated financial or operational risk.
        """
    )

with highlight3:
    st.info(
        """
        **Lending Need**

        Identifies SMEs that may require additional lending or financial support.
        """
    )   

def load_csv(name):
    path = os.path.join(OUTPUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def show_img(name, caption=""):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, width="stretch")
    else:
        st.info(f"{name} not found.")


# ── Tabs ───────────────────────────────────────────────────────────────────
st.divider()

st.markdown("## 📈 Detailed Performance Analysis")

st.caption(
    "The sections below provide detailed evaluation metrics, statistical validation, confusion matrices and explainability results for the trained models."
)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Baseline Metrics", "📈 ROC & PR Curves",
    "🔲 Confusion Matrices", "🧪 Statistical Tests", "🔍 SHAP Importance"
])

# ── Tab 1: Baseline metrics ────────────────────────────────────────────────
with tab1:
    df = load_csv("baseline_results_summary.csv")
    if df is not None:
        st.subheader("Performance Comparison Across All Models")
        col_order = [c for c in ["Label","Model","f1","roc_auc","precision","recall",
                                   "accuracy","avg_precision","brier_score"] if c in df.columns]
        styled = (
            df[col_order]
            .style
            .background_gradient(subset=[c for c in ["f1","roc_auc","avg_precision"] if c in col_order],
                                  cmap="RdYlGn")
            .format({c: "{:.4f}" for c in col_order if c not in ["Label","Model"]})
        )
        st.dataframe(styled, width="stretch")

        st.markdown("---")
        st.subheader("Recommended Production Models")
        rec = load_csv("recommended_models.csv")
        if rec is not None:
            st.dataframe(rec, width="stretch")

    st.markdown("---")
    show_img("baseline_performance_chart.png", "F1, Precision, Recall, Accuracy — all models")

# ── Tab 2: ROC & PR ────────────────────────────────────────────────────────
with tab2:
    st.info(
        """
        **ROC and Precision–Recall Analysis**

        These curves show how well each model separates positive and negative cases.
        Precision–Recall curves are especially useful for the imbalanced Risk Signal
        and Lending Need labels.
        """
    )

    c1, c2 = st.columns(2)
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
            width="stretch"
        )

# ── Tab 3: Confusion matrices ──────────────────────────────────────────────
with tab3:
    st.info(
        """
        **Confusion Matrix Analysis**

        The confusion matrices illustrate how accurately each model classifies
        positive and negative cases for every prediction task, helping identify
        false positives and false negatives.
        """
    )
    show_img("confusion_matrices.png", "Sample (100k) models")

# ── Tab 4: Statistical tests ───────────────────────────────────────────────
with tab4:
    st.info(
        """
        **Statistical Validation**

        Bootstrap confidence intervals and paired statistical tests were used to
        verify that the observed performance differences between models are
        statistically reliable rather than due to random variation.
        """
    )
    st.subheader("Bootstrap 95% Confidence Intervals (n=1000)")
    boot = load_csv("bootstrap_ci.csv")
    if boot is not None:
        st.dataframe(
            boot.style.format({c: "{:.4f}" for c in boot.columns
                                if c not in ["Label","Model","Training"]}),
            width="stretch"
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
        styled_mc = mc.style.map(sig_color, subset=["Significant"]) \
                            .format({"chi2": "{:.2f}", "p_value": "{:.4f}"})
        st.dataframe(styled_mc, width="stretch")

    st.markdown("---")
    st.subheader("Bayesian Model Comparison")
    st.caption("P(Model A better than B) via Beta posterior sampling (100k samples)")
    bay = load_csv("bayesian_comparison.csv")
    if bay is not None:
        st.dataframe(
            bay.style.format({"P_A_better": "{:.3f}", "P_B_better": "{:.3f}"}),
            width="stretch"
        )

# ── Tab 5: SHAP ───────────────────────────────────────────────────────────
with tab5:
    st.info(
        """
        **Model Explainability**

        SHAP values explain how each feature contributes to model predictions,
        improving transparency and helping stakeholders understand the factors
        influencing business decisions.
        """
    )
    st.subheader("XGBoost Feature Importance")
    show_img("feature_importance.png")

    st.markdown("---")
    st.subheader("SHAP Beeswarm Plots")
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
        st.dataframe(cross, width="stretch")
st.divider()

st.success(
    """
    **Executive Conclusion**

    The evaluation demonstrates that the selected models provide strong predictive
    performance across the three business objectives. The statistical tests,
    confusion matrices and SHAP analysis also support reliable, transparent and
    explainable model deployment.
    """
)