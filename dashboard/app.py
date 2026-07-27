import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(
    page_title="Lloyds SME Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stHeader"] {
    background: transparent !important;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(0, 106, 77, 0.10), transparent 28%),
        linear-gradient(180deg, #f7faf9 0%, #eef4f1 100%);
}
/* Force readable colours in the main dashboard */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div {
    color: #17362d;
}

.stApp h1,
.stApp h2,
.stApp h3 {
    color: #10231d !important;
}

[data-testid="stMain"] {
    color: #17362d;
}

[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label {
    color: #4f665e;
}

[data-testid="stMain"] code {
    color: #ffffff !important;
    background-color: #123329 !important;
}

[data-testid="stAlert"] p {
    color: inherit !important;
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003f2f 0%, #006a4d 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.10);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebarNav"] a {
    border-radius: 10px;
    margin: 0.3rem 0.55rem;
    padding: 0.65rem 0.8rem;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(255, 255, 255, 0.12);
}

.hero {
    background: linear-gradient(135deg, #003f2f 0%, #006a4d 70%, #0b8a66 100%);
    border-radius: 22px;
    padding: 2.4rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 18px 45px rgba(0, 63, 47, 0.22);
}

.hero-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.55rem;
    line-height: 1.1;
    font-weight: 800;
    color: white !important;
    margin-bottom: 0.8rem;
}

.hero-title,
.hero-title * {
    color: white !important;
}

.hero-subtitle,
.hero-subtitle * {
    color: rgba(255, 255, 255, 0.90) !important;
}

.hero-badge,
.hero-badge * {
    color: white !important;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.86);
    font-size: 1.02rem;
    max-width: 850px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce8e2;
    border-radius: 16px;
    padding: 1rem 1.15rem;
    box-shadow: 0 9px 24px rgba(20, 46, 37, 0.07);
}

[data-testid="stMetricLabel"] {
    font-weight: 700;
    color: #496159;
}

[data-testid="stMetricValue"] {
    color: #10231d;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    background: #e5f4ed;
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    width: fit-content;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.94);
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(20, 46, 37, 0.07);
}

h1, h2, h3 {
    color: #10231d;
}

hr {
    border: 0;
    border-top: 1px solid #d8e4de;
    margin: 2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:1rem;">
        <span class="hero-badge" style="margin-bottom:0;">LLOYDS BANKING GROUP</span>
        <span class="hero-badge" style="margin-bottom:0;background:rgba(255,255,255,0.28);font-weight:800;letter-spacing:0.06em;">GROUP 33</span>
    </div>
    <div class="hero-title">🏦 SME Intelligence Platform</div>
    <div class="hero-subtitle">
        A decision-support dashboard combining Companies House data,
        machine learning, SHAP explainability and generative AI to identify
        growth opportunities, business risk and lending potential across UK SMEs.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies Analysed", "3,145,434", "UK SMEs")
c2.metric("Best Growth F1", "0.9341", "XGBoost")
c3.metric("Best Risk AUC", "1.0000", "XGBoost")
c4.metric("Best Lending F1", "1.0000", "Random Forest")

st.write("")

col_a, col_b = st.columns([1.65, 1], gap="large")

with col_a:
    with st.container(border=True):
        st.caption("PLATFORM OVERVIEW")
        st.header("What this platform does")

        st.write(
            "The platform analyses more than 3.1 million UK companies and "
            "converts company information into three clear commercial signals."
        )

        signal_1, model_1, score_1 = st.columns([2, 1, 1])
        signal_1.markdown("**🌱 Growth Opportunity**")
        model_1.markdown("`XGBoost`")
        score_1.markdown("**F1 0.9341**")

        st.divider()

        signal_2, model_2, score_2 = st.columns([2, 1, 1])
        signal_2.markdown("**⚠️ Risk Signal**")
        model_2.markdown("`XGBoost`")
        score_2.markdown("**F1 0.9973**")

        st.divider()

        signal_3, model_3, score_3 = st.columns([2, 1, 1])
        signal_3.markdown("**💷 Lending Need Proxy**")
        model_3.markdown("`XGBoost`")
        score_3.markdown("**F1 1.0000**")

        st.divider()

        st.success(
            "Explainable by design — SHAP analysis identifies the features "
            "contributing most strongly to every prediction."
        )

        st.info(
            "Statistically validated using McNemar's test, bootstrap confidence "
            "intervals and Bayesian model comparison."
        )

with col_b:
    with st.container(border=True):
        st.caption("EXPLORE THE PLATFORM")
        st.header("Navigation")

        st.subheader(" Predictor")
        st.write(
            "Browse companies, generate predictions, inspect SHAP explanations "
            "and produce a full business intelligence report."
        )

        st.divider()

        st.subheader("📊 Performance")
        st.write(
            "Review model metrics, ROC and precision-recall curves, confusion "
            "matrices and statistical validation."
        )

        st.divider()

        st.subheader(" AI Agent")
        st.write(
            "Interact with an agentic AI assistant powered by Groq and OpenAI routing."
        )

        st.divider()

        st.caption("LLM GATEWAY")
        st.write("⚡ Quick explanations via Groq")
        st.write("🧾 Full reports via OpenAI")
        st.write("🔁 Automatic fallback on failure")

st.divider()

st.caption(
    "Data source: UK Companies House bulk download · Public company data only · "
    "No personal data processed"
)