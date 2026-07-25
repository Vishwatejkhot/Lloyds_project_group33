import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(
    page_title="Lloyds SME Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #006A4D !important; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] p { color: white !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #A8D5BA !important; }
.metric-card { background: #f0f7f4; border-left: 4px solid #006A4D;
               border-radius: 6px; padding: 14px 18px; margin-bottom: 10px; }
.badge-high   { background:#006A4D; color:white; padding:2px 10px; border-radius:12px; font-size:13px; }
.badge-med    { background:#FFA500; color:white; padding:2px 10px; border-radius:12px; font-size:13px; }
.badge-low    { background:#DC3545; color:white; padding:2px 10px; border-radius:12px; font-size:13px; }
.llm-tag { font-size:11px; color:#888; border:1px solid #ddd;
           border-radius:8px; padding:1px 8px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Home ───────────────────────────────────────────────────────────────────
st.title("🏦 Lloyds SME Intelligence Platform")
st.caption("Group 33 · Companies House ML · Powered by XGBoost + SHAP + Generative AI")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies Analysed", "3,145,434", "UK SMEs")
c2.metric("Best Growth F1", "0.9341", "XGBoost")
c3.metric("Best Risk AUC", "1.0000", "XGBoost")
c4.metric("Best Lending F1", "1.0000", "Random Forest")

st.markdown("---")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
### What this platform does
Using machine learning trained on 3.1 million UK companies from Companies House,
this dashboard identifies SME prospects for Lloyds Banking Group across three signals:

| Signal | Model | F1 Score |
|--------|-------|----------|
| 🌱 Growth Opportunity | XGBoost | 0.9341 |
| ⚠️ Risk Signal (14:1 imbalance) | XGBoost | 0.9973 |
| 💷 Lending Need Proxy | XGBoost | 1.0000 |

Statistical validation includes McNemar's test, 1000-iteration bootstrap CI, and Bayesian model comparison.
SHAP explainability identifies the key features driving each prediction.
""")

with col_b:
    st.markdown("""
### Navigation
**🔮 Predictor**
Browse companies, run predictions, SHAP waterfall, AI explanation & full BI report.

**📊 Performance**
Model metrics, ROC/PR curves, statistical tests.

**🤖 AI Agent**
Chat with an agentic AI powered by Groq + OpenAI routing.

---
**LLM Gateway**
- Quick explain → Groq (Llama 3.1 70B)
- Full reports  → OpenAI (GPT-4o-mini)
- Auto fallback on failure
""")

st.markdown("---")
st.caption("Data: UK Companies House bulk download (public). No personal data processed.")
