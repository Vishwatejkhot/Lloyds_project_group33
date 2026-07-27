"""
AI Agent — agentic chat with tool calling, LLM routing transparency, report generation.
"""
import os, sys
from textwrap import dedent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from agents.report_agent import run_agent, generate_full_report
from utils.pdf_generator import pdf_download_button
st.set_page_config(
    page_title="AI Agent · Lloyds SME",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
<style>
/* Fix AI chat text visibility */
[data-testid="stChatMessage"] {
    color: #111827 !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] span {
    color: #111827 !important;
}

[data-testid="stChatMessage"] table,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    color: #111827 !important;
}
.stApp {
    background:
        radial-gradient(circle at top right, rgba(0, 166, 77, 0.10), transparent 28%),
        linear-gradient(180deg, #f7faf9 0%, #eef4f1 100%);
}

.block-container {
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003f2f 0%, #006a4d 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.agent-hero {
    background: linear-gradient(135deg, #003f2f 0%, #006a4d 70%, #0b8a66 100%);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    box-shadow: 0 18px 42px rgba(0, 63, 47, 0.20);
    margin-bottom: 1.4rem;
}

.agent-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 999px;
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.38rem 0.78rem;
    margin-bottom: 0.9rem;
}

.agent-title {
    color: white !important;
    font-size: 2.45rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 0.6rem;
}

.agent-subtitle {
    color: rgba(255, 255, 255, 0.84) !important;
    font-size: 1rem;
    max-width: 900px;
    margin: 0;
}

[data-testid="stExpander"] {
    background: white;
    border: 1px solid #dce8e2;
    border-radius: 14px;
}

[data-testid="stExpander"] summary {
    background: #f5fbf8 !important;
    color: #006a4d !important;
    font-weight: 700;
    padding: 12px 16px;
}

[data-testid="stExpander"] summary * {
    color: #006a4d !important;
}

[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] em,
[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] strong {
    color: #334155 !important;
    opacity: 1 !important;
}

[data-testid="stChatMessage"] {
    border: 1px solid rgba(17, 24, 39, 0.10);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    background: rgba(255, 255, 255, 0.90);
    box-shadow: 0 8px 24px rgba(20, 46, 37, 0.06);
}

[data-testid="stChatInput"] {
    border-radius: 14px;
}

.model-tag {
    display: inline-block;
    font-size: 11px;
    color: #006a4d;
    background: #e8f5ef;
    border: 1px solid #b9dfcf;
    border-radius: 999px;
    padding: 3px 9px;
    margin-top: 6px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="agent-hero">'
    '<div class="agent-badge">LLOYDS SME DECISION SUPPORT</div>'
    '<div class="agent-title">🤖 AI Banking Intelligence Agent</div>'
    '<p class="agent-subtitle">'
    'Ask questions about model performance, sector opportunities, SHAP drivers and company-level analysis.'
    'The agent automatically routes requests through Groq or OpenAI and can generate professional banking reports.'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report_context" not in st.session_state:
    st.session_state.last_report_context = None

# ── Sidebar 
with st.sidebar:
    st.header("Agent Settings")
    task_mode = st.selectbox(
        "Task mode (affects LLM routing)",
        ["chat", "report", "explain", "analysis"],
        help="chat/explain → Groq first | report/analysis → OpenAI first"
    )
    st.markdown("---")
    st.markdown("**LLM Gateway routing:**")
    st.markdown("- `chat / explain` → Groq gpt-oss-120b → OpenAI fallback")
    st.markdown("- `report / analysis` → GPT-4o-mini → Groq fallback")
    st.markdown("---")
    st.markdown("**Available tools:**")
    st.markdown("- `get_model_performance` — F1, AUC, precision, recall")
    st.markdown("- `get_sector_stats` — high-growth companies by sector")
    st.markdown("- `get_shap_importance` — cross-label feature drivers")
    st.markdown("---")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("📄 Generate Company Report", disabled=st.session_state.last_report_context is None):
        ctx = st.session_state.last_report_context
        with st.spinner("Generating report via OpenAI…"):
            try:
                report, model_used = generate_full_report(
                    {"growth": ctx["growth"], "risk": ctx["risk"], "lending": ctx["lending"]},
                    {k: [(f, v) for f, v in vals] for k, vals in ctx.get("top_features", {}).items()},
                    ctx.get("sector", "Unknown")
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Full BI Report** (from Predictor page context)\n\n{report}",
                    "model": model_used,
                })
                st.session_state["agent_report_text"] = report
                st.session_state["agent_report_ctx"]  = ctx
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if "agent_report_text" in st.session_state:
        st.markdown("---")
        ctx_r = st.session_state["agent_report_ctx"]
        pdf_download_button(
            report_text=st.session_state["agent_report_text"],
            sector=ctx_r.get("sector", "Unknown"),
            growth=ctx_r.get("growth", 0.0),
            risk=ctx_r.get("risk", 0.0),
            lending=ctx_r.get("lending", 0.0),
            top_features={k: [(f, v) for f, v in vals]
                          for k, vals in ctx_r.get("top_features", {}).items()},
            key="agent_pdf_dl",
        )

# ── Company context badge 
ctx = st.session_state.last_report_context
if ctx:
    st.info(
        f"📎 Company context loaded from Predictor — "
        f"Sector: **{ctx['sector']}** · "
        f"Growth: **{ctx['growth']:.1%}** · "
        f"Risk: **{ctx['risk']:.1%}** · "
        f"Lending: **{ctx['lending']:.1%}**"
    )

# ── Suggested prompts 
with st.expander("💡 Suggested prompts", expanded=True):
    st.markdown("""
- *Which model performs best on the risk signal label and why?*
- *Which BCB sectors have the most high-growth SME opportunities?*
- *What are the top SHAP features driving lending need predictions?*
- *Compare XGBoost vs Random Forest performance across all labels.*
- *Generate a summary of the Bayesian model comparison findings.*
- *What does a growth probability of 85% mean for a banking relationship manager?*
""")

# ── Chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "model" in msg and msg["role"] == "assistant":
            st.markdown(
                f'<span style="font-size:11px;color:#888;border:1px solid #ddd;'
                f'border-radius:8px;padding:1px 8px;">via {msg["model"]}</span>',
                unsafe_allow_html=True
            )

# ── Chat input 
prompt = st.chat_input("Ask the agent about models, sectors, features, or company analysis…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking…"):
            try:
                response, model_used = run_agent(
                    prompt,
                    context=st.session_state.last_report_context,
                    task=task_mode,
                )
                st.markdown(response)
                st.markdown(
                    f'<span style="font-size:11px;color:#888;border:1px solid #ddd;'
                    f'border-radius:8px;padding:1px 8px;">via {model_used}</span>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "model": model_used,
                })
            except Exception as e:
                err = str(e)
                st.error(f"Agent error: {err}")
                if "api_key" in err.lower() or "authentication" in err.lower():
                    st.warning("Add your API keys to `dashboard/.env` (copy from `.env.example`).")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {err}",
                })
