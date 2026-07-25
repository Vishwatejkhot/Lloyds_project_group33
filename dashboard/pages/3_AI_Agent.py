"""
AI Agent — agentic chat with tool calling, LLM routing transparency, report generation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from agents.report_agent import run_agent, generate_full_report

st.set_page_config(page_title="AI Agent · Lloyds SME", page_icon="🤖", layout="wide")
st.title("🤖 AI Banking Intelligence Agent")
st.caption("Agentic AI · Tool calling · LLM Gateway (Groq ↔ OpenAI) · ReAct pattern")

# ── Init session state ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report_context" not in st.session_state:
    st.session_state.last_report_context = None

# ── Sidebar ────────────────────────────────────────────────────────────────
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
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ── Company context badge ──────────────────────────────────────────────────
ctx = st.session_state.last_report_context
if ctx:
    st.info(
        f"📎 Company context loaded from Predictor — "
        f"Sector: **{ctx['sector']}** · "
        f"Growth: **{ctx['growth']:.1%}** · "
        f"Risk: **{ctx['risk']:.1%}** · "
        f"Lending: **{ctx['lending']:.1%}**"
    )

# ── Suggested prompts ──────────────────────────────────────────────────────
with st.expander("💡 Suggested prompts"):
    st.markdown("""
- *Which model performs best on the risk signal label and why?*
- *Which BCB sectors have the most high-growth SME opportunities?*
- *What are the top SHAP features driving lending need predictions?*
- *Compare XGBoost vs Random Forest performance across all labels.*
- *Generate a summary of the Bayesian model comparison findings.*
- *What does a growth probability of 85% mean for a banking relationship manager?*
""")

# ── Chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "model" in msg and msg["role"] == "assistant":
            st.markdown(
                f'<span style="font-size:11px;color:#888;border:1px solid #ddd;'
                f'border-radius:8px;padding:1px 8px;">via {msg["model"]}</span>',
                unsafe_allow_html=True
            )

# ── Chat input ─────────────────────────────────────────────────────────────
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
