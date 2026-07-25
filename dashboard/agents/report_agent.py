"""
ReAct-style agent for generating reports and answering banking intelligence queries.
Uses LiteLLM tool-calling (compatible with both OpenAI and Groq function-calling models).
"""
import json
import os
import pandas as pd
from .gateway import call_llm, model_label

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

_SYSTEM = """You are a senior SME banking analyst at Lloyds Banking Group.
You have access to ML predictions (growth opportunity, risk signal, lending need) and SHAP explainability data for 3.1 million UK companies from Companies House.

Guidelines:
- Use professional banking language
- Cite ML probability scores explicitly (e.g. "Growth probability: 87.3%")
- Reference SHAP feature drivers when explaining predictions
- Sector context matters: compare to sector averages where possible
- Recommendations: High Priority / Monitor / Low Priority
- Keep reports concise and actionable (300–500 words)
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_performance",
            "description": "Retrieve model performance metrics (F1, AUC, precision, recall) for all 9 models across 3 labels.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_stats",
            "description": "Get high-growth company counts by BCB sector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Number of top sectors to return", "default": 7}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shap_importance",
            "description": "Get cross-label SHAP feature importance rankings.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _execute_tool(name: str, args: dict) -> str:
    try:
        if name == "get_model_performance":
            path = os.path.join(OUTPUT_DIR, "baseline_results_summary.csv")
            df = pd.read_csv(path)
            return df[["Label", "Model", "f1", "roc_auc", "precision", "recall"]].round(4).to_string(index=False)

        if name == "get_sector_stats":
            path = os.path.join(OUTPUT_DIR, "sector_opportunity_count.csv")
            df = pd.read_csv(path)
            top_n = args.get("top_n", 7)
            return df.head(top_n).to_string(index=False)

        if name == "get_shap_importance":
            path = os.path.join(OUTPUT_DIR, "shap_cross_label_importance.csv")
            df = pd.read_csv(path)
            return df.head(15).to_string(index=False)

    except Exception as exc:
        return f"Tool error: {exc}"
    return "Unknown tool"


def run_agent(user_query: str, context: dict | None = None, task: str = "chat") -> tuple[str, str]:
    """
    Run the ReAct agent. Returns (response_text, model_used_label).
    context: optional dict with current company prediction data.
    """
    ctx_str = ""
    if context:
        ctx_str = f"\n\nCurrent company context:\n{json.dumps(context, indent=2)}"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_query + ctx_str},
    ]

    MAX_ITER = 4
    model_used = "unknown"

    for _ in range(MAX_ITER):
        response, model_used = call_llm(messages, task=task, tools=_TOOLS, tool_choice="auto")
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content, model_label(model_used)

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool and append results
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _execute_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Final synthesis without tools
    final, model_used = call_llm(messages, task=task)
    return final.choices[0].message.content, model_label(model_used)


def quick_explain(label: str, prob: float, top_features: list) -> tuple[str, str]:
    """One-paragraph LLM explanation for a single prediction. Uses Groq (fast)."""
    feat_lines = "\n".join(f"  - {f}: {v:+.4f}" for f, v in top_features[:6])
    prompt = f"""A UK SME company received a {label.replace('_', ' ')} ML probability of {prob:.1%}.

Top SHAP feature contributions (positive = pushes score up):
{feat_lines}

In 2–3 sentences, explain what this prediction means for a Lloyds business banking analyst.
Be specific about which features matter most and what action they suggest."""

    response, model = call_llm(
        [{"role": "system", "content": "You are a concise Lloyds banking analyst."},
         {"role": "user", "content": prompt}],
        task="explain",
    )
    return response.choices[0].message.content, model_label(model)


def generate_full_report(preds: dict, top_features: dict, sector: str) -> tuple[str, str]:
    """Full structured BI report for a company. Uses OpenAI (quality)."""
    shap_section = ""
    for lbl, feats in top_features.items():
        top3 = ", ".join(f"{f} ({v:+.3f})" for f, v in feats[:3])
        shap_section += f"  {lbl.title()}: {top3}\n"

    prompt = f"""Generate a professional Lloyds Banking Group SME Prospect Report.

COMPANY DATA
  Sector: {sector}
  Growth Opportunity: {preds.get('growth', 0):.1%}
  Risk Signal:        {preds.get('risk', 0):.1%}
  Lending Need:       {preds.get('lending', 0):.1%}

TOP SHAP DRIVERS
{shap_section}
Write a structured report with these exact sections:
1. Executive Summary (2 sentences max)
2. Growth Opportunity Assessment
3. Risk Profile
4. Lending Need Evaluation
5. Key Prediction Drivers (reference SHAP features by name)
6. Recommendation (High Priority / Monitor / Low Priority — with justification)

Professional tone. Actionable for a business banking relationship manager."""

    response, model = call_llm(
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": prompt}],
        task="report",
        max_tokens=800,
    )
    return response.choices[0].message.content, model_label(model)
