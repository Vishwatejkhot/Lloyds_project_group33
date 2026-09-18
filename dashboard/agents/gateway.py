"""
LLM Gateway — routes between Groq (fast) and OpenAI (quality) via LiteLLM.

Routing strategy:
  explain  → Groq first (speed), OpenAI fallback
  report   → OpenAI first (quality), Groq fallback
  chat     → Groq first (interactive), OpenAI fallback
  analysis → OpenAI first (depth), Groq fallback
"""
import os
import litellm
from litellm import completion

litellm.suppress_debug_info = True

_GROQ_MODEL = "groq/openai/gpt-oss-120b"

_ROUTES = {
    "explain":  [_GROQ_MODEL, "gpt-4o-mini"],
    "report":   ["gpt-4o-mini", _GROQ_MODEL],
    "chat":     [_GROQ_MODEL, "gpt-4o-mini"],
    "analysis": ["gpt-4o-mini", _GROQ_MODEL],
}


def _api_key(model: str) -> str:
    name = "GROQ_API_KEY" if "groq" in model else "OPENAI_API_KEY"
    key = os.getenv(name, "")
    if key:
        return key
    # Fall back to Streamlit Cloud's secrets manager (no local .env there)
    try:
        import streamlit as st
        return st.secrets.get(name, "")
    except Exception:
        return ""


def call_llm(
    messages: list,
    task: str = "chat",
    stream: bool = False,
    tools: list | None = None,
    tool_choice=None,
    max_tokens: int = 1024,
):
    """
    Route the LLM call through the appropriate provider.
    Returns (response, model_used_str).
    Raises RuntimeError if all routes fail, listing every provider's error.
    """
    route = _ROUTES.get(task, _ROUTES["chat"])
    errors = []

    for model in route:
        key = _api_key(model)
        if not key:
            errors.append(f"{model}: API key not set (check .env)")
            continue
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                api_key=key,
                stream=stream,
                timeout=60,
                max_tokens=max_tokens,
            )
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            response = completion(**kwargs)  # type: ignore[operator]
            return response, model
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            continue

    raise RuntimeError("All LLM routes failed.\n" + "\n".join(errors))


def model_label(model_str: str) -> str:
    """Human-readable label for display in UI."""
    if "groq" in model_str:
        return f"Groq · {model_str.replace('groq/', '')}"
    return f"OpenAI · {model_str}"
