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
    if "groq" in model:
        return os.getenv("GROQ_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


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
    Raises RuntimeError if all routes fail.
    """
    route = _ROUTES.get(task, _ROUTES["chat"])
    last_err = None

    for model in route:
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                api_key=_api_key(model),
                stream=stream,
                timeout=60,
                max_tokens=max_tokens,
            )
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            response = completion(**kwargs)
            return response, model
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"All LLM routes failed. Last error: {last_err}")


def model_label(model_str: str) -> str:
    """Human-readable label for display in UI."""
    if "groq" in model_str:
        return f"Groq · {model_str.replace('groq/', '')}"
    return f"OpenAI · {model_str}"
