"""Shared chat completions for extraction, bridging and answer generation."""

import os

from openai import OpenAI

from .orcarouter import api_key, api_url


def llm_provider():
    provider = os.getenv("INDEXRAG_LLM_PROVIDER", "openai").lower()
    if provider not in ("openai", "orcarouter"):
        raise ValueError("LLM provider must be openai or orcarouter.")
    return provider


def resolve_llm_model(model=None):
    provider = llm_provider()
    return (
        model
        or os.getenv("INDEXRAG_LLM_MODEL")
        or ("deepseek/deepseek-v4.1-flash" if provider == "orcarouter" else "gpt-4o-mini")
    )


def chat_completion(*, messages, model=None, **kwargs):
    provider = llm_provider()
    options = {"api_key": api_key(), "base_url": api_url()} if provider == "orcarouter" else {}
    model = resolve_llm_model(model)
    # Keep the default model's reasoning from consuming short/structured output budgets.
    # https://api-docs.deepseek.com/guides/thinking_mode/
    if provider == "orcarouter" and model == "deepseek/deepseek-v4.1-flash" and "reasoning_effort" not in kwargs:
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra
    with OpenAI(**options) as client:
        result = client.chat.completions.create(model=model, messages=messages, **kwargs)
    if not result.choices or not result.choices[0].message.content:
        raise ValueError("LLM returned no answer text; check the model and output token budget.")
    return result
