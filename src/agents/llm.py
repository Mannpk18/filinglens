"""Thin shared wrapper around the Anthropic SDK so every agent calls the model
the same way and tracing/cost-logging can be added in one place."""

from __future__ import annotations

from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def _client():
    from anthropic import Anthropic

    return Anthropic(api_key=settings.anthropic_api_key)


def call_llm(
    system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
) -> str:
    """Single-turn call. temperature=0 by default: agent nodes here are doing
    extraction/classification/verification, not creative writing, and
    determinism makes the eval harness's results reproducible run to run."""
    resp = _client().messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
