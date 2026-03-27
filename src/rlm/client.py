"""Thin LLM client wrapper. Extracts token usage from responses."""

import os

from openai import AsyncOpenAI

from rlm.types import TokenUsage


def make_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from environment variables."""
    base_url = os.environ.get("RLM_BASE_URL")
    api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def extract_usage(response) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )
