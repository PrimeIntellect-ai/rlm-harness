"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
import os
from typing import Any, Awaitable, Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from rlm.types import TokenUsage

_RETRYABLE: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

# Widely-spaced delays (seconds) between attempts; total ~5 min wall budget.
_RETRY_DELAYS: tuple[int, ...] = (15, 30, 60, 90, 120)


def make_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from environment variables.

    Tags every outbound request with ``X-RLM-Depth: <RLM_DEPTH>`` so an
    interceptor (e.g. verifiers' interception server) can distinguish
    parent-agent calls (depth 0) from sub-agent calls (depth >= 1) and
    decide whether to record them in the rollout's trajectory.
    """
    base_url = os.environ.get("RLM_BASE_URL")
    api_key = (
        os.environ.get("RLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    depth = os.environ.get("RLM_DEPTH", "0")
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=int(os.environ.get("RLM_SDK_MAX_RETRIES", 5)),
        default_headers={"X-RLM-Depth": depth},
    )


async def call_with_retries(
    func: Callable[..., Awaitable[Any]], /, **kwargs: Any
) -> Any:
    """Call ``func(**kwargs)`` with widely-spaced retries on transient errors.

    Extends the SDK's retry set with ``NotFoundError`` to ride out intermittent
    tunnel/proxy 404s that the SDK itself does not retry.
    """
    for delay in _RETRY_DELAYS:
        try:
            return await func(**kwargs)
        except _RETRYABLE:
            await asyncio.sleep(delay)
    return await func(**kwargs)


def extract_usage(response) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )
