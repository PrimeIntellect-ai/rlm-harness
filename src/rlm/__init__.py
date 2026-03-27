"""rlm — A minimalistic CLI agent for true recursion."""

from rlm.engine import RLMEngine
from rlm.types import RLMResult

__all__ = ["run", "batch", "RLMEngine", "RLMResult"]


def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent. Blocking convenience wrapper."""
    import asyncio

    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.run(prompt))


def batch(prompts: list[str], **kwargs) -> list[RLMResult]:
    """Run multiple rlm agents in parallel. Blocking convenience wrapper."""
    import asyncio

    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.batch(prompts))
