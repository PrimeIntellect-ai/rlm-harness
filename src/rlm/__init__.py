"""rlm — A minimalistic CLI agent for true recursion."""

import asyncio
import os

from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult

__all__ = ["run", "batch", "RLMEngine", "RLMResult"]


def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent. Blocking convenience wrapper."""
    # When called as a sub-agent (depth > 0), create a child session dir
    # under the parent so sibling calls nest correctly. We mutate the env
    # var and restore it after — safe because kernel calls are serialized.
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    depth = int(os.environ.get("RLM_DEPTH", "0"))
    if parent_dir and depth > 0:
        child_dir = Session(parent_dir).child_dir()
        os.environ["RLM_SESSION_DIR"] = str(child_dir)
    engine = RLMEngine(**kwargs)
    result = asyncio.run(engine.run(prompt))
    if parent_dir and depth > 0:
        os.environ["RLM_SESSION_DIR"] = parent_dir
    return result


def batch(prompts: list[str], **kwargs) -> list[RLMResult]:
    """Run multiple rlm agents in parallel. Blocking convenience wrapper."""
    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.batch(prompts))
