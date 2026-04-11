"""rlm — A minimalistic CLI agent for true recursion."""

import asyncio
import os

from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult

__all__ = ["run", "batch", "RLMEngine", "RLMResult"]


def _child_session() -> Session | None:
    """Create a child session under the parent if RLM_SESSION_DIR is set."""
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    if not parent_dir:
        return None
    parent = Session(parent_dir)
    child_dir = parent.child_dir()
    return Session(child_dir)


def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent. Blocking convenience wrapper.

    If called from within an rlm session (RLM_SESSION_DIR set),
    automatically creates a child session.
    """
    if "session" not in kwargs:
        kwargs["session"] = _child_session()
    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.run(prompt))


def batch(prompts: list[str], **kwargs) -> list[RLMResult]:
    """Run multiple rlm agents in parallel. Blocking convenience wrapper.

    If called from within an rlm session (RLM_SESSION_DIR set),
    uses the parent session for child dir creation.
    """
    if "session" not in kwargs:
        parent_dir = os.environ.get("RLM_SESSION_DIR")
        if parent_dir:
            kwargs["session"] = Session(parent_dir)
    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.batch(prompts))
