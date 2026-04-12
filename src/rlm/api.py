"""Public Python API for running rlm agents."""

import asyncio
import os

from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult


def _child_session() -> Session | None:
    """If inside a parent session (depth > 0), create a child under it."""
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    depth = int(os.environ.get("RLM_DEPTH", "0"))
    if parent_dir and depth > 0:
        return Session(Session.child_dir(parent_dir))
    return None


def batch(prompts: str | list[str], **kwargs) -> list[RLMResult]:
    """Run one or more rlm agents. Always returns a list of results."""
    normalized = [prompts] if isinstance(prompts, str) else prompts
    if len(normalized) == 1 and "session" not in kwargs:
        child = _child_session()
        if child:
            kwargs["session"] = child
    engine = RLMEngine(**kwargs)
    return asyncio.run(engine.batch(normalized))
