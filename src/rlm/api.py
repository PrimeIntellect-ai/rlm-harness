"""Public Python API for running rlm agents."""

from __future__ import annotations

import asyncio
import os

from rlm._agent_limit import (
    AgentLimitReached,
    acquire_slot,
    acquire_slot_blocking,
    release_slot,
)
from rlm._async_runtime import Handle, Registry, close_all_registries
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult


def _child_session(name: str | None = None) -> Session | None:
    """If inside a parent session (depth > 0), create a child under it."""
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    depth = int(os.environ.get("RLM_DEPTH", "0"))
    if parent_dir and depth > 0:
        return Session(Session.child_dir(parent_dir, name=name))
    return None


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent to completion (blocking).

    A sub-agent (depth > 0) counts against the live-agent cap and **waits** for a
    free slot (unlike ``send``, which raises). The root rollout is not capped.
    """
    if "session" not in kwargs:
        child = _child_session()
        if child:
            kwargs["session"] = child
    marker = None
    if int(os.environ.get("RLM_DEPTH", "0")) > 0:
        marker = await acquire_slot_blocking()
    try:
        engine = RLMEngine(**kwargs)
        return await engine.run(prompt)
    finally:
        release_slot(marker)


# --- Background, named, persistent sub-agents ------------------------------

# Per-kernel registry of named sub-agents. This module is imported fresh in each
# IPython kernel, so the registry is naturally per-kernel (hierarchical): each
# agent owns the registry of the children it spawned.
_REGISTRY = Registry(name_seed=os.environ.get("RLM_SESSION_DIR", ""))


class _RlmProcessor:
    """Stateful processor: one live RLMEngine continued across sends.

    The engine is created and set up lazily on the first item so ``send``
    returns without blocking on kernel startup. Re-sending the same name appends
    a turn to the same engine (a multi-turn conversation).
    """

    def __init__(self, session: Session | None, engine_kwargs: dict, marker=None):
        self._session = session
        self._engine_kwargs = engine_kwargs
        self._marker = marker
        self._engine: RLMEngine | None = None

    async def process(self, prompt: str) -> RLMResult:
        if self._engine is None:
            kwargs = dict(self._engine_kwargs)
            if self._session is not None:
                kwargs.setdefault("session", self._session)
            self._engine = RLMEngine(**kwargs)
            self._engine.setup()
        return await self._engine.advance(prompt)

    async def teardown(self) -> None:
        try:
            if self._engine is not None:
                await self._engine.aclose()
        finally:
            release_slot(self._marker)


def send(
    prompt: str,
    name: str | None = None,
    max_tokens: int | None = None,
    **engine_kwargs,
) -> Handle:
    """Start or continue a named, persistent background sub-agent.

    Returns a handle immediately; poll it with ``handle.poll()`` (or
    ``rlm.get(name)`` from another cell). Re-sending the same ``name`` appends a
    turn to the same agent (multi-turn). ``name=None`` draws a deterministic
    auto-name. ``engine_kwargs`` (e.g. ``model=...``) are forwarded to the
    sub-agent's ``RLMEngine`` and apply only when the agent is first created.

    ``max_tokens`` caps the sub-agent's completion-token budget. It is clamped
    to the ``RLM_SUB_MAX_TOKENS`` ceiling (the user-set maximum), which is also
    the default when ``max_tokens`` is omitted.
    """
    ceiling = int(os.environ.get("RLM_SUB_MAX_TOKENS", "0")) or None
    if ceiling is not None:
        max_tokens = min(max_tokens, ceiling) if max_tokens is not None else ceiling
    if max_tokens is not None:
        engine_kwargs.setdefault("max_tokens", max_tokens)

    # Reserve a live-agent slot for a *new* agent (continuation reuses its slot).
    # At capacity, raise rather than silently spawning — distinct from a runtime
    # failure surfaced via poll().error.
    is_new = name is None or _REGISTRY.get(name) is None
    marker = None
    if is_new:
        granted, marker = acquire_slot()
        if not granted:
            raise AgentLimitReached(
                "live sub-agent cap (RLM_MAX_LIVE_AGENTS) reached; "
                "dismiss an agent before starting another"
            )

    holder: dict[str, Session | None] = {}

    def session_dir_factory(agent_name: str):
        session = _child_session(name=agent_name)
        holder["session"] = session
        return session.dir if session is not None else None

    def processor_factory(agent_name: str) -> _RlmProcessor:
        return _RlmProcessor(holder.get("session"), engine_kwargs, marker=marker)

    return _REGISTRY.send(
        prompt,
        name=name,
        processor_factory=processor_factory,
        session_dir_factory=session_dir_factory,
    )


def _drain_agents() -> None:
    """Synchronously close every background agent in this kernel (all registries).

    Invoked by the engine's teardown cascade as a cell executed in the kernel
    where the registries live. nest_asyncio makes ``run_until_complete``
    reentrant, so each child finalizes its session and cascade-closes its own
    grandchildren before the caller shuts this kernel down.
    """
    asyncio.get_event_loop().run_until_complete(close_all_registries())
