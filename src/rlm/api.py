"""Public Python API for running rlm agents."""

from __future__ import annotations

import asyncio
import os

from rlm._agent_limit import (
    RUNNING,
    TOTAL,
    AgentLimitReached,
    acquire_slot,
    acquire_slot_blocking,
    release_slot,
)
from rlm._async_runtime import Handle, Registry, close_all_registries
from rlm.engine import RLMEngine
from rlm.session import Session, _sanitize_name
from rlm.types import RLMResult


def _is_subagent() -> bool:
    """True below the root rollout (depth > 0), so this agent counts against the
    live-agent caps; the root rollout (depth 0) is uncapped."""
    return int(os.environ.get("RLM_DEPTH", "0")) > 0


def _child_session(name: str | None = None) -> Session | None:
    """If inside a parent session (depth > 0), create a child under it."""
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    if parent_dir and _is_subagent():
        return Session(Session.child_dir(parent_dir, name=name))
    return None


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent to completion (blocking).

    A sub-agent (depth > 0) counts against both caps and **waits** for a slot in
    each (total, then running) — unlike ``send``, which raises on the total cap.
    The root rollout is not capped.
    """
    if "session" not in kwargs:
        child = _child_session()
        if child:
            kwargs["session"] = child
    total_marker = running_marker = None
    if _is_subagent():
        total_marker = await acquire_slot_blocking(TOTAL)
        running_marker = await acquire_slot_blocking(RUNNING)
    try:
        engine = RLMEngine(**kwargs)
        return await engine.run(prompt)
    finally:
        release_slot(running_marker)
        release_slot(total_marker)


# --- Background, named, persistent sub-agents ------------------------------

# Per-kernel registry of named sub-agents. This module is imported fresh in each
# IPython kernel, so the registry is naturally per-kernel (hierarchical): each
# agent owns the registry of the children it spawned.
_REGISTRY = Registry()


class _RlmProcessor:
    """Stateful processor: one live RLMEngine continued across sends.

    The engine is created and set up lazily on the first item so ``send``
    returns without blocking on kernel startup. Re-sending the same name appends
    a turn to the same engine (a multi-turn conversation).

    Slots: the engine holds a *total* slot from creation to teardown, and a
    *running* slot only while a turn executes (acquired per ``advance``). A turn
    that errors reaps the engine (kernel + total slot) eagerly, leaving the worker
    in its error state so the model can still read ``poll().error``.
    """

    def __init__(self, session: Session | None, engine_kwargs: dict, total_marker=None):
        self._session = session
        self._engine_kwargs = engine_kwargs
        self._total_marker = total_marker
        self._engine: RLMEngine | None = None

    async def process(self, prompt: str) -> RLMResult:
        running_marker = None
        try:
            if self._engine is None:
                kwargs = dict(self._engine_kwargs)
                if self._session is not None:
                    kwargs.setdefault("session", self._session)
                self._engine = RLMEngine(**kwargs)
                self._engine.setup()
            if _is_subagent():
                running_marker = await acquire_slot_blocking(RUNNING)
            return await self._engine.advance(prompt)
        except Exception:
            await self._reap()  # any failure (construct/setup/advance) reaps it
            raise
        finally:
            release_slot(running_marker)

    async def _reap(self) -> None:
        """Shut the engine's kernel and free its total slot (idempotent)."""
        try:
            if self._engine is not None:
                await self._engine.aclose()
        finally:
            release_slot(self._total_marker)
            self._total_marker = None

    async def teardown(self) -> None:
        await self._reap()


def send(
    prompt: str,
    name: str | None = None,
    max_tokens: int | None = None,
    **engine_kwargs,
) -> Handle:
    """Start or continue a named, persistent background sub-agent.

    Returns a handle immediately; keep it in a variable and ``handle.poll()`` it
    from a later cell. Re-sending the same ``name`` appends a turn to the same
    agent (multi-turn). ``name`` is canonicalized to a filesystem-safe form, so
    names differing only in unsafe characters (``foo/bar`` and ``foo-bar``)
    refer to the same agent. ``name=None`` draws a deterministic auto-name.
    ``engine_kwargs`` (e.g. ``model=...``) are forwarded to the sub-agent's
    ``RLMEngine`` and apply only when the agent is first created.

    ``max_tokens`` caps the sub-agent's completion-token budget. It is clamped
    to the ``RLM_SUB_MAX_TOKENS`` ceiling (the user-set maximum), which is also
    the default when ``max_tokens`` is omitted.
    """
    # Canonicalize so the registry key matches the session-dir suffix
    # (child_dir sanitizes too): names that differ only in unsafe characters
    # address one agent and one transcript, not two.
    if name is not None:
        name = _sanitize_name(name)

    ceiling = int(os.environ.get("RLM_SUB_MAX_TOKENS", "0")) or None
    if ceiling is not None:
        max_tokens = min(max_tokens, ceiling) if max_tokens is not None else ceiling
    if max_tokens is not None:
        engine_kwargs.setdefault("max_tokens", max_tokens)

    # Reserve a *total* (resident) slot for a new agent; continuation reuses it.
    # At capacity, raise rather than silently spawning — distinct from a runtime
    # failure surfaced via poll().error. The per-turn running slot is taken inside
    # the processor.
    is_new = name is None or _REGISTRY.get(name) is None
    total_marker = None
    if is_new and _is_subagent():
        granted, total_marker = acquire_slot(TOTAL)
        if not granted:
            raise AgentLimitReached(
                "total sub-agent cap (RLM_MAX_LIVE_AGENTS) reached; "
                "reuse an existing agent (re-send its name) instead of starting another"
            )

    holder: dict[str, Session | None] = {}

    def session_dir_factory(agent_name: str):
        session = _child_session(name=agent_name)
        holder["session"] = session
        return session.dir if session is not None else None

    def processor_factory(agent_name: str) -> _RlmProcessor:
        return _RlmProcessor(
            holder.get("session"), engine_kwargs, total_marker=total_marker
        )

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
