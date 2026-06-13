"""Public Python API for running rlm agents."""

from __future__ import annotations

import asyncio

from rlm.agent_limit import (
    RUNNING,
    TOTAL,
    AgentLimitReached,
    acquire_slot,
    acquire_slot_blocking,
    release_slot,
)
from rlm.async_runtime import BackgroundWorker, Handle, Registry, close_all_registries
from rlm.config import get_config
from rlm.engine import RLMEngine
from rlm.session import Session, sanitize_name
from rlm.types import RLMResult


def is_subagent() -> bool:
    """True below the root rollout (depth > 0), so this agent counts against the
    live-agent caps; the root rollout (depth 0) is uncapped."""
    return get_config().depth > 0


def child_session(name: str | None = None) -> Session | None:
    """If inside a parent session (depth > 0), create a child under it."""
    parent_dir = get_config().session_dir
    if parent_dir and is_subagent():
        return Session(Session.child_dir(parent_dir, name=name))
    return None


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent to completion (blocking).

    A sub-agent (depth > 0) counts against both caps and **waits** for a slot in
    each (total, then running) — unlike ``send``, which raises on the total cap.
    The root rollout is not capped.
    """
    if "session" not in kwargs:
        child = child_session()
        if child:
            kwargs["session"] = child
    total_marker = running_marker = None
    try:
        # Acquire inside the try so a cancellation/error between the two
        # blocking acquires still releases whatever was already reserved.
        if is_subagent():
            total_marker = await acquire_slot_blocking(TOTAL)
            running_marker = await acquire_slot_blocking(RUNNING)
        engine = RLMEngine(**kwargs)
        return await engine.run(prompt)
    finally:
        release_slot(running_marker)
        release_slot(total_marker)


# --- Background, named, persistent sub-agents ------------------------------

# Per-kernel registry of named sub-agents. This module is imported fresh in each
# IPython kernel, so the registry is naturally per-kernel (hierarchical): each
# agent owns the registry of the children it spawned.
REGISTRY = Registry()


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
            if is_subagent():
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
    refer to the same agent. ``name=None`` draws a random auto-name (a uuid hex).
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
        name = sanitize_name(name)

    # A non-positive request means "no explicit budget": treat it as omitted so it
    # falls back to the ceiling / default, instead of 0 disabling the budget by
    # truthiness or a negative stopping the sub-agent after one turn (B7).
    if max_tokens is not None and max_tokens <= 0:
        max_tokens = None
    ceiling = get_config().sub_max_tokens
    if ceiling is not None:
        max_tokens = min(max_tokens, ceiling) if max_tokens is not None else ceiling
    if max_tokens is not None:
        engine_kwargs.setdefault("max_tokens", max_tokens)

    def worker_factory(agent_name: str) -> BackgroundWorker:
        # Called only for a *new* agent (the registry reuses a live worker for a
        # continuation), so this is the single place a resident slot is reserved
        # and — on any creation failure — released. At capacity, raise rather than
        # silently spawning (distinct from a runtime failure surfaced via
        # poll().error). Once the worker is returned, its _RlmProcessor owns the
        # slot and reaps it on error / teardown; the per-turn running slot is
        # taken inside the processor.
        total_marker = None
        if is_subagent():
            granted, total_marker = acquire_slot(TOTAL)
            if not granted:
                raise AgentLimitReached(
                    "total sub-agent cap (RLM_MAX_LIVE_AGENTS) reached; "
                    "reuse an existing agent (re-send its name) instead of starting another"
                )
        try:
            session = child_session(name=agent_name)
            session_dir = session.dir if session is not None else None
            processor = _RlmProcessor(session, engine_kwargs, total_marker=total_marker)
            return BackgroundWorker(agent_name, processor, session_dir=session_dir)
        except Exception:
            release_slot(total_marker)
            raise

    return REGISTRY.send(prompt, name=name, worker_factory=worker_factory)


def drain_agents() -> None:
    """Synchronously close every background agent in this kernel (all registries).

    Invoked by the engine's teardown cascade as a cell executed in the kernel
    where the registries live. nest_asyncio makes ``run_until_complete``
    reentrant, so each child finalizes its session and cascade-closes its own
    grandchildren before the caller shuts this kernel down.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.run_until_complete(close_all_registries())
