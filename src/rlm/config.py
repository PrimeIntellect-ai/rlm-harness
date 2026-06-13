"""Centralized configuration.

rlm's behavioral knobs come from ``RLM_*`` environment variables.
:func:`load_config` reads and validates them once, caches the result, and
returns an immutable :class:`Config`. Code reads it either by being handed one
(``RLMEngine`` stores ``self.config`` and sources its attributes from it) or,
for leaf / cross-process helpers, via :func:`get_config`.

Environment variables remain the transport across process boundaries: each
IPython kernel is a fresh subprocess that inherits the parent's environment and
builds its own :class:`Config`. ``IPythonREPL._inject_startup`` overrides the
three per-kernel values (session dir, depth, live-agents dir) in the child's env
*before* rlm is imported there, so the child's :func:`load_config` sees them.

Two cohesive domains keep their own env handling and are intentionally NOT here:

- **provider credentials** (``rlm.client.resolve_provider``) — keeps secrets out
  of a shared, repr-able object;
- **the active builtin-tool set** (``rlm.tools.registry``) — validated against
  the live (test-patched) registry, not a static snapshot.

Caching note: the cache is process-lifetime. In production the environment is
fixed once per process (the harness/CLI before the root runs; ``_inject_startup``
before a kernel imports rlm), so caching is safe. When the environment changes
after first load — the CLI applying a flag, or tests monkeypatching between
cases — call :func:`reload_config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL = "openai/gpt-5-mini"
DEFAULT_EXEC_TIMEOUT = 300
DEFAULT_AGENT_WAIT_TIMEOUT = 300.0
DEFAULT_SDK_MAX_RETRIES = 5


def env_int(name: str, default: int) -> int:
    """Int from env; missing → ``default``. Malformed raises (loud misconfig)."""
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def positive_or_none(name: str) -> int | None:
    """``int(env)`` if positive, else ``None`` (missing / ``0`` / negative).

    Strict: a malformed value raises. Used for budgets, which have no
    "disabled on bad input" contract.
    """
    value = env_int(name, 0)
    return value if value > 0 else None


def cap(name: str) -> int | None:
    """Live-agent cap from env: a positive int enables it, anything else disables.

    Tolerant by contract — a cap is "active only when a positive int, otherwise
    every acquire is granted" — so missing / empty / ``0`` / negative / malformed
    all map to ``None`` (disabled) rather than crashing on the acquire hot path
    when a harness templates an unset var to e.g. ``""``.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_summarize_at_tokens(value: int | str | None) -> int | None:
    """Normalize ``summarize_at_tokens`` to a positive int or ``None``.

    Accepts ``None`` / empty string (disabled), an ``int``, or a numeric ``str``
    (from the env var). Anything else raises ``ValueError``.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("summarize_at_tokens must be an int")
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"summarize_at_tokens must be an int (got {value!r})"
            ) from exc
    elif isinstance(value, int):
        parsed = value
    else:
        raise ValueError(
            f"summarize_at_tokens must be int or None (got {type(value).__name__})"
        )
    if parsed <= 0:
        raise ValueError(f"summarize_at_tokens must be positive (got {parsed})")
    return parsed


def agent_wait_timeout() -> float | None:
    """Seconds to wait for a slot before proceeding over the cap; ``0`` → forever.

    Tolerant like the caps: missing / empty / malformed → the default, so a bad
    value can't crash the slot-acquire path.
    """
    raw = os.environ.get("RLM_AGENT_WAIT_TIMEOUT")
    if not raw:
        return DEFAULT_AGENT_WAIT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_AGENT_WAIT_TIMEOUT
    return value if value > 0 else None


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the ``RLM_*`` environment.

    Optional string paths (``session_dir``, ``live_agents_dir``,
    ``system_prompt_path``, ``append_to_system_prompt``) are stored raw — they
    may be ``None`` or ``""`` — so consumers keep their existing truthiness
    checks. ``RLMEngine`` folds its constructor overrides in via
    :func:`dataclasses.replace`.
    """

    # Model / budgets
    model: str
    max_tokens: int | None
    sub_max_tokens: int | None
    summarize_at_tokens: int | None
    # Recursion / identity (per-process; set in env before a kernel imports rlm)
    depth: int
    max_depth: int
    session_dir: str | None
    # Execution
    exec_timeout: int
    max_output: int
    max_tool_output_chars: int
    # System prompt
    system_prompt_path: str | None
    append_to_system_prompt: str | None
    # Live-agent caps
    max_live_agents: int | None
    max_running_agents: int | None
    agent_wait_timeout: float | None
    live_agents_dir: str | None
    # Misc
    home: Path
    sdk_max_retries: int
    allow_git: bool


def build() -> Config:
    max_output = env_int("RLM_MAX_OUTPUT", -1)
    if max_output == 0:
        raise ValueError(
            "RLM_MAX_OUTPUT must be positive, or -1 to disable truncation"
        )
    return Config(
        model=os.environ.get("RLM_MODEL", DEFAULT_MODEL),
        max_tokens=positive_or_none("RLM_MAX_TOKENS"),
        sub_max_tokens=positive_or_none("RLM_SUB_MAX_TOKENS"),
        summarize_at_tokens=parse_summarize_at_tokens(
            os.environ.get("RLM_SUMMARIZE_AT_TOKENS")
        ),
        depth=env_int("RLM_DEPTH", 0),
        max_depth=env_int("RLM_MAX_DEPTH", 0),
        session_dir=os.environ.get("RLM_SESSION_DIR"),
        exec_timeout=env_int("RLM_EXEC_TIMEOUT", DEFAULT_EXEC_TIMEOUT),
        max_output=max_output,
        max_tool_output_chars=env_int("RLM_MAX_TOOL_OUTPUT_CHARS", -1),
        system_prompt_path=os.environ.get("RLM_SYSTEM_PROMPT_PATH"),
        append_to_system_prompt=os.environ.get("RLM_APPEND_TO_SYSTEM_PROMPT"),
        max_live_agents=cap("RLM_MAX_LIVE_AGENTS"),
        max_running_agents=cap("RLM_MAX_RUNNING_AGENTS"),
        agent_wait_timeout=agent_wait_timeout(),
        live_agents_dir=os.environ.get("RLM_LIVE_AGENTS_DIR"),
        home=Path(os.environ.get("RLM_HOME") or Path.home() / ".rlm"),
        sdk_max_retries=env_int("RLM_SDK_MAX_RETRIES", DEFAULT_SDK_MAX_RETRIES),
        allow_git=os.environ.get("RLM_ALLOW_GIT") == "1",
    )


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load (and cache) the process-wide config from the environment."""
    return build()


def get_config() -> Config:
    """Return the cached process-wide config (loads it on first use)."""
    return load_config()


def reload_config() -> Config:
    """Drop the cache and reload from the current environment.

    Use when the environment changes after first load (CLI flags, tests).
    """
    load_config.cache_clear()
    return load_config()


def with_overrides(base: Config, **overrides) -> Config:
    """Return ``base`` with the given non-``None`` fields replaced."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
