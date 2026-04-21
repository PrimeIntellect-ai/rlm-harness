"""Builtin tool registry."""

from __future__ import annotations

import os

from rlm.tools.base import BuiltinTool
from rlm.tools.ipython import IpythonTool
from rlm.tools.summarize import SummarizeTool


_BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (IpythonTool(), SummarizeTool())
_TOOLS_BY_NAME: dict[str, BuiltinTool] = {tool.name: tool for tool in _BUILTIN_TOOLS}


def _active_tool_names() -> list[str]:
    """Resolve the active builtin-tool names from the ``RLM_TOOLS`` env var.

    Unset → ``["ipython", "summarize"]`` (convenience default for direct CLI
    use; all other callers — notably the verifiers rlm_harness — are
    expected to set ``RLM_TOOLS`` explicitly so the tool set is a single
    source of truth).
    Comma-separated list → that subset, preserving user-specified order.
    Empty string → no tools (pure chat mode).
    Unknown names → ``ValueError``.
    Duplicates and surrounding whitespace are tolerated (deduped / stripped).
    """
    raw = os.environ.get("RLM_TOOLS")
    if raw is None:
        return ["ipython", "summarize"]
    names: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if name and name not in names:
            names.append(name)
    unknown = [n for n in names if n not in _TOOLS_BY_NAME]
    if unknown:
        raise ValueError(
            f"RLM_TOOLS contains unknown tool(s): {unknown}. "
            f"Available: {sorted(_TOOLS_BY_NAME)}"
        )
    return names


def get_active_builtin_tools() -> list[BuiltinTool]:
    """Return active builtin-tool instances (respects ``RLM_TOOLS``)."""
    return [_TOOLS_BY_NAME[name] for name in _active_tool_names()]


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas for active builtins."""
    return [tool.schema() for tool in get_active_builtin_tools()]


def get_builtin_tool(name: str) -> BuiltinTool | None:
    """Look up an active builtin tool handler by name.

    Returns None for unknown names or tools excluded by ``RLM_TOOLS``, so
    the engine's unknown-tool error path catches both cases identically.
    """
    if name not in _active_tool_names():
        return None
    return _TOOLS_BY_NAME.get(name)
