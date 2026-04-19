"""Builtin tool registry."""

from __future__ import annotations

from rlm.tools.base import BuiltinTool
from rlm.tools.ipython import IpythonTool
from rlm.tools.summarize import SummarizeTool


_BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (IpythonTool(), SummarizeTool())
_TOOLS_BY_NAME = {tool.name: tool for tool in _BUILTIN_TOOLS}


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas with runtime defaults baked in."""
    return [tool.schema() for tool in _BUILTIN_TOOLS]


def get_builtin_tool(name: str) -> BuiltinTool | None:
    """Look up a builtin tool handler by name."""
    return _TOOLS_BY_NAME.get(name)
