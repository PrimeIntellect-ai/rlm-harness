"""Shared builtin tool types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rlm.types import BuiltinMetricEvent, RLMMetrics, TokenUsage


@dataclass
class ToolOutcome:
    """Result of executing a builtin tool."""

    content: str
    drop_turns: int = 0
    flush_repl_state: bool = False
    metric_events: list[BuiltinMetricEvent] = field(default_factory=list)


@dataclass
class ToolContext:
    """Runtime context passed to builtin tool handlers."""

    messages: list[dict[str, Any]]
    metrics: RLMMetrics
    total_usage: TokenUsage
    last_prompt_tokens: int
    exec_timeout: int
    repl: Any | None = None
    state: dict[str, Any] = field(default_factory=dict)
    cwd: str = ""


class BuiltinTool(Protocol):
    """Interface for builtin OpenAI tool handlers."""

    name: str

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI schema for this tool."""

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        """Execute the tool and return a string tool result plus side effects."""

    def prompt_lines(self, *, max_turns_in_context: int | None) -> list[str]:
        """Return the lines this tool contributes to the system prompt.

        May return an empty list when the tool has nothing to describe given
        the current configuration (e.g. summarize only adds a line when a
        context-turn limit is in effect).
        """
