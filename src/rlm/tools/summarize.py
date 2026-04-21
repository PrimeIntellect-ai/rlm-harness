"""Builtin summarize tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.types import SummarizeApplied, SummarizeRejected


SUMMARIZE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": (
            "Summarize and drop old turns from context to free up space. "
            "A turn is one assistant response plus all its tool results. "
            "Dropping num_turns removes the oldest complete turns from context. "
            "Optionally flush the persistent IPython state after summarization."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_turns": {
                    "type": "integer",
                    "description": "Number of oldest turns to drop from context.",
                },
                "summary": {
                    "type": "string",
                    "description": "Your summary of the content being dropped.",
                },
                "flush_repl_state": {
                    "type": "boolean",
                    "description": (
                        "If true, restart the persistent IPython kernel after "
                        "summarization so Python state is reset."
                    ),
                },
            },
            "required": ["num_turns", "summary"],
        },
    },
}


@dataclass
class SummarizeState:
    """Runtime state for the summarize builtin tool."""

    summaries: list[str] = field(default_factory=list)
    dropped_turn_count: int = 0
    turn_at_last_summarize: int = 0


def summarize_drop_slice_bounds(
    messages: list[dict[str, Any]], num_turns: int
) -> tuple[int, int]:
    """Return the message slice that would be dropped for ``num_turns``."""
    start = 0
    while start < len(messages) and messages[start]["role"] != "assistant":
        start += 1

    end = start
    for _ in range(num_turns):
        if end >= len(messages) or messages[end]["role"] != "assistant":
            break
        end += 1
        while end < len(messages) and messages[end]["role"] == "tool":
            end += 1
    return start, end


class SummarizeTool:
    """Builtin tool handler for context summarization."""

    name = "summarize"

    def schema(self) -> dict[str, Any]:
        return SUMMARIZE_SCHEMA

    def prompt_lines(self, *, max_turns_in_context: int | None) -> list[str]:
        # Only mention summarize when there's a concrete limit to stay within;
        # otherwise the tool is present but unmotivated.
        if max_turns_in_context is None:
            return []
        return ["Use `summarize` to drop older turns and stay within this limit."]

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        state = context.state.setdefault("summarize", SummarizeState())
        if not isinstance(state, SummarizeState):
            state = SummarizeState()
            context.state["summarize"] = state

        num_turns = args.get("num_turns")
        summary = self._as_text(args.get("summary", ""))
        flush_repl_state = bool(args.get("flush_repl_state", False))
        droppable = self._count_droppable_turns(context.messages)

        if not isinstance(num_turns, int) or num_turns <= 0:
            return ToolOutcome(
                content=(
                    "[no-op] num_turns is required and must be > 0 "
                    f"(got {num_turns}). No context was dropped."
                ),
                metric_events=[SummarizeRejected()],
            )
        if num_turns > droppable:
            return ToolOutcome(
                content=(
                    f"[no-op] num_turns={num_turns} exceeds droppable turns "
                    f"({droppable}). No context was dropped."
                ),
                metric_events=[SummarizeRejected()],
            )

        summary_chars = len(summary)
        dropped_messages = self._dropped_message_slice(context.messages, num_turns)
        dropped_chars = self._count_content_chars(dropped_messages)

        state.turn_at_last_summarize = context.metrics.turns
        start = state.dropped_turn_count
        end = start + num_turns - 1
        state.summaries.append(f"[turns {start}-{end}] {summary}")
        state.dropped_turn_count += num_turns

        return ToolOutcome(
            content="\n\n".join(state.summaries),
            drop_turns=num_turns,
            flush_repl_state=flush_repl_state,
            metric_events=[
                SummarizeApplied(
                    num_turns=num_turns,
                    summary_chars=summary_chars,
                    dropped_chars=dropped_chars,
                    turns_since_last_summarize=context.metrics.turns_since_last_summarize,
                ),
            ],
        )

    @staticmethod
    def _count_droppable_turns(messages: list[dict[str, Any]]) -> int:
        return max(
            0, sum(1 for message in messages if message["role"] == "assistant") - 1
        )

    @staticmethod
    def _as_text(value: Any) -> str:
        return value if isinstance(value, str) else str(value)

    @classmethod
    def _dropped_message_slice(
        cls, messages: list[dict[str, Any]], num_turns: int
    ) -> list[dict[str, Any]]:
        start, end = summarize_drop_slice_bounds(messages, num_turns)
        return messages[start:end]

    @classmethod
    def _count_content_chars(cls, messages: list[dict[str, Any]]) -> int:
        return sum(cls._message_chars(message) for message in messages)

    @classmethod
    def _message_chars(cls, message: dict[str, Any]) -> int:
        total = cls._content_chars(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            total += sum(cls._tool_call_chars(tool_call) for tool_call in tool_calls)
        return total

    @classmethod
    def _content_chars(cls, content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(cls._content_chars(item) for item in content)
        if isinstance(content, dict):
            total = 0
            for field in ("text", "input_text", "output_text"):
                value = content.get(field)
                if isinstance(value, str):
                    total += len(value)
            nested_content = content.get("content")
            if nested_content is not None:
                total += cls._content_chars(nested_content)
            return total
        return 0

    @classmethod
    def _tool_call_chars(cls, tool_call: Any) -> int:
        if isinstance(tool_call, dict):
            function = tool_call.get("function")
        else:
            function = getattr(tool_call, "function", None)
        if function is None:
            return 0
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
        total = 0
        if name is not None:
            total += len(cls._as_text(name))
        if arguments is not None:
            total += len(cls._as_text(arguments))
        return total
