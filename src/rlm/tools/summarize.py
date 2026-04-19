"""Builtin summarize tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome


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


class SummarizeTool:
    """Builtin tool handler for context summarization."""

    name = "summarize"

    def schema(self) -> dict[str, Any]:
        return SUMMARIZE_SCHEMA

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
            context.metrics.summarize_rejected_count += 1
            return ToolOutcome(
                content=(
                    "[no-op] num_turns is required and must be > 0 "
                    f"(got {num_turns}). No context was dropped."
                )
            )
        if num_turns > droppable:
            context.metrics.summarize_rejected_count += 1
            return ToolOutcome(
                content=(
                    f"[no-op] num_turns={num_turns} exceeds droppable turns "
                    f"({droppable}). No context was dropped."
                )
            )

        context.metrics.summarize_count += 1
        context.metrics.summarize_prompt_tokens_before.append(
            context.last_prompt_tokens
        )
        context.metrics.summarize_completion_tokens_before.append(
            context.total_usage.completion_tokens
        )
        context.metrics.turns_between_summarizes.append(
            context.metrics.turns_since_last_summarize
        )
        context.metrics.summarize_summary_lengths.append(len(summary))
        context.metrics.summarize_total_turns_dropped += num_turns

        if context.metrics.turns > 0:
            prompt_per_turn = context.last_prompt_tokens / context.metrics.turns
            completion_per_turn = (
                context.total_usage.completion_tokens / context.metrics.turns
            )
            context.metrics.summarize_prompt_tokens_dropped.append(
                int(prompt_per_turn * num_turns)
            )
            context.metrics.summarize_completion_tokens_dropped.append(
                int(completion_per_turn * num_turns)
            )

        state.turn_at_last_summarize = context.metrics.turns
        start = state.dropped_turn_count
        end = start + num_turns - 1
        state.summaries.append(f"[turns {start}-{end}] {summary}")
        state.dropped_turn_count += num_turns

        return ToolOutcome(
            content="\n\n".join(state.summaries),
            drop_turns=num_turns,
            flush_repl_state=flush_repl_state,
        )

    @staticmethod
    def _count_droppable_turns(messages: list[dict[str, Any]]) -> int:
        return max(
            0, sum(1 for message in messages if message["role"] == "assistant") - 1
        )

    @staticmethod
    def _as_text(value: Any) -> str:
        return value if isinstance(value, str) else str(value)
