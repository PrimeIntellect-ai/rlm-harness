"""Core data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class BuiltinToolCalled:
    name: str


@dataclass(frozen=True)
class IpythonExecuted:
    input_chars: int
    input_loc: int


@dataclass(frozen=True)
class SummarizeRejected:
    pass


@dataclass(frozen=True)
class SummarizeApplied:
    num_turns: int
    summary_chars: int
    dropped_chars: int
    prompt_tokens_before: int
    completion_tokens_before: int
    prompt_tokens_dropped: int
    completion_tokens_dropped: int
    turns_since_last_summarize: int


BuiltinMetricEvent = (
    BuiltinToolCalled | IpythonExecuted | SummarizeRejected | SummarizeApplied
)


@dataclass
class RLMMetrics:
    """Metrics tracked during an rlm session."""

    # Turn metrics
    turns: int = 0
    turns_since_last_summarize: int = 0
    turns_between_summarizes: list[int] = field(default_factory=list)

    # Summarize metrics
    summarize_count: int = 0
    summarize_rejected_count: int = 0
    summarize_total_turns_dropped: int = 0
    summarize_summary_lengths: list[int] = field(default_factory=list)
    summarize_chars_dropped_total: int = 0
    summarize_summary_chars_total: int = 0

    # Builtin tool metrics
    builtin_tool_calls_total: int = 0
    ipython_calls: int = 0
    ipython_input_chars_total: int = 0
    ipython_input_chars_mean: float = 0.0
    ipython_input_loc_total: int = 0
    ipython_input_loc_mean: float = 0.0

    # Token metrics before/dropped per summarize
    summarize_prompt_tokens_before: list[int] = field(default_factory=list)
    summarize_completion_tokens_before: list[int] = field(default_factory=list)
    summarize_prompt_tokens_dropped: list[int] = field(default_factory=list)
    summarize_completion_tokens_dropped: list[int] = field(default_factory=list)

    # This agent's token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Per-turn token counts from the API (for computing tool result tokens etc.)
    # tool_result_tokens[i] ≈ prompt_tokens_per_turn[i+1] - prompt_tokens_per_turn[i] - completion_tokens_per_turn[i]
    prompt_tokens_per_turn: list[int] = field(default_factory=list)
    completion_tokens_per_turn: list[int] = field(default_factory=list)

    # Aggregated from children
    sub_rlm_prompt_tokens: int = 0
    sub_rlm_completion_tokens: int = 0
    sub_rlm_count: int = 0

    # Budget tracking
    max_turns: int = 0
    max_tokens: int = 0
    stop_reason: str = ""  # "done", "max_turns", "token_budget", "multiple_tool_calls", "context_limit", "depth_limit"

    def record(self, event: BuiltinMetricEvent) -> None:
        if isinstance(event, BuiltinToolCalled):
            self.builtin_tool_calls_total += 1
            if event.name == "ipython":
                self.ipython_calls += 1
        elif isinstance(event, IpythonExecuted):
            self.ipython_input_chars_total += event.input_chars
            self.ipython_input_loc_total += event.input_loc
        elif isinstance(event, SummarizeRejected):
            self.summarize_rejected_count += 1
        elif isinstance(event, SummarizeApplied):
            self.summarize_count += 1
            self.turns_between_summarizes.append(event.turns_since_last_summarize)
            self.summarize_total_turns_dropped += event.num_turns
            self.summarize_summary_lengths.append(event.summary_chars)
            self.summarize_chars_dropped_total += event.dropped_chars
            self.summarize_summary_chars_total += event.summary_chars
            self.summarize_prompt_tokens_before.append(event.prompt_tokens_before)
            self.summarize_completion_tokens_before.append(
                event.completion_tokens_before
            )
            self.summarize_prompt_tokens_dropped.append(event.prompt_tokens_dropped)
            self.summarize_completion_tokens_dropped.append(
                event.completion_tokens_dropped
            )
        else:
            raise TypeError(f"Unsupported builtin metric event: {type(event)!r}")

        self._refresh_derived_metrics()

    def _refresh_derived_metrics(self) -> None:
        self.ipython_input_chars_mean = (
            self.ipython_input_chars_total / self.ipython_calls
            if self.ipython_calls
            else 0.0
        )
        self.ipython_input_loc_mean = (
            self.ipython_input_loc_total / self.ipython_calls
            if self.ipython_calls
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        self._refresh_derived_metrics()
        return asdict(self)


@dataclass
class RLMResult:
    answer: str
    session_dir: Path | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
