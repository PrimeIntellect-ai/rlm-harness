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
    turns_since_last_summarize: int


BuiltinMetricEvent = IpythonExecuted | SummarizeRejected | SummarizeApplied


@dataclass
class RLMMetrics:
    """Metrics tracked during an rlm session."""

    # Turn metrics
    turns: int = 0
    turns_since_last_summarize: int = 0
    turns_between_summarizes: list[int] = field(default_factory=list)

    # Summarize metrics
    summarize_rejected_count: int = 0
    summarize_turns_dropped_total: int = 0
    summarize_summary_lengths: list[int] = field(default_factory=list)
    summarize_chars_dropped_total: int = 0
    summarize_summary_chars_total: int = 0

    # IPython input size metrics
    ipython_input_chars_total: int = 0
    ipython_input_chars_mean: float = 0.0
    ipython_input_loc_total: int = 0
    ipython_input_loc_mean: float = 0.0

    # Internal counters for derived metrics
    _ipython_call_count: int = field(default=0, repr=False)

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

    stop_reason: str = ""  # "done", "max_turns", "token_budget", "multiple_tool_calls", "context_limit", "depth_limit"

    def record(self, event: BuiltinMetricEvent) -> None:
        if isinstance(event, IpythonExecuted):
            self._ipython_call_count += 1
            self.ipython_input_chars_total += event.input_chars
            self.ipython_input_loc_total += event.input_loc
        elif isinstance(event, SummarizeRejected):
            self.summarize_rejected_count += 1
        elif isinstance(event, SummarizeApplied):
            self.turns_between_summarizes.append(event.turns_since_last_summarize)
            self.summarize_turns_dropped_total += event.num_turns
            self.summarize_summary_lengths.append(event.summary_chars)
            self.summarize_chars_dropped_total += event.dropped_chars
            self.summarize_summary_chars_total += event.summary_chars
        else:
            raise TypeError(f"Unsupported builtin metric event: {type(event)!r}")

        self._refresh_derived_metrics()

    def _refresh_derived_metrics(self) -> None:
        self.ipython_input_chars_mean = (
            self.ipython_input_chars_total / self._ipython_call_count
            if self._ipython_call_count
            else 0.0
        )
        self.ipython_input_loc_mean = (
            self.ipython_input_loc_total / self._ipython_call_count
            if self._ipython_call_count
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        self._refresh_derived_metrics()
        return {
            key: value for key, value in asdict(self).items() if not key.startswith("_")
        }


@dataclass
class RLMResult:
    answer: str
    session_dir: Path | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
