"""Core data types."""

from __future__ import annotations

from collections import deque
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
    turns_since_last_summarize: int = 0
    turns_between_summarizes_mean: float = 0.0

    # Summarize metrics
    summarize_rejected_count: int = 0
    summarize_turns_dropped_mean: float = 0.0
    summarize_chars_dropped_mean: float = 0.0
    summarize_summary_chars_mean: float = 0.0

    # IPython input size metrics
    ipython_input_chars_mean: float = 0.0
    ipython_input_loc_mean: float = 0.0

    # Internal counters for derived metrics
    _turns: int = field(default=0, repr=False)
    _ipython_call_count: int = field(default=0, repr=False)
    _ipython_input_chars_total: int = field(default=0, repr=False)
    _ipython_input_loc_total: int = field(default=0, repr=False)
    _summarize_applied_count: int = field(default=0, repr=False)
    _turns_between_summarizes_total: int = field(default=0, repr=False)
    _summarize_turns_dropped_total: int = field(default=0, repr=False)
    _summarize_chars_dropped_total: int = field(default=0, repr=False)
    _summarize_summary_chars_total: int = field(default=0, repr=False)

    # Public work/cost token metrics
    sub_rlm_input_tokens: int = 0
    sub_rlm_output_tokens: int = 0

    # Root terminal branch context
    final_input_tokens: int = 0
    final_output_tokens: int = 0

    # Root branch context exposure
    branch_input_tokens_mean: float = 0.0
    branch_input_tokens_max: int = 0
    branch_output_tokens_mean: float = 0.0
    branch_output_tokens_max: int = 0

    # Skill-CLI invocations from inside the ipython REPL
    programmatic_tool_calls_python: int = 0
    programmatic_tool_calls_bash: int = 0
    sub_rlm_programmatic_tool_calls_python: int = 0
    sub_rlm_programmatic_tool_calls_bash: int = 0

    stop_reason: str = ""  # "done", "max_turns", "token_budget", "multiple_tool_calls", "context_limit", "depth_limit"

    @property
    def turns(self) -> int:
        return self._turns

    @turns.setter
    def turns(self, value: int) -> None:
        self._turns = value

    # Internal context-token tracker state
    _retained_completion_tokens: deque[int] = field(default_factory=deque, repr=False)
    _retained_completion_tokens_total: int = field(default=0, repr=False)
    _current_branch_input_tokens: int | None = field(default=None, repr=False)
    _current_branch_output_tokens: int | None = field(default=None, repr=False)
    _branch_count: int = field(default=0, repr=False)
    _branch_input_tokens_sum: int = field(default=0, repr=False)
    _branch_input_tokens_max: int = field(default=0, repr=False)
    _branch_output_tokens_sum: int = field(default=0, repr=False)
    _branch_output_tokens_max: int = field(default=0, repr=False)
    _root_input_tokens: int = field(default=0, repr=False)
    _root_output_tokens: int = field(default=0, repr=False)
    _sub_rlm_count: int = field(default=0, repr=False)
    _sub_rlm_final_input_tokens: int = field(default=0, repr=False)
    _sub_rlm_final_output_tokens: int = field(default=0, repr=False)
    _sub_rlm_branch_count: int = field(default=0, repr=False)
    _sub_rlm_branch_input_tokens_sum: int = field(default=0, repr=False)
    _sub_rlm_branch_input_tokens_max: int = field(default=0, repr=False)
    _sub_rlm_branch_output_tokens_sum: int = field(default=0, repr=False)
    _sub_rlm_branch_output_tokens_max: int = field(default=0, repr=False)

    def note_root_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._root_input_tokens = prompt_tokens
        self._root_output_tokens = completion_tokens
        self._refresh_derived_metrics()

    def note_assistant_turn(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._retained_completion_tokens.append(completion_tokens)
        self._retained_completion_tokens_total += completion_tokens

        turn_total_tokens = prompt_tokens + completion_tokens
        visible_output_tokens = self._retained_completion_tokens_total
        visible_input_tokens = max(0, turn_total_tokens - visible_output_tokens)

        self._current_branch_input_tokens = visible_input_tokens
        self._current_branch_output_tokens = visible_output_tokens
        self._refresh_derived_metrics()

    def finalize_current_branch(self) -> None:
        if (
            self._current_branch_input_tokens is None
            or self._current_branch_output_tokens is None
        ):
            return

        self._branch_count += 1
        self._branch_input_tokens_sum += self._current_branch_input_tokens
        self._branch_output_tokens_sum += self._current_branch_output_tokens
        self._branch_input_tokens_max = max(
            self._branch_input_tokens_max,
            self._current_branch_input_tokens,
        )
        self._branch_output_tokens_max = max(
            self._branch_output_tokens_max,
            self._current_branch_output_tokens,
        )

        self.final_input_tokens = self._current_branch_input_tokens
        self.final_output_tokens = self._current_branch_output_tokens
        self._current_branch_input_tokens = None
        self._current_branch_output_tokens = None
        self._refresh_derived_metrics()

    def apply_child_aggregates(self, stats: dict[str, int]) -> None:
        self._sub_rlm_count = stats.get("session_count", 0)
        self.sub_rlm_input_tokens = stats.get("input_tokens_total", 0)
        self.sub_rlm_output_tokens = stats.get("output_tokens_total", 0)
        self._sub_rlm_final_input_tokens = stats.get("final_input_tokens_total", 0)
        self._sub_rlm_final_output_tokens = stats.get("final_output_tokens_total", 0)
        self._sub_rlm_branch_count = stats.get("branch_count", 0)
        self._sub_rlm_branch_input_tokens_sum = stats.get("branch_input_tokens_sum", 0)
        self._sub_rlm_branch_input_tokens_max = stats.get("branch_input_tokens_max", 0)
        self._sub_rlm_branch_output_tokens_sum = stats.get(
            "branch_output_tokens_sum", 0
        )
        self._sub_rlm_branch_output_tokens_max = stats.get(
            "branch_output_tokens_max", 0
        )
        self._refresh_derived_metrics()

    def context_token_stats(self) -> dict[str, int]:
        self._refresh_derived_metrics()
        return {
            "session_count": 1 + self._sub_rlm_count,
            "input_tokens_total": self._root_input_tokens + self.sub_rlm_input_tokens,
            "output_tokens_total": self._root_output_tokens
            + self.sub_rlm_output_tokens,
            "final_input_tokens_total": self.final_input_tokens
            + self._sub_rlm_final_input_tokens,
            "final_output_tokens_total": self.final_output_tokens
            + self._sub_rlm_final_output_tokens,
            "branch_count": self._branch_count + self._sub_rlm_branch_count,
            "branch_input_tokens_sum": self._branch_input_tokens_sum
            + self._sub_rlm_branch_input_tokens_sum,
            "branch_input_tokens_max": max(
                self._branch_input_tokens_max,
                self._sub_rlm_branch_input_tokens_max,
            ),
            "branch_output_tokens_sum": self._branch_output_tokens_sum
            + self._sub_rlm_branch_output_tokens_sum,
            "branch_output_tokens_max": max(
                self._branch_output_tokens_max,
                self._sub_rlm_branch_output_tokens_max,
            ),
        }

    def record(self, event: BuiltinMetricEvent) -> None:
        if isinstance(event, IpythonExecuted):
            self._ipython_call_count += 1
            self._ipython_input_chars_total += event.input_chars
            self._ipython_input_loc_total += event.input_loc
        elif isinstance(event, SummarizeRejected):
            self.summarize_rejected_count += 1
        elif isinstance(event, SummarizeApplied):
            self._summarize_applied_count += 1
            self._turns_between_summarizes_total += event.turns_since_last_summarize
            self._summarize_turns_dropped_total += event.num_turns
            self._summarize_chars_dropped_total += event.dropped_chars
            self._summarize_summary_chars_total += event.summary_chars
            self.finalize_current_branch()
            for _ in range(min(event.num_turns, len(self._retained_completion_tokens))):
                self._retained_completion_tokens_total -= (
                    self._retained_completion_tokens.popleft()
                )
        else:
            raise TypeError(f"Unsupported builtin metric event: {type(event)!r}")

        self._refresh_derived_metrics()

    def _refresh_derived_metrics(self) -> None:
        if self._ipython_call_count:
            self.ipython_input_chars_mean = (
                self._ipython_input_chars_total / self._ipython_call_count
            )
            self.ipython_input_loc_mean = (
                self._ipython_input_loc_total / self._ipython_call_count
            )
        if self._summarize_applied_count:
            self.turns_between_summarizes_mean = (
                self._turns_between_summarizes_total / self._summarize_applied_count
            )
            self.summarize_turns_dropped_mean = (
                self._summarize_turns_dropped_total / self._summarize_applied_count
            )
            self.summarize_chars_dropped_mean = (
                self._summarize_chars_dropped_total / self._summarize_applied_count
            )
            self.summarize_summary_chars_mean = (
                self._summarize_summary_chars_total / self._summarize_applied_count
            )

        if self._branch_count:
            self.branch_input_tokens_mean = (
                self._branch_input_tokens_sum / self._branch_count
            )
            self.branch_input_tokens_max = self._branch_input_tokens_max
            self.branch_output_tokens_mean = (
                self._branch_output_tokens_sum / self._branch_count
            )
            self.branch_output_tokens_max = self._branch_output_tokens_max

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
