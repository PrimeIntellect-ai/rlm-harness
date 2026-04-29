"""Core data types."""

from __future__ import annotations

import json
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
class CompactionApplied:
    """Emitted when the engine auto-compacts context after a summary turn."""

    num_turns_dropped: int
    dropped_chars: int
    summary_chars: int
    turns_since_last_compaction: int


BuiltinMetricEvent = IpythonExecuted | CompactionApplied


@dataclass
class ProgrammaticToolCallStats:
    """Programmatic tool-call counts (skill CLIs invoked from the ipython REPL)."""

    python_total: int = 0
    bash_total: int = 0
    by_tool_python: dict[str, int] = field(default_factory=dict)
    by_tool_bash: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_log(cls, log_path: Path) -> ProgrammaticToolCallStats:
        """Count programmatic tool calls from a session-local JSONL log.

        Untrusted input (written by child processes that may exit mid-line),
        so malformed entries are skipped.
        """
        stats = cls()
        try:
            f = open(log_path)
        except FileNotFoundError:
            return stats
        with f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tool = entry.get("tool")
                source = entry.get("source")
                if not isinstance(tool, str) or source not in {"python", "bash"}:
                    continue

                if source == "python":
                    stats.python_total += 1
                    stats.by_tool_python[tool] = stats.by_tool_python.get(tool, 0) + 1
                else:
                    stats.bash_total += 1
                    stats.by_tool_bash[tool] = stats.by_tool_bash.get(tool, 0) + 1

        return stats

    @classmethod
    def from_meta(cls, meta: dict) -> ProgrammaticToolCallStats:
        """Load tool-call stats previously persisted via ``to_dict``."""
        raw = meta.get("programmatic_tool_call_stats", {})
        return cls(
            python_total=int(raw.get("python_total", 0)),
            bash_total=int(raw.get("bash_total", 0)),
            by_tool_python=dict(raw.get("by_tool_python", {})),
            by_tool_bash=dict(raw.get("by_tool_bash", {})),
        )

    def merge(self, other: ProgrammaticToolCallStats) -> ProgrammaticToolCallStats:
        """Return a merged copy of this stats object and *other*."""
        merged = ProgrammaticToolCallStats(
            python_total=self.python_total + other.python_total,
            bash_total=self.bash_total + other.bash_total,
            by_tool_python=self.by_tool_python.copy(),
            by_tool_bash=self.by_tool_bash.copy(),
        )
        for tool_name, count in other.by_tool_python.items():
            merged.by_tool_python[tool_name] = (
                merged.by_tool_python.get(tool_name, 0) + count
            )
        for tool_name, count in other.by_tool_bash.items():
            merged.by_tool_bash[tool_name] = (
                merged.by_tool_bash.get(tool_name, 0) + count
            )
        return merged

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_context_token_stats() -> dict[str, int]:
    return {
        "session_count": 0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "final_input_tokens_total": 0,
        "final_output_tokens_total": 0,
        "branch_count": 0,
        "branch_input_tokens_sum": 0,
        "branch_input_tokens_max": 0,
        "branch_output_tokens_sum": 0,
        "branch_output_tokens_max": 0,
        "tool_response_tokens_total": 0,
    }


@dataclass
class ChildSessionAggregate:
    """Aggregated stats across all recursive descendants of a session."""

    context_token_stats: dict[str, int] = field(
        default_factory=_empty_context_token_stats
    )
    tool_call_stats: ProgrammaticToolCallStats = field(
        default_factory=ProgrammaticToolCallStats
    )

    _CONTEXT_TOKEN_SUM_KEYS = (
        "session_count",
        "input_tokens_total",
        "output_tokens_total",
        "final_input_tokens_total",
        "final_output_tokens_total",
        "branch_count",
        "branch_input_tokens_sum",
        "branch_output_tokens_sum",
        "tool_response_tokens_total",
    )
    _CONTEXT_TOKEN_MAX_KEYS = ("branch_input_tokens_max", "branch_output_tokens_max")

    def absorb(
        self,
        ctx_stats: dict[str, int],
        tool_stats: ProgrammaticToolCallStats,
    ) -> None:
        """Combine a child session's stats into this aggregate."""
        for key in self._CONTEXT_TOKEN_SUM_KEYS:
            self.context_token_stats[key] += int(ctx_stats.get(key, 0))
        for key in self._CONTEXT_TOKEN_MAX_KEYS:
            self.context_token_stats[key] = max(
                self.context_token_stats[key],
                int(ctx_stats.get(key, 0)),
            )
        self.tool_call_stats = self.tool_call_stats.merge(tool_stats)


@dataclass
class RLMMetrics:
    """Metrics tracked during an rlm session."""

    # Compaction metrics (auto-summarization at a token threshold)
    compactions_count: int = 0
    has_compacted: int = 0  # 1 if compactions_count > 0, else 0; aggregates as the per-batch fraction of rollouts that compacted
    turns_since_last_compaction: int = 0
    turns_between_compactions_mean: float = 0.0
    compaction_turns_dropped_mean: float = 0.0
    compaction_chars_dropped_mean: float = 0.0
    compaction_summary_chars_mean: float = 0.0

    # IPython input size metrics
    ipython_input_chars_mean: float = 0.0
    ipython_input_loc_mean: float = 0.0

    # Internal counters for derived metrics
    _turns: int = field(default=0, repr=False)
    _ipython_call_count: int = field(default=0, repr=False)
    _ipython_input_chars_total: int = field(default=0, repr=False)
    _ipython_input_loc_total: int = field(default=0, repr=False)
    _turns_between_compactions_total: int = field(default=0, repr=False)
    _compaction_turns_dropped_total: int = field(default=0, repr=False)
    _compaction_chars_dropped_total: int = field(default=0, repr=False)
    _compaction_summary_chars_total: int = field(default=0, repr=False)

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

    # Tool-response tokens: sum of bytes the agent ingested via tool results
    # over the rollout, parent + all descendants, with no double-counting
    # across turns, branches, or sub-RLMs. Per branch, the parent's own
    # contribution is (visible_input_at_last_turn -
    # visible_input_at_first_turn), which strips the system + initial-user
    # baseline; sub-RLM contributions bubble up via context_token_stats.
    total_tool_response_tokens: int = 0

    # Skill-CLI invocations from inside the ipython REPL
    programmatic_tool_calls_python: int = 0
    programmatic_tool_calls_bash: int = 0
    sub_rlm_programmatic_tool_calls_python: int = 0
    sub_rlm_programmatic_tool_calls_bash: int = 0

    stop_reason: str = ""  # "done", "max_turns", "token_budget", "multiple_tool_calls", "invalid_tool_args", "depth_limit", "request_too_large"

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
    # Carry the previous main-loop turn's prompt/completion sizes so the next
    # turn can derive what was appended in between via prompt-token delta.
    # Reset to 0 at branch boundaries (compaction / initial branch).
    _prev_turn_prompt_tokens: int = field(default=0, repr=False)
    _prev_turn_completion_tokens: int = field(default=0, repr=False)
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
    _tool_response_tokens: int = field(default=0, repr=False)
    _sub_rlm_tool_response_tokens: int = field(default=0, repr=False)
    _sub_rlm_enabled: bool = field(default=False, repr=False)

    def note_root_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._root_input_tokens = prompt_tokens
        self._root_output_tokens = completion_tokens
        self._refresh_derived_metrics()

    def note_assistant_turn(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        prev_appended_role: str | None = None,
    ) -> None:
        """Record one main-loop turn's API usage.

        ``prev_appended_role`` names the role of whatever the engine
        appended to ``messages`` between the previous and current API
        call (typically ``"tool"``). When it equals ``"tool"``, the
        prompt-token growth — minus the previous turn's completion
        (which is the just-appended assistant message) — is attributed
        to ``_tool_response_tokens``. Pass ``None`` on the first turn
        of any branch (no preceding turn) to skip the attribution; this
        naturally excludes the system + initial-user baseline.
        """
        if prev_appended_role == "tool":
            delta = (
                prompt_tokens
                - self._prev_turn_prompt_tokens
                - self._prev_turn_completion_tokens
            )
            self._tool_response_tokens += max(0, delta)

        self._prev_turn_prompt_tokens = prompt_tokens
        self._prev_turn_completion_tokens = completion_tokens

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

        # Branch boundary: drop prev-turn carry so the next turn's
        # prompt-token delta isn't attributed across a fresh-context jump.
        self._prev_turn_prompt_tokens = 0
        self._prev_turn_completion_tokens = 0

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
        self._sub_rlm_tool_response_tokens = stats.get("tool_response_tokens_total", 0)
        self._refresh_derived_metrics()

    def apply_programmatic_tool_call_stats(
        self,
        direct: ProgrammaticToolCallStats,
        child: ProgrammaticToolCallStats,
    ) -> None:
        self.programmatic_tool_calls_python = direct.python_total
        self.programmatic_tool_calls_bash = direct.bash_total
        self.sub_rlm_programmatic_tool_calls_python = child.python_total
        self.sub_rlm_programmatic_tool_calls_bash = child.bash_total

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
            "tool_response_tokens_total": (
                self._tool_response_tokens + self._sub_rlm_tool_response_tokens
            ),
        }

    def record(self, event: BuiltinMetricEvent) -> None:
        if isinstance(event, IpythonExecuted):
            self._ipython_call_count += 1
            self._ipython_input_chars_total += event.input_chars
            self._ipython_input_loc_total += event.input_loc
        elif isinstance(event, CompactionApplied):
            self.compactions_count += 1
            self._turns_between_compactions_total += event.turns_since_last_compaction
            self._compaction_turns_dropped_total += event.num_turns_dropped
            self._compaction_chars_dropped_total += event.dropped_chars
            self._compaction_summary_chars_total += event.summary_chars
            # End the old branch for token accounting, then reset the retained
            # completion-token window so the next branch starts clean.
            self.finalize_current_branch()
            self._retained_completion_tokens.clear()
            self._retained_completion_tokens_total = 0
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
        self.has_compacted = 1 if self.compactions_count > 0 else 0
        if self.compactions_count:
            self.turns_between_compactions_mean = (
                self._turns_between_compactions_total / self.compactions_count
            )
            self.compaction_turns_dropped_mean = (
                self._compaction_turns_dropped_total / self.compactions_count
            )
            self.compaction_chars_dropped_mean = (
                self._compaction_chars_dropped_total / self.compactions_count
            )
            self.compaction_summary_chars_mean = (
                self._compaction_summary_chars_total / self.compactions_count
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

        self.total_tool_response_tokens = (
            self._tool_response_tokens + self._sub_rlm_tool_response_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        self._refresh_derived_metrics()
        sub_rlm_enabled = self._sub_rlm_enabled
        return {
            key: value
            for key, value in asdict(self).items()
            if not key.startswith("_")
            and (sub_rlm_enabled or not key.startswith("sub_rlm_"))
        }


@dataclass
class RLMResult:
    answer: str
    session_dir: Path | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
