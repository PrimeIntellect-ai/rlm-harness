"""Helpers for aggregating session-scoped metrics from session artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProgrammaticToolCallStats:
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


class SessionMetricsAggregator:
    """Aggregate session-scoped metrics from local logs and recursive children.

    All session-scoped aggregation (context-token stats, programmatic
    tool-call stats) goes through here, so ``Session`` stays a pure
    file/log container.
    """

    _SUM_KEYS = (
        "session_count",
        "input_tokens_total",
        "output_tokens_total",
        "final_input_tokens_total",
        "final_output_tokens_total",
        "branch_count",
        "branch_input_tokens_sum",
        "branch_output_tokens_sum",
    )
    _MAX_KEYS = ("branch_input_tokens_max", "branch_output_tokens_max")

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

    def direct_programmatic_tool_call_stats(self) -> ProgrammaticToolCallStats:
        return ProgrammaticToolCallStats.from_log(
            self.session_dir / "programmatic_tool_calls.jsonl"
        )

    def aggregate_child_metrics(self) -> ChildSessionAggregate:
        """Single pass over ``sub-*/meta.json``; collects both stat bundles."""
        aggregate = ChildSessionAggregate()

        for child_dir in self.session_dir.glob("sub-*"):
            meta_path = child_dir / "meta.json"
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except FileNotFoundError:
                continue

            ctx_stats = meta.get("context_token_stats")
            if not isinstance(ctx_stats, dict):
                raise RuntimeError(
                    f"Missing context_token_stats in child session meta: {meta_path}"
                )
            for key in self._SUM_KEYS:
                aggregate.context_token_stats[key] += int(ctx_stats.get(key, 0))
            for key in self._MAX_KEYS:
                aggregate.context_token_stats[key] = max(
                    aggregate.context_token_stats[key],
                    int(ctx_stats.get(key, 0)),
                )

            aggregate.tool_call_stats = aggregate.tool_call_stats.merge(
                ProgrammaticToolCallStats.from_meta(meta)
            )

        return aggregate
