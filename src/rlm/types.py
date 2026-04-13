"""Core data types."""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


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

    # Token metrics before/dropped per summarize
    summarize_prompt_tokens_before: list[int] = field(default_factory=list)
    summarize_completion_tokens_before: list[int] = field(default_factory=list)
    summarize_prompt_tokens_dropped: list[int] = field(default_factory=list)
    summarize_completion_tokens_dropped: list[int] = field(default_factory=list)

    # This agent's token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Environment/tool response tokens (estimated from prompt token deltas)
    # Computed as: prompt_tokens[N+1] - prompt_tokens[N] - completion_tokens[N]
    # Skipped on turns where summarize dropped context (delta unreliable)
    tool_result_tokens: int = 0

    # Aggregated from children
    sub_rlm_prompt_tokens: int = 0
    sub_rlm_completion_tokens: int = 0
    sub_rlm_count: int = 0

    # Budget tracking
    max_turns: int = 0
    max_tokens: int = 0
    turn_budget_warnings: int = 0
    token_budget_warnings: int = 0
    stop_reason: str = ""  # "done", "max_turns", "token_budget", "depth_limit"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RLMResult:
    answer: str
    session_dir: Path | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
