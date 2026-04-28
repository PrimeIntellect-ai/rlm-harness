"""Tests for RLMMetrics token-tracking, especially total_tool_response_tokens."""

from rlm.types import CompactionApplied, RLMMetrics


def test_total_tool_response_tokens_single_turn_no_tools():
    """One-turn rollout, model answered without a tool call: zero tool tokens."""
    m = RLMMetrics()
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    m.finalize_current_branch()
    assert m.total_tool_response_tokens == 0


def test_total_tool_response_tokens_multi_turn_with_tools():
    """Multi-turn branch: tool tokens = sum of per-turn deltas where role == tool.

    The engine signals each appendage's role. delta = curr.prompt - prev.prompt
    - prev.completion. Only "tool" appendages count toward the total.
    """
    m = RLMMetrics()
    # Turn 0: prompt = 100 (system + user), completion = 50. No prior turn.
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    # Turn 1: prompt = 100 + 50 (asst_0) + 30 (tool_0) = 180. completion = 40.
    # delta = 180 - 100 - 50 = 30 → tool_response_tokens += 30.
    m.note_assistant_turn(
        prompt_tokens=180, completion_tokens=40, prev_appended_role="tool"
    )
    # Turn 2: prompt = 180 + 40 (asst_1) + 70 (tool_1) = 290. completion = 20.
    # delta = 290 - 180 - 40 = 70 → tool_response_tokens += 70.
    m.note_assistant_turn(
        prompt_tokens=290, completion_tokens=20, prev_appended_role="tool"
    )
    m.finalize_current_branch()
    assert m.total_tool_response_tokens == 100


def test_total_tool_response_tokens_user_message_not_counted():
    """Non-tool appendages (e.g. mid-branch user message) are excluded."""
    m = RLMMetrics()
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    # Imagine a future env injects a user message here. delta = 30 but not counted.
    m.note_assistant_turn(
        prompt_tokens=180, completion_tokens=40, prev_appended_role="user"
    )
    # Now a tool result follows: delta = 70, counted.
    m.note_assistant_turn(
        prompt_tokens=290, completion_tokens=20, prev_appended_role="tool"
    )
    m.finalize_current_branch()
    assert m.total_tool_response_tokens == 70


def test_total_tool_response_tokens_across_compactions():
    """prev-turn carry resets at branch boundaries; tool tokens still summed correctly."""
    m = RLMMetrics()
    # Branch 1
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    m.note_assistant_turn(
        prompt_tokens=180, completion_tokens=40, prev_appended_role="tool"
    )  # +30
    m.record(
        CompactionApplied(
            num_turns_dropped=2,
            dropped_chars=0,
            summary_chars=0,
            turns_since_last_compaction=2,
        )
    )
    # Branch 2 — fresh context. First turn has no prior turn (None).
    m.note_assistant_turn(prompt_tokens=80, completion_tokens=30)
    m.note_assistant_turn(
        prompt_tokens=130, completion_tokens=20, prev_appended_role="tool"
    )  # +20
    m.note_assistant_turn(
        prompt_tokens=200, completion_tokens=10, prev_appended_role="tool"
    )  # +50
    m.finalize_current_branch()
    # Branch 1: 30 + Branch 2: 20 + 50 = 100.
    assert m.total_tool_response_tokens == 100


def test_total_tool_response_tokens_includes_sub_rlm_aggregate():
    """total_tool_response_tokens = parent's own + descendants', via context_token_stats."""
    m = RLMMetrics()
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    m.note_assistant_turn(
        prompt_tokens=200, completion_tokens=30, prev_appended_role="tool"
    )  # +50
    m.finalize_current_branch()

    m.apply_child_aggregates(
        {
            "session_count": 2,
            "input_tokens_total": 1000,
            "output_tokens_total": 200,
            "final_input_tokens_total": 500,
            "final_output_tokens_total": 100,
            "branch_count": 2,
            "branch_input_tokens_sum": 800,
            "branch_input_tokens_max": 600,
            "branch_output_tokens_sum": 200,
            "branch_output_tokens_max": 150,
            "tool_response_tokens_total": 300,
        }
    )

    # 50 (parent) + 300 (descendants) = 350.
    assert m.total_tool_response_tokens == 350


def test_context_token_stats_bubbles_subtree_total():
    """Child's context_token_stats() reports parent + own sub-RLM tool tokens."""
    m = RLMMetrics()
    m.note_assistant_turn(prompt_tokens=100, completion_tokens=50)
    m.note_assistant_turn(
        prompt_tokens=180, completion_tokens=40, prev_appended_role="tool"
    )  # +30
    m.finalize_current_branch()
    m.apply_child_aggregates(
        {
            "session_count": 1,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "final_input_tokens_total": 0,
            "final_output_tokens_total": 0,
            "branch_count": 0,
            "branch_input_tokens_sum": 0,
            "branch_input_tokens_max": 0,
            "branch_output_tokens_sum": 0,
            "branch_output_tokens_max": 0,
            "tool_response_tokens_total": 70,
        }
    )

    stats = m.context_token_stats()
    assert stats["tool_response_tokens_total"] == 100


def test_total_tool_response_tokens_handles_empty_finalize():
    """Finalize before any turn: no contribution, no crash."""
    m = RLMMetrics()
    m.finalize_current_branch()
    assert m.total_tool_response_tokens == 0
