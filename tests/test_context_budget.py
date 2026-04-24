"""Tests for the context-token ceiling (``max_context_tokens``).

Drives ``RLMEngine`` with scripted responses whose ``usage.prompt_tokens``
is set explicitly, so we can verify:

- the engine passes ``max_completion_tokens`` derived from remaining budget
- an overshoot on a normal turn rolls back the last tool result and fires
  compaction, discarding the overshot response (hard path)
- soft compaction fires on a tool_call turn before execution: the tool
  is not run, a skip stub replaces the would-be tool result, and
  compaction runs on messages that fit under the ceiling by construction
- compaction-call overshoot terminates cleanly
  (``stop_reason = "context_budget_exceeded"``)
- with no ceiling set, none of the above happens (huge ``prompt_tokens``
  just pass through)
"""

from __future__ import annotations

import pytest
from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    DummyUsage,
)

from rlm.engine import (
    OVERSHOT_TOOL_RESULT_STUB,
    POST_COMPACTION_FRAMING,
    SOFT_COMPACT_SKIPPED_STUB,
    _BUDGET_MARGIN_TOKENS,
    RLMEngine,
)


@pytest.fixture
def no_tools(monkeypatch):
    """Disable all tools so the engine doesn't spin up an ipython kernel."""
    monkeypatch.setenv("RLM_TOOLS", "")


async def test_completion_budget_kwarg_on_each_call(session, register_add_tool):
    """``max_completion_tokens`` = max_context - last_prompt - margin on every call."""
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("add", {"a": 1, "b": 2})],
            usage=DummyUsage(prompt_tokens=200, completion_tokens=30),
        ),
        DummyMessage(
            content="done",
            usage=DummyUsage(prompt_tokens=500, completion_tokens=20),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        max_context_tokens=10_000,
    )

    await engine.run("go")

    # First call: _last_prompt_tokens starts at 0, so cap is 10000 - 0 - margin.
    assert (
        client.calls[0]["max_completion_tokens"] == 10_000 - 0 - _BUDGET_MARGIN_TOKENS
    )
    # Second call: after first response, _last_prompt_tokens is 200.
    assert (
        client.calls[1]["max_completion_tokens"] == 10_000 - 200 - _BUDGET_MARGIN_TOKENS
    )


async def test_no_ceiling_no_budget_kwarg(session, register_add_tool):
    """Without ``max_context_tokens`` set, no ``max_completion_tokens`` is sent."""
    messages = [
        DummyMessage(
            content="done", usage=DummyUsage(prompt_tokens=999_999, completion_tokens=5)
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    result = await engine.run("anything")

    assert "max_completion_tokens" not in client.calls[0]
    assert engine._metrics.overshoot_rollbacks == 0
    assert engine._metrics.stop_reason == "done"
    assert result.answer == "done"


async def test_overshoot_rolls_back_and_compacts(session, register_add_tool):
    """A tool result that overshoots → rollback to stub → compaction → loop continues."""
    messages = [
        # T1: model calls the tool. Prompt small, no overshoot yet.
        DummyMessage(
            tool_calls=[DummyToolCall("add", {"a": 2, "b": 3})],
            usage=DummyUsage(prompt_tokens=300, completion_tokens=50),
        ),
        # T2: after tool result lands, the next call's prompt_tokens overshoots.
        # This response will be discarded.
        DummyMessage(
            content="ignored",
            usage=DummyUsage(prompt_tokens=15_000, completion_tokens=20),
        ),
        # Compaction call: post-rollback context is small again.
        DummyMessage(
            content="handoff summary",
            usage=DummyUsage(prompt_tokens=400, completion_tokens=30),
        ),
        # T3: fresh context after compaction. Model finishes.
        DummyMessage(
            content="final answer",
            usage=DummyUsage(prompt_tokens=600, completion_tokens=10),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        max_context_tokens=10_000,
    )

    result = await engine.run("please do")

    assert engine._metrics.overshoot_rollbacks == 1
    assert engine._metrics.compactions_count == 1
    assert engine._metrics.stop_reason == "done"
    assert result.answer == "final answer"

    # The compaction call saw the tool_result replaced by the stub.
    compaction_msgs = client.calls[2]["messages"]
    tool_msgs = [m for m in compaction_msgs if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["content"] == OVERSHOT_TOOL_RESULT_STUB

    # Post-compaction, messages are rebuilt as [system, user(framing+summary)].
    post_msgs = client.calls[3]["messages"]
    assert post_msgs[0]["role"] == "system"
    assert post_msgs[1]["role"] == "user"
    assert POST_COMPACTION_FRAMING in post_msgs[1]["content"]
    assert "handoff summary" in post_msgs[1]["content"]
    # No dangling tool messages referencing the discarded assistant turn.
    assert "ignored" not in post_msgs[1]["content"]


async def test_overshoot_with_nothing_to_roll_back_is_terminal(session, no_tools):
    """Initial user prompt alone overshoots → rollout fails cleanly."""
    messages = [
        DummyMessage(
            content="junk",
            usage=DummyUsage(prompt_tokens=20_000, completion_tokens=5),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        max_context_tokens=10_000,
    )

    result = await engine.run("enormous seed task")

    assert engine._metrics.stop_reason == "context_budget_exceeded"
    assert engine._metrics.overshoot_rollbacks == 0
    assert "no tool result to roll back" in result.answer


async def test_soft_compaction_skips_tool_and_compacts(session, register_add_tool):
    """Soft threshold crossed + tool_call present → tool is NOT run; stub + compact.

    The model's assistant message (with its tool_call) is preserved, but
    instead of executing the tool we append SOFT_COMPACT_SKIPPED_STUB as
    the tool result. Compaction runs on those messages (prompt ≈
    usage.prompt_tokens + small), guaranteed to fit under the ceiling.
    The loop continues on the compacted state.
    """
    messages = [
        # T1: small call, tool executes normally (add returns "5").
        DummyMessage(
            tool_calls=[DummyToolCall("add", {"a": 2, "b": 3}, id="call_a")],
            usage=DummyUsage(prompt_tokens=200, completion_tokens=30),
        ),
        # T2: prompt_tokens crosses summarize_at (4000) but NOT
        # max_context (10000). Response has a tool_call. Soft path
        # fires BEFORE running the tool: stub is appended, compaction
        # fires.
        DummyMessage(
            tool_calls=[DummyToolCall("add", {"a": 4, "b": 5}, id="call_b")],
            usage=DummyUsage(prompt_tokens=4_500, completion_tokens=30),
        ),
        # Compaction call: fits by construction (no fat tool_result in
        # messages because the tool was never run).
        DummyMessage(
            content="handoff summary",
            usage=DummyUsage(prompt_tokens=800, completion_tokens=30),
        ),
        # T3: post-compaction, model finishes.
        DummyMessage(
            content="final answer",
            usage=DummyUsage(prompt_tokens=400, completion_tokens=10),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        summarize_at_tokens=4_000,
        max_context_tokens=10_000,
    )

    result = await engine.run("do the thing")

    assert engine._metrics.compactions_count == 1
    assert engine._metrics.overshoot_rollbacks == 0  # soft path doesn't roll back
    assert engine._metrics.stop_reason == "done"
    assert result.answer == "final answer"

    # Compaction call (index 2) should have seen the skip stub as T2's tool result.
    compaction_msgs = client.calls[2]["messages"]
    tool_msgs = [m for m in compaction_msgs if m.get("role") == "tool"]
    assert any(m["content"] == SOFT_COMPACT_SKIPPED_STUB for m in tool_msgs)
    # T1's real tool result ("5") is still there at this point — only
    # T2's would-be result is the stub.
    assert any(m["content"] == "5" for m in tool_msgs)
    # The real `add` tool got called only once (for T1), not twice.
    assert engine._metrics._ipython_call_count == 0  # no ipython — we only ran add


async def test_soft_compaction_skips_on_done_response_is_unreachable(
    session, register_add_tool
):
    """Done responses (no tool_call) can't trigger soft compaction.

    Even if prompt_tokens >= summarize_at_tokens, a response without
    tool_calls ends the rollout with ``stop_reason = "done"``; soft
    compaction's placement after the done-check means it's never
    reached. Protects final answers from being discarded.
    """
    messages = [
        DummyMessage(
            content="final answer",
            usage=DummyUsage(prompt_tokens=5_000, completion_tokens=10),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        summarize_at_tokens=4_000,
        max_context_tokens=10_000,
    )

    result = await engine.run("easy")

    assert engine._metrics.stop_reason == "done"
    assert engine._metrics.compactions_count == 0
    assert result.answer == "final answer"


async def test_compaction_call_overshoot_is_terminal(session, register_add_tool):
    """If compaction itself overshoots, rollout fails with context_budget_exceeded."""
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("add", {"a": 1, "b": 1})],
            usage=DummyUsage(prompt_tokens=300, completion_tokens=20),
        ),
        # T2 overshoots → triggers rollback + compaction.
        DummyMessage(
            content="discarded",
            usage=DummyUsage(prompt_tokens=15_000, completion_tokens=10),
        ),
        # Compaction call ALSO overshoots (e.g., system prompt alone is big).
        DummyMessage(
            content="never used",
            usage=DummyUsage(prompt_tokens=11_000, completion_tokens=5),
        ),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        max_context_tokens=10_000,
    )

    result = await engine.run("go")

    assert engine._metrics.overshoot_rollbacks == 1
    assert engine._metrics.compactions_count == 0
    assert engine._metrics.stop_reason == "context_budget_exceeded"
    assert "compaction call exceeded max_context_tokens" in result.answer


def test_validation_summarize_must_be_below_ceiling():
    """``summarize_at_tokens`` >= ``max_context_tokens`` is rejected at init.

    A dummy client is passed so ``make_client()`` isn't invoked — otherwise
    CI without ``OPENAI_API_KEY`` set would fail at the ``AsyncOpenAI()``
    construction, long after the validation we're actually testing.
    """
    dummy = DummyClient([])
    with pytest.raises(ValueError, match="summarize_at_tokens"):
        RLMEngine(summarize_at_tokens=5_000, max_context_tokens=5_000, client=dummy)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="summarize_at_tokens"):
        RLMEngine(summarize_at_tokens=6_000, max_context_tokens=5_000, client=dummy)  # type: ignore[arg-type]
    # Valid combo does not raise.
    RLMEngine(summarize_at_tokens=4_000, max_context_tokens=5_000, client=dummy)  # type: ignore[arg-type]
