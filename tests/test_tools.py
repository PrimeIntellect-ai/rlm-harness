"""Tests for tools: OpenAI-style tool calls dispatched by the engine.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a dummy
``add(a, b)`` tool (registered per-test, not part of the package) to
exercise how the harness reacts to model tool-call behavior:

- valid tool call   → dispatched, result fed back as a ``tool`` message
- invalid tool call → loop short-circuits, final answer reports the error
"""

from __future__ import annotations

import pytest
from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    show_tool_result,
    tool_result,
)

from rlm.engine import RLMEngine


async def test_valid_tool(session, register_add_tool):
    """Valid tool call: engine dispatches add() and feeds the result back."""
    prompt = "add 2 and 3"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("add", {"a": 2, "b": 3})]),
        DummyMessage(content="the sum is 5"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    show_tool_result(tool_result(client))
    assert tool_result(client) == "5"
    assert result.answer == "the sum is 5"
    assert result.turns == 2


async def test_invalid_tool_args(session, register_add_tool):
    """Malformed JSON in a tool call: rollout short-circuits with an error answer."""
    prompt = "add 2 and 3"
    messages = [DummyMessage(tool_calls=[DummyToolCall("add", "not-valid-json")])]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    # Harness short-circuits — no second round-trip to the model.
    assert len(client.calls) == 1
    assert "invalid JSON arguments" in result.answer
    assert result.turns == 1
    assert engine._metrics.stop_reason == "invalid_tool_args"


async def test_unknown_tool_terminates(session, register_add_tool):
    """Unknown tool name: rollout short-circuits with stop_reason=unknown_tool."""
    prompt = "do something"
    messages = [DummyMessage(tool_calls=[DummyToolCall("ipytron", {})])]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert len(client.calls) == 1
    assert "unknown tool 'ipytron'" in result.answer
    assert result.turns == 1
    assert engine._metrics.stop_reason == "unknown_tool"


async def test_unknown_tool_lenient_with_env(session, register_add_tool, monkeypatch):
    """RLM_ALLOW_UNKNOWN_TOOL=1: error is fed back and the loop continues."""
    monkeypatch.setenv("RLM_ALLOW_UNKNOWN_TOOL", "1")
    prompt = "do something"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipytron", {})]),
        DummyMessage(content="ok, giving up"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert len(client.calls) == 2
    assert "Error: unknown tool 'ipytron'" in tool_result(client)
    assert result.answer == "ok, giving up"
    assert result.turns == 2
    assert engine._metrics.stop_reason == "done"


async def test_tool_raises(session, register_boom_tool):
    """Tool raising from execute() propagates out of engine.run()."""
    prompt = "set off the boom tool"
    messages = [DummyMessage(tool_calls=[DummyToolCall("boom", {})])]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    with pytest.raises(RuntimeError, match="boom"):
        await engine.run(prompt)


def test_bash_tool_handles_non_utf8_output():
    """Bash command stdout containing invalid UTF-8 must not crash the tool.

    `printf '\\xf0'` produces a single 0xf0 byte — a 4-byte UTF-8 lead
    with no continuation bytes. Without ``errors="replace"`` on the
    subprocess decode, this raises ``UnicodeDecodeError`` and propagates
    up as an AgentError that ends the rollout.
    """
    from rlm.tools.base import ToolContext
    from rlm.tools.bash import BashTool
    from rlm.types import RLMMetrics, TokenUsage

    tool = BashTool()
    ctx = ToolContext(
        messages=[],
        metrics=RLMMetrics(),
        total_usage=TokenUsage(),
        last_prompt_tokens=0,
        exec_timeout=10,
    )
    outcome = tool.execute({"command": "printf '\\xf0'"}, ctx)
    assert "�" in outcome.content
