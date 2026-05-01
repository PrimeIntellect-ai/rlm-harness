"""Tests for tools: OpenAI-style tool calls dispatched by the engine.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a dummy
``add(a, b)`` tool (registered per-test, not part of the package) to
exercise how the harness reacts to model tool-call behavior.
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
    """Default (lenient): parse error is fed back and the loop continues."""
    prompt = "add 2 and 3"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("add", "not-valid-json")]),
        DummyMessage(content="ok, giving up"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert len(client.calls) == 2
    assert "Error: invalid JSON arguments for tool 'add'" in tool_result(client)
    assert result.answer == "ok, giving up"
    assert result.turns == 2


async def test_unknown_tool(session, register_add_tool):
    """Default (lenient): unknown-tool error is fed back and the loop continues."""
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


async def test_multiple_tool_calls(session, register_add_tool):
    """Default (lenient): each tool_call_id receives an error and the loop continues."""
    prompt = "add things"
    messages = [
        DummyMessage(
            tool_calls=[
                DummyToolCall("add", {"a": 1, "b": 2}, id="call_0"),
                DummyToolCall("add", {"a": 3, "b": 4}, id="call_1"),
            ]
        ),
        DummyMessage(content="ok, giving up"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert len(client.calls) == 2
    request_messages = client.calls[1]["messages"]
    tool_messages = [m for m in request_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert all(
        "only one tool call per turn allowed" in m["content"] for m in tool_messages
    )
    assert {m["tool_call_id"] for m in tool_messages} == {"call_0", "call_1"}
    assert result.answer == "ok, giving up"
    assert result.turns == 2


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
