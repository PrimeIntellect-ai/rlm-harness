"""Tests for tools: OpenAI-style tool calls dispatched by the engine.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a dummy
``add(a, b)`` tool (registered per-test, not part of the package) to
exercise how the harness reacts to model tool-call behavior:

- valid tool call   → dispatched, result fed back as a ``tool`` message
- invalid tool call → loop short-circuits, final answer reports the error
"""

from __future__ import annotations

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result

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
