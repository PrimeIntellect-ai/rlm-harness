"""Tests for skills: skills invoked from inside the IPython tool.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a real
skill fixture on PATH; the engine starts its own IPython kernel to
dispatch the ``ipython`` tool call, so the code inside the kernel hits
the real skill the model would see:

- python form → ``await add(...)`` calls into the imported skill's ``run``
- bash form   → ``!add ...`` runs the CLI directly from PATH

Skill fixtures live under ``tests/fixtures/skills/<name>/``.
Kernel startup is ~700ms per test; keep the set here small.
"""

from __future__ import annotations

from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    show_tool_result,
    tool_result,
)

from rlm.engine import RLMEngine


async def test_valid_python_skill(session):
    """Python form: ``await add(...)`` in an ipython tool call hits the real skill CLI."""
    prompt = "compute 2 + 3"
    messages = [
        DummyMessage(
            tool_calls=[
                DummyToolCall("ipython", {"code": "print(await add(a=2, b=3))"})
            ]
        ),
        DummyMessage(content="the sum is 5"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    show_tool_result(tool_result(client))
    assert tool_result(client).strip() == "5"
    assert result.answer == "the sum is 5"


async def test_valid_bash_skill(session):
    """Bash form: ``!add ...`` in an ipython tool call runs the skill CLI via IPython's shell escape."""
    prompt = "compute 2 + 3"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "!add --a 2 --b 3"})]
        ),
        DummyMessage(content="the sum is 5"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    show_tool_result(tool_result(client))
    assert tool_result(client).strip() == "5"
    assert result.answer == "the sum is 5"


async def test_invalid_skill_args(session):
    """Skill called with a missing arg: the TypeError traceback is returned to the model."""
    prompt = "compute 2 + ?"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "print(await add(a=2))"})]
        ),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "TypeError" in output
    assert "missing 1 required positional argument: 'b'" in output
    assert result.answer == "the call failed"


async def test_skill_raises(session):
    """Skill raising inside the kernel: the traceback is returned to the model."""
    prompt = "set off the boom skill"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "await boom()"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "boom" in output
    assert result.answer == "the call failed"
