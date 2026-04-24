"""Tests for skills: skills invoked from inside the IPython tool.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a real
skill fixture on PATH; the engine starts its own IPython kernel to
dispatch the ``ipython`` tool call, so the code inside the kernel hits
the real skill the model would see:

- python form → ``await <skill>(...)`` calls into the imported skill's ``run``
- bash form   → ``!<skill> ...`` runs the CLI directly from PATH

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


async def test_python_skill_valid(session):
    """Python form: ``await say(...)`` in an ipython tool call hits the real skill."""
    prompt = "say hello"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "await say(s='hello')"})]
        ),
        DummyMessage(content="ok"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "hello" in output
    assert result.answer == "ok"


async def test_bash_skill_valid(session):
    """Bash form: ``!say ...`` in an ipython tool call runs the skill CLI via IPython's shell escape."""
    prompt = "say hello"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!say --s hello"})]),
        DummyMessage(content="ok"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "hello" in output
    assert result.answer == "ok"


async def test_python_skill_invalid_args(session):
    """Python form: missing required arg → ``TypeError`` traceback is returned to the model."""
    prompt = "say something"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "await say()"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "TypeError" in output
    assert "missing 1 required positional argument: 's'" in output
    assert result.answer == "the call failed"


async def test_bash_skill_invalid_args(session):
    """Bash form: missing required arg → argparse prints a usage error on stderr."""
    prompt = "say something"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!say"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "the following arguments are required: --s" in output
    assert result.answer == "the call failed"


async def test_python_skill_raises(session):
    """Python form: ``await boom()`` raising inside the kernel surfaces the traceback."""
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


async def test_bash_skill_raises(session):
    """Bash form: ``!boom`` raising in the CLI surfaces the traceback on stderr."""
    prompt = "set off the boom skill"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!boom"})]),
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


async def test_python_skill_halt_on_raise(session):
    """Python form: a raise halts the cell — ``say`` after ``await boom()`` is not executed."""
    # Source shows ``'R' + 'A' + 'N'``; say would print ``RAN`` only if it ran.
    code = "await boom(); await say(s='R' + 'A' + 'N')"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    await engine.run("try boom then say")

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "RAN" not in output


async def test_bash_skill_halt_on_raise(session):
    """Bash form: ``&&`` short-circuits — ``say`` is not executed after ``boom`` fails."""
    code = "!boom && say --s RAN"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    await engine.run("try boom then say")

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "RAN" not in output
