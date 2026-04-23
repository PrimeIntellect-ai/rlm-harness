"""Tests for skills: skills invoked from inside the IPython tool.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a real
skill fixture on PATH; the engine starts its own IPython kernel to
dispatch the ``ipython`` tool call, so the code inside the kernel hits
the real ``kernel_shim → subprocess`` path the model takes:

- python form → ``await add(...)`` dispatches through the kernel shim
- bash form   → ``!add ...`` runs the CLI directly from PATH

Skill fixtures live under ``tests/fixtures/skills/<name>/``.
Kernel startup is ~700ms per test; keep the set here small.
"""

from __future__ import annotations

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result

from rlm.engine import RLMEngine


async def test_valid_python_skill(session, register_add_skill):
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

    assert tool_result(client).strip() == "5"
    assert result.answer == "the sum is 5"


async def test_valid_bash_skill(session, register_add_skill):
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

    assert tool_result(client).strip() == "5"
    assert result.answer == "the sum is 5"
