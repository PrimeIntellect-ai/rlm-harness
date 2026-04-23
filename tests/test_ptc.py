"""Tests for programmatic tool calls: skills invoked from inside the IPython tool.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a real
skill fixture on PATH; the engine starts its own IPython kernel to
dispatch the ``ipython`` tool call, so the code inside the kernel hits
the real ``kernel_shim → subprocess`` path the model takes:

- python form → ``await add(...)`` dispatches through the kernel shim
- bash form   → ``!add ...`` runs the CLI directly from PATH
- env vars    → ``RLM_TOOL_CALL_SOURCE`` is set to ``"python"`` on the await path

Skill fixtures live under ``tests/fixtures/skills/<name>/``.
Kernel startup is ~700ms per test; keep the set here small.
"""

from __future__ import annotations

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result

from rlm.engine import RLMEngine


async def test_valid_python_ptc(register_add_ptc, session):
    prompt = "compute 2 + 3"
    messages = [
        DummyMessage(
            tool_calls=[
                DummyToolCall("ipython", {"code": "print(await add(a=2, b=3))"})
            ]
        ),
        DummyMessage(content="done"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert tool_result(client).strip() == "5"
    assert result.answer == "done"


async def test_valid_bash_ptc(register_add_ptc, session):
    prompt = "compute 2 + 3"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "!add --a 2 --b 3"})]
        ),
        DummyMessage(content="done"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert "5" in tool_result(client)
    assert result.answer == "done"


async def test_source_env_propagates(install_skill, session):
    install_skill("whoami_src")
    prompt = "check env"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "print(await whoami_src())"})]
        ),
        DummyMessage(content="done"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    assert tool_result(client).strip() == "python"
    assert result.answer == "done"
