"""Tests for named, persistent background sub-agents (rlm.send / poll).

Driven with a scripted DummyClient and RLM_TOOLS="" so no sub-kernel starts.
"""

import asyncio

import pytest
from conftest import DummyClient, DummyMessage

from rlm._async_runtime import FINISHED
from rlm._agent_limit import AgentLimitReached


async def _settle(handle, *, want=FINISHED, tries=400):
    for _ in range(tries):
        if handle.poll().status == want:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"agent did not reach {want!r}: {handle.poll()!r}")


async def test_send_runs_named_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")  # no sub-kernel
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    from rlm import api

    client = DummyClient([DummyMessage(content="first answer")])
    handle = api.send("hello", name="probe", client=client)
    result = await handle.wait()

    assert result.answer == "first answer"
    # transcript path is nested under the parent session as sub-<name>
    assert handle.session_dir is not None
    assert handle.session_dir.name == "sub-probe"
    assert handle.session_dir.parent == tmp_path

    handle.dismiss()
    await asyncio.sleep(0.05)
    assert api._REGISTRY.get("probe") is None  # dismiss removes from the registry


async def test_send_same_name_continues_conversation(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    from rlm import api

    client = DummyClient(
        [DummyMessage(content="answer one"), DummyMessage(content="answer two")]
    )
    handle = api.send("first", name="chat", client=client)
    assert (await handle.wait()).answer == "answer one"

    api.send("second", name="chat")  # continue: same engine + client
    assert (await handle.wait()).answer == "answer two"

    handle.dismiss()
    await asyncio.sleep(0.05)


def test_engine_max_tokens_kwarg_overrides_env(monkeypatch):
    from rlm.engine import RLMEngine

    monkeypatch.setenv("RLM_MAX_TOKENS", "100")
    assert RLMEngine(max_tokens=5).max_tokens == 5
    assert RLMEngine().max_tokens == 100
    monkeypatch.delenv("RLM_MAX_TOKENS", raising=False)
    assert RLMEngine().max_tokens is None


async def test_send_respects_live_agent_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import api

    h1 = api.send(
        "task a",
        name="a",
        client=DummyClient([DummyMessage(content="a1"), DummyMessage(content="a2")]),
    )
    # a second *distinct* agent exceeds the cap of 1
    with pytest.raises(AgentLimitReached):
        api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))

    # continuing the same agent needs no new slot
    api.send("more a", name="a")  # no raise

    # freeing the slot lets a new agent start
    h1.dismiss()
    await asyncio.sleep(0.05)
    h2 = api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))
    assert (await h2.wait()).answer == "b1"
    h2.dismiss()
    await asyncio.sleep(0.05)
