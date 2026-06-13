"""Tests for named, persistent background sub-agents (rlm.send / poll).

Driven with a scripted DummyClient and RLM_TOOLS="" so no sub-kernel starts.
"""

import asyncio

import pytest
from conftest import DummyClient, DummyMessage

from rlm._async_runtime import ERROR, FINISHED
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

    await api._REGISTRY.close_all()
    assert api._REGISTRY.get("probe") is None  # teardown clears the registry


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

    await api._REGISTRY.close_all()


async def test_send_collapses_names_that_sanitize_alike(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    from rlm import api

    client = DummyClient(
        [DummyMessage(content="answer one"), DummyMessage(content="answer two")]
    )
    # "foo/bar" and "foo-bar" both sanitize to sub-foo-bar; they must address
    # one worker + one transcript, not two engines writing the same dir.
    h1 = api.send("first", name="foo/bar", client=client)
    assert (await h1.wait()).answer == "answer one"

    h2 = api.send("second", name="foo-bar")  # same sanitized key → continues h1
    assert (await h2.wait()).answer == "answer two"  # reuses h1's engine + client

    assert h1.name == h2.name == "foo-bar"
    assert h1.session_dir == h2.session_dir
    assert h2.session_dir.name == "sub-foo-bar"

    await api._REGISTRY.close_all()


def test_engine_max_tokens_kwarg_overrides_env(monkeypatch):
    from rlm.config import reload_config
    from rlm.engine import RLMEngine

    monkeypatch.setenv("RLM_MAX_TOKENS", "100")
    reload_config()
    assert RLMEngine(max_tokens=5).max_tokens == 5
    assert RLMEngine().max_tokens == 100
    monkeypatch.delenv("RLM_MAX_TOKENS", raising=False)
    reload_config()
    assert RLMEngine().max_tokens is None


async def test_send_respects_live_agent_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import api

    api.send(
        "task a",
        name="a",
        client=DummyClient([DummyMessage(content="a1"), DummyMessage(content="a2")]),
    )
    # a second *distinct* agent exceeds the cap of 1
    with pytest.raises(AgentLimitReached):
        api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))

    # continuing the same agent needs no new slot
    api.send("more a", name="a")  # no raise (reuses a's slot)

    # teardown is the only thing that frees slots now; afterward a new agent starts
    await api._REGISTRY.close_all()
    h2 = api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))
    assert (await h2.wait()).answer == "b1"
    await api._REGISTRY.close_all()


async def test_errored_agent_frees_its_total_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import api

    class _BoomClient:  # raises when the engine reaches for the chat API
        @property
        def chat(self):
            raise RuntimeError("boom")

    h1 = api.send("task a", name="a", client=_BoomClient())
    await _settle(h1, want=ERROR)
    assert isinstance(h1.poll().error, RuntimeError)

    # the errored agent reaped its kernel and freed its total slot, so a new
    # agent fits under the cap of 1 — no dismiss needed
    h2 = api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))
    assert (await h2.wait()).answer == "b1"
    await api._REGISTRY.close_all()


async def test_construction_failure_frees_total_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import api

    # A bad engine kwarg makes RLMEngine(**kwargs) raise *during construction* —
    # before advance(), the path that used to skip the reap and leak the slot.
    h1 = api.send("task a", name="a", summarize_at_tokens="not-an-int")
    await _settle(h1, want=ERROR)
    assert isinstance(h1.poll().error, ValueError)

    # the total slot taken in send() was freed, so a new agent fits under cap=1
    h2 = api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))
    assert (await h2.wait()).answer == "b1"
    await api._REGISTRY.close_all()


async def test_registration_failure_frees_total_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import api

    # Child-session creation raises *inside* _REGISTRY.send — after send() has
    # already reserved the total slot, on the path the processor teardown can't
    # reach to release it.
    def _boom(name=None):
        raise OSError("boom")

    original = api._child_session
    monkeypatch.setattr(api, "_child_session", _boom)
    with pytest.raises(OSError, match="boom"):
        api.send("task a", name="a")
    monkeypatch.setattr(api, "_child_session", original)

    # the reserved slot was released despite the failure, so a new agent fits (cap=1)
    h = api.send("task b", name="b", client=DummyClient([DummyMessage(content="b1")]))
    assert (await h.wait()).answer == "b1"
    await api._REGISTRY.close_all()


async def test_one_off_run_waits_for_a_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")
    monkeypatch.setenv("RLM_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path / ".live_agents"))
    from rlm import _agent_limit as lim
    from rlm import api

    granted, held = lim.acquire_slot(lim.TOTAL)  # occupy the only slot
    assert granted

    started = asyncio.ensure_future(
        api.run("hi", client=DummyClient([DummyMessage(content="done")]))
    )
    await asyncio.sleep(0.4)
    assert not started.done()  # the one-off blocks until a slot frees

    lim.release_slot(held)
    result = await asyncio.wait_for(started, timeout=3)
    assert result.answer == "done"


async def test_engine_resumes_from_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_TOOLS", "")  # no sub-kernel
    monkeypatch.delenv("RLM_DEPTH", raising=False)
    monkeypatch.delenv("RLM_MAX_DEPTH", raising=False)
    from rlm.engine import RLMEngine
    from rlm.session import Session

    sdir = tmp_path / "agent"
    e1 = RLMEngine(
        session=Session(sdir),
        client=DummyClient([DummyMessage(content="first answer")]),
    )
    await e1.run("original task")

    # the engine object is gone (as after a kernel restart); a fresh engine on
    # the same session dir resumes the conversation from the on-disk view.
    e2 = RLMEngine(
        session=Session(sdir),
        client=DummyClient([DummyMessage(content="second answer")]),
    )
    e2.setup()
    contents = [m.get("content") for m in e2._messages]
    assert e2._messages[0]["role"] == "system"
    assert "original task" in contents  # prior turn restored from disk
    assert any("kernel was restarted" in (c or "") for c in contents)  # warned

    result = await e2.advance("follow-up")
    assert result.answer == "second answer"  # continues with the restored context
    await e2.aclose()
