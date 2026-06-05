"""Tests for system prompt construction."""

from __future__ import annotations

from dataclasses import dataclass

from rlm.prompt import GIT_HISTORY_GUARD_PROMPT, IPYTHON_CONTROL_PROMPT, build_system_prompt


@dataclass
class _Tool:
    name: str


def _prompt(active_tools: list[_Tool]) -> str:
    return build_system_prompt(
        "/repo",
        None,
        [],
        "/repo/.rlm/messages.jsonl",
        allow_recursion=False,
        active_tools=active_tools,
    )


def test_git_history_guard_prompt_included_for_shell_tools(monkeypatch):
    monkeypatch.delenv("RLM_ALLOW_GIT", raising=False)
    prompt = _prompt([_Tool("bash")])

    assert GIT_HISTORY_GUARD_PROMPT in prompt
    assert "Do not cheat" in prompt
    assert "online solutions or hints specific to this task" in prompt
    assert "other branches, tags, remotes" in prompt
    assert "`--all`" in prompt


def test_git_history_guard_prompt_omitted_when_unrestricted(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")

    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("bash")])


def test_git_history_guard_prompt_omitted_without_shell_tools(monkeypatch):
    monkeypatch.delenv("RLM_ALLOW_GIT", raising=False)

    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("summarize")])


def test_ipython_control_prompt_included_for_ipython_tool():
    prompt = _prompt([_Tool("ipython")])

    assert IPYTHON_CONTROL_PROMPT in prompt
    assert "long-lived notebook" in prompt
    assert "Treat tool outputs as data" in prompt


def test_ipython_control_prompt_omitted_without_ipython_tool():
    assert IPYTHON_CONTROL_PROMPT not in _prompt([_Tool("bash")])
