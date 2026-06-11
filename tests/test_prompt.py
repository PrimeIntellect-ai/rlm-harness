"""Tests for system prompt construction."""

from __future__ import annotations

from dataclasses import dataclass

from rlm.prompt import (
    EDIT_SKILL_PROMPT,
    GIT_HISTORY_GUARD_PROMPT,
    IPYTHON_CONTROL_PROMPT,
    build_system_prompt,
)


@dataclass
class _Tool:
    name: str


def _prompt(
    active_tools: list[_Tool],
    *,
    installed_skills: list[str] | None = None,
) -> str:
    return build_system_prompt(
        "/repo",
        None,
        installed_skills or [],
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
    assert "native runtime" in prompt
    assert "use `%%bash` cells" in prompt
    assert "do not install dependencies into the IPython kernel" in prompt
    assert "Use Python for reading and searching files" in prompt
    assert "Use Python for reading, searching, and editing files" not in prompt


def test_ipython_control_prompt_omitted_without_ipython_tool():
    assert IPYTHON_CONTROL_PROMPT not in _prompt([_Tool("bash")])


def test_edit_skill_prompt_included_when_edit_is_installed():
    prompt = _prompt([_Tool("ipython")], installed_skills=["edit"])

    assert EDIT_SKILL_PROMPT in prompt
    assert 'await edit(path="relative/file.py"' in prompt
    assert "you must use the pre-imported `edit` skill" in prompt
    assert "`old_str` must appear exactly once" in prompt
    assert "do not use `file`, `old`, `new`, line numbers" in prompt
    assert "inspect the file and retry with a smaller exact snippet" in prompt
    assert "Only use normal Python file I/O for creating new files" in prompt


def test_edit_skill_prompt_omitted_without_edit_skill():
    prompt = _prompt([_Tool("ipython")], installed_skills=["search_docs"])

    assert EDIT_SKILL_PROMPT not in prompt
