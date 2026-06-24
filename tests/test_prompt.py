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
        "/repo/.agent/messages.jsonl",
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
    assert "must be the first line of the code cell" in prompt
    assert "do not install dependencies into the IPython kernel" in prompt


def test_ipython_control_prompt_omitted_without_ipython_tool():
    assert IPYTHON_CONTROL_PROMPT not in _prompt([_Tool("bash")])


def test_edit_skill_prompt_included_only_when_edit_is_installed():
    prompt = _prompt([_Tool("ipython")], installed_skills=["edit"])

    assert EDIT_SKILL_PROMPT in prompt
    assert 'await edit(path="pkg/file.py", old_str=old, new_str=new)' in prompt
    assert "triple double quotes" in prompt
    assert EDIT_SKILL_PROMPT not in _prompt(
        [_Tool("ipython")], installed_skills=["search_docs"]
    )


def test_skill_pre_imported_language_present_with_ipython():
    prompt = _prompt([_Tool("ipython")], installed_skills=["websearch"])

    assert "Installed skills (pre-imported): `websearch`." in prompt
    assert "async function" in prompt
    assert "also available as a shell command" in prompt


def test_skill_usage_language_requires_ipython():
    """Without ipython there is no kernel, so skills are not pre-imported.

    The async-function language must drop, but the shell-command surface is
    still real in bash mode.
    """
    prompt = _prompt([_Tool("bash")], installed_skills=["websearch"])

    assert "Installed skills: `websearch`." in prompt
    assert "(pre-imported)" not in prompt
    assert "async function" not in prompt
    assert "also available as a shell command" in prompt


def test_skill_shell_command_language_requires_shell_tool():
    """With neither ipython nor bash, skills can't be invoked — list names only."""
    prompt = _prompt([_Tool("edit")], installed_skills=["websearch"])

    assert "Installed skills: `websearch`." in prompt
    assert "(pre-imported)" not in prompt
    assert "also available as a shell command" not in prompt


def test_edit_skill_prompt_omitted_without_ipython():
    """EDIT_SKILL_PROMPT is IPython-specific; drop it when no kernel runs."""
    assert EDIT_SKILL_PROMPT not in _prompt([_Tool("bash")], installed_skills=["edit"])
