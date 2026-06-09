"""Tests for system prompt construction."""

from __future__ import annotations

from dataclasses import dataclass

from rlm.prompt import (
    GIT_HISTORY_GUARD_PROMPT,
    IPYTHON_CONTROL_PROMPT,
    build_system_prompt,
    resolve_prompt_input,
)


@dataclass
class _Tool:
    name: str


def _prompt(active_tools: list[_Tool], **kwargs) -> str:
    return build_system_prompt(
        "/repo",
        None,
        [],
        "/repo/.rlm/messages.jsonl",
        allow_recursion=False,
        active_tools=active_tools,
        **kwargs,
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


def test_ipython_control_prompt_omitted_without_ipython_tool():
    assert IPYTHON_CONTROL_PROMPT not in _prompt([_Tool("bash")])


def test_append_system_prompt_literal():
    prompt = _prompt([_Tool("ipython")], append_system_prompt="Always run tests.")
    assert "Always run tests." in prompt


def test_append_system_prompt_from_file(tmp_path):
    extra = tmp_path / "extra.md"
    extra.write_text("Never use global variables.")
    prompt = _prompt([_Tool("ipython")], append_system_prompt=str(extra))
    assert "Never use global variables." in prompt


def test_append_system_prompt_nonexistent_path_used_as_literal():
    # A path that doesn't exist should be treated as literal text
    prompt = _prompt(
        [_Tool("ipython")],
        append_system_prompt="/no/such/path/and/certainly/not/a/file.md",
    )
    assert "/no/such/path/and/certainly/not/a/file.md" in prompt


def test_append_system_prompt_omitted_when_none():
    prompt = _prompt([_Tool("ipython")])
    assert prompt.endswith("Call at most one built-in tool per turn.")


def test_resolve_prompt_input_returns_file_contents(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("hello from file")
    assert resolve_prompt_input(str(f)) == "hello from file"


def test_resolve_prompt_input_returns_literal_for_nonexistent():
    assert resolve_prompt_input("not a file") == "not a file"


def test_resolve_prompt_input_empty_string():
    assert resolve_prompt_input("") == ""
