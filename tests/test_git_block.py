"""Tests for the tool-level git block."""

from __future__ import annotations

from rlm.tools.base import ToolContext
from rlm.tools.bash import BashTool
from rlm.tools.git_block import (
    find_blocked_command,
    find_blocked_in_ipython,
    refusal,
)
from rlm.types import RLMMetrics, TokenUsage


REFUSAL = "Bash command 'git' is not allowed. Please use a different command or tool."


def _ctx() -> ToolContext:
    return ToolContext(
        messages=[],
        metrics=RLMMetrics(),
        total_usage=TokenUsage(),
        last_prompt_tokens=0,
        exec_timeout=10,
    )


# --- find_blocked_command (bash predicate, mirrors mini_swe_agent_plus) ---


def test_plain_git_status_blocked():
    assert find_blocked_command("git status") == "git"


def test_chained_git_after_cd_blocked():
    # mini_swe_agent_plus's docstring example: "cd /testbed && git diff"
    assert find_blocked_command("cd /testbed && git diff") == "git"


def test_pipe_separator_blocked():
    assert find_blocked_command("ls | git foo") == "git"


def test_or_separator_blocked():
    assert find_blocked_command("false || git status") == "git"


def test_semicolon_separator_blocked():
    assert find_blocked_command("ls; git log") == "git"


def test_non_git_command_unaffected():
    assert find_blocked_command("ls -la") is None


def test_command_substring_unaffected():
    # "github" starts with "git" but is not the git binary
    assert find_blocked_command("echo github") is None


def test_allow_git_env_var_disables_block(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    assert find_blocked_command("git status") is None


def test_refusal_message_matches_existing_shim():
    assert refusal("git") == REFUSAL


# --- find_blocked_in_ipython ---


def test_ipython_shell_escape_blocked():
    assert find_blocked_in_ipython("!git status") == "git"


def test_ipython_double_shell_escape_blocked():
    assert find_blocked_in_ipython("!!git diff") == "git"


def test_ipython_chained_shell_escape_blocked():
    assert find_blocked_in_ipython("!cd /repo && git log") == "git"


def test_ipython_bash_cell_magic_blocked():
    code = "%%bash\ncd /repo\ngit status"
    assert find_blocked_in_ipython(code) == "git"


def test_ipython_sx_line_magic_blocked():
    assert find_blocked_in_ipython("%sx git status") == "git"


def test_ipython_pure_python_unaffected():
    # No shell escape ⇒ nothing to block, even if literal "git" appears
    assert find_blocked_in_ipython("x = 'git status'\nprint(x)") is None


def test_ipython_non_git_shell_escape_unaffected():
    assert find_blocked_in_ipython("!ls -la") is None


def test_ipython_subprocess_run_documented_bypass():
    # Pure-Python subprocess calls are NOT detected (documented limitation).
    code = "import subprocess; subprocess.run(['git', 'status'])"
    assert find_blocked_in_ipython(code) is None


def test_ipython_allow_git_env_var(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    assert find_blocked_in_ipython("!git status") is None


# --- BashTool integration ---


def test_bash_tool_refuses_git():
    outcome = BashTool().execute({"command": "git status"}, _ctx())
    assert outcome.content == REFUSAL


def test_bash_tool_refuses_chained_git():
    outcome = BashTool().execute({"command": "cd /tmp && git diff"}, _ctx())
    assert outcome.content == REFUSAL


def test_bash_tool_runs_non_git_command():
    outcome = BashTool().execute({"command": "echo hello"}, _ctx())
    assert "hello" in outcome.content


def test_bash_tool_allow_git_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    # ``git --version`` is a harmless probe that doesn't need a repo. If
    # the env doesn't have git installed at all, fall back to verifying
    # the refusal didn't fire.
    outcome = BashTool().execute({"command": "git --version"}, _ctx())
    assert outcome.content != REFUSAL


# --- IpythonTool integration ---


def test_ipython_tool_refuses_git():
    """IpythonTool.execute short-circuits with the refusal before touching the REPL."""
    from rlm.tools.ipython import IpythonTool

    ctx = _ctx()
    # context.repl stays None; the refusal must fire before the
    # "REPL is not available" branch.
    ctx.repl = object()  # truthy sentinel — never used
    outcome = IpythonTool().execute({"code": "!git status"}, ctx)
    assert outcome.content == REFUSAL


def test_ipython_tool_refuses_bash_cell_magic_git():
    from rlm.tools.ipython import IpythonTool

    ctx = _ctx()
    ctx.repl = object()
    outcome = IpythonTool().execute({"code": "%%bash\ncd /repo && git diff"}, ctx)
    assert outcome.content == REFUSAL


def test_ipython_tool_passes_through_non_git(monkeypatch):
    """When the code has no shell escapes, the tool delegates to the REPL."""
    from rlm.tools.ipython import IpythonTool

    class StubRepl:
        def execute(self, code, timeout):
            return f"ran: {code}"

    ctx = _ctx()
    ctx.repl = StubRepl()
    outcome = IpythonTool().execute({"code": "print(1+1)"}, ctx)
    assert outcome.content == "ran: print(1+1)"
