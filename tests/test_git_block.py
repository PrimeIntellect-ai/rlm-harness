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


def test_ipython_allow_git_env_var(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    assert find_blocked_in_ipython("!git status") is None


# --- find_blocked_python (AST-based detection of Python git invocations) ---


def test_python_subprocess_run_list_blocked():
    code = "import subprocess\nsubprocess.run(['git', 'log'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_run_shell_string_blocked():
    code = "import subprocess\nsubprocess.run('git log', shell=True)"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_popen_blocked():
    code = "import subprocess\nsubprocess.Popen(['git', 'diff'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_call_blocked():
    code = "import subprocess\nsubprocess.call(['git', 'status'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_check_call_blocked():
    code = "import subprocess\nsubprocess.check_call(['git', 'status'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_check_output_blocked():
    code = "import subprocess\nsubprocess.check_output(['git', 'log'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_os_system_blocked():
    assert find_blocked_in_ipython("import os\nos.system('git fetch')") == "git"


def test_python_os_popen_blocked():
    assert find_blocked_in_ipython("import os\nos.popen('git status')") == "git"


def test_python_subprocess_module_alias_blocked():
    code = "import subprocess as sp\nsp.run(['git', 'status'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_from_subprocess_import_run_blocked():
    code = "from subprocess import run\nrun(['git', 'log'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_single_hop_assignment_alias_blocked():
    code = "import subprocess\nr = subprocess.run\nr(['git', 'log'])"
    assert find_blocked_in_ipython(code) == "git"


def test_python_subprocess_run_non_git_unaffected():
    code = "import subprocess\nsubprocess.run(['pytest'])"
    assert find_blocked_in_ipython(code) is None


def test_python_string_literal_no_call_unaffected():
    assert find_blocked_in_ipython("x = 'git status'") is None


def test_python_command_starting_with_git_word_unaffected():
    # "github" is not the git binary; first arg literal "github" must not match.
    code = "import subprocess\nsubprocess.run(['github', 'status'])"
    assert find_blocked_in_ipython(code) is None


def test_python_shell_string_with_leading_whitespace_blocked():
    # Whitespace in front of the string still resolves first token to "git".
    code = "import subprocess\nsubprocess.run('   git log', shell=True)"
    assert find_blocked_in_ipython(code) == "git"


def test_python_allow_git_env_var_disables_ast(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    code = "import subprocess\nsubprocess.run(['git', 'log'])"
    assert find_blocked_in_ipython(code) is None


def test_python_getattr_documented_bypass():
    # Dynamic attribute lookup is intentionally NOT caught — pin behavior.
    code = "import subprocess\ngetattr(subprocess, 'run')(['git', 'log'])"
    assert find_blocked_in_ipython(code) is None


def test_python_dunder_import_documented_bypass():
    # ``__import__("subprocess").run(...)`` is also not caught.
    code = "__import__('subprocess').run(['git', 'log'])"
    assert find_blocked_in_ipython(code) is None


def test_python_multi_hop_alias_documented_bypass():
    # Two-hop reassignment is out of scope; r2 isn't tracked back to subprocess.run.
    code = "import subprocess\nr1 = subprocess.run\nr2 = r1\nr2(['git', 'log'])\n"
    assert find_blocked_in_ipython(code) is None


def test_python_syntax_error_does_not_refuse():
    # Malformed Python: AST scan returns None so the REPL surfaces the SyntaxError.
    assert find_blocked_in_ipython("def broken(:\n    pass") is None


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
