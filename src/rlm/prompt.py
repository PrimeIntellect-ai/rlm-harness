"""System prompt construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.tools.git_block import allow_git

if TYPE_CHECKING:
    from rlm.tools.base import BuiltinTool


# Importable names of the base toolkit declared in pyproject.toml.
# Surfaced in the system prompt so the agent knows what's available
# without probing — keep in sync with the dependency list.
BASE_TOOLKIT = (
    "requests",
    "httpx",
    "yaml",
    "tomli",
    "dotenv",
    "pandas",
    "numpy",
    "scipy",
    "bs4",
    "lxml",
    "pydantic",
)

SHELL_TOOL_NAMES = frozenset({"bash", "ipython"})


def _ipython_usage_prompt() -> str:
    return (
        "- Treat `ipython` as your scratchpad: hold Python state, jot intermediate "
        "results, keep notes between turns, manage context. Each call is one IPython "
        "cell.\n"
        "- Cells default to Python. To run a shell command, prefix with `!` "
        "(single line) or `%%bash` (multi-line, must be the cell's first line). "
        "Prefer `%%bash`; use `!` only for one-liners.\n"
        "- Antipatterns (each raises `SyntaxError` — the cell is parsed as Python):\n"
        "    BAD:  find /app -name '*.py'    GOOD: !find /app -name '*.py'\n"
        "    BAD:  grep -rn 'foo' /app       GOOD: !grep -rn 'foo' /app\n"
        "    BAD:  cat > /tmp/x.py <<EOF     GOOD: a `%%bash` cell with the heredoc inside\n"
        "  Also: don't wrap a shell command in `subprocess.run([...])` when `!cmd` "
        "or `%%bash` does the same thing in fewer tokens.\n"
        f"- The kernel has these Python imports available by default: "
        f"{', '.join(BASE_TOOLKIT)}."
    )


GIT_HISTORY_GUARD_PROMPT = (
    "Do not cheat by using online solutions or hints specific to this task, or "
    "by copying or inferring solutions from other branches, tags, remotes, "
    "reflogs, or broad git history in the project. Broad-history `git log` "
    "options such as `--all`, `-all`, `--branches`, `--remotes`, `--tags`, "
    "`--glob`, `--alternate-refs`, `--reflog`, `--walk-reflogs`, or `-g` will "
    "be refused."
)


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    messages_path: str,
    *,
    allow_recursion: bool,
    active_tools: list[BuiltinTool],
) -> str:
    """Build the system prompt.

    Layout: role → environment (cwd, log path, skills) → capabilities
    (recursion) → tool API. Keep it tight: the model also receives the
    per-tool schemas, so redundant tool guidance here just inflates
    every request.
    """
    parts: list[str] = [
        "You are a coding agent. You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.",
        "When you are done, stop calling tools and state your final answer.",
        "",
        f"Working directory: {cwd}",
    ]

    skill_lines: list[str] = []
    if skills_dir:
        skill_lines.append(
            f"Local skills live under {skills_dir}. Read their SKILL.md files when helpful."
        )
    if installed_skills:
        installed = ", ".join(f"`{skill}`" for skill in installed_skills)
        skill_lines.append(f"Installed skills (pre-imported): {installed}.")
        skill_lines.append(
            "Each skill is an async function by the same name. "
            "Inspect with `help(<skill>)` or `inspect.signature(<skill>.run)`."
        )
        skill_lines.append(
            "Each skill is also available as a shell command by the same name: `<skill> ...`. "
            "Discover its CLI usage with `<skill> --help`."
        )
    if skill_lines:
        parts.extend(["", *skill_lines])

    if allow_recursion:
        parts.extend(
            [
                "",
                "A callable `rlm` is already in your global namespace — call it directly with `await rlm('sub-task')` to spawn a recursive sub-agent. Returns an `RLMResult` with `.answer` (string), `.usage`, `.turns`, and `.session_dir`.",
                "For parallel sub-agents, use normal Python async patterns such as `await asyncio.gather(rlm('task1'), rlm('task2'))`.",
            ]
        )

    if any(tool.name == "ipython" for tool in active_tools):
        parts.extend(["", _ipython_usage_prompt()])

    if _should_include_git_history_guard(active_tools):
        parts.extend(["", GIT_HISTORY_GUARD_PROMPT])

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    return "\n".join(parts)


def _should_include_git_history_guard(active_tools: list["BuiltinTool"]) -> bool:
    if allow_git():
        return False
    return any(tool.name in SHELL_TOOL_NAMES for tool in active_tools)
