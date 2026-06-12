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
GIT_HISTORY_GUARD_PROMPT = (
    "Do not cheat by using online solutions or hints specific to this task, or "
    "by copying or inferring solutions from other branches, tags, remotes, "
    "reflogs, or broad git history in the project. Broad-history `git log` "
    "options such as `--all`, `-all`, `--branches`, `--remotes`, `--tags`, "
    "`--glob`, `--alternate-refs`, `--reflog`, `--walk-reflogs`, or `-g` will "
    "be refused."
)
IPYTHON_CONTROL_PROMPT = (
    "IPython is the agent's long-lived notebook: a persistent control "
    "environment for reasoning, context management, state, tool orchestration, "
    "and recursive subcalls. Use it to keep intermediate variables, inspect "
    "and transform outputs, write small helper functions, and preserve useful "
    "state across turns or compaction.\n\n"
    "Do not assume IPython is the native runtime of the external thing being "
    "investigated. A repository, package, service, dataset, paper, website, "
    "benchmark, or API may have its own environment and normal interface. "
    "Evaluate external systems through their own interface, then use IPython "
    "to coordinate the process and analyze what comes back.\n\n"
    "When running shell commands from IPython, use `%%bash` cells. This is a "
    "hard syntax rule: if an IPython tool call contains `%%bash`, the `code` "
    "argument must begin with exactly `%%bash\n` as its first seven "
    "characters. No leading blank line, no leading spaces, no comments, no "
    "imports, and no Python statements may appear before it. If a cell starts "
    "with `# note\n%%bash`, IPython parses the whole cell as Python and the "
    "shell command fails with `SyntaxError`.\n\n"
    "Before every IPython tool call, silently check: does `code` contain "
    "`%%bash`? If yes, the first line must be exactly `%%bash`. If not, "
    "rewrite the tool call before sending it.\n\n"
    "Correct:\n"
    "```python\n"
    "%%bash\n"
    "# explanation belongs here, after the magic\n"
    "cd /workspace/project && pytest -q\n"
    "```\n"
    "Wrong:\n"
    "```python\n"
    "# explanation before the magic breaks IPython\n"
    "%%bash\n"
    "cd /workspace/project && pytest -q\n"
    "```\n"
    "Wrong:\n"
    "```python\n"
    "\n"
    "%%bash\n"
    "cd /workspace/project && pytest -q\n"
    "```\n\n"
    "Avoid `!cmd` shell escapes for project commands so shell behavior is "
    "explicit and multi-line commands share one shell context.\n\n"
    "Important: do not install dependencies into the IPython kernel just to "
    "make an external project import or run there. If a project import, test, "
    "script, CLI, or dependency check is needed, run it through that project's "
    "own environment and normal command interface. For example, in a Python "
    "repo use its documented commands, `uv run ...`, `.venv/bin/python ...`, "
    "or the active project interpreter from the repo root. Treat failures from "
    "that native environment as the relevant result."
    "\n\n"
    "Use Python for reading and searching files — it gives you reusable "
    "variables you can slice, filter, and act on without re-reading. Always "
    "assign read/search results to named variables so you can revisit them "
    "later."
)
EDIT_SKILL_PROMPT = (
    "For targeted modifications to existing files, you must use the "
    "pre-imported `edit` skill from IPython instead of manual Python file "
    "writes. Read and inspect files with normal Python, then make the change "
    "with `await edit(path=\"relative/file.py\", old_str=old, new_str=new)`, "
    "where `old` and `new` are exact strings. Inline string literals are fine "
    "when they are valid Python. The target `old_str` must appear exactly "
    "once. Inside IPython `code`, never write native tool-call markup like "
    "`<tool_call>`, `<arg_key>`, or `<arg_value>`; use Python "
    "`await edit(...)` instead. "
    "The supported keyword arguments are `path`, `old_str`, `new_str`, "
    "and optional `cwd`; do not use `file`, `old`, `new`, line numbers, "
    "`after`, or `insert`. If an edit call fails because the string is not "
    "found or not unique, inspect the file and retry with a smaller exact "
    "snippet before falling back. Only use normal Python file I/O for creating "
    "new files or for broad generated rewrites that cannot be expressed as "
    "one or more exact replacements.\n\n"
    "Before calling `edit`, make sure the Python string syntax is valid. Do "
    "not wrap text that contains `\"\"\"` inside a `\"\"\"...\"\"\"` string; "
    "that creates a SyntaxError before `edit` runs. If the target text "
    "contains triple double quotes, use triple single quotes (`'''...'''`) or "
    "assign `old`/`new` from inspected file slices. If the text contains both "
    "triple quote styles or quoting becomes complex, build `old` and `new` as "
    "variables first, then pass those variables to `await edit(...)`.\n\n"
    "Good targeted edit pattern:\n"
    "```python\n"
    "from pathlib import Path\n"
    "text = Path(\"pkg/module.py\").read_text()\n"
    "print(text[text.index(\"def broken\") : text.index(\"def next_func\")])\n"
    "old = '''def broken():\n"
    "    \"\"\"Docstring with triple double quotes.\"\"\"\n"
    "    return False\n"
    "'''\n"
    "new = '''def broken():\n"
    "    \"\"\"Docstring with triple double quotes.\"\"\"\n"
    "    return True\n"
    "'''\n"
    "await edit(path=\"pkg/module.py\", old_str=old, new_str=new)\n"
    "```\n"
    "Do not do this for targeted edits:\n"
    "```python\n"
    "await edit(path=\"pkg/module.py\", old_str=\"\"\"def broken():\n"
    "    \"\"\"Docstring closes the outer string early.\"\"\"\n"
    "    return False\n"
    "\"\"\", new_str=\"\"\"...\n"
    "\"\"\")\n"
    "Path(\"pkg/module.py\").write_text(text.replace(old, new))\n"
    "open(\"pkg/module.py\", \"w\").write(new_contents)\n"
    "```"
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
        "You are a general purpose agent that uses code to solve tasks.",
        "You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.",
        "When you are done, stop calling tools and state your final answer.",
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {messages_path}",
        f"Pre-installed Python packages: {', '.join(BASE_TOOLKIT)}.",
        "Install additional packages with `uv pip install <pkg>` (this is a uv-managed venv with no pip module).",
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
        if "edit" in installed_skills:
            skill_lines.append(EDIT_SKILL_PROMPT)
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

    if _has_tool(active_tools, "ipython"):
        parts.extend(["", IPYTHON_CONTROL_PROMPT])

    if _should_include_git_history_guard(active_tools):
        parts.extend(["", GIT_HISTORY_GUARD_PROMPT])

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    return "\n".join(parts)


def _should_include_git_history_guard(active_tools: list["BuiltinTool"]) -> bool:
    if allow_git():
        return False
    return any(tool.name in SHELL_TOOL_NAMES for tool in active_tools)


def _has_tool(active_tools: list["BuiltinTool"], name: str) -> bool:
    return any(tool.name == name for tool in active_tools)
