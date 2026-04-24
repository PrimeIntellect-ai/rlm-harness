"""System prompt construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.tools.base import BuiltinTool


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
        "You are a coding agent. You solve tasks by writing and executing code, observing results, and iterating one step at a time.",
        "When you are done, stop calling tools and state your final answer.",
        "A Python project's interpreter can be in `PATH`. If not use the appropriate `.venv`.",
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {messages_path}",
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
                "The `rlm` module is pre-imported. `await rlm('sub-task')` spawns a recursive sub-agent and returns an `RLMResult` with `.answer` (string), `.usage`, `.turns`, and `.session_dir`.",
                "For parallel sub-agents, use normal Python async patterns such as `await asyncio.gather(rlm('task1'), rlm('task2'))`.",
            ]
        )

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    active_tool_names = {tool.name for tool in active_tools}
    if "bash" in active_tool_names:
        parts.extend(
            [
                "",
                "When you are confident the task is complete and verified, signal the"
                " end of the rollout by issuing exactly one `bash` tool call with the"
                " command:",
                "",
                '    echo "TASK_FINISHED"',
                "",
                "Emit no further tool calls or text after this. You cannot continue"
                " working on this task after submitting.",
            ]
        )

    return "\n".join(parts)
