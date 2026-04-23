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
    max_turns_in_context: int | None,
    active_tools: list[BuiltinTool],
) -> str:
    """Build the system prompt.

    Layout: role → work instructions → environment (cwd, log path, skills) →
    constraints (max turns) → capabilities (recursion) → tool API. Tools come
    last so the model has already seen any constraints that motivate them
    (e.g. summarize makes sense only once the turns-in-context limit is known).
    """
    parts: list[str] = [
        "You are a coding agent. You solve tasks by writing and executing code, observing results, and iterating.",
        "",
        "Work one step at a time: execute code, read the output, then decide your next step.",
        "When you are done, stop calling tools and state your final answer.",
        f"Working directory: {cwd}",
        "",
        f"Your conversation is logged to {messages_path}.",
    ]

    skill_lines: list[str] = []
    if skills_dir:
        skill_lines.append(
            f"Local skills live under {skills_dir}. Read their SKILL.md files when helpful."
        )
    if installed_skills:
        installed = ", ".join(f"`{skill}`" for skill in installed_skills)
        skill_lines.append(f"Installed skills: {installed}.")
        skill_lines.append(
            "Each skill is a Python package with an async `run` function. "
            "Call it with `import <skill>; await <skill>.run(...)`. "
            "Inspect its schema via `<skill>.PARAMETERS`."
        )
        skill_lines.append(
            "Each skill is also available as a shell command by the same name: `!<skill> ...`. "
            "Discover its CLI usage with `!<skill> --help`."
        )
    if skill_lines:
        parts.extend(["", *skill_lines])

    if max_turns_in_context is not None:
        parts.extend(
            [
                "",
                f"The current context may contain at most {max_turns_in_context} assistant turns.",
            ]
        )
        if any(tool.name == "summarize" for tool in active_tools):
            parts.append(
                "Use `summarize` to drop older turns and stay within this limit."
            )

    if allow_recursion:
        parts.extend(
            [
                "",
                "Spawn a recursive sub-agent with `from rlm import run; result = await run('sub-task')`. `run` returns an `RLMResult`; read `result.answer` for the sub-agent's final answer (string).",
                "For parallel sub-agents: `await asyncio.gather(run('task1'), run('task2'))`.",
            ]
        )

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    return "\n".join(parts)
