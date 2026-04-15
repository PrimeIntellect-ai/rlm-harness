"""System prompt construction."""

from __future__ import annotations


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    messages_path: str,
    *,
    allow_recursion: bool,
    max_turns_in_context: int | None,
    summarize_enabled: bool,
) -> str:
    """Build the system prompt."""
    parts = [
        "You are a coding agent. You solve tasks by writing and executing code, observing results, and iterating.",
        "Call at most one built-in tool per turn.",
        "",
        "You have a persistent IPython session. Variables, imports, and function definitions persist across calls.",
        "Use !command for shell commands (e.g. !git status, !ls -la, !pip install foo).",
        "Use %%bash for multi-line shell scripts.",
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
        skill_lines.append(
            f"Installed skills: {installed}."
        )
        skill_lines.append(
            "Each skill is available as a Python module by the same name: `import <skill>`. "
            "Inspect its schema via `<skill>.PARAMETERS`, and call its async entrypoint via `<skill>.run(...)`."
        )
        skill_lines.append(
            "Each skill is also available as a shell command by the same name: `<skill> ...`. "
            "Discover its CLI usage with `<skill> --help`."
        )
    if skill_lines:
        parts[10:10] = [*skill_lines, ""]

    if max_turns_in_context is not None:
        parts.extend(
            [
                "",
                "The current context may contain at most "
                f"{max_turns_in_context} assistant turns.",
            ]
        )
        if summarize_enabled:
            parts.append(
                "Use `summarize` to drop older turns and stay within this limit."
            )

    if allow_recursion:
        parts[6:6] = [
            "The `rlm` module is pre-imported. Call `await rlm.run('sub-task')` to spawn a recursive sub-agent.",
            "For parallel sub-agents, use normal Python async patterns such as `await asyncio.gather(rlm.run('task1'), rlm.run('task2'))`.",
            "",
        ]

    return "\n".join(parts)
