"""System prompt construction."""

from __future__ import annotations


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
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

    if skills_dir:
        parts[10:10] = [
            f"Local skills live under {skills_dir}. Read their SKILL.md files when helpful.",
            "Installed skills are importable directly by name, e.g. `import websearch` then `await websearch.run(...)`.",
            "If a skill exposes a shell command, invoke it by the same name, e.g. `websearch --help`.",
            "",
        ]
    else:
        parts[10:10] = [
            "Installed skills may be importable directly by name, e.g. `import websearch` then `await websearch.run(...)`.",
            "If a skill exposes a shell command, invoke it by the same name, e.g. `websearch --help`.",
            "",
        ]

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
