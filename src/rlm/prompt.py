"""System prompt construction."""

from __future__ import annotations


def build_system_prompt(
    cwd: str,
    skills_dir: str,
    messages_path: str,
    *,
    allow_recursion: bool,
) -> str:
    """Build the system prompt."""
    parts = [
        "You are a coding agent. You solve tasks by writing and executing code, observing results, and iterating.",
        "",
        "You have a persistent IPython session. Variables, imports, and function definitions persist across calls.",
        "Use !command for shell commands (e.g. !git status, !ls -la, !pip install foo).",
        "Use %%bash for multi-line shell scripts.",
        "",
        "Work one step at a time: execute code, read the output, then decide your next step.",
        "When you are done, stop calling tools and state your final answer.",
        f"Working directory: {cwd}",
        "",
        f"Programmatic tools are available as Python modules in {skills_dir}.",
        "Each subdirectory is a tool. Run with --help to see usage.",
        "",
        f"Your conversation is logged to {messages_path}.",
    ]

    if allow_recursion:
        parts[6:6] = [
            "The `rlm` module is pre-imported. Call rlm.batch(['sub-task']) to spawn a recursive sub-agent.",
            "Pass multiple tasks to rlm.batch(['task1', 'task2', ...]) to run sub-agents in parallel.",
            "",
        ]

    return "\n".join(parts)
