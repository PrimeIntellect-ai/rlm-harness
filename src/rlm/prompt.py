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
        "Each subdirectory is a tool. Import them in Python and await run(...), or invoke them via their CLI with --help.",
        "",
        f"Your conversation is logged to {messages_path}.",
    ]

    if allow_recursion:
        parts[6:6] = [
            "The `rlm` module is pre-imported. Call `await rlm.run('sub-task')` to spawn a recursive sub-agent.",
            "For parallel sub-agents, use normal Python async patterns such as `await asyncio.gather(rlm.run('task1'), rlm.run('task2'))`.",
            "",
        ]

    return "\n".join(parts)
