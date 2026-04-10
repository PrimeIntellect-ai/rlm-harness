"""System prompt construction."""

from __future__ import annotations

import os


def build_system_prompt(cwd: str, skills_dir: str, messages_path: str) -> str:
    """Build the system prompt."""
    verbosity = os.environ.get("RLM_SYSTEM_PROMPT_VERBOSITY", "medium")

    parts = [
        "You are a coding agent. You solve tasks by calling tools, observing results, and iterating.",
        "",
        "Work one step at a time: call a tool, read the output, then decide your next step.",
        "When you are done, stop calling tools and state your final answer.",
        f"Working directory: {cwd}",
        "",
        f"Programmatic tools are available as Python modules in {skills_dir}.",
        "Each subdirectory is a tool. Run with --help to see usage.",
        "",
        f"Your conversation is logged to {messages_path}.",
    ]

    if verbosity == "heavy":
        parts.append("")
        parts.append("## Important guidelines")
        parts.append(
            "- NEVER claim you are done until you have seen tool output confirming success"
        )
        parts.append(
            "- Make small, incremental tool calls — don't try to do everything at once"
        )
        parts.append("- If a command fails, read the error and adjust your approach")
        parts.append("- For large outputs, use grep/head/tail to focus on what matters")

    return "\n".join(parts)
