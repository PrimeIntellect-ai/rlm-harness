"""System prompt construction."""

from __future__ import annotations

import os
from types import ModuleType


def build_system_prompt(skills: dict[str, ModuleType], cwd: str) -> str:
    """Build system prompt from loaded skill modules."""
    verbosity = os.environ.get("RLM_SYSTEM_PROMPT_VERBOSITY", "medium")

    parts = [
        "You are a coding agent. You solve tasks by calling tools, observing results, and iterating.",
        "",
        "Work one step at a time: call a tool, read the output, then decide your next step.",
        "When you are done, stop calling tools and state your final answer.",
        f"Working directory: {cwd}",
    ]

    # Tool docs — collected from each skill's SKILL.md body
    parts.append("")
    parts.append("## Available tools")

    for name, skill in skills.items():
        prompt = getattr(skill, "PROMPT", "")
        if prompt:
            parts.append("")
            parts.append(prompt)

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
