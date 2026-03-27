"""System prompt construction."""

import os


def build_system_prompt(tools: list[str], cwd: str) -> str:
    """Build system prompt based on active tools."""
    verbosity = os.environ.get("RLM_SYSTEM_PROMPT_VERBOSITY", "medium")

    parts = [
        "You are a coding agent. You solve tasks by calling tools, observing results, and iterating.",
        "",
        "Work one step at a time: call a tool, read the output, then decide your next step.",
        "When you are done, stop calling tools and state your final answer.",
        f"Working directory: {cwd}",
    ]

    # Tool docs
    parts.append("")
    parts.append("## Available tools")

    if "bash" in tools:
        parts.append("")
        parts.append("### bash")
        parts.append("Run any shell command. Examples:")
        parts.append('  bash(command="ls -la src/")')
        parts.append('  bash(command="python -m pytest tests/ -x")')
        parts.append('  bash(command="grep -rn TODO src/")')

    if "edit" in tools:
        parts.append("")
        parts.append("### edit")
        parts.append("Replace a unique string in a file. old_str must appear exactly once.")
        parts.append('  edit(path="src/auth.py", old_str="return self._token", new_str="return self._generate_token()")')

    # Delegation docs (only if bash is available)
    if "bash" in tools:
        parts.append("")
        parts.append("## Sub-agent delegation")
        parts.append("")
        parts.append("For complex tasks, you can delegate sub-tasks to child agents by invoking `rlm` via bash:")
        parts.append("")
        parts.append("Single sub-task:")
        parts.append('  bash(command=\'rlm "check auth.py for security issues"\')')
        parts.append("")
        parts.append("Parallel sub-tasks:")
        parts.append('  bash(command=\'rlm --batch "check auth.py" "check login.py" "check session.py"\')')
        parts.append("")
        parts.append("Each child agent has the same capabilities as you. Use delegation when:")
        parts.append("- Sub-tasks are independent and can run in parallel")
        parts.append("- You want to give a focused task a fresh context window")

    if verbosity == "heavy":
        parts.append("")
        parts.append("## Important guidelines")
        parts.append("- NEVER claim you are done until you have seen tool output confirming success")
        parts.append("- Make small, incremental tool calls — don't try to do everything at once")
        parts.append("- If a command fails, read the error and adjust your approach")
        parts.append("- For large outputs, use grep/head/tail to focus on what matters")

    return "\n".join(parts)
