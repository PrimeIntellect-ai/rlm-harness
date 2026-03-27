"""Tool definitions and execution."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.session import Session

# -- Tool schemas (OpenAI function-calling format) --

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for file exploration, running tests, installing packages, "
            "and invoking `rlm` for sub-tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. "
            "old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit."},
                "old_str": {"type": "string", "description": "The exact string to find (must be unique)."},
                "new_str": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

ALL_TOOLS = {"bash": BASH_TOOL, "edit": EDIT_TOOL}

_RLM_CMD_RE = re.compile(r"\brlm\s")


def get_active_tools(allowed: list[str]) -> list[dict]:
    """Return tool schemas for the allowed tool names."""
    return [ALL_TOOLS[name] for name in allowed if name in ALL_TOOLS]


def run_bash(
    command: str,
    *,
    cwd: str,
    session: Session | None = None,
    timeout: int = 120,
    max_output: int = 8192,
) -> str:
    """Execute a bash command. Detects `rlm` invocations and sets up child sessions."""
    env = os.environ.copy()

    # Detect rlm sub-invocation → create child session dir
    if session and _RLM_CMD_RE.search(command):
        child_dir = session.child_dir()
        env["RLM_SESSION_DIR"] = str(child_dir)
        env["RLM_DEPTH"] = str(int(env.get("RLM_DEPTH", "0")) + 1)
        # Propagate sub-tools if set
        sub_tools = env.get("RLM_SUB_TOOLS")
        if sub_tools:
            env["RLM_TOOLS"] = sub_tools
        session.log_sub_spawn(child_dir.name, command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"

    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"

    # Truncate large output
    if len(output) > max_output:
        half = max_output // 2
        total = len(output)
        output = (
            output[:half]
            + f"\n... [output truncated, {total} chars total] ...\n"
            + output[-half:]
        )

    return output


def run_edit(
    path: str,
    old_str: str,
    new_str: str,
    *,
    cwd: str,
) -> str:
    """Safe single-occurrence string replacement."""
    filepath = Path(cwd) / path
    if not filepath.exists():
        return f"Error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: found {count} occurrences, need exactly 1"

    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"
