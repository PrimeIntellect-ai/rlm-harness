"""Tool definitions and execution."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.session import Session


def _find_skills_dir() -> Path:
    """Locate the skills/ directory. Works for both editable and installed layouts."""
    base = Path(__file__).resolve().parent
    # Editable install: src/rlm/tools.py → ../../skills
    candidate = base.parent.parent / "skills"
    if candidate.is_dir():
        return candidate
    # Installed wheel: site-packages/rlm/tools.py → ../skills
    candidate = base.parent / "skills"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Could not find skills/ directory")


SKILLS_DIR = _find_skills_dir()

# -- Non-programmatic tool schemas --


BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for file exploration, running tests, installing packages, "
            "and invoking programmatic tools or `rlm` sub-agents."
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

_RLM_CMD_RE = re.compile(r"\brlm\s")


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


# Handled specially in the engine — mutates the messages list instead of
# returning output from a subprocess.
SUMMARIZE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": (
            "Summarize and drop old turns from context to free up space. "
            "A turn is one assistant response plus all its tool results. "
            "Dropping num_turns removes the oldest complete turns from context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_turns": {
                    "type": "integer",
                    "description": "Number of oldest turns to drop from context.",
                },
                "summary": {
                    "type": "string",
                    "description": "Your summary of the content being dropped.",
                },
            },
            "required": ["num_turns", "summary"],
        },
    },
}


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas for non-programmatic tools."""
    return [BASH_SCHEMA, SUMMARIZE_SCHEMA]
