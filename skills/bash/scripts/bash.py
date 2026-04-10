"""Bash tool — run shell commands."""

from __future__ import annotations

import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.session import Session

PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        }
    },
    "required": ["command"],
}

_RLM_CMD_RE = re.compile(r"\brlm\s")


def run(
    command: str,
    *,
    cwd: str,
    session: Session | None = None,
    timeout: int = 120,
    max_output: int = 8192,
    **_,
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


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="bash")
    parser.add_argument("command", help="The shell command to execute.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory.")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Seconds before timeout."
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=8192,
        help="Truncate output to this many chars.",
    )
    args = parser.parse_args()

    print(
        run(
            args.command, cwd=args.cwd, timeout=args.timeout, max_output=args.max_output
        )
    )


if __name__ == "__main__":
    main()
