"""Builtin bash tool — stateless shell command execution.

Each call spawns a fresh ``bash -c`` subprocess. No state is shared across
calls (no cwd changes, no exports, no background processes). For stateful
shell work, the ipython tool's ``!command`` / ``%%bash`` are still available.
"""

from __future__ import annotations

import copy
import os
import subprocess
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_command, refusal


BASH_TIMEOUT_MAX_SECONDS = 600


BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command. Stateless: each call runs in a fresh "
            "shell, so cd / exports / background processes don't persist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": None,  # filled by schema()
                },
            },
            "required": ["command"],
        },
    },
}


class BashTool:
    """Builtin tool handler for stateless bash command execution."""

    name = "bash"

    def schema(self) -> dict[str, Any]:
        timeout = min(
            int(os.environ.get("RLM_EXEC_TIMEOUT", "300")),
            BASH_TIMEOUT_MAX_SECONDS,
        )
        schema = copy.deepcopy(BASH_SCHEMA)
        schema["function"]["parameters"]["properties"]["timeout"]["description"] = (
            f"Optional timeout in seconds. Default: {timeout}s. "
            f"Maximum: {BASH_TIMEOUT_MAX_SECONDS}s."
        )
        return schema

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        command = args.get("command", "")
        if not isinstance(command, str):
            command = str(command)

        timeout = args.get("timeout")
        if timeout is None:
            timeout = context.exec_timeout
        else:
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = context.exec_timeout
        timeout = min(timeout, BASH_TIMEOUT_MAX_SECONDS)

        blocked = find_blocked_command(command)
        if blocked is not None:
            return ToolOutcome(content=refusal(blocked))

        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                # Replace invalid bytes with U+FFFD instead of raising.
                # Without this, any non-UTF-8 byte in stdout (binary
                # files, locale-encoded paths, truncated multibyte
                # sequences) crashes the tool and propagates out as an
                # AgentError that ends the rollout.
                errors="replace",
                timeout=timeout,
                cwd=context.cwd or None,
            )
        except subprocess.TimeoutExpired:
            return ToolOutcome(content=f"[command timed out after {timeout}s]")

        parts: list[str] = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(proc.stderr)
        if proc.returncode != 0:
            parts.append(f"[exit code: {proc.returncode}]")
        return ToolOutcome(content="\n".join(parts) if parts else "")
