"""Builtin edit tool — safe single-occurrence string replacement.

Ported from the ``edit`` skill in research-environments/rlm_swe. Kept as a
builtin so it's available to any env that opts in via RLM_TOOLS, not just
envs that ship the skill.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome


EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly "
            "once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to cwd or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must appear exactly once).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string.",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


class EditTool:
    """Builtin tool handler for single-occurrence file edits."""

    name = "edit"

    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(EDIT_SCHEMA)

    def prompt_lines(self, *, max_turns_in_context: int | None) -> list[str]:
        # Static tool — usage info lives in the OpenAI tool description.
        return []

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        path = args.get("path")
        old_str = args.get("old_str")
        new_str = args.get("new_str")

        if not isinstance(path, str) or not path:
            return ToolOutcome(content="Error: 'path' is required")
        if not isinstance(old_str, str):
            return ToolOutcome(content="Error: 'old_str' must be a string")
        if not isinstance(new_str, str):
            return ToolOutcome(content="Error: 'new_str' must be a string")

        base_dir = Path(context.cwd) if context.cwd else Path.cwd()
        filepath = base_dir / path
        if not filepath.exists():
            return ToolOutcome(content=f"Error: {path} not found")
        try:
            content = filepath.read_text()
        except Exception as exc:
            return ToolOutcome(content=f"Error reading {path}: {exc}")

        count = content.count(old_str)
        if count == 0:
            return ToolOutcome(content=f"Error: string not found in {path}")
        if count > 1:
            return ToolOutcome(
                content=f"Error: found {count} occurrences, need exactly 1"
            )

        try:
            filepath.write_text(content.replace(old_str, new_str, 1))
        except Exception as exc:
            return ToolOutcome(content=f"Error writing {path}: {exc}")
        return ToolOutcome(content=f"Edited {path}")
