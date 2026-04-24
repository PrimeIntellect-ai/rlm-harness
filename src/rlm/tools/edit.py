"""Builtin edit tool — safe single-occurrence string replacement.

Ported from the ``edit_via_str_replace`` tool in mini-swe-agent-plus:
on ambiguity the tool reports the line numbers of every match so the
model can add surrounding context and retry; on success it echoes a
context snippet with line numbers so the model can confirm the edit
landed in the right spot without re-reading the file.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome


_CONTEXT_LINES = 3


EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly "
            "once in the file; otherwise the tool reports the line numbers "
            "of every match so you can add surrounding context and retry."
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


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return positions
        positions.append(idx)
        start = idx + len(needle)


def _line_number(text: str, char_index: int) -> int:
    return text.count("\n", 0, char_index) + 1


def _snippet(new_content: str, replacement_line: int, new_str: str) -> str:
    lines = new_content.split("\n")
    extra = new_str.count("\n")
    start = max(1, replacement_line - _CONTEXT_LINES)
    end = min(len(lines), replacement_line + _CONTEXT_LINES + extra)
    width = len(str(end))
    return "\n".join(f"{i:>{width}} | {lines[i - 1]}" for i in range(start, end + 1))


def _atomic_write(path: Path, data: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as f:
        tmp_path = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


class EditTool:
    """Builtin tool handler for single-occurrence file edits."""

    name = "edit"

    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(EDIT_SCHEMA)

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

        positions = _find_all(content, old_str)
        if not positions:
            return ToolOutcome(
                content=f"Error: old_str did not appear verbatim in {path}."
            )
        if len(positions) > 1:
            line_nums = [_line_number(content, i) for i in positions]
            return ToolOutcome(
                content=(
                    f"Error: found {len(positions)} occurrences of old_str in "
                    f"{path} at lines {line_nums}. Expand old_str with "
                    f"surrounding context so it matches exactly once."
                )
            )

        replace_start = positions[0]
        replacement_line = _line_number(content, replace_start)
        new_content = (
            content[:replace_start] + new_str + content[replace_start + len(old_str) :]
        )

        try:
            _atomic_write(filepath, new_content)
        except Exception as exc:
            return ToolOutcome(content=f"Error writing {path}: {exc}")

        snippet = _snippet(new_content, replacement_line, new_str)
        return ToolOutcome(
            content=(
                f"Edited {path} at line {replacement_line}.\n"
                f"{snippet}\n"
                f"Review the changes and make sure they are as expected."
            )
        )
