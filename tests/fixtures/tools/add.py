"""Test-only native ``add`` tool. Registered per-test via ``register_add_ntc``."""

from __future__ import annotations

from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome


class AddTool:
    name = "add"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(content=str(args["a"] + args["b"]))
