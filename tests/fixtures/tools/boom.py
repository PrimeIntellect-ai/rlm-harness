"""Test-only ``boom`` tool that always raises. Registered session-wide via ``register_fixture_tools``."""

from __future__ import annotations

from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome


class BoomTool:
    name = "boom"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "boom",
                "description": "Always raises RuntimeError.",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        raise RuntimeError("boom")
