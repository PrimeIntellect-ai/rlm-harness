"""Test skill: always raises."""

from __future__ import annotations

PARAMETERS = {"type": "object", "properties": {}}


async def run(**_) -> None:
    raise RuntimeError("boom")


def main() -> None:
    raise RuntimeError("boom")
