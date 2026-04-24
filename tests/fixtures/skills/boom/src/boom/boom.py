"""Test skill: always raises."""

from __future__ import annotations


async def run() -> None:
    """Always raise ``RuntimeError``."""
    raise RuntimeError("boom")
