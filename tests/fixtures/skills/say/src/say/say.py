"""Test skill: echo a string."""

from __future__ import annotations


async def run(s: str) -> str:
    """Echo a string.

    Args:
        s: The string to echo back.

    Returns:
        ``s`` unchanged.
    """
    return s
