"""Test skill: print a string."""

from __future__ import annotations

import argparse

PARAMETERS = {
    "type": "object",
    "properties": {"s": {"type": "string"}},
    "required": ["s"],
}


async def run(s: str, **_) -> None:
    print(s)


def main() -> None:
    parser = argparse.ArgumentParser(prog="say")
    parser.add_argument("--s", type=str, required=True)
    args = parser.parse_args()
    print(args.s)
