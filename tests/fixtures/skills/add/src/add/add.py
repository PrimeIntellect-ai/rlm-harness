"""Test skill: add two integers."""

from __future__ import annotations

import argparse

PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    },
    "required": ["a", "b"],
}


async def run(a: int, b: int, **_) -> int:
    return int(a) + int(b)


def main() -> None:
    parser = argparse.ArgumentParser(prog="add")
    parser.add_argument("--a", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    args = parser.parse_args()
    print(args.a + args.b)
