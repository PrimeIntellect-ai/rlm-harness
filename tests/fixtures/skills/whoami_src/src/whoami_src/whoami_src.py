"""Test skill: echo RLM_TOOL_CALL_SOURCE (set by kernel_shim on the python path)."""

from __future__ import annotations

import argparse
import os

PARAMETERS = {"type": "object", "properties": {}}


async def run(**_) -> str:
    return os.environ.get("RLM_TOOL_CALL_SOURCE", "<unset>")


def main() -> None:
    argparse.ArgumentParser(prog="whoami_src").parse_args()
    print(os.environ.get("RLM_TOOL_CALL_SOURCE", "<unset>"))
