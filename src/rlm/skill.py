"""Shared CLI entry for skill packages.

A skill is a Python package exposing an async ``run(...)``. Its bash CLI
comes from ``tyro.cli(run)`` via a single shared entry point —
``[project.scripts]`` in each skill's ``pyproject.toml`` just points at
``rlm.skill:cli``.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import tyro


def run_cli(func: Any, prog: str | None = None) -> None:
    """Run ``func`` as a CLI with arguments parsed from the signature.

    Async returns are awaited; non-``None`` return values are printed.
    """
    result = tyro.cli(func, prog=prog or Path(sys.argv[0]).name)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if result is not None:
        print(result)


def cli() -> None:
    """Entry point for skill console scripts.

    Dispatches based on ``sys.argv[0]`` basename — invoked as ``say`` it
    imports the ``say`` module and runs its ``run`` via ``run_cli``.
    """
    name = Path(sys.argv[0]).name
    module = importlib.import_module(name)
    run_cli(module.run, prog=name)
