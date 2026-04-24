"""Skill-side schema derivation, tool wrapper, and CLI entry.

A skill is a Python package exposing an async ``run(...)``. Its OpenAI
tool schema, its CLI, and its tool-call dispatch all derive from that
one function's signature and docstring.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import site
import sys
import traceback
from pathlib import Path
from typing import Any

import tyro
from agents.function_schema import function_schema

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.skills import get_installed_skills


def build_tool_schema(module: Any) -> dict:
    """Build an OpenAI tool definition from a skill module's ``run`` function.

    Name comes from the module (so ``say.run`` surfaces as tool ``say``);
    description + parameter schema come from ``function_schema(run)`` —
    which parses the Google-style docstring for the summary and per-param
    descriptions and builds the JSON schema from type hints.
    """
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"skill {module.__name__!r} has no callable run()")
    fs = function_schema(run)
    return {
        "type": "function",
        "function": {
            "name": module.__name__,
            "description": fs.description or "",
            "parameters": fs.params_json_schema,
        },
    }


def run_cli(func: Any, prog: str | None = None) -> None:
    """Auto-generate a CLI from ``func``'s signature via tyro and run it.

    ``tyro.cli`` turns the signature + Google-style docstring into a
    typed argparse-style CLI. Async returns are awaited; non-``None``
    return values are printed.
    """
    result = tyro.cli(func, prog=prog or Path(sys.argv[0]).name)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if result is not None:
        print(result)


def cli() -> None:
    """Entry point for ``rlm-skill-<name>`` console scripts.

    Dispatches based on ``sys.argv[0]`` basename — e.g. invoked as
    ``say`` it imports the ``say`` module and runs its ``run`` via
    ``run_cli``. Keeps skill ``pyproject.toml``s to one line under
    ``[project.scripts]``.
    """
    name = Path(sys.argv[0]).name
    module = importlib.import_module(name)
    run_cli(module.run, prog=name)


class SkillTool:
    """Wrap a skill module as a BuiltinTool-compatible handler.

    ``run()``'s return value is the tool result. On raise, the traceback
    becomes the tool result so the model can observe the error.
    """

    def __init__(self, module: Any) -> None:
        self._module = module
        self.name = module.__name__
        self._schema = build_tool_schema(module)

    def schema(self) -> dict:
        return self._schema

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        try:
            result = asyncio.run(self._module.run(**args))
        except Exception:
            return ToolOutcome(content=traceback.format_exc())
        return ToolOutcome(content="" if result is None else str(result))


def _refresh_site_paths() -> None:
    """Re-process site-packages ``.pth`` files for post-startup installs.

    Skills may be installed after the interpreter started (e.g. pytest
    session fixtures), which leaves editable installs' ``.pth`` files
    unapplied to ``sys.path``. Reapply them and drop import caches.
    """
    for entry in list(sys.path):
        if entry.endswith("site-packages"):
            site.addsitedir(entry)
    importlib.invalidate_caches()


def get_skill_tools() -> list[SkillTool]:
    """Instantiate a SkillTool for each installed skill."""
    _refresh_site_paths()
    return [SkillTool(importlib.import_module(name)) for name in get_installed_skills()]
