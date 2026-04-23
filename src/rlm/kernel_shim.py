"""Skill shims for external ipython kernels.

Registers lightweight proxy modules so that ``import edit``,
``edit.PARAMETERS``, and ``await edit.run(...)`` work in the ipython
kernel — delegating to the skill CLIs on PATH via subprocess.

Shims are always installed regardless of whether a same-named module
exists, guaranteeing the kernel uses the uploaded skills.

Usage (called from _inject_startup)::

    from rlm.kernel_shim import install_shims
    install_shims("/task/rlm-skills")
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any


def _read_parameters(skill_src: Path) -> dict:
    """Extract the PARAMETERS dict from skill source without importing it."""
    for pyfile in skill_src.rglob("*.py"):
        try:
            tree = ast.parse(pyfile.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PARAMETERS"
            ):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
    return {}


def _maybe_decode(out: str) -> Any:
    """Decode structured skill output transparently.

    Skills communicate via subprocess stdout, so their return values
    always arrive as strings. When a skill prints a valid JSON value
    (list, dict, number, bool, null, or a quoted string), decode it so
    callers get the native Python object — avoids the
    ``if events:`` footgun where ``"[]"`` (a non-empty string) is truthy
    even though the underlying list is empty.

    Plain text output (anything that isn't valid JSON) is returned
    as-is, preserving existing behaviour for skills that return
    free-form strings.
    """
    if not out:
        return out
    try:
        return json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return out


def _make_run(cli_name: str):
    """Create an async run() that delegates to the skill CLI."""

    async def run(**kwargs) -> Any:
        cmd = [cli_name]
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, (list, tuple)):
                cmd.append(flag)
                cmd.extend(str(v) for v in value)
            elif isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        env = os.environ.copy()
        env["RLM_TOOL_CALL_SOURCE"] = "python"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode().strip()
        if proc.returncode != 0:
            err = stderr.decode().strip()
            return err or out or f"{cli_name} exited with code {proc.returncode}"
        return _maybe_decode(out)

    return run


def _make_proxy(name: str, parameters: dict) -> types.ModuleType:
    """Create a proxy module for a skill."""
    mod = types.ModuleType(name)
    mod.__doc__ = f"Proxy for the {name} skill (delegates to CLI)."
    mod.__path__ = []  # make it look like a package
    mod.PARAMETERS = parameters
    mod.run = _make_run(name)
    return mod


def install_shims(skills_dir: str) -> list[str]:
    """Register proxy modules for all skills found in *skills_dir*.

    Always installs shims regardless of whether a same-named module is
    already importable — this guarantees the kernel uses the rlm
    checkout's version of each skill, not an unrelated package.

    Returns the list of skill names that were shimmed.
    """
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        return []

    shimmed = []
    for skill_dir in sorted(skills_path.iterdir()):
        if not (skill_dir / "pyproject.toml").is_file():
            continue
        src = skill_dir / "src"
        if not src.is_dir():
            continue
        # The importable name is the subdirectory under src/
        for candidate in src.iterdir():
            if candidate.is_dir() and candidate.name != "__pycache__":
                name = candidate.name
                break
        else:
            continue

        # Skip if the CLI isn't on PATH
        if not shutil.which(name):
            continue

        parameters = _read_parameters(src)
        sys.modules[name] = _make_proxy(name, parameters)
        shimmed.append(name)

    # Also shim `rlm` itself for sub-agent recursion
    if shutil.which("rlm"):
        mod = types.ModuleType("rlm")
        mod.__doc__ = "Proxy for the rlm CLI (sub-agent recursion)."
        mod.__path__ = []

        async def _rlm_run(prompt: str, **kwargs) -> types.SimpleNamespace:
            cmd = ["rlm", prompt]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return types.SimpleNamespace(answer=stdout.decode().strip())

        mod.run = _rlm_run
        sys.modules["rlm"] = mod
        shimmed.append("rlm")

    return shimmed
