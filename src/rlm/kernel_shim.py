"""Skill shims for external ipython kernels.

Registers lightweight proxy modules so that ``import edit``,
``edit.PARAMETERS``, and ``await edit.run(...)`` work in the ipython
kernel — delegating to the skill CLIs on PATH via subprocess.

Shims are always installed regardless of whether a same-named module
exists, guaranteeing the kernel uses the rlm checkout's skills.

Usage (called from _inject_startup)::

    from rlm.kernel_shim import install_shims
    install_shims("/tmp/rlm-checkout/skills")
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import sys
import types
from pathlib import Path


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


def _make_run(cli_name: str):
    """Create an async run() that delegates to the skill CLI."""

    async def run(**kwargs) -> str:
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode().strip()
        if proc.returncode != 0:
            err = stderr.decode().strip()
            return err or out or f"{cli_name} exited with code {proc.returncode}"
        return out

    return run


def _make_proxy(name: str, parameters: dict) -> types.ModuleType:
    """Create a proxy module for a skill."""
    mod = types.ModuleType(name)
    mod.__doc__ = f"Proxy for the {name} skill (delegates to CLI)."
    mod.__path__ = []  # make it look like a package
    mod.PARAMETERS = parameters
    mod.run = _make_run(name)
    return mod


def _parse_enabled_tools() -> frozenset[str] | None:
    """Mirror of rlm.tools._parse_enabled_tools (duplicated to keep this
    module independent of rlm.__init__, which pulls in openai)."""
    raw = os.environ.get("RLM_ENABLED_TOOLS", "edit").strip()
    if raw == "*":
        return None
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def install_shims(skills_dir: str) -> list[str]:
    """Register proxy modules for all skills found in *skills_dir*.

    Respects ``RLM_ENABLED_TOOLS`` so that disabled skills are neither
    importable nor listed in the system prompt. The CLI itself remains
    on PATH — we only gate the Python shim and the prompt exposure.

    Returns the list of skill names that were shimmed.
    """
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        return []

    enabled = _parse_enabled_tools()

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

        if enabled is not None and name not in enabled:
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
