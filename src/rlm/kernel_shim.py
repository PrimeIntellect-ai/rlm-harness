"""Skill shims for external ipython kernels.

Registers modules so that ``import edit``, ``edit.PARAMETERS``, and
``await edit.run(...)`` work in the ipython kernel. Two dispatch paths:

* **In-process** — preferred. If the skill is importable from the
  kernel's Python (i.e. it lives in the same venv as rlm, which is the
  default layout produced by ``install.sh``), the real module is
  registered directly. ``await edit.run(**kwargs)`` then calls the
  skill's Python function with no process boundary: kwargs stay typed,
  exceptions surface as native Python exceptions with real tracebacks,
  and return values travel as native objects.
* **Subprocess** — fallback for skills installed in an isolated venv
  (not importable from the kernel). A proxy module shells out to the
  skill's CLI on PATH, translating kwargs to ``--flag value`` pairs.

The in-process path only wins when the importable module resolves to a
file under the skill's own ``src/`` directory — that guards against a
same-named PyPI package on ``sys.path`` shadowing the skill.

Usage (called from _inject_startup)::

    from rlm.kernel_shim import install_shims
    install_shims("/task/rlm-skills")
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
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
        return out

    return run


def _make_proxy(name: str, parameters: dict) -> types.ModuleType:
    """Create a subprocess-dispatched proxy module for a skill."""
    mod = types.ModuleType(name)
    mod.__doc__ = f"Proxy for the {name} skill (delegates to CLI)."
    mod.__path__ = []  # make it look like a package
    mod.PARAMETERS = parameters
    mod.run = _make_run(name)
    return mod


def _import_skill_module(skill_dir: Path, name: str) -> types.ModuleType | None:
    """Import *name* only if it resolves to a file under ``skill_dir/src``.

    Guards against a same-named PyPI package on ``sys.path`` shadowing
    the skill: if the importable module lives somewhere else, we refuse
    and let the caller fall back to the subprocess proxy.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    try:
        origin = Path(spec.origin).resolve()
        src_root = (skill_dir / "src").resolve()
    except OSError:
        return None
    if not origin.is_relative_to(src_root):
        return None
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def install_shims(skills_dir: str) -> list[str]:
    """Register modules for all skills found in *skills_dir*.

    Prefers in-process import when the skill resolves to a file under
    its own ``src/`` directory — this gives the kernel native Python
    semantics (typed kwargs, real exceptions, native return values)
    instead of the subprocess/CLI boundary. Falls back to the
    subprocess CLI proxy when the skill isn't importable from the
    kernel's Python (e.g. it's in an isolated venv).

    Returns the list of skill names that were registered.
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

        real_mod = _import_skill_module(skill_dir, name)
        if real_mod is not None and callable(getattr(real_mod, "run", None)):
            sys.modules[name] = real_mod
            shimmed.append(name)
            continue

        # Subprocess fallback — skill is not importable from the kernel.
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
