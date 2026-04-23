"""Tests for in-process vs subprocess dispatch in kernel_shim.install_shims."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from rlm.kernel_shim import install_shims

PARAMETERS = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string"},
    },
    "required": ["title", "date"],
}


def _write_skill_package(skills_dir: Path, name: str, run_src: str) -> Path:
    """Write a minimal skill package under *skills_dir*/name/."""
    skill = skills_dir / name
    src = skill / "src" / name
    src.mkdir(parents=True, exist_ok=True)
    (skill / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "rlm-skill-{name}"
            version = "0.1.0"
            requires-python = ">=3.10"

            [project.scripts]
            {name} = "{name}:main"
            """
        ).strip()
        + "\n"
    )
    (src / "__init__.py").write_text(run_src)
    return skill


def _write_cli_on_path(bin_dir: Path, name: str, required: list[str]) -> Path:
    """Write an executable argparse CLI to *bin_dir*/name and return its path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    lines = [
        f"#!{sys.executable}",
        "import argparse, json",
        "p = argparse.ArgumentParser(prog=" + repr(name) + ")",
    ]
    for f in required:
        lines.append(f"p.add_argument('--{f.replace('_','-')}', required=True)")
    lines.append("args = p.parse_args()")
    lines.append("print(json.dumps(vars(args)))")
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def clean_modules():
    """Snapshot sys.modules; restore after test so skill names don't leak."""
    before = dict(sys.modules)
    yield
    for name in list(sys.modules):
        if name not in before:
            del sys.modules[name]
        else:
            sys.modules[name] = before[name]


@pytest.fixture
def cli_path(tmp_path, monkeypatch):
    """Fresh bin dir prepended to PATH; returns the dir path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH','')}")
    return bin_dir


# ---------- in-process path ----------

_IN_PROCESS_SKILL_SRC = textwrap.dedent(
    """
    PARAMETERS = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["title", "date"],
    }

    async def run(title: str, date: str) -> dict:
        if date.count("-") != 2:
            raise ValueError(f"bad date format: {date!r} (expected YYYY-MM-DD)")
        return {"title": title, "date": date}
    """
).lstrip()


def test_in_process_valid_call_returns_native_value(
    tmp_path, monkeypatch, clean_modules, cli_path
):
    """A valid Python call returns the skill's real return value (dict, not str)."""
    skills = tmp_path / "skills"
    skill = _write_skill_package(skills, "in_proc_tool", _IN_PROCESS_SKILL_SRC)
    monkeypatch.syspath_prepend(str(skill / "src"))
    # CLI must be resolvable for parity with subprocess path; it's irrelevant
    # here because in-process should win.
    _write_cli_on_path(cli_path, "in_proc_tool", ["title", "date"])

    shimmed = install_shims(str(skills))
    assert "in_proc_tool" in shimmed

    mod = sys.modules["in_proc_tool"]
    assert getattr(mod, "__file__", None) and "in_proc_tool" in mod.__file__
    result = asyncio.run(mod.run(title="Standup", date="2025-07-14"))
    assert result == {"title": "Standup", "date": "2025-07-14"}


def test_in_process_missing_kwarg_raises_native_typeerror(
    tmp_path, monkeypatch, clean_modules, cli_path
):
    """Missing required kwarg surfaces as a native Python TypeError."""
    skills = tmp_path / "skills"
    skill = _write_skill_package(skills, "in_proc_tool", _IN_PROCESS_SKILL_SRC)
    monkeypatch.syspath_prepend(str(skill / "src"))
    _write_cli_on_path(cli_path, "in_proc_tool", ["title", "date"])

    install_shims(str(skills))
    mod = sys.modules["in_proc_tool"]
    with pytest.raises(TypeError, match="date"):
        asyncio.run(mod.run(title="Standup"))


def test_in_process_skill_raised_exception_propagates(
    tmp_path, monkeypatch, clean_modules, cli_path
):
    """An exception raised inside the skill reaches the caller unchanged."""
    skills = tmp_path / "skills"
    skill = _write_skill_package(skills, "in_proc_tool", _IN_PROCESS_SKILL_SRC)
    monkeypatch.syspath_prepend(str(skill / "src"))
    _write_cli_on_path(cli_path, "in_proc_tool", ["title", "date"])

    install_shims(str(skills))
    mod = sys.modules["in_proc_tool"]
    with pytest.raises(ValueError, match="bad date format"):
        asyncio.run(mod.run(title="Standup", date="2025/07/14"))


# ---------- subprocess fallback path ----------

def test_subprocess_fallback_missing_flag_returns_argparse_text(
    tmp_path, clean_modules, cli_path
):
    """When the skill isn't importable, the shim uses subprocess and
    argparse's stderr comes back as a string (matching the shell-call
    experience, since the Python call translates kwargs to CLI flags)."""
    skills = tmp_path / "skills"
    # Skill package exists on disk but its src/ is NOT on sys.path, so the
    # in-process import path refuses.
    _write_skill_package(skills, "sub_tool", "PARAMETERS = {}\n")
    _write_cli_on_path(cli_path, "sub_tool", ["title", "date"])

    shimmed = install_shims(str(skills))
    assert "sub_tool" in shimmed

    mod = sys.modules["sub_tool"]
    # Proxy module — no __file__ pointing into the skill.
    assert getattr(mod, "__file__", None) is None

    result = asyncio.run(mod.run())  # no kwargs -> CLI gets no flags
    assert isinstance(result, str)
    # argparse-style stderr message from our tiny CLI.
    assert "required" in result.lower() and "--title" in result


def test_subprocess_bash_call_gets_argparse_directly(cli_path):
    """Independently confirm the CLI itself emits argparse text on stderr
    for the bash/shell call site. No shim involved."""
    _write_cli_on_path(cli_path, "bash_tool", ["title", "date"])
    cli = cli_path / "bash_tool"

    proc = subprocess.run([str(cli)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "required" in proc.stderr.lower()
    assert "--title" in proc.stderr


# ---------- shadow guard ----------

def test_same_name_pypi_does_not_hijack_in_process_path(
    tmp_path, monkeypatch, clean_modules, cli_path
):
    """If `<name>` is importable but resolves outside the skill's src/,
    the shim refuses in-process and falls back to subprocess."""
    # 1. A decoy module with the same name, importable from a DIFFERENT path.
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "shadow_tool.py").write_text(
        "async def run(**kwargs):\n    return 'HIJACKED'\n"
    )
    monkeypatch.syspath_prepend(str(decoy_dir))

    # 2. A legitimate skill dir whose src/ is NOT on sys.path.
    skills = tmp_path / "skills"
    _write_skill_package(skills, "shadow_tool", "PARAMETERS = {}\n")
    _write_cli_on_path(cli_path, "shadow_tool", ["title", "date"])

    shimmed = install_shims(str(skills))
    assert "shadow_tool" in shimmed

    # The shim must NOT register the decoy — it falls back to subprocess.
    mod = sys.modules["shadow_tool"]
    assert getattr(mod, "__file__", None) is None  # proxy, not the decoy
    assert "decoy" not in str(mod)

    result = asyncio.run(mod.run())
    assert isinstance(result, str)
    assert "HIJACKED" not in result
    assert "required" in result.lower()
