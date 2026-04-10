"""Skill loader — discovers and imports tool skills from the skills/ directory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _find_skills_dir() -> Path:
    """Locate the skills/ directory. Works for both editable and installed layouts."""
    base = Path(__file__).resolve().parent
    # Editable install: src/rlm/tools.py → ../../skills
    candidate = base.parent.parent / "skills"
    if candidate.is_dir():
        return candidate
    # Installed wheel: site-packages/rlm/tools.py → ../skills
    candidate = base.parent / "skills"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Could not find skills/ directory")


_SKILLS_DIR = _find_skills_dir()


def _parse_skill_md(path: Path) -> tuple[dict[str, str], str]:
    """Read SKILL.md and return (frontmatter dict, body text)."""
    if not path.exists():
        return {}, ""
    text = path.read_text()
    if text.startswith("---"):
        _, fm_raw, body = text.split("---", 2)
        # Simple key: value parser — no YAML dependency needed
        meta = {}
        for line in fm_raw.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        return meta, body.strip()
    return {}, text.strip()


def _load_skill_module(skill_name: str) -> ModuleType:
    """Dynamically import a skill's scripts package."""
    scripts_dir = _SKILLS_DIR / skill_name / "scripts"
    init_path = scripts_dir / "__init__.py"
    module_name = f"rlm_skill_{skill_name}"

    # Register the scripts package so relative imports work
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(scripts_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod

    # Load the tool module (e.g. bash.py) — this is what exposes SCHEMA and run()
    inner_path = scripts_dir / f"{skill_name}.py"
    inner_spec = importlib.util.spec_from_file_location(
        f"{module_name}.{skill_name}",
        inner_path,
    )
    inner_mod = importlib.util.module_from_spec(inner_spec)
    sys.modules[f"{module_name}.{skill_name}"] = inner_mod
    inner_spec.loader.exec_module(inner_mod)

    spec.loader.exec_module(mod)

    # Read SKILL.md: frontmatter provides name/description, body provides prompt
    skill_md = _SKILLS_DIR / skill_name / "SKILL.md"
    meta, body = _parse_skill_md(skill_md)
    inner_mod.PROMPT = body

    # Build full OpenAI schema from frontmatter + PARAMETERS from the module
    inner_mod.SCHEMA = {
        "type": "function",
        "function": {
            "name": meta.get("name", skill_name),
            "description": meta.get("description", ""),
            "parameters": inner_mod.PARAMETERS,
        },
    }

    return inner_mod


def load_skills(allowed: list[str]) -> dict[str, ModuleType]:
    """Load and return skill modules for the allowed tool names.

    Returns a dict mapping skill name to its module, which exposes SCHEMA and run().
    """
    skills = {}
    for name in allowed:
        skill_dir = _SKILLS_DIR / name
        if not skill_dir.is_dir():
            continue
        skills[name] = _load_skill_module(name)
    return skills


def get_active_tools(skills: dict[str, ModuleType]) -> list[dict]:
    """Return OpenAI tool schemas from loaded skill modules."""
    return [skills[name].SCHEMA for name in skills]
