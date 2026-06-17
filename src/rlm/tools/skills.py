"""Uploaded-skill discovery helpers."""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

TASK_SKILLS_DIR = Path("/task/rlm-skills")


def _find_skills_dir() -> Path | None:
    """Locate the uploaded skills directory when available."""
    return TASK_SKILLS_DIR if TASK_SKILLS_DIR.is_dir() else None


SKILLS_DIR = _find_skills_dir()


def _normalize_skill_name(name: str) -> str:
    """Normalize a discovered skill token to the import/CLI form."""
    return name.replace("-", "_")


def _allowed_skill_names() -> list[str] | None:
    """Return the RLM_SKILLS allowlist, or None when all skills are enabled."""
    raw = os.environ.get("RLM_SKILLS")
    if raw is None:
        return None
    return [
        _normalize_skill_name(token.strip())
        for token in raw.split(",")
        if token.strip()
    ]


def get_installed_skills() -> list[str]:
    """Return installed skill names discovered from distribution metadata."""
    skills: set[str] = set()
    prefix = "rlm-skill-"
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.startswith(prefix):
            skills.add(_normalize_skill_name(name[len(prefix) :]))

    installed = sorted(skills)
    allowed = _allowed_skill_names()
    if allowed is None:
        return installed

    unknown = sorted(set(allowed).difference(installed))
    if unknown:
        raise ValueError(
            f"RLM_SKILLS contains unknown skill(s): {unknown}. "
            f"Installed: {installed}"
        )
    return [skill for skill in installed if skill in allowed]
