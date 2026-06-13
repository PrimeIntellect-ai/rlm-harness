"""Uploaded-skill discovery helpers."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


TASK_SKILLS_DIR = Path("/task/rlm-skills")


def find_skills_dir() -> Path | None:
    """Locate the uploaded skills directory when available."""
    return TASK_SKILLS_DIR if TASK_SKILLS_DIR.is_dir() else None


SKILLS_DIR = find_skills_dir()


def normalize_skill_name(name: str) -> str:
    """Normalize a discovered skill token to the import/CLI form."""
    return name.replace("-", "_")


def get_installed_skills() -> list[str]:
    """Return installed skill names discovered from distribution metadata."""
    skills: set[str] = set()
    prefix = "rlm-skill-"
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.startswith(prefix):
            skills.add(normalize_skill_name(name[len(prefix) :]))
    return sorted(skills)
