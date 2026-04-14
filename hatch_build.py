"""Build-time metadata hook for aggregating skill dependencies."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface

BASE_DEPENDENCIES = [
    "openai>=1.0",
    "ipykernel",
    "jupyter_client",
    "nest_asyncio",
]
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_name(requirement: str) -> str:
    match = _REQ_NAME_RE.match(requirement)
    if not match:
        raise ValueError(f"Invalid dependency string: {requirement!r}")
    return match.group(1).lower().replace("_", "-")


def _iter_skill_dependency_lists(root: Path) -> list[tuple[Path, list[str]]]:
    skills_root = root / "skills"
    dependency_lists: list[tuple[Path, list[str]]] = []
    if not skills_root.is_dir():
        return dependency_lists

    for pyproject_path in sorted(skills_root.glob("*/pyproject.toml")):
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        dependencies = data.get("project", {}).get("dependencies", [])
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list) or any(
            not isinstance(dep, str) for dep in dependencies
        ):
            raise ValueError(
                f"{pyproject_path} must define [project].dependencies as a list of strings"
            )
        dependency_lists.append((pyproject_path, dependencies))

    return dependency_lists


class CustomMetadataHook(MetadataHookInterface):
    """Aggregate root dependencies with per-skill dependencies."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        dependencies = list(BASE_DEPENDENCIES)
        seen = {_requirement_name(dep): dep for dep in dependencies}

        for pyproject_path, skill_deps in _iter_skill_dependency_lists(Path(self.root)):
            for dep in skill_deps:
                name = _requirement_name(dep)
                existing = seen.get(name)
                if existing is not None and existing != dep:
                    raise ValueError(
                        f"Conflicting dependency constraints for {name!r}: "
                        f"{existing!r} vs {dep!r} from {pyproject_path}"
                    )
                if existing is None:
                    seen[name] = dep
                    dependencies.append(dep)

        metadata["dependencies"] = dependencies
