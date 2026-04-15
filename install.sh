#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq curl
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 --clear "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

SKILL_PATHS_FILE="$VENV_DIR/.skill-paths"

"$VENV_PYTHON" - "$ROOT_DIR" >"$SKILL_PATHS_FILE" <<'PY'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

root_dir = Path(sys.argv[1])
skills_dir = root_dir / "skills"
errors: list[str] = []
installable: list[Path] = []
seen_imports: dict[str, str] = {}
seen_scripts: dict[str, str] = {}

if skills_dir.is_dir():
    for skill_dir in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
        pyproject_path = skill_dir / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        skill_name = skill_dir.name
        skill_md = skill_dir / "SKILL.md"
        package_init = skill_dir / "src" / skill_name / "__init__.py"

        if not skill_md.exists():
            errors.append(f"{skill_name}: missing SKILL.md")
        if not package_init.exists():
            errors.append(
                f"{skill_name}: expected import package at src/{skill_name}/__init__.py"
            )

        project = tomllib.loads(pyproject_path.read_text()).get("project") or {}
        expected_dist_name = f"rlm-skill-{skill_name}"
        if project.get("name") != expected_dist_name:
            errors.append(
                f"{skill_name}: project.name must be {expected_dist_name!r}"
            )

        scripts = project.get("scripts") or {}
        if set(scripts) != {skill_name}:
            errors.append(
                f"{skill_name}: project.scripts must contain exactly one entry named {skill_name!r}"
            )
        else:
            script_target = scripts[skill_name]
            target_module = script_target.split(":", 1)[0] if isinstance(script_target, str) else ""
            if target_module not in {skill_name, f"{skill_name}.cli"} and not target_module.startswith(
                f"{skill_name}."
            ):
                errors.append(
                    f"{skill_name}: console script must target the top-level {skill_name!r} package"
                )

        previous_import = seen_imports.get(skill_name)
        if previous_import:
            errors.append(
                f"duplicate import name {skill_name!r}: {previous_import} and {skill_name}"
            )
        else:
            seen_imports[skill_name] = skill_name

        for script_name in scripts:
            previous_script = seen_scripts.get(script_name)
            if previous_script:
                errors.append(
                    f"duplicate console script {script_name!r}: {previous_script} and {skill_name}"
                )
            else:
                seen_scripts[script_name] = skill_name

        installable.append(skill_dir)

if errors:
    for error in errors:
        print(f"install.sh validation error: {error}", file=sys.stderr)
    raise SystemExit(1)

for skill_dir in installable:
    print(skill_dir)
PY

uv pip install --python "$VENV_PYTHON" -e "$ROOT_DIR"

while IFS= read -r skill_path; do
    [ -n "$skill_path" ] || continue
    uv pip install --python "$VENV_PYTHON" -e "$skill_path"
done <"$SKILL_PATHS_FILE"

echo "Installed rlm into $VENV_DIR"
echo "Activate it with: source \"$VENV_DIR/bin/activate\""
