"""Default Python packages pre-installed in the rlm tool venv.

`default_packages.txt` is the single source of truth, read by both
`install.sh` (to build `uv tool install --with ...` args) and the system
prompt (to tell the agent which libraries are already importable).
"""

from __future__ import annotations

from importlib.resources import files
from typing import NamedTuple


class DefaultPackage(NamedTuple):
    install: str
    imp: str


def _parse(text: str) -> list[DefaultPackage]:
    packages: list[DefaultPackage] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        install = parts[0]
        imp = parts[1] if len(parts) > 1 else install
        packages.append(DefaultPackage(install, imp))
    return packages


DEFAULT_PACKAGES: list[DefaultPackage] = _parse(
    files("rlm").joinpath("default_packages.txt").read_text()
)
