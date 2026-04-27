"""Block ``git`` at the tool-call level.

Mirrors the predicate in mini_swe_agent_plus's ``execute_bash.py``:

- split a bash command on ``&&``, ``||``, ``;`` and ``|``
- if the first whitespace-separated token of any segment is ``git``, refuse

The ``RLM_ALLOW_GIT=1`` env var disables the check entirely.

The same refusal string the existing sandbox shim emits is reused here so
that agent-visible logs (and any rubrics keyed on it) stay consistent
across the two block strategies.
"""

from __future__ import annotations

import os
import re

REFUSAL_TEMPLATE = (
    "Bash command '{cmd}' is not allowed. Please use a different command or tool."
)

# Reuse the mini_swe_agent_plus separators verbatim so behavior matches.
_SEPARATORS = re.compile(r"&&|\|\||;|\|")

_BLOCKED = ("git",)


def allow_git() -> bool:
    return os.environ.get("RLM_ALLOW_GIT") == "1"


def find_blocked_command(command: str) -> str | None:
    """Return the offending token if ``command`` invokes a blocked binary.

    Splits on ``&&``, ``||``, ``;``, ``|`` so chained calls like
    ``cd /repo && git status`` are caught the same way the reference
    implementation catches them. Returns ``None`` if nothing is blocked
    or if ``RLM_ALLOW_GIT=1``.
    """
    if allow_git():
        return None
    for segment in _SEPARATORS.split(command):
        tokens = segment.strip().split()
        if tokens and tokens[0] in _BLOCKED:
            return tokens[0]
    return None


def refusal(cmd: str) -> str:
    return REFUSAL_TEMPLATE.format(cmd=cmd)


# IPython shell-escape lines: ``!cmd`` and ``!!cmd``. Leading whitespace
# is allowed (IPython accepts indented shell escapes inside blocks).
_SHELL_ESCAPE_RE = re.compile(r"^\s*!{1,2}(?P<rest>.*)$")
# ``%sx``, ``%system`` line magics and equivalents that shell out.
_SHELL_LINE_MAGIC_RE = re.compile(r"^\s*%(?:sx|system)\s+(?P<rest>.*)$")
# ``%%bash`` / ``%%sh`` cell magic header — the whole cell body is shell.
_SHELL_CELL_MAGIC_RE = re.compile(r"^\s*%%(?:bash|sh)\b")


def find_blocked_in_ipython(code: str) -> str | None:
    """Scan IPython ``code`` for blocked commands in shell-escape lines.

    Detects:

    - ``!cmd`` / ``!!cmd`` shell escapes on any line
    - ``%sx`` / ``%system`` line magics
    - ``%%bash`` / ``%%sh`` cell magic — whole cell body is bash

    Each extracted bash fragment is passed through ``find_blocked_command``
    so chained calls (``!cd repo && git log``) refuse with the same
    predicate as the bash tool. Pure Python (``subprocess.run([...])``,
    ``os.system(...)``) is *not* detected here — see PR description for
    the documented bypass.
    """
    if allow_git():
        return None

    lines = code.splitlines()
    in_bash_cell = False
    for line in lines:
        if in_bash_cell:
            blocked = find_blocked_command(line)
            if blocked is not None:
                return blocked
            continue
        if _SHELL_CELL_MAGIC_RE.match(line):
            in_bash_cell = True
            continue
        m = _SHELL_ESCAPE_RE.match(line) or _SHELL_LINE_MAGIC_RE.match(line)
        if m:
            blocked = find_blocked_command(m.group("rest"))
            if blocked is not None:
                return blocked
    return None
