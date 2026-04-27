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

import ast
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
    """Scan IPython ``code`` for blocked commands.

    Two passes, both honoring ``RLM_ALLOW_GIT=1``:

    1. Shell-escape scan — ``!cmd`` / ``!!cmd``, ``%sx`` / ``%system``
       line magics, ``%%bash`` / ``%%sh`` cell magic. Each extracted
       bash fragment goes through ``find_blocked_command``.
    2. Pure-Python AST scan via :func:`find_blocked_python` — catches
       ``subprocess.run([\"git\", ...])`` / ``os.system(\"git ...\")``
       and the obvious aliases. See that function for documented
       bypasses (dynamic ``getattr``, multi-hop reassignment, etc.).
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

    return find_blocked_python(code)


# Statically-resolved fully-qualified callees that shell out to ``git``
# when invoked with a literal ``git ...`` first positional argument.
_BLOCKED_PY_CALLS = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "os.system",
        "os.popen",
    }
)


def _first_arg_starts_with_git(node: ast.Call) -> bool:
    """Return True iff ``node.args[0]`` is a literal that begins with ``git``."""
    if not node.args:
        return False
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        tokens = arg.value.strip().split()
        return bool(tokens) and tokens[0] in _BLOCKED
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        first = arg.elts[0]
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value in _BLOCKED
        )
    return False


class _GitCallFinder(ast.NodeVisitor):
    """Single-pass AST walker tracking simple aliases for blocked callees.

    Tracks three alias sources:

    - ``import subprocess as sp`` — module-level name remap.
    - ``from subprocess import run`` — bare-name binding.
    - ``r = subprocess.run`` — single-hop assignment of a known callee.

    Multi-hop chains (``r1 = subprocess.run; r2 = r1; r2(...)``) and
    dynamic forms (``getattr(subprocess, \"run\")(...)``,
    ``__import__(\"subprocess\").run(...)``) are explicitly out of scope.
    """

    def __init__(self) -> None:
        # Maps local name -> canonical "module.attr" string.
        self.module_aliases: dict[str, str] = {"subprocess": "subprocess", "os": "os"}
        # Maps local name -> blocked callee fqn (e.g. "run" -> "subprocess.run").
        self.callable_aliases: dict[str, str] = {}
        self.found = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in {"subprocess", "os"}:
                self.module_aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"subprocess", "os"}:
            for alias in node.names:
                fqn = f"{node.module}.{alias.name}"
                if fqn in _BLOCKED_PY_CALLS:
                    self.callable_aliases[alias.asname or alias.name] = fqn
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only track single-hop aliases: ``r = subprocess.run``. Chained
        # reassignment (``r2 = r1``) is intentionally not propagated.
        fqn = None
        if isinstance(node.value, ast.Attribute) and isinstance(
            node.value.value, ast.Name
        ):
            module = self.module_aliases.get(node.value.value.id)
            if module is not None:
                fqn = f"{module}.{node.value.attr}"
        if fqn in _BLOCKED_PY_CALLS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.callable_aliases[target.id] = fqn
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fqn = self._resolve_callee(node.func)
        if fqn in _BLOCKED_PY_CALLS and _first_arg_starts_with_git(node):
            self.found = True
        self.generic_visit(node)

    def _resolve_callee(self, expr: ast.AST) -> str | None:
        """Return ``module.attr`` if ``expr`` resolves to a tracked callee."""
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            module = self.module_aliases.get(expr.value.id)
            if module is not None:
                return f"{module}.{expr.attr}"
        if isinstance(expr, ast.Name):
            return self.callable_aliases.get(expr.id)
        return None


def find_blocked_python(code: str) -> str | None:
    """Detect pure-Python git invocations in an ipython cell via AST walk.

    Returns the offending token (``\"git\"``) if a blocked call is found,
    else ``None``. Honors ``RLM_ALLOW_GIT=1``. Returns ``None`` on
    ``SyntaxError`` so the normal exec path surfaces the parse error.
    """
    if allow_git():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    finder = _GitCallFinder()
    finder.visit(tree)
    return _BLOCKED[0] if finder.found else None
