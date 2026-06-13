"""Restrict history-wide ``git log`` access at the tool-call level.

Ordinary git commands are allowed. The guard refuses ``git log`` invocations
that ask for non-current-branch history, such as ``git log --all``. This keeps
the agent's visibility close to current-branch history while preserving useful
commands like ``git status`` and ``git diff``.

- split a bash command on ``&&``, ``||``, ``;`` and ``|``
- if a segment invokes ``git log`` with a restricted history flag, refuse

The ``RLM_ALLOW_GIT=1`` env var disables the check entirely for backwards
compatibility with environments that already opted out of git restrictions.
"""

from __future__ import annotations

import ast
import re
import shlex

from rlm.config import get_config

REFUSAL_TEMPLATE = (
    "Git history option '{cmd}' is not allowed. Use current-branch history only."
)

# Reuse the mini_swe_agent_plus separators verbatim so behavior matches.
SEPARATORS = re.compile(r"&&|\|\||;|\|")

RESTRICTED_LOG_OPTIONS = {
    "--all",
    "-all",
    "--alternate-refs",
    "--reflog",
    "--walk-reflogs",
    "-g",
}
RESTRICTED_LOG_OPTION_PREFIXES = (
    "--branches",
    "--glob",
    "--remotes",
    "--tags",
)

GIT_GLOBAL_OPTIONS_WITH_VALUE = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--work-tree",
}


def allow_git() -> bool:
    return get_config().allow_git


def find_blocked_command(command: str) -> str | None:
    """Return the offending token if ``command`` asks for broad git history.

    Splits on ``&&``, ``||``, ``;``, ``|`` so chained calls like
    ``cd /repo && git log --all`` are caught. Returns ``None`` if nothing is
    blocked or if ``RLM_ALLOW_GIT=1``.
    """
    if allow_git():
        return None
    for segment in SEPARATORS.split(command):
        blocked = find_blocked_git_log_option(split_segment(segment))
        if blocked is not None:
            return blocked
    return None


def refusal(cmd: str) -> str:
    return REFUSAL_TEMPLATE.format(cmd=cmd)


def split_segment(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.strip().split()


def is_git_binary(token: str) -> bool:
    return token == "git" or token.rsplit("/", 1)[-1] == "git"


def skip_git_global_options(argv: list[str], index: int) -> int:
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return index + 1
        if not token.startswith("-"):
            return index

        option = token.split("=", 1)[0]
        if option in GIT_GLOBAL_OPTIONS_WITH_VALUE and "=" not in token:
            index += 2
        else:
            index += 1
    return index


def is_restricted_log_option(token: str) -> bool:
    if token in RESTRICTED_LOG_OPTIONS:
        return True
    return any(
        token == option or token.startswith(f"{option}=")
        for option in RESTRICTED_LOG_OPTION_PREFIXES
    )


def find_blocked_git_log_option(argv: list[str]) -> str | None:
    if not argv or not is_git_binary(argv[0]):
        return None

    subcommand_index = skip_git_global_options(argv, 1)
    if subcommand_index >= len(argv) or argv[subcommand_index] != "log":
        return None

    for token in argv[subcommand_index + 1 :]:
        if token == "--":
            return None
        if is_restricted_log_option(token):
            return token
    return None


# IPython shell-escape lines: ``!cmd`` and ``!!cmd``. Leading whitespace
# is allowed (IPython accepts indented shell escapes inside blocks).
SHELL_ESCAPE_RE = re.compile(r"^\s*!{1,2}(?P<rest>.*)$")
# ``%sx``, ``%system`` line magics and equivalents that shell out.
SHELL_LINE_MAGIC_RE = re.compile(r"^\s*%(?:sx|system)\s+(?P<rest>.*)$")
# ``%%bash`` / ``%%sh`` cell magic header — the whole cell body is shell.
SHELL_CELL_MAGIC_RE = re.compile(r"^\s*%%(?:bash|sh)\b")
# Any IPython line magic — used by the AST pre-pass to drop ipython-only
# lines so ``ast.parse`` doesn't choke on them.
ANY_LINE_MAGIC_RE = re.compile(r"^\s*%[A-Za-z]")
# Any IPython cell magic header — same purpose.
ANY_CELL_MAGIC_RE = re.compile(r"^\s*%%[A-Za-z]")


def find_blocked_in_ipython(code: str) -> str | None:
    """Scan IPython ``code`` for blocked commands.

    Two passes, both honoring ``RLM_ALLOW_GIT=1``:

    1. Shell-escape scan — ``!cmd`` / ``!!cmd``, ``%sx`` / ``%system``
       line magics, ``%%bash`` / ``%%sh`` cell magic. Each extracted
       bash fragment goes through ``find_blocked_command``.
    2. Pure-Python AST scan via :func:`find_blocked_python` — catches
       restricted literal subprocess / ``os.system`` git-log invocations
       and the obvious aliases. See that function for documented bypasses
       (dynamic ``getattr``, multi-hop reassignment, etc.).
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
        if SHELL_CELL_MAGIC_RE.match(line):
            in_bash_cell = True
            continue
        m = SHELL_ESCAPE_RE.match(line) or SHELL_LINE_MAGIC_RE.match(line)
        if m:
            blocked = find_blocked_command(m.group("rest"))
            if blocked is not None:
                return blocked

    return find_blocked_python(code)


# Statically-resolved fully-qualified callees that shell out when invoked
# with a literal first positional argument.
BLOCKED_PY_CALLS = frozenset(
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


def blocked_option_from_python_call(node: ast.Call) -> str | None:
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return find_blocked_command(arg.value)
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        argv: list[str] = []
        for elt in arg.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                return None
            argv.append(elt.value)
        return find_blocked_git_log_option(argv)
    return None


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
        self.found: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in {"subprocess", "os"}:
                self.module_aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"subprocess", "os"}:
            for alias in node.names:
                fqn = f"{node.module}.{alias.name}"
                if fqn in BLOCKED_PY_CALLS:
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
        if fqn in BLOCKED_PY_CALLS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.callable_aliases[target.id] = fqn
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fqn = self._resolve_callee(node.func)
        if fqn in BLOCKED_PY_CALLS:
            blocked = blocked_option_from_python_call(node)
            if blocked is not None:
                self.found = blocked
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


def strip_ipython_only(code: str) -> str:
    """Drop ipython-only lines so the remainder is pure Python for ``ast.parse``.

    Removes ``!cmd`` / ``!!cmd`` shell escapes, line magics (``%foo``),
    and any ``%%cellmagic`` header plus its body. Trailing ``?`` / ``??``
    object-inspection markers are stripped from the line tail rather
    than dropping the whole line, so ``subprocess.run?`` becomes
    ``subprocess.run`` and still parses. All other Python lines are
    preserved verbatim.
    """
    out: list[str] = []
    for line in code.splitlines():
        # Drop only the cell-magic HEADER, not the body — magics like
        # ``%%timeit`` / ``%%capture`` execute their body as Python and
        # would otherwise hide ``subprocess.run([\"git\", ...])`` calls.
        # Bash-bodied magics (``%%bash`` / ``%%sh``) are caught earlier
        # by the shell-escape pre-pass, so dropping just the header here
        # is safe.
        if ANY_CELL_MAGIC_RE.match(line):
            continue
        if SHELL_ESCAPE_RE.match(line) or ANY_LINE_MAGIC_RE.match(line):
            continue
        stripped = line.rstrip()
        if stripped.endswith("?"):
            line = stripped.rstrip("?")
        out.append(line)
    return "\n".join(out)


def find_blocked_python(code: str) -> str | None:
    """Detect restricted pure-Python git invocations via AST walk.

    Returns the offending token (``\"--all\"`` etc.) if a blocked call is found,
    else ``None``. Honors ``RLM_ALLOW_GIT=1``. Ipython-only syntax
    (``!cmd``, ``%magic``, ``obj?``) is stripped before parsing so
    cells mixing ipython and Python still get scanned. Returns ``None``
    on ``SyntaxError`` so the normal exec path surfaces the parse error.
    """
    if allow_git():
        return None
    try:
        tree = ast.parse(strip_ipython_only(code))
    except SyntaxError:
        return None
    finder = _GitCallFinder()
    finder.visit(tree)
    return finder.found
