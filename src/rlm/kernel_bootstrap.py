"""Kernel-side bootstrap: wrap skills and ``rlm`` as await-callable modules.

Run inside the IPython kernel by ``IPythonREPL._inject_startup``. The injected
cell sets the per-kernel env vars, applies ``nest_asyncio``, then calls
:func:`build_namespace` and merges the result into the kernel's global namespace
(so the model can ``await <skill>(...)`` / ``await rlm(...)`` directly).

This lives in a real module — rather than an exec'd string literal — so the
wrapping logic is importable, unit-testable, and type-checked. ``_inject_startup``
keeps only a tiny stub that can't be moved here, because ``globals().update(...)``
must run in the kernel's interactive namespace, not this module's.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import time
import types

from rlm.async_runtime import attach_background


def log_programmatic_call(tool_name: str, source: str) -> None:
    """Append one programmatic-tool-call record to the session log.

    Matches the line format written by ``install.sh``'s bash wrapper so
    ``ProgrammaticToolCallStats.from_log`` parses both sources identically.
    """
    session_dir = os.environ.get("RLM_SESSION_DIR", "")
    if not session_dir:
        return
    try:
        with open(os.path.join(session_dir, "programmatic_tool_calls.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {"tool": tool_name, "source": source, "timestamp": time.time()}
                )
                + "\n"
            )
    except OSError:
        pass


class _CallableModule(types.ModuleType):
    """Module subclass making ``await <mod>(...)`` shorthand for ``<mod>.run(...)``.

    ``__call__`` is looked up on the type, not the instance, so the override has
    to live on the class.
    """

    async def __call__(self, *args, **kwargs):
        return await self.run(*args, **kwargs)


def wrap_callable(mod, log_source: str | None):
    """Return an await-callable clone of ``mod`` that exposes its ``run`` API.

    ``log_source`` is ``'python'`` for skills (each call is logged to
    ``programmatic_tool_calls.jsonl``) and ``None`` for ``rlm`` (already
    aggregated via ``Session.aggregate_child_metrics``).
    """
    wrapped = _CallableModule(mod.__name__)
    wrapped.__dict__.update(mod.__dict__)
    if log_source is not None:
        _original_run = wrapped.run

        @functools.wraps(_original_run)
        async def _logged_run(*args, **kwargs):
            log_programmatic_call(mod.__name__, log_source)
            return await _original_run(*args, **kwargs)

        wrapped.run = _logged_run
    # Mirror run's signature and docstring onto the module so
    # ``inspect.signature(<skill>)`` and ``help(<skill>)`` expose the real API
    # surface instead of ``_CallableModule.__call__``'s ``(*args, **kwargs)`` and
    # the file-level module docstring.
    wrapped.__signature__ = inspect.signature(wrapped.run)
    wrapped.__doc__ = wrapped.run.__doc__
    sys.modules[mod.__name__] = wrapped
    return wrapped


def build_namespace(installed_skills: list[str], allow_recursion: bool) -> dict:
    """Wrap each installed skill (and ``rlm`` when recursion is allowed).

    Returns the ``{name: module}`` mapping the caller injects into the kernel's
    global namespace.
    """
    namespace: dict = {}
    for name in installed_skills:
        skill = wrap_callable(__import__(name), "python")
        attach_background(skill, skill.run)
        namespace[name] = skill
    if allow_recursion:
        namespace["rlm"] = wrap_callable(__import__("rlm"), None)
    return namespace
