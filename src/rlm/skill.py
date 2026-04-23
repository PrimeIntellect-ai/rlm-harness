"""Public helpers for building rlm skills."""

from __future__ import annotations

import sys
import types


def callable_module(name: str) -> None:
    """Make a skill module directly awaitable.

    Call from the skill package's ``__init__.py`` after ``run`` is bound
    at module scope::

        from .edit import PARAMETERS, main, run
        from rlm.skill import callable_module
        callable_module(__name__)

    After this runs, ``await edit(...)`` is equivalent to
    ``await edit.run(...)``; ``edit.run`` and ``edit.PARAMETERS`` stay
    accessible unchanged.
    """

    class _CallableSkill(types.ModuleType):
        async def __call__(self, *args, **kwargs):
            return await self.run(*args, **kwargs)

    sys.modules[name].__class__ = _CallableSkill
