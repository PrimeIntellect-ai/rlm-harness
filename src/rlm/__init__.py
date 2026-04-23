"""rlm — A minimalistic CLI agent for true recursion."""

from rlm.api import run
from rlm.engine import RLMEngine
from rlm.skill import callable_module
from rlm.types import RLMMetrics, RLMResult

__all__ = ["callable_module", "run", "RLMEngine", "RLMMetrics", "RLMResult"]

# Opt the rlm module itself into the callable-shorthand helper we ship
# for skill authors, so `await rlm('sub-task')` works identically to
# `await rlm.run('sub-task')` inside an IPython cell.
callable_module(__name__)
