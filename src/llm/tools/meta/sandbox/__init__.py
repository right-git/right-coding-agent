"""Sandboxed Python-subset interpreter for LLM tool orchestration.

This is the "run_tools" engine: LLM-written scripts are parsed with `ast`
and executed by a tree-walking interpreter against a whitelist of nodes,
builtins, and methods, under strict resource budgets. See `interpreter.py`
for the execution core, `policy.py` / `limits.py` for what is allowed and
how much of it, and `python -m src.llm.tools.meta.sandbox` for a runnable
demo.
"""

from .errors import ExecError, SandboxError
from .interpreter import Interpreter
from .limits import (
    MAX_MEMORY_BYTES,
    MAX_OPS,
    MAX_PARALLEL,
    MAX_SLEEP_CALL,
    MAX_TOTAL_SLEEP,
    MAX_WALL_TIME,
)

__all__ = [
    "ExecError",
    "Interpreter",
    "MAX_MEMORY_BYTES",
    "MAX_OPS",
    "MAX_PARALLEL",
    "MAX_SLEEP_CALL",
    "MAX_TOTAL_SLEEP",
    "MAX_WALL_TIME",
    "SandboxError",
]
