"""The builtin environment scripts run against.

Names listed here must stay in sync with `RESERVED_SCRIPT_NAMES` in
`src.llm.tools.meta.registry` — a registered tool shadowed by a builtin would be
silently unreachable from scripts.
"""

import asyncio
import random as _random
import time
from typing import Any

from .errors import SandboxError
from .guards import safe_range
from .limits import MAX_SLEEP_CALL, MAX_TOTAL_SLEEP


def make_builtins(interp) -> dict[str, Any]:
    async def _sleep(seconds):
        seconds = float(seconds)
        if seconds < 0 or seconds > MAX_SLEEP_CALL:
            raise SandboxError(f"sleep() must be 0..{MAX_SLEEP_CALL}s")
        interp.total_sleep += seconds
        if interp.total_sleep > MAX_TOTAL_SLEEP:
            raise SandboxError(f"total sleep budget exceeded ({MAX_TOTAL_SLEEP}s)")
        await asyncio.sleep(seconds)

    return {
        # async primitives
        "sleep": _sleep,
        # numeric / misc
        "random": _random.random,
        "randint": _random.randint,
        "choice": _random.choice,
        "now": time.time,
        # pure builtins
        "len": len,
        "range": safe_range,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "reversed": lambda x: list(reversed(x)),
        "enumerate": enumerate,
        "zip": zip,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type_name": lambda x: type(x).__name__,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": None,  # replaced per-run: appends to interp.logs
        # exception class allowed in `except`
        "Exception": Exception,
    }
