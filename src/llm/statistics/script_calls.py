"""The per-run counter of registry-tool invocations made by a script.

`run_tools` opens the counter with `counting_script_calls()`; every tool
call the script makes bumps it via `count_script_call()`. The total is
reported in the run's result JSON as `tool_calls`, where the usage footer
(`src.llm.statistics.usage.turn_usage_from_messages`) picks it back up.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_script_call_counter: ContextVar[list[int] | None] = ContextVar("script_tool_calls", default=None)


@contextmanager
def counting_script_calls():
    """Open a tool-call counter for one run_tools call and yield it."""
    counter = [0]
    token = _script_call_counter.set(counter)
    try:
        yield counter
    finally:
        _script_call_counter.reset(token)


def count_script_call() -> None:
    """Bump the active run's counter; a no-op when no run is counting."""
    counter = _script_call_counter.get()
    if counter is not None:
        counter[0] += 1
