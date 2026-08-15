"""Error taxonomy of the sandbox.

The split matters to script authors (the LLM): `ExecError` is an ordinary
runtime failure their `try/except Exception` may handle, while `SandboxError`
is a policy or budget violation that always aborts the run.
"""


class SandboxError(Exception):
    """FATAL: policy violation / resource budget. Not catchable by LLM code."""


class ExecError(Exception):
    """Runtime error (tool failure, bad key, etc). Catchable by LLM try/except."""
