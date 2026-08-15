"""LangChain tool for running shell commands."""

import asyncio

from langchain_core.tools import tool

from .service import DEFAULT_TIMEOUT, CommandRunner

_runner = CommandRunner()


@tool(parse_docstring=True, return_direct=False)
async def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute a shell command and return its combined stdout and stderr.

    Commands are stateless — the working directory and environment reset
    between calls, so chain dependent steps with `&&` inside one command.
    A non-zero exit code is reported as an `[exit code N]` prefix.

    Args:
        command: The command to run, for example `uv run python -m unittest discover -s tests`.
        timeout: Seconds to wait before killing the command (default 30).

    Returns:
        The command output, truncated when very long, or an error message.
    """
    try:
        return await asyncio.to_thread(_runner.run, command, timeout)
    except Exception as error:
        return f"Tool call failed, error: {error}"
