"""The process-wide default registry and the tools it ships with."""

from ..computer.tool import COMPUTER_TOOLS
from ..files.tool import FILE_TOOLS
from ..parser.tool import web_fetch, web_search
from ..shell.tool import bash
from .registry import ToolRegistry

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """The process-wide tool registry, created on first use."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry([web_fetch, web_search, *FILE_TOOLS, bash, *COMPUTER_TOOLS])
    return _registry


def set_registry(registry: ToolRegistry | None) -> None:
    """Replace the shared registry (used by tests and embedders)."""
    global _registry
    _registry = registry
