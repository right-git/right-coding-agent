"""The process-wide default registry and the tools it ships with."""

from ..computer.tool import COMPUTER_TOOLS, VISION_TOOLS
from ..files.tool import FILE_TOOLS
from ..parser.tool import web_fetch, web_search
from ..shell.tool import bash
from .registry import ToolRegistry

_registry: ToolRegistry | None = None


def default_tools() -> list:
    """Every default tool, minus the ones whose heavy models are disabled.

    Without ENABLE_VISION_MODEL the locator-driven screen tools are not
    registered at all — nothing in the process can pull the multi-GB vision
    model; screenshots, typing, keys, and scrolling stay available.
    """
    # Imported here so `import src.llm.tools` keeps working without a .env.
    from src.config.settings import settings

    tools = [web_fetch, web_search, *FILE_TOOLS, bash, *COMPUTER_TOOLS]
    if not settings.enable_vision_model:
        vision_names = {tool.name for tool in VISION_TOOLS}
        tools = [tool for tool in tools if tool.name not in vision_names]
    return tools


def get_registry() -> ToolRegistry:
    """The process-wide tool registry, created on first use."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry(default_tools())
    return _registry


def set_registry(registry: ToolRegistry | None) -> None:
    """Replace the shared registry (used by tests and embedders)."""
    global _registry
    _registry = registry
