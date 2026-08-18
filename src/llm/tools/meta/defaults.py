"""The process-wide default registry and the tools it ships with."""

from ..computer.tool import COMPUTER_TOOLS, VISION_TOOLS
from ..files.tool import FILE_TOOLS
from ..parser.tool import web_fetch, web_search
from ..shell.tool import bash
from .registry import ToolRegistry

_registry: ToolRegistry | None = None


def _mcp_servers_configured() -> bool:
    """True when at least one MCP server is configured (project or user).

    Imported inside the function — `defaults.py` sits under `meta`, and `mcp`
    imports back into `meta` (registry, attachments), so a module-level
    import here would close an import cycle.
    """
    from ..mcp.config import load_mcp_servers

    try:
        return bool(load_mcp_servers())
    except Exception:
        return False


def default_tools() -> list:
    """Every default tool, minus the ones whose heavy models are disabled.

    Without ENABLE_VISION_MODEL the locator-driven screen tools are not
    registered at all — nothing in the process can pull the multi-GB vision
    model; screenshots, typing, keys, and scrolling stay available. The MCP
    resource tools (`mcp_list_resources`/`mcp_read_resource`) are appended
    only when at least one MCP server is configured, so an agent with no
    `.mcp.json`/`~/.right-agent/mcp.json` never sees tools it cannot use.
    """
    # Imported here so `import src.llm.tools` keeps working without a .env.
    from src.config.settings import settings

    tools = [web_fetch, web_search, *FILE_TOOLS, bash, *COMPUTER_TOOLS]
    if not settings.enable_vision_model:
        vision_names = {tool.name for tool in VISION_TOOLS}
        tools = [tool for tool in tools if tool.name not in vision_names]
    if _mcp_servers_configured():
        from ..mcp.tool import MCP_SERVICE_TOOLS

        tools = [*tools, *MCP_SERVICE_TOOLS]
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
