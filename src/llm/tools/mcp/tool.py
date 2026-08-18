"""LangChain tools that expose MCP resources: list them, read one.

Resources are static or dynamic content an MCP server hands back by URI
(config files, logs, database rows, ...) — distinct from remote MCP tools
(adapted straight into the registry by `adapter.py`) and prompts (surfaced
as slash commands by `McpManager.prompt_commands`). These two ordinary
registry tools are the model's only way to browse and pull resources in.

Both import the manager lazily (inside the function body) so this module
never has to import `McpManager` — or anything from `meta` beyond
`attach_image` — at import time, keeping the `meta` <-> `mcp` import cycle
one-directional.
"""

from langchain_core.tools import tool

from ..meta.attachments import attach_image


def _read_field(value, name, default=None):
    """Read an attribute or dict key, tolerating either shape."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _serialize_resource_content(item, *, server: str, parts: list[str]) -> None:
    """Render one resource content block into `parts`, routing images out of band."""
    text = _read_field(item, "text")
    if text is not None:
        parts.append(str(text))
        return
    blob = _read_field(item, "blob")
    mime = _read_field(item, "mimeType") or ""
    if blob is None:
        parts.append("(empty resource content)")
        return
    if mime.startswith("image/"):
        uri = str(_read_field(item, "uri", "") or "")
        if attach_image(blob, mime, label=f"{server}:{uri}"):
            parts.append("[image attached — you will see it right after this result]")
        else:
            parts.append(f"[image resource ({mime}, {len(blob)} base64 chars) — no attachment channel open]")
        return
    parts.append(f"[binary resource ({mime}, {len(blob)} chars)]")


@tool(parse_docstring=True)
async def mcp_list_resources(server: str = "") -> str:
    """List resources exposed by connected MCP servers.

    Resources are content a server can hand back by URI — config, logs,
    records, and the like. Use mcp_read_resource with a uri from this list
    to fetch one.

    Args:
        server: Name of one connected MCP server to list, or empty for
            every connected server.

    Returns:
        One line per resource as "server uri (mime) name - description",
        or a "[mcp error] ..." string on failure.
    """
    try:
        from .manager import get_mcp_manager

        manager = get_mcp_manager()
        resources = await manager.list_resources(server or None)
        if not resources:
            return "No resources available."
        lines = []
        for resource in resources:
            mime = resource.get("mime_type") or "unknown"
            name = resource.get("name") or ""
            description = resource.get("description") or ""
            lines.append(f"{resource.get('server', '')} {resource.get('uri', '')} ({mime}) {name} - {description}")
        return "\n".join(lines)
    except Exception as error:
        return f"[mcp error] {error}"


@tool(parse_docstring=True)
async def mcp_read_resource(server: str, uri: str) -> str:
    """Read one resource from a connected MCP server by its uri.

    Get the uri from mcp_list_resources first. Text content comes back as
    plain text, image content is attached for you to see, and other binary
    content is reported as a size and mime type rather than fetched inline.

    Args:
        server: Name of the connected MCP server that owns the resource.
        uri: The resource's uri, exactly as listed by mcp_list_resources.

    Returns:
        The resource content as text, or a "[mcp error] ..." string on
        failure.
    """
    try:
        from .manager import get_mcp_manager

        manager = get_mcp_manager()
        result = await manager.read_resource(server, uri)
        parts: list[str] = []
        for item in _read_field(result, "contents", []) or []:
            _serialize_resource_content(item, server=server, parts=parts)
        return "\n".join(parts) if parts else "(empty resource)"
    except Exception as error:
        return f"[mcp error] {error}"


MCP_SERVICE_TOOLS = [mcp_list_resources, mcp_read_resource]
