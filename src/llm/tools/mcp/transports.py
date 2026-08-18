"""The one module that touches SDK transport entry points directly.

Verified live against the installed SDK, `mcp` 2.0.0 (see
`tests/test_mcp_sdk_surface.py`) — several names and shapes moved since the
1.x-era draft this was written against:

- The streamable-http export is `streamable_http_client` (not
  `streamablehttp_client`).
- `streamable_http_client(url, *, http_client=None, terminate_on_close=True)`
  takes no `headers`/`auth` kwargs at all — those now live on the underlying
  `httpx2.AsyncClient`. Headers/auth for the http transport go into a client
  built by `create_mcp_http_client` and handed in via `http_client=`; because
  the SDK only manages that client's lifecycle when it built it itself
  (`http_client=None`), a caller-supplied client must be opened and closed
  here too — otherwise a reconnecting persistent connection leaks a socket
  per attempt.
- `streamable_http_client` yields a 2-tuple `(read, write)`, not a 1.x
  3-tuple with a `get_session_id` callable.
- `sse_client` is unchanged in shape from the 1.x draft: it still takes
  `(url, headers=, auth=)` directly and yields a 2-tuple. SSE is legacy
  transport; header/OAuth auth both still work there without the
  httpx-client indirection the http transport now needs.
"""

from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from .config import McpServerConfig


@asynccontextmanager
async def default_session_factory(config: McpServerConfig, auth: Any | None = None):
    """Enter one transport + `ClientSession` for `config`; yields the session.

    The session is entered but not initialized — `McpManager` calls
    `session.initialize()` itself so it can bound that step with its own
    timeout.
    """
    if config.transport == "stdio":
        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env={**get_default_environment(), **config.env},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
        return
    if config.transport == "http":
        http_client = create_mcp_http_client(headers=config.headers or None, auth=auth)
        async with http_client:
            async with streamable_http_client(config.url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session
        return
    if config.transport == "sse":
        async with sse_client(config.url, headers=config.headers or None, auth=auth) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
        return
    raise ValueError(f"Unsupported MCP transport: {config.transport}")
