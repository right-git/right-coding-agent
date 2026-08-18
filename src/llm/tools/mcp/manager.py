"""Persistent MCP connections: one background task per configured server.

The task enters the transport + session context, initializes, registers the
adapted tools into the shared ToolRegistry, then parks on a stop event; the
context managers unwind in the same task that entered them (anyio requires
it). A stdio server's subprocess therefore lives for the whole REPL session.
Every failure is contained: statuses record it, callers get error strings,
and nothing here ever breaks startup or a turn.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import Any

from src.config.logging import logger
from src.config.settings import settings

from ..meta.defaults import get_registry
from ..meta.registry import MCP_SOURCE_PREFIX, ToolRegistry
from .adapter import build_mcp_tool, build_prompt_command
from .config import McpServerConfig, load_mcp_servers

# A stopping connection gets this long to unwind its contexts before it is
# cancelled outright — a wedged stdio child must never hold up REPL exit.
STOP_TIMEOUT = 5.0
# The session's own read timeout fires first; this outer guard only catches a
# transport that never answers at all.
TOOL_TIMEOUT_GRACE = 5.0


class ServerState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    NEEDS_AUTH = "needs auth"
    DISCONNECTED = "disconnected"


@dataclass
class ServerStatus:
    name: str
    transport: str
    scope: str
    state: ServerState
    error: str | None = None
    tool_count: int = 0
    prompt_count: int = 0
    resource_count: int = 0


@dataclass
class _Connection:
    config: McpServerConfig
    state: ServerState = ServerState.DISCONNECTED
    error: str | None = None
    session: object | None = None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None
    ready: asyncio.Event | None = None
    registered: list[str] = field(default_factory=list)
    tools: list = field(default_factory=list)
    prompts: list = field(default_factory=list)
    resources: list = field(default_factory=list)


@asynccontextmanager
async def _unavailable_session_factory(config: McpServerConfig, auth: Any | None = None):
    """Placeholder default until the real transports module lands.

    Raising here (rather than at import time) keeps `import manager` working
    everywhere; tests inject their own factory.
    """
    raise RuntimeError("no session factory configured — MCP transports are not wired up yet")
    yield None  # pragma: no cover - unreachable; keeps this an async generator


def _first_sentence(text: str) -> str:
    """First sentence of a description, whitespace-normalized."""
    normalized = " ".join((text or "").split())
    head, separator, _ = normalized.partition(". ")
    return f"{head}." if separator else normalized


def _read_field(value: Any, name: str, default=None):
    """Read an attribute or dict key, tolerating either shape."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class McpManager:
    """Owns one persistent connection per configured MCP server."""

    def __init__(
        self,
        configs: dict[str, McpServerConfig] | None = None,
        session_factory: Callable[..., Any] | None = None,
        registry: ToolRegistry | None = None,
        on_status: Callable[[str, ServerState], None] | None = None,
    ) -> None:
        self._configs = dict(configs) if configs is not None else load_mcp_servers()
        self._session_factory = session_factory or _unavailable_session_factory
        self._registry = registry
        # Public: the REPL assigns its status reporter after construction.
        self.on_status = on_status
        self._connections: dict[str, _Connection] = {
            name: _Connection(config) for name, config in self._configs.items()
        }

    # ---------------------------------------------------------------- state

    def _registry_or_default(self) -> ToolRegistry:
        return self._registry if self._registry is not None else get_registry()

    def _require(self, name: str) -> _Connection:
        connection = self._connections.get(name)
        if connection is None:
            raise ValueError(f"unknown MCP server '{name}'")
        return connection

    def _set_state(self, conn: _Connection, state: ServerState, error: str | None = None) -> None:
        conn.state = state
        conn.error = error
        logger.info("MCP server [{}] is {}{}", conn.config.name, state.value, f": {error}" if error else "")
        if self.on_status is None:
            return
        try:
            self.on_status(conn.config.name, state)
        except Exception:
            logger.exception("MCP status callback failed for server [{}]", conn.config.name)

    def _auth_for(self, config: McpServerConfig) -> Any | None:
        """OAuth provider for this server; none until the oauth layer lands."""
        return None

    def _failure_state(self, config: McpServerConfig, error: Exception) -> ServerState:
        """Which state a connection failure lands in (oauth adds NEEDS_AUTH)."""
        return ServerState.FAILED

    # ----------------------------------------------------------- connection

    async def _load_inventory(self, conn: _Connection, session: Any) -> None:
        """Tools, prompts, and resources of one server.

        Only `list_tools` is required: a server legitimately lacking the
        prompts or resources capability answers those with an error.
        """
        listed = await session.list_tools()
        conn.tools = list(_read_field(listed, "tools", []) or [])
        conn.prompts = []
        conn.resources = []
        try:
            listed_prompts = await session.list_prompts()
            conn.prompts = list(_read_field(listed_prompts, "prompts", []) or [])
        except Exception as error:
            logger.debug("MCP server [{}] lists no prompts: {}", conn.config.name, error)
        try:
            listed_resources = await session.list_resources()
            conn.resources = list(_read_field(listed_resources, "resources", []) or [])
        except Exception as error:
            logger.debug("MCP server [{}] lists no resources: {}", conn.config.name, error)

    def _register_tools(self, conn: _Connection) -> None:
        """Adapt every remote tool into the registry under `mcp:<server>`."""
        self._unregister_tools(conn)
        registry = self._registry_or_default()
        source = f"{MCP_SOURCE_PREFIX}{conn.config.name}"
        # The reconnect-retry wrapper, not the bare session call: a session
        # that died between turns heals itself on the next script call.
        call = partial(self.call_tool, conn.config.name)
        for remote_tool in conn.tools:
            try:
                tool_obj = build_mcp_tool(conn.config.name, remote_tool, call)
                # Reconnects must not hit the registry's duplicate-name guard.
                registry.unregister(tool_obj.name)
                registry.register(tool_obj, source=source)
                conn.registered.append(tool_obj.name)
            except Exception:
                logger.exception(
                    "Failed to register MCP tool server [{}] tool [{}]",
                    conn.config.name,
                    _read_field(remote_tool, "name"),
                )

    def _unregister_tools(self, conn: _Connection) -> None:
        if not conn.registered:
            return
        try:
            registry = self._registry_or_default()
            for name in conn.registered:
                registry.unregister(name)
        except Exception:
            logger.exception("Failed to unregister MCP tools of server [{}]", conn.config.name)
        conn.registered = []

    async def _run_connection(self, conn: _Connection) -> None:
        """The owning task: enters, serves, and unwinds one session."""
        conn.stop_event = asyncio.Event()
        try:
            async with self._session_factory(conn.config, self._auth_for(conn.config)) as session:
                await asyncio.wait_for(session.initialize(), timeout=settings.mcp_connect_timeout)
                conn.session = session
                await asyncio.wait_for(self._load_inventory(conn, session), timeout=settings.mcp_connect_timeout)
                self._register_tools(conn)
                self._set_state(conn, ServerState.CONNECTED)
                conn.ready.set()
                await conn.stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("MCP server [{}] failed: {}", conn.config.name, error)
            self._set_state(conn, self._failure_state(conn.config, error), str(error))
        finally:
            self._unregister_tools(conn)
            conn.session = None
            if conn.ready is not None:
                conn.ready.set()
            if conn.state == ServerState.CONNECTED:
                self._set_state(conn, ServerState.DISCONNECTED)

    async def _connect(self, conn: _Connection) -> None:
        if conn.task is not None and not conn.task.done():
            return
        conn.ready = asyncio.Event()
        self._set_state(conn, ServerState.CONNECTING)
        conn.task = asyncio.create_task(self._run_connection(conn), name=f"mcp:{conn.config.name}")
        await conn.ready.wait()

    async def _disconnect(self, conn: _Connection) -> None:
        task, conn.task = conn.task, None
        if task is None:
            return
        if conn.stop_event is not None:
            conn.stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=STOP_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning("MCP server [{}] did not stop in time; cancelled", conn.config.name)
        except Exception as error:
            logger.warning("MCP server [{}] stopped with an error: {}", conn.config.name, error)

    # -------------------------------------------------------------- control

    async def start(self) -> None:
        """Connect every configured server concurrently; never raises."""
        if not self._connections:
            return
        results = await asyncio.gather(
            *(self._connect(conn) for conn in self._connections.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("MCP connect raised: {}", result)

    async def stop(self) -> None:
        """Stop every connection; tools unregister as the tasks unwind."""
        if not self._connections:
            return
        await asyncio.gather(
            *(self._disconnect(conn) for conn in self._connections.values()),
            return_exceptions=True,
        )

    async def reconnect(self, name: str) -> ServerStatus:
        conn = self._require(name)
        await self._disconnect(conn)
        await self._connect(conn)
        return self._status_of(conn)

    # ----------------------------------------------------------------- rpcs

    def _session_of(self, server: str) -> tuple[_Connection, Any]:
        conn = self._require(server)
        if conn.session is None:
            raise ConnectionError(
                f"MCP server '{server}' is {conn.state.value}" + (f": {conn.error}" if conn.error else "")
            )
        return conn, conn.session

    async def _call_via_session(self, server: str, tool: str, arguments: dict) -> Any:
        _, session = self._session_of(server)
        return await asyncio.wait_for(
            session.call_tool(tool, arguments, read_timeout_seconds=timedelta(seconds=settings.mcp_tool_timeout)),
            timeout=settings.mcp_tool_timeout + TOOL_TIMEOUT_GRACE,
        )

    async def call_tool(self, server: str, tool: str, arguments: dict) -> Any:
        """One tool call, healing a dead session with a single reconnect."""
        conn = self._require(server)
        try:
            return await self._call_via_session(server, tool, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as first_error:
            logger.warning("MCP call failed, reconnecting once server [{}] tool [{}]: {}", server, tool, first_error)
            await self.reconnect(server)
            if conn.state != ServerState.CONNECTED:
                raise ConnectionError(f"server '{server}' is {conn.state.value}: {conn.error}") from first_error
            return await self._call_via_session(server, tool, arguments)

    async def get_prompt(self, server: str, prompt: str, arguments: dict) -> str:
        """One prompt, flattened to the text the REPL sends as a user turn."""
        _, session = self._session_of(server)
        result = await asyncio.wait_for(
            session.get_prompt(prompt, arguments or None),
            timeout=settings.mcp_tool_timeout + TOOL_TIMEOUT_GRACE,
        )
        parts: list[str] = []
        for message in _read_field(result, "messages", []) or []:
            content = _read_field(message, "content")
            text = _read_field(content, "text")
            parts.append(str(text) if text is not None else str(content))
        return "\n\n".join(part for part in parts if part)

    async def list_resources(self, server: str | None = None) -> list[dict]:
        """Cached resource inventory of one or of every connected server."""
        if server is not None:
            connections = [self._require(server)]
        else:
            connections = [conn for conn in self._connections.values() if conn.state == ServerState.CONNECTED]
        rows: list[dict] = []
        for conn in connections:
            for resource in conn.resources:
                rows.append(
                    {
                        "server": conn.config.name,
                        "uri": str(_read_field(resource, "uri", "") or ""),
                        "name": _read_field(resource, "name", "") or "",
                        "description": _read_field(resource, "description", "") or "",
                        "mime_type": _read_field(resource, "mimeType", "") or "",
                    }
                )
        return rows

    async def read_resource(self, server: str, uri: str) -> Any:
        """The raw SDK read result; the resource tool serializes it."""
        _, session = self._session_of(server)
        return await asyncio.wait_for(
            session.read_resource(uri),
            timeout=settings.mcp_tool_timeout + TOOL_TIMEOUT_GRACE,
        )

    # ------------------------------------------------------------ reporting

    def _status_of(self, conn: _Connection) -> ServerStatus:
        return ServerStatus(
            name=conn.config.name,
            transport=conn.config.transport,
            scope=conn.config.scope,
            state=conn.state,
            error=conn.error,
            tool_count=len(conn.tools),
            prompt_count=len(conn.prompts),
            resource_count=len(conn.resources),
        )

    def statuses(self) -> list[ServerStatus]:
        return [self._status_of(conn) for conn in self._connections.values()]

    def prompt_commands(self) -> list[tuple[str, str]]:
        """`("/mcp__server__prompt", "description")` for every known prompt."""
        commands: list[tuple[str, str]] = []
        for conn in self._connections.values():
            for prompt in conn.prompts:
                name = str(_read_field(prompt, "name", "") or "")
                if not name:
                    continue
                command = build_prompt_command(conn.config.name, name)
                commands.append((command, _first_sentence(_read_field(prompt, "description", "") or "")))
        return commands

    def find_prompt(self, command: str) -> tuple[str, Any] | None:
        """The (server, prompt) behind a slash command, or None.

        Name sanitization is not reversible, so each candidate's command is
        rebuilt and compared instead of parsing the command apart.
        """
        wanted = (command or "").strip()
        if not wanted:
            return None
        for conn in self._connections.values():
            for prompt in conn.prompts:
                name = str(_read_field(prompt, "name", "") or "")
                if name and build_prompt_command(conn.config.name, name) == wanted:
                    return conn.config.name, prompt
        return None


_manager: McpManager | None = None


def get_mcp_manager() -> McpManager:
    """The process-wide manager, built from the configured servers."""
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager


def set_mcp_manager(manager: McpManager | None) -> None:
    """Replace the shared manager (used by tests and embedders)."""
    global _manager
    _manager = manager
