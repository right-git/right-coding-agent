"""Persistent MCP connections: one background task per configured server.

The task enters the transport + session context, initializes, registers the
adapted tools into the shared ToolRegistry, then parks on a stop event; the
context managers unwind in the same task that entered them (anyio requires
it). A stdio server's subprocess therefore lives for the whole REPL session.
Every failure is contained: statuses record it, callers get error strings,
and nothing here ever breaks startup or a turn.

Reconnects are **generational**. Scripts call tools concurrently (`parallel`),
so several callers can discover the same dead session at once: a per-connection
lock serializes teardown+setup, a caller that finds the connection already
healed reuses it instead of tearing it down again, and every connection task
touches the shared `_Connection` only while it is still the current generation
(`conn.task is asyncio.current_task()`). Without that guard a slow-unwinding
old task would unregister the new generation's tools and null its live session.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any

from src.config.logging import logger
from src.config.settings import settings

from ..meta.defaults import get_registry
from ..meta.registry import MCP_SOURCE_PREFIX, ToolRegistry
from .adapter import build_mcp_tool, build_prompt_command
from .config import McpServerConfig, load_mcp_servers
from .oauth import CALLBACK_TIMEOUT, NeedsInteractiveAuth, build_oauth_provider, clear_tokens, has_stored_tokens
from .transports import default_session_factory
from .utils import read_field as _read_field

# A stopping connection gets this long to unwind its contexts before it is
# cancelled outright — a wedged stdio child must never hold up REPL exit.
STOP_TIMEOUT = 5.0
# The session's own read timeout fires first; this outer guard only catches a
# transport that never answers at all.
TOOL_TIMEOUT_GRACE = 5.0
# How much longer than the connect timeout `_connect` waits for the task to
# report ready: the setup steps are individually bounded, but a transport that
# hangs on context *exit* would otherwise never release the waiter, and a hung
# `start()` is a hung REPL startup.
CONNECT_GRACE = 10.0
# Room for the ordinary handshake on either side of the human's consent.
LOGIN_CONNECT_GRACE = 30.0
# An interactive login's own connect budget. The OAuth dance runs lazily inside
# the transport's FIRST request — so it happens inside the connect timeout, and
# the ordinary 30 s one would abort while the user is still typing a password
# (reporting "connection setup timed out", and stopping the callback server out
# from under the browser's pending redirect). Derived from the callback wait so
# the two can never drift apart.
LOGIN_CONNECT_TIMEOUT = CALLBACK_TIMEOUT + LOGIN_CONNECT_GRACE

# Word-bounded so "1401" in an unrelated identifier is not read as a 401.
_UNAUTHORIZED_RE = re.compile(r"\b401\b")


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
    # Bumped on every connect attempt; a task that is no longer the current
    # generation must not touch the shared fields above.
    generation: int = 0
    # Serializes teardown+setup; created lazily inside the running loop.
    lock: asyncio.Lock | None = None
    # One-shot auth provider for the next connect: `login` stashes its
    # interactive (browser-opening) provider here and `_auth_for` consumes it.
    # Set and cleared under the connection lock, so it can never leak into a
    # reconnect that some other caller started.
    override_auth: Any | None = None


def _first_sentence(text: str) -> str:
    """First sentence of a description, whitespace-normalized."""
    normalized = " ".join((text or "").split())
    head, separator, _ = normalized.partition(". ")
    return f"{head}." if separator else normalized


def _walk(error: BaseException | None, depth: int = 8):
    """Yield `error` and every exception it wraps or was caused by.

    A failure raised deep in a transport surfaces several layers away: anyio
    task groups re-raise as `ExceptionGroup`, and httpx/the SDK re-raise with
    the original attached as `__cause__`/`__context__`. Anything that inspects
    only the outermost object is inspecting the wrapper, not the failure.
    """
    if error is None or depth <= 0:
        return
    yield error
    nested = list(getattr(error, "exceptions", ()) or ())
    nested += [error.__cause__, error.__context__]
    for item in nested:
        yield from _walk(item, depth - 1)


def _carries(error: BaseException | None, wanted: type[BaseException]) -> bool:
    """True when `error` is, wraps, or was caused by a `wanted` exception."""
    return any(isinstance(item, wanted) for item in _walk(error))


def _carries_unauthorized(error: BaseException | None) -> bool:
    """True when anything in the chain is an HTTP 401.

    Both shapes are checked because both occur: httpx-style errors carry a
    `response.status_code`, while transports that stringify the status leave
    only text. The text match is word-bounded so a "1401" in some unrelated id
    cannot masquerade as an authorization failure — and it runs per wrapped
    exception, since `str(ExceptionGroup(...))` is only "… (2 sub-exceptions)"
    and would hide the real 401 underneath it.
    """
    for item in _walk(error):
        status = getattr(getattr(item, "response", None), "status_code", None)
        if status is None:
            status = getattr(item, "status_code", None)
        if status == 401:
            return True
        if _UNAUTHORIZED_RE.search(str(item)):
            return True
    return False


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
        self._session_factory = session_factory or default_session_factory
        self._registry = registry
        # Public: the REPL assigns its status reporter after construction.
        self.on_status = on_status
        self._stopping = False
        self._connections: dict[str, _Connection] = {
            name: _Connection(config) for name, config in self._configs.items()
        }

    # ---------------------------------------------------------------- state

    def _registry_or_default(self) -> ToolRegistry:
        return self._registry if self._registry is not None else get_registry()

    @staticmethod
    def _lock_of(conn: _Connection) -> asyncio.Lock:
        """The connection's reconnect lock, created inside the running loop.

        Never in `__init__`: a manager built before a loop exists (or reused
        across `asyncio.run` calls) would carry a lock bound to a dead one.
        """
        if conn.lock is None:
            conn.lock = asyncio.Lock()
        return conn.lock

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
        """The auth provider one connect attempt should use, if any.

        `login`'s interactive provider wins and is consumed once — every later
        connect falls back to the silent one. Otherwise a remote server with
        tokens on file gets a non-interactive provider, which refreshes an
        expired token without asking and raises `NeedsInteractiveAuth` if the
        server wants a fresh consent. A server with nothing stored connects
        unauthenticated: that first 401 is what puts it in `NEEDS_AUTH`.
        """
        conn = self._connections.get(config.name)
        if conn is not None and conn.override_auth is not None:
            provider, conn.override_auth = conn.override_auth, None
            return provider
        if config.transport not in ("http", "sse") or not config.url:
            return None
        try:
            if not has_stored_tokens(config.name, config.url):
                return None
            return build_oauth_provider(config, interactive=False)[0]
        except Exception:
            # An unreadable token file must cost at most an unauthenticated
            # attempt (and a NEEDS_AUTH status), never a crashed connection.
            logger.exception("Could not build an OAuth provider for MCP server [{}]", config.name)
            return None

    def _failure_state(self, config: McpServerConfig, error: Exception) -> ServerState:
        """Which state a connection failure lands in.

        `NeedsInteractiveAuth` is definitive. Otherwise a remote server that
        answered 401 while we have no token on file is a server asking to be
        logged into; with tokens on file the same 401 means a revoked grant or
        a broken server, and showing the real error beats a misleading hint.
        """
        if _carries(error, NeedsInteractiveAuth):
            return ServerState.NEEDS_AUTH
        if config.transport in ("http", "sse") and _carries_unauthorized(error):
            try:
                if not has_stored_tokens(config.name, config.url or ""):
                    return ServerState.NEEDS_AUTH
            except Exception:
                logger.exception("Could not read stored MCP tokens for server [{}]", config.name)
        return ServerState.FAILED

    @staticmethod
    def _failure_error(config: McpServerConfig, state: ServerState, error: Exception) -> str:
        """The error text a failed connection reports."""
        if state == ServerState.NEEDS_AUTH:
            return f"needs auth — run /mcp login {config.name}"
        return str(error)

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

    def _owns(self, conn: _Connection) -> bool:
        """True while the running task is still this connection's generation.

        A task whose connection has been rebuilt underneath it (its `_connect`
        timed out, or it was abandoned mid-teardown) must not write to the
        shared `_Connection`: it would unregister the live generation's tools
        and null its session.
        """
        return conn.task is asyncio.current_task()

    async def _run_connection(
        self,
        conn: _Connection,
        ready: asyncio.Event,
        stop_event: asyncio.Event,
        connect_timeout: float | None = None,
    ) -> None:
        """The owning task: enters, serves, and unwinds one session.

        `ready`/`stop_event` are passed in rather than read off `conn`, so a
        late-unwinding task can never signal a newer generation's events, and
        `connect_timeout` likewise: an interactive login needs a far longer
        budget than a background reconnect, and reading it off shared state
        would leak that budget into whatever connects next.
        """
        timeout = settings.mcp_connect_timeout if connect_timeout is None else connect_timeout
        try:
            async with self._session_factory(conn.config, self._auth_for(conn.config)) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                await asyncio.wait_for(self._load_inventory(conn, session), timeout=timeout)
                if not self._owns(conn):
                    logger.warning("MCP server [{}] connected into a stale generation; unwinding", conn.config.name)
                    return
                conn.session = session
                self._register_tools(conn)
                self._set_state(conn, ServerState.CONNECTED)
                ready.set()
                await stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("MCP server [{}] failed: {}", conn.config.name, error)
            if self._owns(conn):
                state = self._failure_state(conn.config, error)
                self._set_state(conn, state, self._failure_error(conn.config, state, error))
        finally:
            if self._owns(conn):
                self._unregister_tools(conn)
                conn.session = None
                if conn.state == ServerState.CONNECTED:
                    self._set_state(conn, ServerState.DISCONNECTED)
            ready.set()

    async def _connect(self, conn: _Connection, connect_timeout: float | None = None) -> None:
        if self._stopping:
            return
        if conn.task is not None and not conn.task.done():
            return
        timeout = settings.mcp_connect_timeout if connect_timeout is None else connect_timeout
        ready = conn.ready = asyncio.Event()
        stop_event = conn.stop_event = asyncio.Event()
        conn.generation += 1
        self._set_state(conn, ServerState.CONNECTING)
        conn.task = asyncio.create_task(
            self._run_connection(conn, ready, stop_event, timeout),
            name=f"mcp:{conn.config.name}",
        )
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout + CONNECT_GRACE)
        except asyncio.TimeoutError:
            # The task stays `conn.task` so stop()/reconnect can still cancel
            # it; it keeps unwinding in the background meanwhile.
            stop_event.set()
            self._set_state(conn, ServerState.FAILED, "connection setup timed out")

    async def _disconnect(self, conn: _Connection) -> None:
        """Stop one connection task; `conn.task` stays set until it unwinds.

        Clearing `conn.task` up front would make the task disown itself in its
        own `finally` (see `_owns`) and skip unregistering.
        """
        task = conn.task
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
        finally:
            if conn.task is task:
                conn.task = None

    # -------------------------------------------------------------- control

    async def start(self) -> None:
        """Connect every configured server concurrently; never raises."""
        self._stopping = False
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
        # Refuses in-flight reconnects a new task, so nothing survives exit.
        self._stopping = True
        if not self._connections:
            return
        await asyncio.gather(
            *(self._disconnect(conn) for conn in self._connections.values()),
            return_exceptions=True,
        )

    async def reconnect(self, name: str) -> ServerStatus:
        """Rebuild one connection, or reuse one another caller just healed.

        The generation seen on entry decides: unchanged means nothing happened
        while we waited for the lock, so this is a real (possibly manual)
        reconnect; changed and connected means a concurrent caller already
        rebuilt the session and tearing it down again would kill a healthy one.
        """
        conn = self._require(name)
        seen = conn.generation
        async with self._lock_of(conn):
            if conn.generation != seen and conn.state == ServerState.CONNECTED:
                logger.debug("MCP server [{}] was already reconnected by another caller", name)
                return self._status_of(conn)
            await self._disconnect(conn)
            await self._connect(conn)
            return self._status_of(conn)

    async def login(self, name: str) -> ServerStatus:
        """Authorize one remote server in the browser, then reconnect it.

        A login is a reconnect that carries an interactive provider, so it
        takes the same per-connection lock: nothing else may tear the session
        down while the user is on the consent screen. The SDK runs the actual
        OAuth flow lazily, during the transport's first request — which is why
        the provider has to be stashed for `_auth_for` (`override_auth`) rather
        than passed down, why the connect runs on `LOGIN_CONNECT_TIMEOUT`
        rather than the ordinary one, and why the callback server must outlive
        `_connect` and be stopped only in the `finally`.
        """
        conn = self._require(name)
        config = conn.config
        if config.transport not in ("http", "sse"):
            raise ValueError(f"MCP server '{name}' is a {config.transport} server; OAuth applies to http/sse servers")
        if not config.url:
            raise ValueError(f"MCP server '{name}' has no url to authorize against")
        async with self._lock_of(conn):
            # Built before the disconnect: a callback port that cannot bind
            # must not cost the user a working connection.
            provider, callback = build_oauth_provider(config, interactive=True)
            try:
                # Inside the try: a failed start still owns a bound socket, and
                # the fixed port would wedge every later login this session.
                if callback is not None:
                    callback.start()
                await self._disconnect(conn)
                conn.override_auth = provider
                await self._connect(conn, connect_timeout=LOGIN_CONNECT_TIMEOUT)
            finally:
                # One-shot: a connect that never consumed it (a timeout, say)
                # must not authorize some later reconnect behind the user.
                conn.override_auth = None
                if callback is not None:
                    callback.stop()
            return self._status_of(conn)

    async def logout(self, name: str) -> ServerStatus:
        """Forget one server's tokens and reconnect it unauthenticated.

        The reconnect is the point: it proves the logout took, and it leaves
        the server in whatever state it really is — usually `NEEDS_AUTH`.
        """
        conn = self._require(name)
        config = conn.config
        async with self._lock_of(conn):
            await self._disconnect(conn)
            if config.transport in ("http", "sse") and config.url:
                try:
                    clear_tokens(config.name, config.url)
                except Exception:
                    logger.exception("Could not clear stored MCP tokens for server [{}]", config.name)
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
            session.call_tool(tool, arguments, read_timeout_seconds=settings.mcp_tool_timeout),
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
