import asyncio
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import settings
from src.llm.tools import ToolRegistry
from src.llm.tools.mcp import manager as manager_module
from src.llm.tools.mcp import oauth as oauth_module
from src.llm.tools.mcp import transports
from src.llm.tools.mcp.config import McpServerConfig
from src.llm.tools.mcp.manager import McpManager, ServerState


def live_connection_tasks():
    """Connection tasks still alive in this loop, by their `mcp:` name."""
    return [task for task in asyncio.all_tasks() if task.get_name().startswith("mcp:")]


def remote_tool(name="click"):
    return SimpleNamespace(
        name=name,
        description=f"{name} tool.",
        inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        annotations=None,
    )


def text_result(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], structuredContent=None, isError=False)


class FakeSession:
    def __init__(self, tools=(), prompts=(), resources=(), fail_calls=0):
        self.tools, self.prompts, self.resources = list(tools), list(prompts), list(resources)
        self.fail_calls = fail_calls
        self.calls = []
        # `read_timeout_seconds` as received by each `call_tool` invocation —
        # kept separate from `self.calls` so existing (name, arguments)
        # assertions elsewhere don't need reshaping.
        self.read_timeouts = []

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def list_resources(self):
        return SimpleNamespace(resources=self.resources)

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.read_timeouts.append(read_timeout_seconds)
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise ConnectionError("stream closed")
        self.calls.append((name, arguments))
        return text_result("done")

    async def get_prompt(self, name, arguments=None):
        message = SimpleNamespace(role="user", content=SimpleNamespace(type="text", text=f"prompt:{name}"))
        return SimpleNamespace(messages=[message])

    async def read_resource(self, uri):
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, text="resource body", mimeType="text/plain")])


class ManagerHarness:
    """One fake server; factory counts connections and can fail first.

    Each connection gets a FRESH session, the way a real reconnect does — a
    shared one cannot tell "retried on the new session" from "retried on the
    dead one". Teardown takes a beat, like a real transport closing.
    """

    TEARDOWN_DELAY = 0.02

    def __init__(self, make_session=None, connect_failures=0):
        self.make_session = make_session or (lambda attempt: FakeSession(tools=[remote_tool()]))
        self.connect_failures = connect_failures
        self.connections = 0
        self.sessions = []
        self.registry = ToolRegistry()
        config = McpServerConfig(name="srv", command="fake")
        self.manager = McpManager(configs={"srv": config}, session_factory=self.factory, registry=self.registry)

    @property
    def session(self):
        """The session of the most recent connection."""
        return self.sessions[-1]

    @asynccontextmanager
    async def factory(self, config, auth=None):
        self.connections += 1
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise ConnectionError("refused")
        session = self.make_session(self.connections)
        self.sessions.append(session)
        try:
            yield session
        finally:
            await asyncio.sleep(self.TEARDOWN_DELAY)


class TestManagerLifecycle(unittest.TestCase):
    def test_start_registers_tools_and_reports_connected(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            try:
                status = harness.manager.statuses()[0]
                self.assertEqual(status.state, ServerState.CONNECTED)
                self.assertEqual(status.tool_count, 1)
                self.assertIsNotNone(harness.registry.get("mcp__srv__click"))
                self.assertEqual(harness.registry.source_of("mcp__srv__click"), "mcp:srv")
            finally:
                await harness.manager.stop()

        asyncio.run(scenario())

    def test_stop_unregisters(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            await harness.manager.stop()
            self.assertIsNone(harness.registry.get("mcp__srv__click"))
            self.assertEqual(harness.manager.statuses()[0].state, ServerState.DISCONNECTED)

        asyncio.run(scenario())

    def test_connect_failure_is_contained(self):
        harness = ManagerHarness(connect_failures=1)

        async def scenario():
            await harness.manager.start()
            status = harness.manager.statuses()[0]
            self.assertEqual(status.state, ServerState.FAILED)
            self.assertIn("refused", status.error)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_reconnect_after_failure_recovers(self):
        harness = ManagerHarness(connect_failures=1)

        async def scenario():
            await harness.manager.start()
            status = await harness.manager.reconnect("srv")
            self.assertEqual(status.state, ServerState.CONNECTED)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_reconnect_replaces_registrations_without_duplicate_error(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            await harness.manager.reconnect("srv")
            self.assertIsNotNone(harness.registry.get("mcp__srv__click"))
            self.assertEqual(harness.connections, 2)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_call_tool_passes_tool_timeout_as_a_plain_float(self):
        """SDK 2.0's `ClientSession.call_tool` wants a bare float, not a

        `timedelta` — regression coverage for the live smoke-test finding
        (mcp 1.x accepted a `timedelta`; 2.0's `read_timeout_seconds` is
        typed `float | None` and a `timedelta` blows up inside anyio).
        """
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            result = await harness.manager.call_tool("srv", "click", {"x": "1"})
            self.assertFalse(result.isError)
            timeout = harness.session.read_timeouts[-1]
            self.assertIsInstance(timeout, float)
            self.assertEqual(timeout, settings.mcp_tool_timeout)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_dead_session_call_reconnects_once_and_retries(self):
        harness = ManagerHarness(
            make_session=lambda attempt: FakeSession(tools=[remote_tool()], fail_calls=1 if attempt == 1 else 0)
        )

        async def scenario():
            await harness.manager.start()
            result = await harness.manager.call_tool("srv", "click", {"x": "1"})
            self.assertEqual(harness.connections, 2)
            self.assertEqual(harness.sessions[0].calls, [])
            self.assertEqual(harness.sessions[1].calls, [("click", {"x": "1"})])
            self.assertFalse(result.isError)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_concurrent_dead_session_calls_keep_one_live_connection(self):
        """Two parallel calls on a dead session must heal it exactly once.

        Both land in the reconnect branch; without serialization the second
        caller starts a second generation and the first one's slow teardown
        then unregisters the survivor's tools and nulls its live session.
        """
        harness = ManagerHarness(
            make_session=lambda attempt: FakeSession(tools=[remote_tool()], fail_calls=99 if attempt == 1 else 0)
        )

        async def scenario():
            await harness.manager.start()
            results = await asyncio.gather(
                harness.manager.call_tool("srv", "click", {"x": "1"}),
                harness.manager.call_tool("srv", "click", {"x": "2"}),
                return_exceptions=True,
            )
            self.assertEqual([r for r in results if isinstance(r, BaseException)], [], results)
            status = harness.manager.statuses()[0]
            self.assertEqual(status.state, ServerState.CONNECTED)
            self.assertEqual(harness.connections, 2)
            self.assertIsNotNone(harness.registry.get("mcp__srv__click"))
            self.assertEqual(len(live_connection_tasks()), 1)
            self.assertEqual(len(harness.sessions[1].calls), 2)
            await harness.manager.stop()
            self.assertEqual(live_connection_tasks(), [])
            self.assertIsNone(harness.registry.get("mcp__srv__click"))

        asyncio.run(scenario())

    def test_hung_connect_times_out_and_start_returns(self):
        """A transport that never finishes setup must not hang startup."""

        @asynccontextmanager
        async def hanging_factory(config, auth=None):
            await asyncio.Event().wait()  # never returns on its own
            yield None  # pragma: no cover

        registry = ToolRegistry()
        manager = McpManager(
            configs={"srv": McpServerConfig(name="srv", command="fake")},
            session_factory=hanging_factory,
            registry=registry,
        )

        async def scenario():
            with (
                patch.object(settings, "mcp_connect_timeout", 0.05),
                patch.object(manager_module, "CONNECT_GRACE", 0.05),
                patch.object(manager_module, "STOP_TIMEOUT", 0.1),
            ):
                await asyncio.wait_for(manager.start(), timeout=5)
                status = manager.statuses()[0]
                self.assertEqual(status.state, ServerState.FAILED)
                self.assertIn("timed out", status.error)
                await asyncio.wait_for(manager.stop(), timeout=5)
            self.assertEqual(live_connection_tasks(), [])

        asyncio.run(scenario())

    def test_get_prompt_flattens_text(self):
        harness = ManagerHarness(
            make_session=lambda attempt: FakeSession(
                tools=[], prompts=[SimpleNamespace(name="review", description="d", arguments=[])]
            )
        )

        async def scenario():
            await harness.manager.start()
            self.assertEqual(harness.manager.prompt_commands(), [("/mcp__srv__review", "d")])
            server, prompt = harness.manager.find_prompt("/mcp__srv__review")
            self.assertEqual(server, "srv")
            text = await harness.manager.get_prompt("srv", "review", {})
            self.assertEqual(text, "prompt:review")
            await harness.manager.stop()

        asyncio.run(scenario())


class _SlowSession(FakeSession):
    """A session whose handshake takes longer than a short connect budget."""

    def __init__(self, delay, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    async def initialize(self):
        await asyncio.sleep(self.delay)
        return None


class _FakeCallbackServer:
    """Records the interactive callback server's lifecycle."""

    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def remote_config(name="remote", transport="http"):
    return McpServerConfig(name=name, transport=transport, url="https://r/mcp")


def remote_manager(config=None, factory=None, registry=None):
    config = config or remote_config()
    return McpManager(configs={config.name: config}, session_factory=factory, registry=registry or ToolRegistry())


class TestAuthWiring(unittest.TestCase):
    """`_auth_for`: stdio never authenticates, http refreshes silently."""

    def test_stdio_never_builds_a_provider(self):
        manager = McpManager(configs={"srv": McpServerConfig(name="srv", command="fake")}, registry=ToolRegistry())
        with patch.object(manager_module, "build_oauth_provider") as build:
            self.assertIsNone(manager._auth_for(manager._connections["srv"].config))
        build.assert_not_called()

    def test_http_without_stored_tokens_stays_unauthenticated(self):
        manager = remote_manager()
        with patch.object(manager_module, "has_stored_tokens", return_value=False):
            with patch.object(manager_module, "build_oauth_provider") as build:
                self.assertIsNone(manager._auth_for(manager._connections["remote"].config))
        build.assert_not_called()

    def test_http_with_stored_tokens_builds_a_silent_provider(self):
        manager = remote_manager()
        provider = object()
        with patch.object(manager_module, "has_stored_tokens", return_value=True):
            with patch.object(manager_module, "build_oauth_provider", return_value=(provider, None)) as build:
                self.assertIs(manager._auth_for(manager._connections["remote"].config), provider)
        self.assertIs(build.call_args.kwargs["interactive"], False)

    def test_override_auth_wins_and_is_consumed_once(self):
        manager = remote_manager()
        conn = manager._connections["remote"]
        provider = object()
        conn.override_auth = provider
        with patch.object(manager_module, "has_stored_tokens", return_value=True):
            with patch.object(manager_module, "build_oauth_provider", return_value=(object(), None)):
                self.assertIs(manager._auth_for(conn.config), provider)
                self.assertIsNone(conn.override_auth)
                # The stash is one-shot: the next connect refreshes normally.
                self.assertIsNotNone(manager._auth_for(conn.config))

    def test_a_broken_token_file_does_not_break_connecting(self):
        manager = remote_manager()
        with patch.object(manager_module, "has_stored_tokens", side_effect=OSError("unreadable")):
            self.assertIsNone(manager._auth_for(manager._connections["remote"].config))


class TestNeedsAuth(unittest.TestCase):
    def needs_auth_manager(self, error, config=None):
        config = config or remote_config()

        @asynccontextmanager
        async def factory(config, auth=None):
            raise error
            yield  # pragma: no cover

        return remote_manager(config=config, factory=factory)

    def run_start(self, manager, stored=False):
        """Start once and report the resulting status.

        `has_stored_tokens` is always patched: unpatched it would read the
        developer's real `~/.right-agent/mcp-tokens.json`.
        """

        async def scenario():
            await manager.start()
            status = manager.statuses()[0]
            await manager.stop()
            return status

        with patch.object(manager_module, "has_stored_tokens", return_value=stored):
            return asyncio.run(scenario())

    def test_needs_interactive_auth_reports_needs_auth(self):
        from src.llm.tools.mcp.oauth import NeedsInteractiveAuth

        status = self.run_start(self.needs_auth_manager(NeedsInteractiveAuth("authorization required")))
        self.assertEqual(status.state, ServerState.NEEDS_AUTH)
        self.assertIn("/mcp login remote", status.error)

    def test_needs_interactive_auth_survives_transport_exception_wrapping(self):
        # anyio task groups re-raise as ExceptionGroup and httpx re-raises with
        # the original as __cause__, so the manager must look inside both.
        from src.llm.tools.mcp.oauth import NeedsInteractiveAuth

        inner = NeedsInteractiveAuth("run /mcp login remote")
        wrapped = ExceptionGroup("transport failed", [RuntimeError("closed"), inner])
        self.assertEqual(self.run_start(self.needs_auth_manager(wrapped)).state, ServerState.NEEDS_AUTH)

        chained = RuntimeError("connection closed")
        chained.__cause__ = inner
        self.assertEqual(self.run_start(self.needs_auth_manager(chained)).state, ServerState.NEEDS_AUTH)

    def test_http_401_without_tokens_reports_needs_auth(self):
        status = self.run_start(self.needs_auth_manager(RuntimeError("HTTP 401 Unauthorized")))
        self.assertEqual(status.state, ServerState.NEEDS_AUTH)
        self.assertIn("/mcp login remote", status.error)

    def test_a_wrapped_401_still_reports_needs_auth(self):
        # `str(ExceptionGroup(...))` is only "… (2 sub-exceptions)", so a check
        # on the outermost exception would miss the real 401 underneath — and
        # wrapped is the likeliest shape to arrive in under anyio.
        wrapped = ExceptionGroup("transport failed", [RuntimeError("closed"), RuntimeError("HTTP 401 Unauthorized")])
        self.assertEqual(self.run_start(self.needs_auth_manager(wrapped)).state, ServerState.NEEDS_AUTH)

    def test_a_401_status_code_without_the_text_reports_needs_auth(self):
        class Response:
            status_code = 401

        class HttpStatusError(RuntimeError):
            response = Response()

        status = self.run_start(self.needs_auth_manager(HttpStatusError("server refused the request")))
        self.assertEqual(status.state, ServerState.NEEDS_AUTH)

    def test_a_401_like_number_is_not_an_authorization_failure(self):
        status = self.run_start(self.needs_auth_manager(RuntimeError("upstream request 1401 failed")))
        self.assertEqual(status.state, ServerState.FAILED)

    def test_http_401_with_stored_tokens_reports_failed(self):
        # Tokens exist and the server still refuses: that is a broken server or
        # a revoked grant, not a missing login — surfacing the real error helps.
        status = self.run_start(self.needs_auth_manager(RuntimeError("HTTP 401 Unauthorized")), stored=True)
        self.assertEqual(status.state, ServerState.FAILED)
        self.assertIn("401", status.error)

    def test_sse_401_without_tokens_reports_needs_auth(self):
        config = McpServerConfig(name="remote", transport="sse", url="https://r/sse")
        status = self.run_start(self.needs_auth_manager(RuntimeError("401"), config=config))
        self.assertEqual(status.state, ServerState.NEEDS_AUTH)

    def test_other_http_errors_report_failed(self):
        status = self.run_start(self.needs_auth_manager(RuntimeError("HTTP 503 Service Unavailable")))
        self.assertEqual(status.state, ServerState.FAILED)

    def test_stdio_failures_never_report_needs_auth(self):
        config = McpServerConfig(name="remote", command="fake")
        status = self.run_start(self.needs_auth_manager(RuntimeError("401"), config=config))
        self.assertEqual(status.state, ServerState.FAILED)


class TestLoginLogout(unittest.TestCase):
    """`login`/`logout` are disconnect+connect cycles under the same lock."""

    def setUp(self):
        self.auths = []
        self.callback = _FakeCallbackServer()
        self.provider = object()

    @asynccontextmanager
    async def factory(self, config, auth=None):
        self.auths.append(auth)
        yield FakeSession(tools=[remote_tool()])

    def manager(self, config=None):
        return remote_manager(config=config, factory=self.factory)

    def patched_build(self):
        return patch.object(manager_module, "build_oauth_provider", return_value=(self.provider, self.callback))

    def test_login_connects_with_the_interactive_provider(self):
        manager = self.manager()

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                await manager.start()
                with self.patched_build() as build:
                    status = await manager.login("remote")
                await manager.stop()
            return status, build

        status, build = asyncio.run(scenario())
        self.assertEqual(status.state, ServerState.CONNECTED)
        self.assertIs(build.call_args.kwargs["interactive"], True)
        # First connect unauthenticated, second one carrying the login provider.
        self.assertEqual(self.auths, [None, self.provider])
        self.assertEqual((self.callback.started, self.callback.stopped), (1, 1))

    def test_login_clears_the_override_after_the_attempt(self):
        manager = self.manager()

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                with self.patched_build():
                    await manager.login("remote")
                override = manager._connections["remote"].override_auth
                await manager.stop()
            return override

        self.assertIsNone(asyncio.run(scenario()))

    def test_login_stops_the_callback_server_when_connecting_fails(self):
        @asynccontextmanager
        async def failing_factory(config, auth=None):
            self.auths.append(auth)
            raise ConnectionError("refused")
            yield  # pragma: no cover

        manager = remote_manager(factory=failing_factory)

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                with self.patched_build():
                    status = await manager.login("remote")
                await manager.stop()
            return status

        status = asyncio.run(scenario())
        self.assertNotEqual(status.state, ServerState.CONNECTED)
        self.assertEqual(self.callback.stopped, 1)

    def test_login_stops_the_callback_server_when_the_provider_flow_raises(self):
        manager = self.manager()

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                with patch.object(manager, "_connect", side_effect=RuntimeError("boom")):
                    with self.patched_build():
                        with self.assertRaises(RuntimeError):
                            await manager.login("remote")
            return manager._connections["remote"].override_auth

        self.assertIsNone(asyncio.run(scenario()))
        self.assertEqual(self.callback.stopped, 1)

    def test_login_on_a_stdio_server_is_refused(self):
        manager = self.manager(config=McpServerConfig(name="remote", command="fake"))

        async def scenario():
            with self.patched_build() as build:
                with self.assertRaises(ValueError):
                    await manager.login("remote")
            return build

        build = asyncio.run(scenario())
        build.assert_not_called()
        self.assertEqual(self.callback.started, 0)

    def test_login_without_a_url_is_refused_before_binding_a_port(self):
        # A urlless remote config would otherwise reach OAuthClientProvider,
        # whose failure used to leak the already-bound callback socket and
        # wedge every later login on the fixed port.
        config = McpServerConfig(name="remote", transport="http", url="https://r/mcp")
        manager = self.manager(config=config)
        object.__setattr__(manager._connections["remote"].config, "url", "")

        async def scenario():
            with self.patched_build() as build:
                with self.assertRaises(ValueError):
                    await manager.login("remote")
            return build

        build = asyncio.run(scenario())
        build.assert_not_called()
        self.assertEqual(self.callback.started, 0)

    def test_login_on_an_unknown_server_raises(self):
        manager = self.manager()

        async def scenario():
            with self.assertRaises(ValueError):
                await manager.login("nope")

        asyncio.run(scenario())

    def test_login_holds_the_connection_lock(self):
        manager = self.manager()
        conn = manager._connections["remote"]

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                async with manager._lock_of(conn):
                    with self.patched_build():
                        login = asyncio.create_task(manager.login("remote"))
                        await asyncio.sleep(0.05)
                        # Blocked on the lock, so the callback server is not up
                        # and no connect has been attempted behind our back.
                        self.assertEqual(self.callback.started, 0)
                        self.assertFalse(login.done())
                    login.cancel()
                    await asyncio.gather(login, return_exceptions=True)

        asyncio.run(scenario())

    def test_login_gets_its_own_connect_budget(self):
        # The OAuth dance runs inside the transport's FIRST request, so it is
        # spent inside the connect timeout. On the ordinary 30 s budget a human
        # typing a password would be cut off — and login's `finally` would stop
        # the callback server while the browser redirect is still in flight.
        @asynccontextmanager
        async def slow_factory(config, auth=None):
            self.auths.append(auth)
            yield _SlowSession(0.3, tools=[remote_tool()])

        manager = remote_manager(factory=slow_factory)

        async def scenario():
            with patch.object(settings, "mcp_connect_timeout", 0.05):
                with patch.object(manager_module, "LOGIN_CONNECT_TIMEOUT", 5.0):
                    with patch.object(manager_module, "has_stored_tokens", return_value=False):
                        await manager.start()
                        ordinary = manager.statuses()[0].state
                        with self.patched_build():
                            after_login = await manager.login("remote")
                        await manager.stop()
            return ordinary, after_login

        ordinary, after_login = asyncio.run(scenario())
        # Same server, same delay: too slow for a background connect, fine for
        # a login. That contrast is the whole point of the separate budget.
        self.assertEqual(ordinary, ServerState.FAILED)
        self.assertEqual(after_login.state, ServerState.CONNECTED)

    def test_the_login_budget_covers_the_whole_callback_wait(self):
        # Derived from CALLBACK_TIMEOUT so the two cannot drift apart.
        self.assertGreater(manager_module.LOGIN_CONNECT_TIMEOUT, oauth_module.CALLBACK_TIMEOUT)
        self.assertGreater(manager_module.LOGIN_CONNECT_TIMEOUT, settings.mcp_connect_timeout)

    def test_logout_clears_tokens_and_reconnects(self):
        manager = self.manager()

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                await manager.start()
                with patch.object(manager_module, "clear_tokens", return_value=True) as clear:
                    status = await manager.logout("remote")
                await manager.stop()
            return status, clear

        status, clear = asyncio.run(scenario())
        self.assertEqual(status.state, ServerState.CONNECTED)
        clear.assert_called_once_with("remote", "https://r/mcp")
        # Reconnected, and without any leftover login provider.
        self.assertEqual(self.auths, [None, None])

    def test_logout_reconnects_even_when_nothing_was_stored(self):
        manager = self.manager()

        async def scenario():
            with patch.object(manager_module, "has_stored_tokens", return_value=False):
                with patch.object(manager_module, "clear_tokens", return_value=False):
                    status = await manager.logout("remote")
                await manager.stop()
            return status

        self.assertEqual(asyncio.run(scenario()).state, ServerState.CONNECTED)

    def test_logout_on_a_stdio_server_just_reconnects(self):
        manager = self.manager(config=McpServerConfig(name="remote", command="fake"))

        async def scenario():
            with patch.object(manager_module, "clear_tokens") as clear:
                status = await manager.logout("remote")
                await manager.stop()
            return status, clear

        status, clear = asyncio.run(scenario())
        self.assertEqual(status.state, ServerState.CONNECTED)
        clear.assert_not_called()


class _FakeStreams:
    """Stand-in for a transport's yielded (read, write[, ...]) tuple."""

    def __init__(self, count):
        self.count = count

    async def __aenter__(self):
        return tuple(object() for _ in range(self.count))

    async def __aexit__(self, *exc):
        return False


class _FakeHttpClient:
    """Stand-in for the `httpx2.AsyncClient` `create_mcp_http_client` returns.

    Only the async-context-manager surface matters: `default_session_factory`
    opens and closes this client itself around the http transport.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestDefaultSessionFactory(unittest.TestCase):
    """Transport selection only — no real process or network I/O.

    Every SDK entry point (`stdio_client`, `create_mcp_http_client`,
    `streamable_http_client`, `sse_client`, `ClientSession`) is patched in
    the `transports` module namespace.
    """

    def run_factory(self, config, auth=None):
        async def scenario():
            async with transports.default_session_factory(config, auth=auth) as session:
                self.assertIsNotNone(session)

        with patch.object(transports, "ClientSession") as session_cls:
            session_cls.return_value.__aenter__ = lambda s: asyncio.sleep(0, result=object())
            session_cls.return_value.__aexit__ = lambda s, *e: asyncio.sleep(0, result=False)
            asyncio.run(scenario())

    def test_stdio_builds_server_params_merged_over_default_environment(self):
        config = McpServerConfig(name="pw", command="npx", args=["-y", "x"], env={"A": "1"})
        with patch.object(transports, "stdio_client", return_value=_FakeStreams(2)) as client:
            self.run_factory(config)
        params = client.call_args.args[0]
        self.assertEqual(params.command, "npx")
        self.assertEqual(params.args, ["-y", "x"])
        self.assertEqual(params.env.get("A"), "1")
        # Merged over the real default environment, not replacing it.
        self.assertIn("PATH", params.env)

    def test_http_builds_httpx_client_with_headers_and_auth_and_passes_it_through(self):
        config = McpServerConfig(name="c", transport="http", url="https://c/mcp", headers={"K": "V"})
        fake_client = _FakeHttpClient()
        sentinel_auth = object()
        with (
            patch.object(transports, "create_mcp_http_client", return_value=fake_client) as make_client,
            patch.object(transports, "streamable_http_client", return_value=_FakeStreams(2)) as client,
        ):
            self.run_factory(config, auth=sentinel_auth)
        make_client.assert_called_once_with(headers={"K": "V"}, auth=sentinel_auth)
        self.assertEqual(client.call_args.args[0], "https://c/mcp")
        self.assertIs(client.call_args.kwargs["http_client"], fake_client)

    def test_sse_selected_for_sse_transport_with_headers_and_auth(self):
        config = McpServerConfig(name="l", transport="sse", url="https://l/sse", headers={"K": "V"})
        sentinel_auth = object()
        with patch.object(transports, "sse_client", return_value=_FakeStreams(2)) as client:
            self.run_factory(config, auth=sentinel_auth)
        self.assertEqual(client.call_args.args[0], "https://l/sse")
        self.assertEqual(client.call_args.kwargs["headers"], {"K": "V"})
        self.assertIs(client.call_args.kwargs["auth"], sentinel_auth)

    def test_unsupported_transport_raises(self):
        # A plain object stands in here: McpServerConfig's own Literal type
        # already forbids constructing a bogus `transport`, so this exercises
        # the factory's own fallback branch in isolation.
        config = SimpleNamespace(transport="carrier-pigeon")

        async def scenario():
            async with transports.default_session_factory(config):
                pass  # pragma: no cover - never reached

        with self.assertRaises(ValueError):
            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()


class DescribeErrorTests(unittest.TestCase):
    """A TaskGroup wrapper must not hide the failure it carries."""

    def test_exception_group_reports_its_leaf(self):
        from src.llm.tools.mcp.manager import describe_error

        group = ExceptionGroup("unhandled errors in a TaskGroup", [ValueError("Token exchange failed (400)")])
        self.assertEqual(describe_error(group), "ValueError: Token exchange failed (400)")

    def test_nested_groups_are_flattened_and_deduped(self):
        from src.llm.tools.mcp.manager import describe_error

        inner = ExceptionGroup("inner", [RuntimeError("boom"), RuntimeError("boom")])
        self.assertEqual(describe_error(ExceptionGroup("outer", [inner])), "RuntimeError: boom")

    def test_a_plain_exception_is_unchanged(self):
        from src.llm.tools.mcp.manager import describe_error

        self.assertEqual(describe_error(ValueError("plain")), "ValueError: plain")
