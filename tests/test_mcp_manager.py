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

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def list_resources(self):
        return SimpleNamespace(resources=self.resources)

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
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


if __name__ == "__main__":
    unittest.main()
