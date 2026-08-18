import asyncio
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry
from src.llm.tools.mcp.config import McpServerConfig
from src.llm.tools.mcp.manager import McpManager, ServerState


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
    """One fake server; factory counts connections and can fail first."""

    def __init__(self, session=None, connect_failures=0):
        self.session = session or FakeSession(tools=[remote_tool()])
        self.connect_failures = connect_failures
        self.connections = 0
        self.registry = ToolRegistry()
        config = McpServerConfig(name="srv", command="fake")
        self.manager = McpManager(configs={"srv": config}, session_factory=self.factory, registry=self.registry)

    @asynccontextmanager
    async def factory(self, config, auth=None):
        self.connections += 1
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise ConnectionError("refused")
        yield self.session


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
        harness = ManagerHarness(session=FakeSession(tools=[remote_tool()], fail_calls=1))

        async def scenario():
            await harness.manager.start()
            result = await harness.manager.call_tool("srv", "click", {"x": "1"})
            self.assertEqual(harness.session.calls, [("click", {"x": "1"})])
            self.assertEqual(harness.connections, 2)
            self.assertFalse(result.isError)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_get_prompt_flattens_text(self):
        session = FakeSession(tools=[], prompts=[SimpleNamespace(name="review", description="d", arguments=[])])
        harness = ManagerHarness(session=session)

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
