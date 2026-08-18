import asyncio
import io
import sys
import unittest
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.manager import ServerState, ServerStatus, set_mcp_manager
from src.ui.commands import CommandHandler, McpAction, run_mcp_action


class FakeManager:
    def __init__(self):
        self.reconnected = []

    def statuses(self):
        return [
            ServerStatus(name="pw", transport="stdio", scope="project", state=ServerState.CONNECTED, tool_count=3),
            ServerStatus(name="ctx", transport="http", scope="user", state=ServerState.FAILED, error="401"),
        ]

    def prompt_commands(self):
        return [("/mcp__pw__review", "Review code")]

    def find_prompt(self, command):
        if command == "/mcp__pw__review":
            return ("pw", type("P", (), {"name": "review", "arguments": []})())
        return None

    async def reconnect(self, name):
        self.reconnected.append(name)
        return self.statuses()[0]

    async def get_prompt(self, server, prompt, arguments):
        return f"PROMPT {server}/{prompt} {arguments}"


class FakeUI:
    def __init__(self):
        self.console = Console(file=io.StringIO(), width=100)


class TestMcpCommand(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)
        self.ui = FakeUI()
        self.handler = CommandHandler(self.ui)

    def output(self):
        return self.ui.console.file.getvalue()

    def test_mcp_renders_status_table(self):
        self.assertIsNone(self.handler.handle("/mcp"))
        out = self.output()
        self.assertIn("pw", out)
        self.assertIn("connected", out)
        self.assertIn("401", out)

    def test_mcp_reconnect_returns_action(self):
        result = self.handler.handle("/mcp reconnect pw")
        self.assertEqual(result, McpAction("reconnect", "pw"))

    def test_prompt_command_returns_action(self):
        result = self.handler.handle("/mcp__pw__review please")
        self.assertEqual(result, McpAction("prompt", "/mcp__pw__review please"))

    def test_unknown_prompt_command_errors(self):
        self.assertIsNone(self.handler.handle("/mcp__nope__x"))
        self.assertIn("unknown", self.output().lower())


class TestRunMcpAction(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.console = Console(file=io.StringIO(), width=100)

    def test_reconnect_awaits_manager(self):
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertEqual(self.manager.reconnected, ["pw"])

    def test_prompt_fetches_text(self):
        action = McpAction("prompt", "/mcp__pw__review please fix")
        out = asyncio.run(run_mcp_action(action, self.manager, self.console))
        self.assertIn("PROMPT pw/review", out)

    def test_failures_print_not_raise(self):
        async def boom(name):
            raise RuntimeError("down")

        self.manager.reconnect = boom
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertIn("down", self.console.file.getvalue())


if __name__ == "__main__":
    unittest.main()
