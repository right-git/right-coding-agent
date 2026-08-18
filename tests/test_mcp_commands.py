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
        return [
            ("/mcp__pw__review", "Review code"),
            # A zero-arg prompt (arguments=None, as the real MCP SDK ships
            # it — not []) with a description that looks like mismatched
            # rich markup, to guard both known review-bot findings at once.
            ("/mcp__pw__greet", "[weird]tag[/mismatch]"),
        ]

    def find_prompt(self, command):
        if command == "/mcp__pw__review":
            return ("pw", type("P", (), {"name": "review", "arguments": []})())
        if command == "/mcp__pw__greet":
            return ("pw", type("P", (), {"name": "greet", "arguments": None})())
        return None

    async def reconnect(self, name):
        self.reconnected.append(name)
        return self.statuses()[0]

    async def get_prompt(self, server, prompt, arguments):
        return f"PROMPT {server}/{prompt} {arguments}"


class FakeUI:
    def __init__(self):
        self.console = Console(file=io.StringIO(), width=100)
        self.available_models = []
        self.model_catalog = {}
        self.commands = CommandHandler(self)


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

    def test_mcp_status_renders_markup_like_prompt_description_verbatim(self):
        # A server-supplied description containing bracket text that looks
        # like mismatched rich markup ("[weird]tag[/mismatch]") must not
        # raise MarkupError and must show up unmangled.
        self.assertIsNone(self.handler.handle("/mcp"))
        self.assertIn("[weird]tag[/mismatch]", self.output())

    def test_unknown_prompt_listing_renders_markup_like_description_verbatim(self):
        self.assertIsNone(self.handler.handle("/mcp__nope__x"))
        self.assertIn("[weird]tag[/mismatch]", self.output())


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

    def test_prompt_with_none_arguments_still_fetches_text(self):
        # Real MCP servers ship `Prompt.arguments=None` (not []) for a
        # zero-argument prompt; `_map_prompt_arguments` must not crash on it.
        action = McpAction("prompt", "/mcp__pw__greet")
        out = asyncio.run(run_mcp_action(action, self.manager, self.console))
        self.assertIn("PROMPT pw/greet", out)

    def test_failures_print_not_raise(self):
        async def boom(name):
            raise RuntimeError("down")

        self.manager.reconnect = boom
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertIn("down", self.console.file.getvalue())

    def test_error_text_with_markup_like_characters_does_not_raise(self):
        # The catch-all except branch prints exception text verbatim; a
        # message containing bracket text must not be parsed as rich markup.
        async def boom(name):
            raise RuntimeError("[weird]tag[/mismatch]")

        self.manager.reconnect = boom
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertIn("[weird]tag[/mismatch]", self.console.file.getvalue())


class TestCompleterMcp(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)

    def completions(self, text):
        from prompt_toolkit.document import Document

        from src.ui.completer import CommandCompleter

        ui = FakeUI()
        ui.available_models = []
        ui.model_catalog = {}
        completer = CommandCompleter(ui)
        return [c.text for c in completer.get_completions(Document(text, len(text)), None)]

    def test_prompt_commands_offered(self):
        self.assertIn("/mcp__pw__review", self.completions("/mcp_"))

    def test_mcp_offered(self):
        self.assertIn("/mcp", self.completions("/mc"))

    def test_mcp_subcommands(self):
        self.assertIn("reconnect", self.completions("/mcp rec"))

    def test_server_names_after_reconnect(self):
        self.assertIn("pw", self.completions("/mcp reconnect p"))


if __name__ == "__main__":
    unittest.main()
