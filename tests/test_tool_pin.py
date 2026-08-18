import sys
import unittest
from io import StringIO
from pathlib import Path

from langchain_core.tools import tool
from prompt_toolkit.document import Document
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry, set_registry
from src.ui.chat import ChatUI, theme
from src.ui.completer import CommandCompleter


@tool(parse_docstring=True)
async def demo_probe(x: str) -> str:
    """Probe something.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


def make_ui():
    ui = ChatUI(model="test/model")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


class TestToolPin(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry()
        registry.register(demo_probe)
        set_registry(registry)
        self.addCleanup(set_registry, None)
        self.ui = make_ui()

    def handle(self, text):
        return self.ui.handle_command(text)

    def output(self):
        return self.ui.console.export_text()

    def test_pin_and_consume(self):
        self.handle("/tool demo_probe")

        content = self.ui.take_user_content("do the thing")

        self.assertIn("do the thing", content)
        self.assertIn("Tool directive", content)
        self.assertIn("demo_probe(x)", content)
        self.assertEqual(self.ui.pinned_tools, [])
        self.assertEqual(self.ui.take_user_content("next"), "next")

    def test_unknown_name_suggests_and_does_not_pin(self):
        # "demo_pro" would be rejected as intended, but note it is a
        # substring of the only registered tool ("demo_probe") — that would
        # actually auto-pin under the single-partial-match rule (see
        # test_unambiguous_partial_name_pins below, and _switch_model's
        # identical "single partial match switches" convention). Use a name
        # with zero substring overlap so this test exercises the true
        # no-match "suggest, don't pin" path.
        self.handle("/tool zzz_nonexistent")

        self.assertEqual(self.ui.pinned_tools, [])

    def test_none_clears(self):
        self.handle("/tool demo_probe")

        self.handle("/tool none")

        self.assertEqual(self.ui.pinned_tools, [])

    def test_pin_with_pending_image_lands_in_text_block(self):
        # Real pending-image dict shape, from `encode_image`
        # (src/ui/clipboard.py) and `ChatUI.take_user_content`
        # (src/ui/chat.py): base64_data/mime_type/width/height — NOT the
        # data_uri shape the brief guessed.
        self.handle("/tool demo_probe")
        self.ui.pending_images.append({"base64_data": "eA==", "mime_type": "image/jpeg", "width": 1, "height": 1})

        content = self.ui.take_user_content("look")

        text_blocks = [b for b in content if b.get("type") == "text"]
        self.assertTrue(any("Tool directive" in b.get("text", "") for b in text_blocks))

    def test_pin_shows_current_pins(self):
        self.handle("/tool demo_probe")

        self.handle("/tool")

        self.assertIn("demo_probe", self.output())

    def test_no_pins_reports_nothing_pinned(self):
        self.handle("/tool")

        self.assertIn("nothing", self.output())

    def test_unambiguous_partial_name_pins(self):
        self.handle("/tool demo_prob")

        self.assertEqual(self.ui.pinned_tools, ["demo_probe"])


class ToolCompletionTests(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry()
        registry.register(demo_probe)
        set_registry(registry)
        self.addCleanup(set_registry, None)
        self.ui = make_ui()
        self.completer = CommandCompleter(self.ui)

    def complete(self, text):
        return [c.text for c in self.completer.get_completions(Document(text), None)]

    def test_tool_argument_completes_registry_names(self):
        self.assertEqual(self.complete("/tool demo"), ["demo_probe"])

    def test_tool_command_name_completes(self):
        self.assertIn("/tool", self.complete("/too"))


if __name__ == "__main__":
    unittest.main()
