import os
import unittest
from io import StringIO

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

from src.ui import ChatUI
from src.ui.chat import theme


def make_ui():
    ui = ChatUI(model="google/gemini-3.7-flash")
    ui.console = Console(
        file=StringIO(),
        record=True,
        force_terminal=False,
        width=200,
        theme=theme,
    )
    return ui


class WelcomeTests(unittest.TestCase):
    def test_welcome_panel_shows_session_info(self):
        ui = make_ui()

        ui.print_welcome()

        rendered = ui.console.export_text()
        self.assertIn("Right Code", rendered)
        self.assertIn("google/gemini-3.7-flash", rendered)
        self.assertIn(os.path.basename(os.getcwd()), rendered)
        self.assertIn("logs.log", rendered)
        self.assertIn("LocateAnything-3B", rendered)
        self.assertIn("/help", rendered)


class ToolCallRenderTests(unittest.TestCase):
    def ai_with_call(self, name, args):
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": "c1", "type": "tool_call"}],
        )

    def test_run_tools_call_shows_label_and_script_lines(self):
        ui = make_ui()
        code = "\n".join(f"line{i}" for i in range(9))

        ui.print_response([self.ai_with_call("run_tools", {"code": code})])

        rendered = ui.console.export_text()
        self.assertIn("● Run tools script", rendered)
        self.assertIn("│ line0", rendered)
        self.assertIn("│ line5", rendered)
        self.assertNotIn("line6", rendered)
        self.assertIn("+3 more lines", rendered)

    def test_search_and_docs_calls_show_their_argument(self):
        ui = make_ui()

        ui.print_response(
            [
                self.ai_with_call("search_tools", {"query": "click button"}),
                self.ai_with_call("get_tool", {"names": ["screen_click", "screen_type"]}),
            ]
        )

        rendered = ui.console.export_text()
        self.assertIn("Search tools · click button", rendered)
        self.assertIn("Read tool docs · screen_click, screen_type", rendered)

    def test_markup_like_content_does_not_break_rendering(self):
        ui = make_ui()

        ui.print_response([self.ai_with_call("run_tools", {"code": 'return ["red", "[bold]"]'})])

        self.assertIn('["red", "[bold]"]', ui.console.export_text())


class ToolResultRenderTests(unittest.TestCase):
    def test_results_are_shown_scrubbed_and_collapsed(self):
        ui = make_ui()
        content = '{"result": "ok",\n "blob": "' + "Q" * 5000 + '"}'

        ui.print_response([ToolMessage(content=content, tool_call_id="c1", name="run_tools")])

        rendered = ui.console.export_text()
        self.assertIn("⎿", rendered)
        self.assertIn("<base64 stripped, 5000 chars>", rendered)
        self.assertNotIn("Q" * 100, rendered)

    def test_failed_results_still_render(self):
        ui = make_ui()

        ui.print_response(
            [
                ToolMessage(
                    content="Tool call failed, error: boom",
                    tool_call_id="c1",
                    status="error",
                )
            ]
        )

        self.assertIn("Tool call failed, error: boom", ui.console.export_text())


if __name__ == "__main__":
    unittest.main()
