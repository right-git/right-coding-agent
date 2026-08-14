import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.computer_tools import (
    COMPUTER_TOOLS,
    get_computer,
    screen_click,
    screen_key,
    screen_locate,
    screen_mark,
    screen_scroll,
    screen_type,
    set_computer,
)
from src.tools.computer_use import ComputerUse, Detection, NullOverlay
from src.tools.computer_use.fakes import (
    RecordingPointer,
    ScriptedLocator,
    StaticScreen,
)


class ComputerToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(set_computer, None)
        self.pointer = RecordingPointer()
        self.overlay = NullOverlay()
        self.directory = Path(directory.name)

    def install(self, responses):
        computer = ComputerUse(
            locator=ScriptedLocator(responses),
            screen=StaticScreen([Image.new("RGB", (200, 100))], (200, 100)),
            pointer=self.pointer,
            overlay=self.overlay,
            output_path=self.directory / "annotated.jpg",
            dpi_aware=False,
            min_target_px=0,  # refinement is covered in tests/test_computer_use.py
        )
        set_computer(computer)
        return computer

    def test_every_tool_is_registered_with_a_stable_name(self):
        self.assertEqual(
            [tool.name for tool in COMPUTER_TOOLS],
            [
                "screen_locate",
                "screen_mark",
                "screen_click",
                "screen_type",
                "screen_key",
                "screen_scroll",
            ],
        )

    def test_the_shared_session_is_created_once(self):
        computer = self.install([[]])

        self.assertIs(get_computer(), computer)
        self.assertIs(get_computer(), computer)

    async def test_locate_reports_boxes_and_centers(self):
        self.install([[Detection("save", (10, 20, 30, 40))]])

        result = await screen_locate.ainvoke({"description": "save button"})

        self.assertEqual(result, "1. save: box=(10, 20, 30, 40), center=(20, 30)")

    async def test_locate_reports_when_nothing_matches(self):
        self.install([[]])

        result = await screen_locate.ainvoke({"description": "missing"})

        self.assertEqual(result, "No matching region was found on the screen.")

    async def test_mark_points_at_the_element_and_keeps_the_note(self):
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        result = await screen_mark.ainvoke(
            {"description": "кнопка рендера", "note": "Запускает просчёт"}
        )

        self.assertIn("render button", result)
        self.assertIn("(20, 30)", result)
        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [])
        self.assertEqual(self.overlay.markers[0].note, "Запускает просчёт")

    async def test_mark_uses_the_supplied_title_as_the_tooltip_heading(self):
        self.install([[Detection("the render button in the export panel", (10, 20, 30, 40))]])

        await screen_mark.ainvoke(
            {
                "description": "кнопка рендера в панели экспорта",
                "note": "Запускает просчёт",
                "title": "Кнопка Render",
            }
        )

        self.assertEqual(self.overlay.markers[0].title, "Кнопка Render")

    async def test_mark_falls_back_to_the_detected_label_without_a_title(self):
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        await screen_mark.ainvoke({"description": "render", "note": "n"})

        self.assertEqual(self.overlay.markers[0].title, "render button")

    async def test_mark_says_so_when_nothing_matches(self):
        self.install([[]])

        result = await screen_mark.ainvoke({"description": "ghost", "note": "n"})

        self.assertEqual(result, "Nothing on screen matched: ghost")
        self.assertEqual(self.overlay.markers, [])

    async def test_click_uses_a_single_click_by_default(self):
        self.install([[Detection("ok", (10, 10, 30, 30))]])

        result = await screen_click.ainvoke({"description": "ok button"})

        self.assertIn("Clicked 'ok'", result)
        self.assertEqual(self.pointer.clicks, [("left", 1, ())])

    async def test_click_can_double_click(self):
        self.install([[Detection("file", (10, 10, 30, 30))]])

        await screen_click.ainvoke({"description": "file", "double": True})

        self.assertEqual(self.pointer.clicks, [("left", 2, ())])

    async def test_typing_scrolling_and_shortcuts_reach_the_pointer(self):
        self.install([[]])

        await screen_type.ainvoke({"text": "hello"})
        await screen_key.ainvoke({"combination": "ctrl+s"})
        await screen_scroll.ainvoke({"direction": "down", "amount": 5})

        self.assertEqual(self.pointer.typed, ["hello"])
        self.assertEqual(self.pointer.keys, ["ctrl+s"])
        self.assertEqual(self.pointer.scrolls, [("down", 5, ())])

    async def test_failures_are_reported_to_the_model_instead_of_raising(self):
        computer = self.install([[]])
        computer.locator.locate = Mock(side_effect=RuntimeError("model unavailable"))

        result = await screen_locate.ainvoke({"description": "anything"})

        self.assertEqual(result, "Tool call failed, error: model unavailable")


if __name__ == "__main__":
    unittest.main()
