import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.computer import (
    COMPUTER_TOOLS,
    get_computer,
    screen_click,
    screen_key,
    screen_locate,
    screen_scroll,
    screen_type,
    set_activity_listener,
    set_computer,
)
from src.llm.tools.computer import ComputerUse, Detection, NullOverlay
from src.llm.tools.computer.fakes import (
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

    async def test_screen_tools_ping_the_activity_listener(self):
        self.install([])
        pings = []
        set_activity_listener(lambda: pings.append(1))
        self.addCleanup(set_activity_listener, None)

        await screen_type.ainvoke({"text": "hi"})
        await screen_key.ainvoke({"combination": "ctrl+a"})

        self.assertEqual(len(pings), 2)

    def test_every_tool_is_registered_with_a_stable_name(self):
        self.assertEqual(
            [tool.name for tool in COMPUTER_TOOLS],
            [
                "screen_locate",
                "screen_screenshot",
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

        self.assertEqual(result, "1. save: box=(10, 20, 30, 40), center=(20, 30), at top-left")

    async def test_locate_reports_when_nothing_matches(self):
        self.install([[]])

        result = await screen_locate.ainvoke({"description": "missing"})

        self.assertEqual(result, "No matching region was found on the screen.")

    async def test_locate_with_mark_finds_and_points_in_one_call(self):
        # Marking is part of locating: one inference finds the element AND
        # shows it — a separate mark tool used to re-run the whole locate.
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        result = await screen_locate.ainvoke(
            {"description": "кнопка рендера", "mark": True, "note": "Запускает просчёт"}
        )

        self.assertIn("render button", result)
        self.assertIn("box=(10, 20, 30, 40)", result)  # locate report stays
        self.assertIn("Marked", result)
        self.assertIn("(20, 30)", result)
        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [])
        self.assertEqual(self.overlay.markers[0].note, "Запускает просчёт")

    async def test_a_note_alone_counts_as_marking_intent(self):
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        await screen_locate.ainvoke({"description": "render", "note": "Запускает просчёт"})

        self.assertEqual(len(self.overlay.markers), 1)

    async def test_locate_without_mark_never_touches_the_screen(self):
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        await screen_locate.ainvoke({"description": "render"})

        self.assertEqual(self.overlay.markers, [])
        self.assertEqual(self.pointer.moves, [])

    async def test_mark_uses_the_supplied_title_as_the_tooltip_heading(self):
        self.install([[Detection("the render button in the export panel", (10, 20, 30, 40))]])

        await screen_locate.ainvoke(
            {
                "description": "кнопка рендера в панели экспорта",
                "mark": True,
                "note": "Запускает просчёт",
                "title": "Кнопка Render",
            }
        )

        self.assertEqual(self.overlay.markers[0].title, "Кнопка Render")

    async def test_mark_falls_back_to_the_detected_label_without_a_title(self):
        self.install([[Detection("render button", (10, 20, 30, 40))]])

        await screen_locate.ainvoke({"description": "render", "mark": True, "note": "n"})

        self.assertEqual(self.overlay.markers[0].title, "render button")

    async def test_mark_says_so_when_nothing_matches(self):
        self.install([[]])

        result = await screen_locate.ainvoke({"description": "ghost", "mark": True, "note": "n"})

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

    def install_two_inputs(self):
        return self.install(
            [
                [
                    Detection("address bar", (10, 5, 190, 15)),
                    Detection("search field", (50, 60, 150, 80)),
                ]
            ]
        )

    async def test_click_refuses_ambiguous_descriptions(self):
        self.install_two_inputs()

        result = await screen_click.ainvoke({"description": "input field"})

        self.assertIn("Did not click: 2 elements matched", result)
        self.assertIn("match=<number>", result)
        self.assertIn("at top-center", result)
        self.assertIn("at bottom-center", result)
        self.assertEqual(self.pointer.clicks, [])

    async def test_click_match_picks_one_of_several(self):
        self.install_two_inputs()

        result = await screen_click.ainvoke({"description": "input field", "match": 2})

        self.assertIn("Clicked 'search field'", result)
        self.assertEqual(self.pointer.clicks, [("left", 1, ())])
        self.assertEqual(self.pointer.moves[-1], (100, 70))

    async def test_click_match_out_of_range_lists_candidates(self):
        self.install_two_inputs()

        result = await screen_click.ainvoke({"description": "input field", "match": 5})

        self.assertIn("match=5 is out of range", result)
        self.assertIn("address bar", result)
        self.assertEqual(self.pointer.clicks, [])

    async def test_mark_refuses_ambiguous_descriptions(self):
        self.install_two_inputs()

        result = await screen_locate.ainvoke({"description": "input field", "mark": True, "note": "n"})

        self.assertIn("Did not mark: 2 elements matched", result)
        self.assertEqual(self.overlay.markers, [])

    async def test_mark_match_picks_one_of_several(self):
        self.install_two_inputs()

        result = await screen_locate.ainvoke({"description": "input field", "mark": True, "note": "n", "match": 1})

        self.assertIn("Marked 'address bar'", result)
        self.assertEqual(self.overlay.markers[0].title, "address bar")

    async def test_locate_accepts_a_named_region(self):
        # 200x100 screen, "bottom-right" third starts at (133, 66); the
        # scripted detection is in crop coordinates and must map back.
        self.install([[Detection("wifi icon", (10, 5, 20, 15))]])

        result = await screen_locate.ainvoke({"description": "wifi icon", "region": "bottom-right"})

        self.assertIn("box=(143, 71, 153, 81)", result)

    async def test_locate_rejects_an_unknown_region_with_the_vocabulary(self):
        self.install([[Detection("x", (0, 0, 5, 5))]])

        result = await screen_locate.ainvoke({"description": "x", "region": "nowhere"})

        self.assertIn("Tool call failed", result)
        self.assertIn("top-bar", result)  # the error teaches the valid names

    async def test_click_accepts_a_region(self):
        self.install([[Detection("ok", (10, 5, 20, 15))]])

        result = await screen_click.ainvoke({"description": "ok", "region": "bottom-right"})

        self.assertIn("Clicked 'ok'", result)
        self.assertEqual(self.pointer.moves[-1], (148, 76))

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
