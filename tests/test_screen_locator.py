import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tools.computer_use import (
    ComputerUse,
    Detection,
    NullOverlay,
    box_center,
    capture_primary_screen,
    clamp_point,
    enable_dpi_awareness,
    move_pointer,
    parse_detections,
    primary_screen_size,
    select_targets,
)
from src.tools.computer_use.cli import parse_command, run_interactive_loop
from src.tools.computer_use.fakes import RecordingPointer, ScriptedLocator, StaticScreen


class ScreenAndPointerAdapterTests(unittest.TestCase):
    def test_capture_primary_screen_converts_non_rgb_image(self):
        source = Image.new("RGBA", (123, 45), (10, 20, 30, 40))
        grabber = Mock(return_value=source)

        screenshot = capture_primary_screen(grabber=grabber)

        grabber.assert_called_once_with(all_screens=False)
        self.assertEqual(screenshot.mode, "RGB")
        self.assertEqual(screenshot.size, source.size)

    def test_capture_primary_screen_returns_existing_rgb_image_unchanged(self):
        source = Image.new("RGB", (123, 45), (10, 20, 30))
        grabber = Mock(return_value=source)

        screenshot = capture_primary_screen(grabber=grabber)

        grabber.assert_called_once_with(all_screens=False)
        self.assertIs(screenshot, source)

    def test_primary_screen_size_reads_system_metrics(self):
        metrics = Mock(side_effect=[1920, 1080])
        grabber = Mock()

        size = primary_screen_size(get_system_metrics=metrics, grabber=grabber)

        self.assertEqual(size, (1920, 1080))
        grabber.assert_not_called()

    def test_primary_screen_size_falls_back_to_a_screenshot(self):
        grabber = Mock(return_value=Image.new("RGB", (800, 600)))

        size = primary_screen_size(
            platform="linux", get_system_metrics=None, grabber=grabber
        )

        self.assertEqual(size, (800, 600))

    def test_move_pointer_uses_injected_setter_with_integer_coordinates(self):
        setter = Mock(return_value=True)

        move_pointer(12, 34, set_cursor_position=setter)

        setter.assert_called_once_with(12, 34)

    def test_move_pointer_casts_coordinates_to_integers(self):
        setter = Mock(return_value=True)

        move_pointer(12.9, 34.1, set_cursor_position=setter)

        setter.assert_called_once_with(12, 34)

    def test_move_pointer_raises_when_setter_reports_failure(self):
        setter = Mock(return_value=False)

        with self.assertRaisesRegex(OSError, "^SetCursorPos failed$"):
            move_pointer(12, 34, set_cursor_position=setter)

    def test_enable_dpi_awareness_prefers_shcore(self):
        libraries = Mock()
        libraries.shcore.SetProcessDpiAwareness.return_value = 0

        enable_dpi_awareness(platform="win32", libraries=libraries)

        libraries.shcore.SetProcessDpiAwareness.assert_called_once_with(2)
        libraries.user32.SetProcessDPIAware.assert_not_called()

    def test_enable_dpi_awareness_falls_back_when_shcore_is_unavailable(self):
        for error in (AttributeError("missing"), OSError("unavailable")):
            with self.subTest(error=type(error).__name__):
                libraries = Mock()
                libraries.shcore.SetProcessDpiAwareness.side_effect = error
                libraries.user32.SetProcessDPIAware.return_value = True

                enable_dpi_awareness(platform="win32", libraries=libraries)

                libraries.user32.SetProcessDPIAware.assert_called_once_with()

    def test_enable_dpi_awareness_falls_back_for_failed_hresult(self):
        for failed_hresult in (0x80004005, -2147467259):
            with self.subTest(failed_hresult=failed_hresult):
                libraries = Mock()
                libraries.shcore.SetProcessDpiAwareness.return_value = failed_hresult
                libraries.user32.SetProcessDPIAware.return_value = True

                enable_dpi_awareness(platform="win32", libraries=libraries)

                libraries.user32.SetProcessDPIAware.assert_called_once_with()

    def test_enable_dpi_awareness_raises_when_legacy_fallback_fails(self):
        libraries = Mock()
        libraries.shcore.SetProcessDpiAwareness.return_value = 0x80004005
        libraries.user32.SetProcessDPIAware.return_value = False

        with self.assertRaisesRegex(OSError, "SetProcessDPIAware failed"):
            enable_dpi_awareness(platform="win32", libraries=libraries)

    def test_enable_dpi_awareness_treats_access_denied_as_already_configured(self):
        for access_denied in (0x80070005, -2147024891):
            with self.subTest(access_denied=access_denied):
                libraries = Mock()
                libraries.shcore.SetProcessDpiAwareness.return_value = access_denied

                enable_dpi_awareness(platform="win32", libraries=libraries)

                libraries.user32.SetProcessDPIAware.assert_not_called()

    def test_enable_dpi_awareness_is_a_non_windows_no_op(self):
        libraries = Mock(spec=[])

        enable_dpi_awareness(platform="linux", libraries=libraries)

        self.assertEqual(libraries.mock_calls, [])

    def test_default_move_pointer_requires_windows(self):
        with patch("src.tools.computer_use.pointer.sys.platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "requires Windows"):
                move_pointer(12, 34)


class ParseDetectionsTests(unittest.TestCase):
    def test_reference_label_and_coordinates_are_scaled_to_the_screenshot(self):
        detections = parse_detections(
            "<ref>submit button</ref><box><100><250><900><750></box>",
            (1919, 1079),
        )

        self.assertEqual(
            detections,
            [Detection(label="submit button", box=(192, 270, 1727, 809))],
        )

    def test_multiple_boxes_preserve_order_and_missing_labels_use_object(self):
        detections = parse_detections(
            "prefix <box><0><0><100><100></box>"
            "<ref>icons</ref>"
            "<box><200><300><400><500></box>"
            "<box><600><700><800><900></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [
                Detection(label="object", box=(0, 0, 100, 100)),
                Detection(label="icons", box=(200, 300, 400, 500)),
                Detection(label="icons", box=(600, 700, 800, 900)),
            ],
        )

    def test_malformed_boxes_are_ignored(self):
        detections = parse_detections(
            "<ref>targets</ref>"
            "<box><10><20><30></box>"
            "<box><-10><+20><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="targets", box=(-10, 20, 300, 400))],
        )

    def test_valid_box_after_unterminated_box_is_recovered(self):
        detections = parse_detections(
            "<box><10><20><30><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="object", box=(100, 200, 300, 400))],
        )

    def test_labeled_box_after_unterminated_box_is_recovered(self):
        detections = parse_detections(
            "<box><10><20><30>"
            "<ref>good</ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="good", box=(100, 200, 300, 400))],
        )

    def test_reference_labels_are_stripped(self):
        detections = parse_detections(
            "<ref>  submit button  </ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="submit button", box=(100, 200, 300, 400))],
        )

    def test_empty_reference_label_uses_object(self):
        detections = parse_detections(
            "<ref></ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="object", box=(100, 200, 300, 400))],
        )


class BoxCenterTests(unittest.TestCase):
    def test_midpoint_is_returned_as_integer_coordinates(self):
        self.assertEqual(box_center((10, 20, 31, 43), (100, 100)), (20, 31))

    def test_midpoint_is_clamped_to_the_screenshot(self):
        self.assertEqual(box_center((-30, -20, -10, -4), (80, 60)), (0, 0))
        self.assertEqual(box_center((90, 70, 110, 90), (80, 60)), (79, 59))

    def test_clamp_point_rejects_empty_images(self):
        with self.assertRaisesRegex(ValueError, "image dimensions must be positive"):
            clamp_point((1, 1), (0, 10))


class SelectTargetsTests(unittest.TestCase):
    def setUp(self):
        self.detections = [
            Detection(label="first", box=(0, 0, 10, 10)),
            Detection(label="second", box=(20, 20, 30, 30)),
        ]

    def test_first_returns_only_the_first_detection(self):
        self.assertEqual(select_targets(self.detections, "first"), self.detections[:1])
        self.assertEqual(select_targets([], "first"), [])

    def test_all_returns_every_detection_in_order(self):
        self.assertEqual(select_targets(self.detections, "all"), self.detections)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported target mode"):
            select_targets(self.detections, "nearest")  # type: ignore[arg-type]


class CommandParsingTests(unittest.TestCase):
    def test_valid_mode_commands_are_trimmed_and_case_insensitive(self):
        for text, expected in {":mode first": "first", "  :MODE ALL  ": "all"}.items():
            with self.subTest(text=text):
                command = parse_command(text)
                self.assertEqual(command.kind, "mode")
                self.assertEqual(command.mode, expected)

    def test_invalid_mode_command_is_rejected_with_guidance(self):
        for text in (":mode nearest", ":mode", ":mode all now"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "Use :mode first or :mode all"):
                    parse_command(text)

    def test_mark_command_splits_description_and_note(self):
        command = parse_command(":mark render button | starts the export")

        self.assertEqual(command.kind, "mark")
        self.assertEqual(command.argument, "render button")
        self.assertEqual(command.note, "starts the export")

    def test_mark_command_without_a_note_keeps_the_description(self):
        command = parse_command(":mark render button")

        self.assertEqual((command.kind, command.argument, command.note),
                         ("mark", "render button", ""))

    def test_mark_command_requires_a_description(self):
        with self.assertRaisesRegex(ValueError, "Use :mark"):
            parse_command(":mark  | note only")

    def test_click_command_carries_the_description(self):
        command = parse_command(":click ok button")

        self.assertEqual((command.kind, command.argument), ("click", "ok button"))

    def test_click_command_requires_a_description(self):
        with self.assertRaisesRegex(ValueError, "Use :click"):
            parse_command(":click")

    def test_exit_and_empty_and_plain_queries_are_classified(self):
        self.assertEqual(parse_command("exit").kind, "exit")
        self.assertEqual(parse_command("QUIT").kind, "exit")
        self.assertEqual(parse_command("   ").kind, "empty")
        self.assertEqual(parse_command(":model button").kind, "query")
        self.assertEqual(parse_command(" save button ").argument, "save button")


class InteractiveLoopTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_path = Path(directory.name) / "annotated.jpg"

    @staticmethod
    def input_from(*values):
        commands = iter(values)

        def read_input(_prompt):
            value = next(commands)
            if isinstance(value, BaseException):
                raise value
            return value

        return read_input

    def build_computer(self, *, locator, images=None, size=(100, 100)):
        self.screen = StaticScreen(images or [Image.new("RGB", size)], size)
        self.pointer = RecordingPointer()
        self.overlay = NullOverlay()
        return ComputerUse(
            locator=locator,
            screen=self.screen,
            pointer=self.pointer,
            overlay=self.overlay,
            output_path=self.output_path,
            dpi_aware=False,
            min_target_px=0,  # these tests cover the loop, not two-stage locating
        )

    def test_each_query_uses_a_fresh_screenshot_and_matching_description(self):
        locator = ScriptedLocator([[]])
        computer = self.build_computer(
            locator=locator,
            images=[Image.new("RGB", (100, 100)), Image.new("RGB", (200, 100))],
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from("first query", "second query", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        self.assertEqual(self.screen.captures, 2)
        self.assertEqual(
            locator.calls,
            [((100, 100), "first query"), ((200, 100), "second query")],
        )
        self.assertTrue(self.output_path.is_file())

    def test_default_mode_moves_only_to_first_clamped_center(self):
        computer = self.build_computer(
            locator=ScriptedLocator(
                [
                    [
                        Detection("first", (-30, -20, -10, -4)),
                        Detection("second", (40, 40, 60, 60)),
                    ]
                ]
            ),
            size=(100, 80),
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from("button", "quit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        self.assertEqual(self.pointer.moves, [(0, 0)])

    def test_all_mode_moves_to_every_center_in_order_and_pauses_between(self):
        pause = Mock()
        computer = self.build_computer(
            locator=ScriptedLocator(
                [
                    [
                        Detection("first", (10, 10, 30, 30)),
                        Detection("second", (40, 40, 60, 60)),
                    ]
                ]
            )
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from(":MODE ALL", "buttons", "ExIt"),
            output_fn=Mock(),
            pause_fn=pause,
        )

        self.assertEqual(self.screen.captures, 1)
        self.assertEqual(self.pointer.moves, [(20, 20), (50, 50)])
        pause.assert_called_once_with(0.35)

    def test_no_detections_saves_image_without_moving_pointer(self):
        output = Mock()
        computer = self.build_computer(locator=ScriptedLocator([[]]))

        run_interactive_loop(
            computer,
            input_fn=self.input_from("missing", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        self.assertTrue(self.output_path.is_file())
        self.assertEqual(self.pointer.moves, [])
        output.assert_any_call("Nothing found; pointer was not moved.")

    def test_empty_input_does_not_capture_screen(self):
        output = Mock()
        input_fn = Mock(side_effect=self.input_from("   ", "exit"))
        computer = self.build_computer(locator=ScriptedLocator([[]]))

        run_interactive_loop(
            computer, input_fn=input_fn, output_fn=output, pause_fn=Mock()
        )

        self.assertEqual(self.screen.captures, 0)
        help_text = output.call_args_list[0].args[0]
        self.assertIn(":mode first", help_text)
        self.assertIn(":mode all", help_text)
        self.assertIn(":mark", help_text)
        self.assertIn(":click", help_text)
        self.assertIn("exit", help_text)
        self.assertIn("Что найти?", input_fn.call_args_list[0].args[0])
        self.assertIn("first", input_fn.call_args_list[0].args[0])

    def test_prints_every_box_and_center_in_detection_order(self):
        output = Mock()
        computer = self.build_computer(
            locator=ScriptedLocator(
                [
                    [
                        Detection("button", (10, 20, 30, 40)),
                        Detection("icon", (50, 60, 90, 100)),
                    ]
                ]
            )
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from("controls", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        output.assert_any_call(
            "1. button: box=(10, 20, 30, 40), center=(20, 30)\n"
            "2. icon: box=(50, 60, 90, 100), center=(70, 80)"
        )

    def test_invalid_mode_command_keeps_default_first_mode(self):
        output = Mock()
        computer = self.build_computer(
            locator=ScriptedLocator(
                [
                    [
                        Detection("first", (10, 10, 30, 30)),
                        Detection("second", (40, 40, 60, 60)),
                    ]
                ]
            )
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from(":mode nearest", "buttons", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        self.assertEqual(self.pointer.moves, [(20, 20)])
        output.assert_any_call("Use :mode first or :mode all")

    def test_mode_near_prefix_is_treated_as_an_ordinary_query(self):
        locator = ScriptedLocator([[]])
        computer = self.build_computer(locator=locator)

        run_interactive_loop(
            computer,
            input_fn=self.input_from(":model button", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        self.assertEqual(locator.calls, [((100, 100), ":model button")])

    def test_mark_command_moves_the_pointer_and_shows_a_tooltip(self):
        output = Mock()
        computer = self.build_computer(
            locator=ScriptedLocator([[Detection("render button", (10, 20, 30, 40))]])
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from(
                ":mark render button | starts the export", "exit"
            ),
            output_fn=output,
            pause_fn=Mock(),
        )

        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [])
        self.assertEqual(len(self.overlay.markers), 1)
        marker = self.overlay.markers[0]
        self.assertEqual(marker.box, (10, 20, 30, 40))
        self.assertEqual(marker.title, "render button")
        self.assertEqual(marker.note, "starts the export")
        self.assertEqual(marker.anchor, (20, 30))
        output.assert_any_call("Marked render button at (20, 30)")

    def test_click_command_clicks_the_center_of_the_match(self):
        output = Mock()
        computer = self.build_computer(
            locator=ScriptedLocator([[Detection("ok", (10, 10, 30, 30))]])
        )

        run_interactive_loop(
            computer,
            input_fn=self.input_from(":click ok button", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        self.assertEqual(self.pointer.moves, [(20, 20)])
        self.assertEqual(self.pointer.clicks, [("left", 1, ())])
        output.assert_any_call("Clicked ok at (20, 20)")

    def test_keyboard_interrupt_reports_stopped_without_capturing(self):
        output = Mock()
        computer = self.build_computer(locator=ScriptedLocator([[]]))

        run_interactive_loop(
            computer,
            input_fn=self.input_from(KeyboardInterrupt()),
            output_fn=output,
            pause_fn=Mock(),
        )

        self.assertEqual(self.screen.captures, 0)
        output.assert_any_call("Stopped.")

    def test_keyboard_interrupt_during_query_reports_stopped_and_returns(self):
        output = Mock()
        locator = Mock()
        locator.locate.side_effect = KeyboardInterrupt
        computer = self.build_computer(locator=locator)

        try:
            run_interactive_loop(
                computer,
                input_fn=self.input_from("button"),
                output_fn=output,
                pause_fn=Mock(),
            )
        except KeyboardInterrupt:
            self.fail("query-stage KeyboardInterrupt must stop the loop cleanly")

        output.assert_any_call("Stopped.")
        self.assertFalse(self.output_path.exists())

    def test_query_failure_is_reported_and_following_query_still_runs(self):
        output = Mock()
        detection = Detection("recovered", (20, 30, 60, 70))
        locator = Mock()
        locator.locate.side_effect = [RuntimeError("model unavailable"), [detection]]
        computer = self.build_computer(locator=locator)

        run_interactive_loop(
            computer,
            input_fn=self.input_from("first", "second", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        output.assert_any_call("Query failed: model unavailable")
        self.assertEqual(self.pointer.moves, [(40, 50)])


if __name__ == "__main__":
    unittest.main()
