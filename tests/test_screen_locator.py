import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_locator import (
    Detection,
    box_center,
    capture_primary_screen,
    enable_dpi_awareness,
    move_pointer,
    parse_detections,
    parse_mode_command,
    run_interactive_loop,
    select_targets,
)


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
        with patch("screen_locator.sys.platform", "linux"):
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
            "<box><10><20><30>"
            "<box><100><200><300><400></box>",
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


class ModeCommandTests(unittest.TestCase):
    def test_valid_mode_commands_are_trimmed_and_case_insensitive(self):
        cases = {
            ":mode first": "first",
            "  :MODE ALL  ": "all",
        }

        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(parse_mode_command(command), expected)

    def test_invalid_mode_command_is_rejected_with_guidance(self):
        for command in (":mode nearest", ":mode", "mode all", ":mode all now"):
            with self.subTest(command=command):
                with self.assertRaisesRegex(
                    ValueError, "Use :mode first or :mode all"
                ):
                    parse_mode_command(command)


class InteractiveLoopTests(unittest.TestCase):
    @staticmethod
    def input_from(*values):
        commands = iter(values)

        def read_input(_prompt):
            value = next(commands)
            if isinstance(value, BaseException):
                raise value
            return value

        return read_input

    def test_each_query_uses_a_fresh_screenshot_and_matching_description(self):
        first_image = Image.new("RGB", (100, 100))
        second_image = Image.new("RGB", (200, 100))
        capture = Mock(side_effect=[first_image, second_image])
        locate = Mock(return_value=[])
        save = Mock()

        run_interactive_loop(
            locate=locate,
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=save,
            input_fn=self.input_from("first query", "second query", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        self.assertEqual(capture.call_count, 2)
        self.assertEqual(
            locate.call_args_list,
            [call(first_image, "first query"), call(second_image, "second query")],
        )
        self.assertEqual(
            save.call_args_list,
            [call(first_image, []), call(second_image, [])],
        )

    def test_default_mode_moves_only_to_first_clamped_center(self):
        image = Image.new("RGB", (100, 80))
        move = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (-30, -20, -10, -4)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from("button", "quit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        move.assert_called_once_with(0, 0)

    def test_all_mode_moves_to_every_center_in_order_and_pauses_between(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        pause = Mock()
        capture = Mock(return_value=image)

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=capture,
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from(":MODE ALL", "buttons", "ExIt"),
            output_fn=Mock(),
            pause_fn=pause,
        )

        capture.assert_called_once_with()
        self.assertEqual(move.call_args_list, [call(20, 20), call(50, 50)])
        pause.assert_called_once_with(0.35)

    def test_no_detections_saves_image_without_moving_pointer(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        save = Mock()
        output = Mock()

        run_interactive_loop(
            locate=Mock(return_value=[]),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=save,
            input_fn=self.input_from("missing", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        save.assert_called_once_with(image, [])
        move.assert_not_called()
        output.assert_any_call("Nothing found; pointer was not moved.")

    def test_empty_input_does_not_capture_screen(self):
        capture = Mock()
        output = Mock()
        input_fn = Mock(side_effect=self.input_from("   ", "exit"))

        run_interactive_loop(
            locate=Mock(),
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=input_fn,
            output_fn=output,
            pause_fn=Mock(),
        )

        capture.assert_not_called()
        help_text = output.call_args_list[0].args[0]
        self.assertIn(":mode first", help_text)
        self.assertIn(":mode all", help_text)
        self.assertIn("exit", help_text)
        self.assertIn("Что найти?", input_fn.call_args_list[0].args[0])
        self.assertIn("first", input_fn.call_args_list[0].args[0])

    def test_prints_every_box_and_center_in_detection_order(self):
        image = Image.new("RGB", (100, 100))
        output = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("button", (10, 20, 30, 40)),
                    Detection("icon", (50, 60, 90, 100)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=self.input_from("controls", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        detection_messages = [
            message.args[0]
            for message in output.call_args_list
            if message.args[0][:1].isdigit()
        ]
        self.assertEqual(
            detection_messages,
            [
                "1. button: box=(10, 20, 30, 40), center=(20, 30)",
                "2. icon: box=(50, 60, 90, 100), center=(70, 80)",
            ],
        )

    def test_invalid_mode_command_keeps_default_first_mode(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        output = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from(":mode nearest", "buttons", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        move.assert_called_once_with(20, 20)
        output.assert_any_call("Use :mode first or :mode all")

    def test_invalid_mode_command_preserves_selected_all_mode(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        pause = Mock()
        output = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from(
                ":mode all", ":mode nearest", "buttons", "exit"
            ),
            output_fn=output,
            pause_fn=pause,
        )

        output.assert_any_call("Use :mode first or :mode all")
        self.assertEqual(move.call_args_list, [call(20, 20), call(50, 50)])
        pause.assert_called_once_with(0.35)

    def test_mode_near_prefix_is_treated_as_an_ordinary_query(self):
        image = Image.new("RGB", (100, 100))
        locate = Mock(return_value=[])
        save = Mock()

        run_interactive_loop(
            locate=locate,
            capture_screen=Mock(return_value=image),
            move_pointer=Mock(),
            save_result=save,
            input_fn=self.input_from(":model button", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        locate.assert_called_once_with(image, ":model button")
        save.assert_called_once_with(image, [])

    def test_keyboard_interrupt_reports_stopped_without_capturing(self):
        capture = Mock()
        output = Mock()

        run_interactive_loop(
            locate=Mock(),
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=self.input_from(KeyboardInterrupt()),
            output_fn=output,
            pause_fn=Mock(),
        )

        capture.assert_not_called()
        output.assert_any_call("Stopped.")

    def test_keyboard_interrupt_during_query_reports_stopped_and_returns(self):
        image = Image.new("RGB", (100, 100))
        output = Mock()
        save = Mock()

        try:
            run_interactive_loop(
                locate=Mock(side_effect=KeyboardInterrupt),
                capture_screen=Mock(return_value=image),
                move_pointer=Mock(),
                save_result=save,
                input_fn=self.input_from("button"),
                output_fn=output,
                pause_fn=Mock(),
            )
        except KeyboardInterrupt:
            self.fail("query-stage KeyboardInterrupt must stop the loop cleanly")

        output.assert_any_call("Stopped.")
        save.assert_not_called()

    def test_query_failure_is_reported_and_following_query_still_runs(self):
        first_image = Image.new("RGB", (100, 100))
        second_image = Image.new("RGB", (200, 100))
        detection = Detection("recovered", (20, 30, 60, 70))
        locate = Mock(side_effect=[RuntimeError("model unavailable"), [detection]])
        save = Mock()
        move = Mock()
        output = Mock()

        run_interactive_loop(
            locate=locate,
            capture_screen=Mock(side_effect=[first_image, second_image]),
            move_pointer=move,
            save_result=save,
            input_fn=self.input_from("first", "second", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        output.assert_any_call("Query failed: model unavailable")
        save.assert_called_once_with(second_image, [detection])
        move.assert_called_once_with(40, 50)


if __name__ == "__main__":
    unittest.main()
