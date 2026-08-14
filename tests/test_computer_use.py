import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tools.computer_use import ComputerUse, Detection, NullOverlay
from src.tools.computer_use.clipboard import MemoryClipboard
from src.tools.computer_use.fakes import (
    RecordingPointer,
    ScriptedLocator,
    StaticScreen,
)
from src.tools.computer_use.overlay import (
    OverlayStyle,
    connector_corner,
    enable_click_through,
    tooltip_placement,
    wrap_note,
)
from src.tools.computer_use.pointer import (
    KeyEvent,
    MOUSEEVENTF_HWHEEL,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP,
    MOUSEEVENTF_WHEEL,
    Pointer,
    TextAction,
    WHEEL_DELTA,
    interpolate,
    key_combination_events,
    key_events,
    resolve_key,
    scroll_command,
    text_actions,
)


VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_RETURN = 0x0D
VK_LEFT = 0x25


class RecordingDispatcher:
    def __init__(self):
        self.events = []

    def mouse(self, flags, data=0):
        self.events.append(("mouse", flags, data))

    def key(self, code, *, pressed, extended=False):
        self.events.append(("key", code, pressed, extended))

    def unicode(self, code_unit, *, pressed):
        self.events.append(("unicode", code_unit, pressed))


class KeyResolutionTests(unittest.TestCase):
    def test_named_keys_are_case_insensitive(self):
        self.assertEqual(resolve_key("Ctrl"), VK_CONTROL)
        self.assertEqual(resolve_key("  ENTER "), VK_RETURN)
        self.assertEqual(resolve_key("f5"), 0x74)

    def test_letters_and_digits_map_to_their_virtual_key_codes(self):
        self.assertEqual(resolve_key("a"), ord("A"))
        self.assertEqual(resolve_key("Z"), ord("Z"))
        self.assertEqual(resolve_key("7"), ord("7"))

    def test_unsupported_names_point_at_type_text(self):
        for name in ("", "  ", "ctrl+s", "±"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_key(name)

    def test_combination_presses_modifiers_first_and_releases_in_reverse(self):
        events = key_combination_events("Ctrl+Shift+S")

        self.assertEqual(
            events,
            [
                KeyEvent(VK_CONTROL, True, False),
                KeyEvent(VK_SHIFT, True, False),
                KeyEvent(ord("S"), True, False),
                KeyEvent(ord("S"), False, False),
                KeyEvent(VK_SHIFT, False, False),
                KeyEvent(VK_CONTROL, False, False),
            ],
        )

    def test_extended_keys_are_flagged(self):
        events = key_combination_events("left")

        self.assertEqual(events, [KeyEvent(VK_LEFT, True, True), KeyEvent(VK_LEFT, False, True)])

    def test_only_modifiers_may_precede_the_final_key(self):
        with self.assertRaisesRegex(ValueError, "Only modifier keys"):
            key_combination_events("a+b")

    def test_empty_combinations_are_rejected(self):
        for combination in ("", "+", "   "):
            with self.subTest(combination=combination):
                with self.assertRaises(ValueError):
                    key_combination_events(combination)

    def test_a_sequence_runs_each_combination_in_order(self):
        events = key_events("ctrl+a delete")

        self.assertEqual(
            [(event.code, event.pressed) for event in events],
            [
                (VK_CONTROL, True),
                (ord("A"), True),
                (ord("A"), False),
                (VK_CONTROL, False),
                (0x2E, True),
                (0x2E, False),
            ],
        )


class TextActionTests(unittest.TestCase):
    def test_plain_text_becomes_utf16_code_units(self):
        self.assertEqual(
            text_actions("Hi"),
            [TextAction("unicode", ord("H")), TextAction("unicode", ord("i"))],
        )

    def test_cyrillic_text_is_supported(self):
        self.assertEqual(text_actions("я"), [TextAction("unicode", ord("я"))])

    def test_newlines_and_tabs_become_key_presses(self):
        self.assertEqual(
            text_actions("\n\t"),
            [TextAction("key", VK_RETURN), TextAction("key", 0x09)],
        )

    def test_unpaired_surrogates_are_rejected_with_a_readable_message(self):
        # Text decoded with the wrong codec arrives carrying lone surrogates.
        broken = "привет".encode("utf-8").decode("ascii", "surrogateescape")

        with self.assertRaises(ValueError) as caught:
            text_actions(broken)

        message = str(caught.exception)
        self.assertIn("unpaired surrogate", message)
        self.assertIn("UTF-8", message)

    def test_astral_characters_are_split_into_surrogate_pairs(self):
        actions = text_actions("😀")

        self.assertEqual([action.kind for action in actions], ["unicode", "unicode"])
        self.assertEqual(
            "".join(
                action.code.to_bytes(2, "little").decode("utf-16-le", "surrogatepass")
                for action in actions
            ).encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass"),
            "😀",
        )


class ScrollCommandTests(unittest.TestCase):
    def test_vertical_and_horizontal_directions_use_matching_wheels(self):
        self.assertEqual(scroll_command("up", 2), (MOUSEEVENTF_WHEEL, 2 * WHEEL_DELTA))
        self.assertEqual(
            scroll_command("down", 2), (MOUSEEVENTF_WHEEL, -2 * WHEEL_DELTA)
        )
        self.assertEqual(
            scroll_command("right", 1), (MOUSEEVENTF_HWHEEL, WHEEL_DELTA)
        )
        self.assertEqual(
            scroll_command("left", 1), (MOUSEEVENTF_HWHEEL, -WHEEL_DELTA)
        )

    def test_invalid_direction_and_amount_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported scroll direction"):
            scroll_command("sideways", 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            scroll_command("up", -1)


class InterpolateTests(unittest.TestCase):
    def test_intermediate_points_end_at_the_target(self):
        self.assertEqual(
            interpolate((0, 0), (10, 20), 2), [(5, 10), (10, 20)]
        )

    def test_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "steps must be positive"):
            interpolate((0, 0), (1, 1), 0)


class PointerTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = RecordingDispatcher()
        self.sleeps = []
        self.moves = []
        self.pointer = Pointer(
            dispatcher=self.dispatcher,
            set_cursor_position=lambda x, y: self.moves.append((x, y)) or True,
            sleep=self.sleeps.append,
            drag_steps=2,
        )

    def test_click_emits_one_down_up_pair(self):
        self.pointer.click()

        self.assertEqual(
            self.dispatcher.events,
            [
                ("mouse", MOUSEEVENTF_LEFTDOWN, 0),
                ("mouse", MOUSEEVENTF_LEFTUP, 0),
            ],
        )

    def test_double_click_pauses_between_clicks(self):
        self.pointer.click("right", count=2)

        self.assertEqual(
            self.dispatcher.events,
            [
                ("mouse", MOUSEEVENTF_RIGHTDOWN, 0),
                ("mouse", MOUSEEVENTF_RIGHTUP, 0),
                ("mouse", MOUSEEVENTF_RIGHTDOWN, 0),
                ("mouse", MOUSEEVENTF_RIGHTUP, 0),
            ],
        )
        self.assertEqual(self.sleeps, [0.02])

    def test_modifiers_wrap_the_click(self):
        self.pointer.click(modifiers=("ctrl", "shift"))

        self.assertEqual(
            self.dispatcher.events,
            [
                ("key", VK_CONTROL, True, False),
                ("key", VK_SHIFT, True, False),
                ("mouse", MOUSEEVENTF_LEFTDOWN, 0),
                ("mouse", MOUSEEVENTF_LEFTUP, 0),
                ("key", VK_SHIFT, False, False),
                ("key", VK_CONTROL, False, False),
            ],
        )

    def test_non_modifier_click_modifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-modifier key"):
            self.pointer.click(modifiers=("a",))

    def test_unsupported_button_and_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported mouse button"):
            self.pointer.click("back")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "click count must be positive"):
            self.pointer.click(count=0)

    def test_drag_holds_the_button_across_interpolated_moves(self):
        self.pointer.drag((0, 0), (10, 10))

        self.assertEqual(self.moves, [(0, 0), (5, 5), (10, 10)])
        self.assertEqual(
            self.dispatcher.events,
            [
                ("mouse", MOUSEEVENTF_LEFTDOWN, 0),
                ("mouse", MOUSEEVENTF_LEFTUP, 0),
            ],
        )

    def test_scroll_sends_the_signed_wheel_delta(self):
        self.pointer.scroll("down", 3)

        self.assertEqual(
            self.dispatcher.events, [("mouse", MOUSEEVENTF_WHEEL, -3 * WHEEL_DELTA)]
        )

    def test_type_text_presses_and_releases_each_code_unit(self):
        self.pointer.type_text("a\n")

        self.assertEqual(
            self.dispatcher.events,
            [
                ("unicode", ord("a"), True),
                ("unicode", ord("a"), False),
                ("key", VK_RETURN, True, False),
                ("key", VK_RETURN, False, False),
            ],
        )

    def test_hold_key_sleeps_between_press_and_release(self):
        self.pointer.hold_key("shift", 0.5)

        self.assertEqual(
            self.dispatcher.events,
            [("key", VK_SHIFT, True, False), ("key", VK_SHIFT, False, False)],
        )
        self.assertEqual(self.sleeps, [0.5])

    def test_hold_key_rejects_negative_durations(self):
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            self.pointer.hold_key("shift", -1)

    def test_move_reports_setter_failures(self):
        pointer = Pointer(
            dispatcher=self.dispatcher, set_cursor_position=lambda x, y: False
        )

        with self.assertRaisesRegex(OSError, "SetCursorPos failed"):
            pointer.move(1, 2)


class OverlayHelperTests(unittest.TestCase):
    def test_wrap_note_wraps_long_lines_and_keeps_author_breaks(self):
        self.assertEqual(
            wrap_note("one two three\n\nfour", width=7),
            ["one two", "three", "four"],
        )

    def test_wrap_note_of_empty_text_is_empty(self):
        self.assertEqual(wrap_note("", width=10), [])

    def test_wrap_note_requires_a_positive_width(self):
        with self.assertRaisesRegex(ValueError, "width must be positive"):
            wrap_note("text", width=0)

    def test_tooltip_sits_below_and_right_of_the_anchor(self):
        self.assertEqual(
            tooltip_placement((100, 100), (200, 80), (1920, 1080), offset=20, margin=10),
            (120, 120),
        )

    def test_tooltip_flips_when_it_would_leave_the_screen(self):
        self.assertEqual(
            tooltip_placement((1900, 1060), (200, 80), (1920, 1080), offset=20, margin=10),
            (1680, 960),
        )

    def test_tooltip_is_clamped_inside_the_margins(self):
        self.assertEqual(
            tooltip_placement((5, 5), (200, 80), (1920, 1080), offset=0, margin=10),
            (10, 10),
        )

    def test_connector_targets_the_corner_facing_the_anchor(self):
        self.assertEqual(connector_corner((10, 10), (100, 100), (50, 40)), (100, 100))
        self.assertEqual(connector_corner((400, 10), (100, 100), (50, 40)), (150, 100))
        self.assertEqual(connector_corner((400, 400), (100, 100), (50, 40)), (150, 140))

    def test_click_through_adds_the_transparent_window_styles(self):
        user32 = Mock()
        user32.GetWindowLongW.return_value = 0x1

        enable_click_through(4242, user32=user32)

        user32.GetWindowLongW.assert_called_once_with(4242, -20)
        _, styles = user32.SetWindowLongW.call_args.args[1:]
        self.assertTrue(styles & 0x00000020)
        self.assertTrue(styles & 0x00080000)
        self.assertTrue(styles & 0x1)

    def test_null_overlay_records_markers(self):
        overlay = NullOverlay()
        self.assertFalse(overlay.visible)

        overlay.show(Mock())
        self.assertTrue(overlay.visible)

        overlay.hide()
        self.assertFalse(overlay.visible)
        self.assertEqual(len(overlay.markers), 1)

    def test_overlay_style_defaults_are_readable(self):
        style = OverlayStyle()

        self.assertGreater(style.wrap_chars, 20)
        self.assertGreater(style.outline_width, 0)


class ComputerUseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.screenshot = Image.new("RGB", (200, 100), "white")
        self.pointer = RecordingPointer()
        self.overlay = NullOverlay()

    def build(self, responses=((),), *, size=(200, 100)):
        self.locator = ScriptedLocator(responses)
        self.screen = StaticScreen([self.screenshot], size)
        return ComputerUse(
            locator=self.locator,
            screen=self.screen,
            pointer=self.pointer,
            overlay=self.overlay,
            output_path=self.directory / "annotated.jpg",
            dpi_aware=False,
            sleep=Mock(),
            min_target_px=0,  # refinement has its own tests below
        )

    def test_get_screenshot_remembers_and_optionally_saves_the_capture(self):
        computer = self.build()
        destination = self.directory / "screen.png"

        screenshot = computer.get_screenshot(save_to=destination)

        self.assertIs(screenshot, self.screenshot)
        self.assertIs(computer.last_screenshot, self.screenshot)
        self.assertTrue(destination.is_file())

    def test_locate_object_returns_every_match_and_records_them(self):
        detections = [Detection("save", (0, 0, 20, 20)), Detection("open", (40, 0, 60, 20))]
        computer = self.build([detections])

        found = computer.locate_object("toolbar buttons")

        self.assertEqual(found, detections)
        self.assertEqual(computer.last_detections, detections)
        self.assertEqual(self.locator.calls, [((200, 100), "toolbar buttons")])

    def test_locate_object_can_trim_to_the_first_match_and_annotate(self):
        detections = [Detection("save", (0, 0, 20, 20)), Detection("open", (40, 0, 60, 20))]
        computer = self.build([detections])
        annotated = self.directory / "boxes.jpg"

        found = computer.locate_object("buttons", mode="first", annotate_to=annotated)

        self.assertEqual(found, detections[:1])
        self.assertTrue(annotated.is_file())

    def test_locate_object_reuses_a_supplied_screenshot(self):
        computer = self.build([[]])
        supplied = Image.new("RGB", (30, 30))

        computer.locate_object("anything", screenshot=supplied)

        self.assertEqual(self.screen.captures, 0)
        self.assertEqual(self.locator.calls, [((30, 30), "anything")])

    def test_find_object_and_locate_point_use_the_first_match(self):
        computer = self.build([[Detection("save", (10, 20, 30, 40))]])

        self.assertEqual(computer.find_object("save"), Detection("save", (10, 20, 30, 40)))
        self.assertEqual(computer.locate_point("save"), (20, 30))

    def test_locate_point_is_none_when_nothing_matches(self):
        computer = self.build([[]])

        self.assertIsNone(computer.locate_point("missing"))

    def test_describe_reports_boxes_and_centers(self):
        computer = self.build([[Detection("save", (10, 20, 30, 40))]])
        computer.locate_object("save")

        self.assertEqual(
            computer.describe(),
            "1. save: box=(10, 20, 30, 40), center=(20, 30)",
        )

    def test_describe_without_detections_says_so(self):
        computer = self.build()

        self.assertEqual(
            computer.describe(), "No matching region was found on the screen."
        )

    def test_save_annotated_requires_a_screenshot(self):
        computer = self.build()

        with self.assertRaisesRegex(RuntimeError, "No screenshot to annotate"):
            computer.save_annotated()

    def test_move_mouse_clamps_to_the_screen(self):
        computer = self.build()

        self.assertEqual(computer.move_mouse(500, -20), (199, 0))
        self.assertEqual(self.pointer.moves, [(199, 0)])

    def test_clicks_move_first_only_when_coordinates_are_given(self):
        computer = self.build()

        computer.left_click(10, 20)
        computer.right_click()
        computer.double_click(30, 40)
        computer.triple_click()
        computer.middle_click(modifiers=("ctrl",))

        self.assertEqual(self.pointer.moves, [(10, 20), (30, 40)])
        self.assertEqual(
            self.pointer.clicks,
            [
                ("left", 1, ()),
                ("right", 1, ()),
                ("left", 2, ()),
                ("left", 3, ()),
                ("middle", 1, ("ctrl",)),
            ],
        )

    def test_scroll_drag_and_keyboard_reach_the_pointer_backend(self):
        computer = self.build()

        computer.scroll("down", 5, 10, 20)
        computer.drag((0, 0), (500, 500))
        computer.type_text("hello")
        computer.key("ctrl+s")
        computer.hold_key("shift", 0.2)

        self.assertEqual(self.pointer.scrolls, [("down", 5, ())])
        self.assertEqual(self.pointer.drags, [((0, 0), (199, 99), "left")])
        self.assertEqual(self.pointer.typed, ["hello"])
        self.assertEqual(self.pointer.keys, ["ctrl+s"])
        self.assertEqual(self.pointer.holds, [("shift", 0.2)])

    def test_wait_rejects_negative_durations(self):
        computer = self.build()

        with self.assertRaisesRegex(ValueError, "must not be negative"):
            computer.wait(-1)

    def test_click_object_clicks_the_center_of_the_match(self):
        computer = self.build([[Detection("save", (10, 20, 30, 40))]])

        detection = computer.click_object("save button")

        self.assertEqual(detection, Detection("save", (10, 20, 30, 40)))
        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [("left", 1, ())])

    def test_click_object_does_nothing_when_no_match_is_found(self):
        computer = self.build([[]])

        self.assertIsNone(computer.click_object("missing"))
        self.assertEqual(self.pointer.moves, [])
        self.assertEqual(self.pointer.clicks, [])

    def test_move_to_object_never_clicks(self):
        computer = self.build([[Detection("save", (10, 20, 30, 40))]])

        computer.move_to_object("save")

        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [])

    def test_type_into_object_can_clear_the_field_first(self):
        computer = self.build([[Detection("search", (10, 20, 30, 40))]])

        computer.type_into_object("search field", "query", clear=True)

        self.assertEqual(self.pointer.clicks, [("left", 1, ())])
        self.assertEqual(self.pointer.keys, ["ctrl+a", "delete"])
        self.assertEqual(self.pointer.typed, ["query"])

    def test_mark_object_points_at_the_element_and_explains_it(self):
        computer = self.build([[Detection("render button", (10, 20, 30, 40))]])

        detection = computer.mark_object(
            "кнопка рендера", "Запускает просчёт таймлайна", duration=3.0
        )

        self.assertEqual(detection.label, "render button")
        self.assertEqual(self.pointer.moves, [(20, 30)])
        self.assertEqual(self.pointer.clicks, [])
        marker = self.overlay.markers[0]
        self.assertEqual(marker.box, (10, 20, 30, 40))
        self.assertEqual(marker.title, "render button")
        self.assertEqual(marker.note, "Запускает просчёт таймлайна")
        self.assertEqual(marker.anchor, (20, 30))
        self.assertEqual(marker.duration, 3.0)

    def test_mark_object_can_use_an_explicit_title_and_leave_the_pointer(self):
        computer = self.build([[Detection("btn", (10, 20, 30, 40))]])

        computer.mark_object(
            "render", "note", title="Кнопка Render", move_pointer=False
        )

        self.assertEqual(self.pointer.moves, [])
        self.assertEqual(self.overlay.markers[0].title, "Кнопка Render")

    def test_mark_object_without_a_match_shows_nothing(self):
        computer = self.build([[]])

        self.assertIsNone(computer.mark_object("missing", "note"))
        self.assertEqual(self.overlay.markers, [])
        self.assertEqual(self.pointer.moves, [])

    def test_mark_point_builds_a_box_around_the_coordinate(self):
        computer = self.build()

        computer.mark_point(50, 60, "Тут", "подсказка", radius=10)

        marker = self.overlay.markers[0]
        self.assertEqual(marker.box, (40, 50, 60, 70))
        self.assertEqual(marker.anchor, (50, 60))
        self.assertEqual(marker.duration, computer.mark_duration)

    def test_clear_marks_hides_without_closing_the_overlay(self):
        computer = self.build()
        computer.mark_point(10, 10, "title")

        computer.clear_marks()

        self.assertFalse(self.overlay.visible)
        self.assertFalse(self.overlay.closed)

    def test_close_releases_the_overlay(self):
        computer = self.build()
        computer.mark_point(10, 10, "title")

        computer.close()

        self.assertFalse(self.overlay.visible)
        self.assertTrue(self.overlay.closed)

    def test_copy_presses_ctrl_c_and_returns_the_clipboard(self):
        computer = self.build()
        computer._clipboard = MemoryClipboard("https://example.com/watch?v=abc")

        value = computer.copy()

        self.assertEqual(self.pointer.keys, ["ctrl+c"])
        self.assertEqual(value, "https://example.com/watch?v=abc")

    def test_copy_address_bar_focuses_the_bar_first(self):
        computer = self.build()
        computer._clipboard = MemoryClipboard("https://youtu.be/xyz")

        value = computer.copy_address_bar()

        self.assertEqual(self.pointer.keys, ["ctrl+l", "ctrl+c"])
        self.assertEqual(value, "https://youtu.be/xyz")

    def test_paste_text_puts_the_text_on_the_clipboard_and_pastes_it(self):
        computer = self.build()
        clipboard = MemoryClipboard()
        computer._clipboard = clipboard

        computer.paste_text("line one\nline two")

        self.assertEqual(clipboard.writes, ["line one\nline two"])
        self.assertEqual(self.pointer.keys, ["ctrl+v"])
        # Typing it would have sent the message on the newline instead.
        self.assertEqual(self.pointer.typed, [])

    def test_an_injected_locator_is_used_as_is(self):
        computer = self.build()

        self.assertIs(computer.locator, self.locator)


class StubLocator:
    """Mimics the real locator: one answer per call, in the coordinates of the
    image it was handed."""

    max_image_side = 768

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sizes = []

    def locate(self, image, description):
        self.sizes.append(image.size)
        index = min(len(self.sizes) - 1, len(self.answers) - 1)
        answer = self.answers[index]
        return [Detection("target", box) for box in answer]


class TwoStageLocateTests(unittest.TestCase):
    SCREEN = (1920, 1080)
    # A text field: 42 px tall on screen, 17 px once the screen is squeezed
    # into 768 px — small enough that the coarse box drifts.
    COARSE = (1225, 892, 1859, 932)
    FINE_IN_CROP = (20, 217, 560, 259)

    def build(self, locator, **options):
        self.pointer = RecordingPointer()
        return ComputerUse(
            locator=locator,
            screen=StaticScreen([Image.new("RGB", self.SCREEN)], self.SCREEN),
            pointer=self.pointer,
            overlay=NullOverlay(),
            dpi_aware=False,
            **options,
        )

    def expected_region(self, box, margin=160):
        return (
            max(0, box[0] - margin),
            max(0, box[1] - margin),
            min(self.SCREEN[0], box[2] + margin),
            min(self.SCREEN[1], box[3] + margin),
        )

    def test_a_small_target_is_located_again_on_a_crop(self):
        locator = StubLocator([self.COARSE], [self.FINE_IN_CROP])
        computer = self.build(locator)

        detections = computer.locate_object("the message input field", mode="first")

        region = self.expected_region(self.COARSE)
        self.assertEqual(locator.sizes[0], self.SCREEN)
        self.assertEqual(
            locator.sizes[1], (region[2] - region[0], region[3] - region[1])
        )
        self.assertEqual(
            detections[0].box,
            (
                self.FINE_IN_CROP[0] + region[0],
                self.FINE_IN_CROP[1] + region[1],
                self.FINE_IN_CROP[2] + region[0],
                self.FINE_IN_CROP[3] + region[1],
            ),
        )
        self.assertEqual(computer.inference_calls, 2)

    def test_a_large_target_is_located_once(self):
        # 400x300 on a 1920-wide screen is 160x120 for the model: plenty.
        large = (700, 400, 1100, 700)
        locator = StubLocator([large])
        computer = self.build(locator)

        detections = computer.locate_object("the export panel", mode="first")

        self.assertEqual(locator.sizes, [self.SCREEN])
        self.assertEqual(detections[0].box, large)
        self.assertEqual(computer.inference_calls, 1)

    def test_refinement_can_be_switched_off(self):
        locator = StubLocator([self.COARSE], [self.FINE_IN_CROP])
        computer = self.build(locator)

        detections = computer.locate_object("field", mode="first", refine=False)

        self.assertEqual(locator.sizes, [self.SCREEN])
        self.assertEqual(detections[0].box, self.COARSE)

    def test_the_coarse_box_is_kept_when_refinement_finds_nothing(self):
        locator = StubLocator([self.COARSE], [])
        computer = self.build(locator)

        detections = computer.locate_object("field", mode="first")

        self.assertEqual(detections[0].box, self.COARSE)
        self.assertEqual(computer.inference_calls, 2)

    def test_only_the_first_few_detections_are_refined(self):
        boxes = [(10, 10, 40, 40), (60, 10, 90, 40), (110, 10, 140, 40)]
        locator = StubLocator(boxes, [(0, 0, 30, 30)])
        computer = self.build(locator, refine_limit=2)

        computer.locate_object("icons", mode="all")

        # one coarse pass plus two refinements, the third box left alone
        self.assertEqual(computer.inference_calls, 3)

    def test_a_region_narrows_the_search_and_maps_coordinates_back(self):
        region = (1200, 290, 1900, 1010)
        locator = StubLocator([(20, 659, 560, 701)])
        computer = self.build(locator)

        detections = computer.locate_object(
            "the message input field", mode="first", region=region, refine=False
        )

        self.assertEqual(locator.sizes, [(700, 720)])
        self.assertEqual(detections[0].box, (1220, 949, 1760, 991))

    def test_a_region_small_enough_to_run_natively_needs_no_second_pass(self):
        # 700x720 fits inside the 768 px the model looks at, so the field is
        # already at full size and refining it again would be wasted work.
        region = (1200, 290, 1900, 1010)
        locator = StubLocator([(20, 659, 560, 701)])
        computer = self.build(locator)

        detections = computer.locate_object(
            "the message input field", mode="first", region=region
        )

        self.assertEqual(computer.inference_calls, 1)
        self.assertEqual(detections[0].box, (1220, 949, 1760, 991))

    def test_a_region_that_is_still_too_large_is_refined(self):
        region = (0, 0, 1920, 900)
        locator = StubLocator([(1225, 602, 1859, 642)], [(20, 217, 560, 259)])
        computer = self.build(locator)

        computer.locate_object("the message input field", mode="first", region=region)

        self.assertEqual(computer.inference_calls, 2)

    def test_repeating_a_query_on_an_unchanged_screen_reuses_the_result(self):
        locator = StubLocator([self.COARSE], [self.FINE_IN_CROP])
        computer = self.build(locator)

        first = computer.locate_object("field", mode="first")
        calls_after_first = computer.inference_calls
        second = computer.locate_object("field", mode="first")

        self.assertEqual(first, second)
        self.assertEqual(computer.inference_calls, calls_after_first)

    def test_a_changed_screen_invalidates_the_cache(self):
        locator = StubLocator([self.COARSE], [self.FINE_IN_CROP])
        computer = self.build(locator)
        computer.locate_object("field", mode="first")
        calls_after_first = computer.inference_calls

        changed = Image.new("RGB", self.SCREEN, "white")
        computer.locate_object("field", mode="first", screenshot=changed)

        self.assertGreater(computer.inference_calls, calls_after_first)

    def test_the_cache_can_be_switched_off(self):
        locator = StubLocator([self.COARSE], [self.FINE_IN_CROP])
        computer = self.build(locator, cache_detections=False)

        computer.locate_object("field", mode="first")
        computer.locate_object("field", mode="first")

        self.assertEqual(computer.inference_calls, 4)


if __name__ == "__main__":
    unittest.main()
