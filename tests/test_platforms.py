import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.computer import platforms
from src.llm.tools.computer.platforms.portable.pointer import (
    NAMED_KEYS,
    PortablePointer,
    split_combination,
)


class RecordingKeyboard:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))

    def type(self, text):
        self.events.append(("type", text))


class SplitCombinationTests(unittest.TestCase):
    def test_splits_modifiers_and_final_key(self):
        self.assertEqual(split_combination("ctrl+shift+s"), ["ctrl", "shift", "s"])

    def test_single_key_needs_no_modifier(self):
        self.assertEqual(split_combination("enter"), ["enter"])

    def test_rejects_non_modifier_prefix(self):
        with self.assertRaisesRegex(ValueError, "modifier"):
            split_combination("a+b")

    def test_rejects_empty_combination(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            split_combination("  ")


class PortablePointerKeyTests(unittest.TestCase):
    def setUp(self):
        self.pointer = PortablePointer(sleep=lambda seconds: None)
        self.keyboard = RecordingKeyboard()
        self.pointer._keyboard = self.keyboard

    def test_named_keys_resolve_to_pynput_keys(self):
        from pynput.keyboard import Key

        self.assertIs(self.pointer.resolve_key("ctrl"), Key.ctrl)
        self.assertIs(self.pointer.resolve_key("Enter"), Key.enter)
        self.assertIs(self.pointer.resolve_key("pgdn"), Key.page_down)

    def test_letters_and_digits_resolve_to_characters(self):
        self.assertEqual(self.pointer.resolve_key("a"), "a")
        self.assertEqual(self.pointer.resolve_key("7"), "7")

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported key name"):
            self.pointer.resolve_key("nosuchkey")

    def test_every_named_key_maps_to_an_existing_attribute_name(self):
        from pynput.keyboard import Key

        # Some names are legitimately absent on some OSes; the mapping itself
        # must only ever point at attribute names pynput defines somewhere.
        missing = [name for name, attr in NAMED_KEYS.items() if not hasattr(Key, attr)]
        self.assertEqual(missing, [])

    def test_ctrl_c_presses_and_releases_in_order(self):
        from pynput.keyboard import Key

        self.pointer.key("ctrl+c")

        self.assertEqual(
            self.keyboard.events,
            [("press", Key.ctrl), ("press", "c"), ("release", "c"), ("release", Key.ctrl)],
        )

    def test_sequences_run_left_to_right(self):
        from pynput.keyboard import Key

        self.pointer.key("ctrl+a delete")

        self.assertEqual(
            self.keyboard.events,
            [
                ("press", Key.ctrl),
                ("press", "a"),
                ("release", "a"),
                ("release", Key.ctrl),
                ("press", Key.delete),
                ("release", Key.delete),
            ],
        )

    def test_type_text_delegates_to_the_keyboard(self):
        self.pointer.type_text("hello, мир!")

        self.assertEqual(self.keyboard.events, [("type", "hello, мир!")])

    def test_hold_key_releases_even_when_sleep_fails(self):
        from pynput.keyboard import Key

        def broken_sleep(seconds):
            raise RuntimeError("boom")

        self.pointer._sleep = broken_sleep
        with self.assertRaises(RuntimeError):
            self.pointer.hold_key("shift", 0.5)

        self.assertEqual(self.keyboard.events, [("press", Key.shift), ("release", Key.shift)])


class FactoryTests(unittest.TestCase):
    def test_windows_factory_returns_native_backends(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only check")
        from src.llm.tools.computer.platforms.windows.clipboard import Win32Clipboard
        from src.llm.tools.computer.platforms.windows.pointer import Pointer
        from src.llm.tools.computer.platforms.windows.screen import PrimaryScreen

        self.assertIsInstance(platforms.default_screen(), PrimaryScreen)
        self.assertIsInstance(platforms.default_pointer(), Pointer)
        self.assertIsInstance(platforms.default_clipboard(), Win32Clipboard)

    def test_package_import_carries_no_windows_modules(self):
        # The computer package must be importable on any OS: nothing in it may
        # eagerly import the Win32 backends (ctypes.wintypes breaks elsewhere).
        import importlib

        module = importlib.import_module("src.llm.tools.computer")
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("platforms.windows", source)


if __name__ == "__main__":
    unittest.main()
