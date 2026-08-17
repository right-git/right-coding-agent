import asyncio
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import process_user_turn
from src.ui.chat import ChatUI
from src.ui.interrupt import EscapeWatcher, TurnCancelled


class InterruptPolicyTests(unittest.TestCase):
    def test_single_presses_cancel_and_spaced_presses_stay_cancels(self):
        from src.ui.interrupt import InterruptPolicy

        policy = InterruptPolicy(window=2.0)
        self.assertEqual(policy.press(now=100.0), "cancel")
        self.assertEqual(policy.press(now=103.0), "cancel")  # window expired

    def test_a_second_press_within_the_window_forces_quit(self):
        from src.ui.interrupt import InterruptPolicy

        policy = InterruptPolicy(window=2.0)
        policy.press(now=100.0)
        self.assertEqual(policy.press(now=101.0), "force")


class SigintHandlerTests(unittest.TestCase):
    """Ctrl+C during a turn: cancel first, force-quit on a double press.

    A turn blocked inside asyncio.to_thread (MPS inference, whisper) cannot
    actually be cancelled until the thread ends — only process death reliably
    frees the microphone, hence the force path.
    """

    def make_handler(self):
        from src.main import make_sigint_handler
        from src.ui.interrupt import InterruptPolicy

        ui = Mock()
        current_turn = {"task": None}
        force_exit = Mock()
        clock = Mock(side_effect=[100.0, 101.0, 200.0, 300.0])
        handler = make_sigint_handler(InterruptPolicy(window=2.0), ui, current_turn, force_exit=force_exit, clock=clock)
        return handler, ui, current_turn, force_exit

    def test_first_press_cancels_the_running_turn(self):
        handler, ui, current_turn, force_exit = self.make_handler()
        task = Mock()
        task.done.return_value = False
        task.cancelling.return_value = 0
        current_turn["task"] = task

        handler()

        task.cancel.assert_called_once()
        force_exit.assert_not_called()
        ui.print_warning.assert_called_once()

    def test_any_later_press_on_a_stuck_turn_forces_quit(self):
        # No 2-second window here: if the turn ignored the first Ctrl+C (a
        # tool call stuck in an executor thread cannot be cancelled), the next
        # press must force-quit no matter how much later it comes.
        handler, ui, current_turn, force_exit = self.make_handler()
        task = Mock()
        task.done.return_value = False
        task.cancelling.return_value = 1  # cancel already requested, still running
        current_turn["task"] = task

        handler()

        force_exit.assert_called_once_with(130)

    def test_second_press_within_the_window_hard_exits(self):
        # No turn running (e.g. stuck outside a turn): the timing window applies.
        handler, ui, current_turn, force_exit = self.make_handler()

        handler()
        handler()

        force_exit.assert_called_once_with(130)

    def test_press_without_a_turn_only_warns(self):
        handler, ui, current_turn, force_exit = self.make_handler()

        handler()

        force_exit.assert_not_called()
        ui.print_warning.assert_called_once()


class EscapeWatcherTests(unittest.TestCase):
    def test_arrow_key_sequences_do_not_look_like_escape(self):
        # In a Unix terminal an arrow key arrives as "\x1b[A" — that must not
        # cancel the turn; only a lone Esc does.
        watcher = EscapeWatcher(read_keys=lambda: "")

        watcher.handle("\x1b[A")
        watcher.handle("\x1bOP")  # SS3 function-key form
        self.assertFalse(watcher.pressed.is_set())

        watcher.handle("\x1b")
        self.assertTrue(watcher.pressed.is_set())

    def test_unix_backspace_del_edits_the_stash(self):
        watcher = EscapeWatcher(read_keys=lambda: "")

        watcher.handle("ab\x7f")

        self.assertEqual(watcher.typed_text, "a")

    @unittest.skipIf(sys.platform == "win32", "termios — не Windows")
    def test_unix_reader_drains_keys_and_restores_the_terminal(self):
        import os
        import pty
        import termios

        from src.ui.interrupt import _UnixKeyReader

        master, slave = pty.openpty()
        try:
            before = termios.tcgetattr(slave)
            reader = _UnixKeyReader(fd=slave)
            reader.start()  # enters cbreak
            os.write(master, "аб\x1b".encode())
            deadline_chars = ""
            for _ in range(50):
                deadline_chars += reader()
                if "\x1b" in deadline_chars:
                    break
            reader.close()

            self.assertIn("аб", deadline_chars)
            self.assertIn("\x1b", deadline_chars)
            # canonical mode and echo are back (macOS ptys tweak unrelated bits)
            restored = termios.tcgetattr(slave)
            essentials = termios.ICANON | termios.ECHO
            self.assertEqual(restored[3] & essentials, before[3] & essentials)
        finally:
            os.close(master)
            os.close(slave)

    @unittest.skipIf(sys.platform == "win32", "unix reader — не Windows")
    def test_create_returns_a_watcher_on_a_unix_tty(self):
        # Esc-cancel used to be Windows-only; on macOS/Linux create() must
        # produce a watcher whenever stdin is a real terminal.
        from unittest.mock import patch

        from src.ui import interrupt as interrupt_module

        fake_stdin = Mock()
        fake_stdin.isatty.return_value = True
        fake_stdin.fileno.return_value = 0
        with patch.object(interrupt_module.sys, "stdin", fake_stdin):
            watcher = EscapeWatcher.create()

        self.assertIsNotNone(watcher)

    def test_escape_sets_the_event_and_typing_is_stashed(self):
        watcher = EscapeWatcher(read_keys=lambda: "")

        watcher.handle("прив")
        watcher.handle("\x08")  # backspace правит запас
        watcher.handle("\x1b")

        self.assertTrue(watcher.pressed.is_set())
        self.assertEqual(watcher.typed_text, "при")

    def test_control_characters_are_ignored(self):
        watcher = EscapeWatcher(read_keys=lambda: "")

        watcher.handle("a\r\nb\t")

        self.assertFalse(watcher.pressed.is_set())
        self.assertEqual(watcher.typed_text, "ab")  # управляющие символы не попадают в запас


class FakeWatcher:
    def __init__(self, pressed: bool):
        self.pressed = threading.Event()
        if pressed:
            self.pressed.set()
        self.typed_text = "хвост"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class RunCancellableTests(unittest.IsolatedAsyncioTestCase):
    async def test_escape_cancels_the_task_and_stashes_typing(self):
        ui = ChatUI(model="m")
        cancelled = []

        async def slow():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        with self.assertRaises(TurnCancelled):
            await ui.run_cancellable(slow(), watcher=FakeWatcher(pressed=True))

        self.assertEqual(cancelled, [True])
        self.assertEqual(ui._type_ahead, "хвост")

    async def test_completed_task_returns_its_result(self):
        ui = ChatUI(model="m")

        async def quick():
            return 42

        self.assertEqual(await ui.run_cancellable(quick(), watcher=FakeWatcher(pressed=False)), 42)

    async def test_a_stuck_cancellation_tells_the_user_why_it_waits(self):
        # A tool call inside asyncio.to_thread cannot be aborted — Esc must
        # explain that the cancel is waiting instead of looking dead.
        ui = ChatUI(model="m")
        ui.CANCEL_GRACE_SECONDS = 0.02
        ui.print_warning = Mock()

        async def stubborn():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0.1)  # "the executor call is still running"
                raise

        with self.assertRaises(TurnCancelled):
            await ui.run_cancellable(stubborn(), watcher=FakeWatcher(pressed=True))

        self.assertTrue(
            any("Ctrl+C" in str(call) for call in ui.print_warning.call_args_list),
            "the stuck-cancel notice was not shown",
        )


class CancelledTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_turn_rolls_history_back(self):
        agents = Mock()
        agents.right_coding_agent = AsyncMock()
        ui = Mock()
        ui.turn_stream.return_value = nullcontext()

        async def cancel(coro):
            coro.close()
            raise TurnCancelled()

        ui.run_cancellable = cancel

        updated = await process_user_turn(agents=agents, ui=ui, messages=[], model="m", user_content="привет")

        self.assertEqual(updated, [])
        ui.print_warning.assert_called_once()
        ui.cancel_voice_turn.assert_called_once()
        ui.print_response.assert_not_called()


if __name__ == "__main__":
    unittest.main()
