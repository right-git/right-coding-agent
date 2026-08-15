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


class EscapeWatcherTests(unittest.TestCase):
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
