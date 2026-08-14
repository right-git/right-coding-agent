import io
import sys
import threading
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout

from src.utils.silence import silenced


class SilencedTests(unittest.TestCase):
    def test_writes_from_the_silenced_thread_are_dropped(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with silenced():
                print("hidden")
            print("visible")

        self.assertNotIn("hidden", buffer.getvalue())
        self.assertIn("visible", buffer.getvalue())

    def test_other_threads_keep_writing(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with silenced():
                worker = threading.Thread(target=lambda: print("from-worker"))
                worker.start()
                worker.join()

        self.assertIn("from-worker", buffer.getvalue())

    def test_warnings_and_stderr_are_muted(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            with silenced():
                warnings.warn("boo", FutureWarning)
                sys.stderr.write("stderr noise")

        self.assertEqual(errors.getvalue(), "")

    def test_reinstalls_after_streams_are_swapped(self):
        first = io.StringIO()
        with redirect_stdout(first):
            with silenced():
                print("hidden one")

        second = io.StringIO()
        with redirect_stdout(second):
            with silenced():
                print("hidden two")
            print("visible two")

        self.assertEqual(first.getvalue(), "")
        self.assertEqual(second.getvalue(), "visible two\n")

    def test_stream_attributes_are_forwarded(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with silenced():
                pass
            self.assertFalse(sys.stdout.isatty())


if __name__ == "__main__":
    unittest.main()
