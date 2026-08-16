import io
import os
import sys
import threading
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.utils.silence import silenced, suppress_native_stderr


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


class SuppressNativeStderrTests(unittest.TestCase):
    """fd-level suppression for C libraries (objc runtime, dylib loaders)."""

    def setUp(self):
        # Keep test probe bytes out of the repo's real logs.native.log.
        import tempfile
        from unittest.mock import patch

        from src.utils import silence

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(silence, "NATIVE_STDERR_LOG", str(Path(directory.name) / "native.log"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def capture_fd2(self):
        read_fd, write_fd = os.pipe()
        saved = os.dup(2)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        return read_fd, saved

    def restore_fd2(self, read_fd, saved):
        os.dup2(saved, 2)
        os.close(saved)
        data = b""
        os.set_blocking(read_fd, False)
        try:
            data = os.read(read_fd, 65536)
        except BlockingIOError:
            pass
        os.close(read_fd)
        return data

    def test_fd_writes_inside_are_dropped_and_restored_after(self):
        read_fd, saved = self.capture_fd2()
        try:
            with suppress_native_stderr():
                os.write(2, b"native noise")
            os.write(2, b"after")
        finally:
            data = self.restore_fd2(read_fd, saved)

        self.assertNotIn(b"native noise", data)
        self.assertIn(b"after", data)

    def test_native_noise_is_captured_to_the_log_file(self):
        # A crashing dylib's abort message must stay diagnosable — the noise
        # goes to a sidecar file, never to a black hole.
        from src.utils import silence

        read_fd, saved = self.capture_fd2()
        try:
            with suppress_native_stderr():
                os.write(2, b"objc noise")
        finally:
            data = self.restore_fd2(read_fd, saved)

        self.assertNotIn(b"objc noise", data)
        self.assertIn(b"objc noise", Path(silence.NATIVE_STDERR_LOG).read_bytes())

    def test_nested_use_restores_only_at_the_outermost_exit(self):
        read_fd, saved = self.capture_fd2()
        try:
            with suppress_native_stderr():
                with suppress_native_stderr():
                    os.write(2, b"inner")
                os.write(2, b"between")  # still one level deep — still muted
            os.write(2, b"after")
        finally:
            data = self.restore_fd2(read_fd, saved)

        self.assertNotIn(b"inner", data)
        self.assertNotIn(b"between", data)
        self.assertIn(b"after", data)


if __name__ == "__main__":
    unittest.main()
