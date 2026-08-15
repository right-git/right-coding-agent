import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.utils.downloads import (
    directory_size,
    download_hf_model,
    format_download,
    report_progress,
    reporting_progress,
    watching_directory_size,
)


class FormatDownloadTests(unittest.TestCase):
    def test_formats_gigabytes_with_percent(self):
        self.assertEqual(format_download(1_200_000_000, 6_400_000_000), "↓ 1.2/6.4 GB 19%")

    def test_formats_megabytes_with_percent(self):
        self.assertEqual(format_download(5_000_000, 38_000_000), "↓ 5/38 MB 13%")

    def test_formats_bytes_alone_when_total_unknown(self):
        self.assertEqual(format_download(1_200_000_000, None), "↓ 1.2 GB")
        self.assertEqual(format_download(5_000_000, 0), "↓ 5 MB")

    def test_percent_is_capped_at_hundred(self):
        self.assertTrue(format_download(2_000_000_000, 1_000_000_000).endswith("100%"))


class ReportingChannelTests(unittest.TestCase):
    def test_report_without_context_is_a_no_op(self):
        report_progress("↓ 1 MB")  # must not raise

    def test_report_reaches_the_callback_only_inside_the_context(self):
        seen = []
        with reporting_progress(seen.append):
            report_progress("a")
        report_progress("b")

        self.assertEqual(seen, ["a"])

    def test_callback_errors_are_swallowed(self):
        def broken(_):
            raise RuntimeError("boom")

        with reporting_progress(broken):
            report_progress("a")  # must not raise


class DirectorySizeTests(unittest.TestCase):
    def test_sums_nested_files_and_tolerates_missing_dir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.bin").write_bytes(b"x" * 10)
            (root / "sub").mkdir()
            (root / "sub" / "b.bin").write_bytes(b"x" * 5)

            self.assertEqual(directory_size(root), 15)
            self.assertEqual(directory_size(root / "missing"), 0)

    def test_watcher_reports_growing_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen = []
            with reporting_progress(seen.append):
                with watching_directory_size(root, total=100, poll_interval=0.01):
                    (root / "a.bin").write_bytes(b"x" * 50)
                    deadline = time.monotonic() + 2
                    while not seen and time.monotonic() < deadline:
                        time.sleep(0.01)

            self.assertTrue(seen, "watcher never reported progress")
            self.assertIn("50%", seen[-1])


def make_fake_api(sha="abc123", files=(("config.json", 10), ("model.bin", 90))):
    siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in files]
    info = SimpleNamespace(sha=sha, siblings=siblings)
    calls = []

    def model_info(repo_id, files_metadata=False):
        calls.append(repo_id)
        return info

    return SimpleNamespace(model_info=model_info, calls=calls)


class DownloadHfModelTests(unittest.TestCase):
    def test_cached_snapshot_is_returned_without_network(self):
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            api = make_fake_api()

            path = download_hf_model(
                "org/name",
                cache_dir=tmp,
                required_files=("config.json",),
                api=api,
                downloader=lambda **kw: self.fail("must not download"),
                resolver=lambda repo, cache: str(snapshot),
            )

            self.assertEqual(path, str(snapshot))
            self.assertEqual(api.calls, [])

    def test_partial_snapshot_falls_through_to_download(self):
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap"
            snapshot.mkdir()  # refs resolve, but the weights file is missing
            api = make_fake_api()
            downloaded = []

            download_hf_model(
                "org/name",
                cache_dir=tmp,
                required_files=("model.bin",),
                api=api,
                downloader=lambda **kw: downloaded.append(kw["filename"]),
                resolver=lambda repo, cache: str(snapshot),
            )

            self.assertEqual(downloaded, ["config.json", "model.bin"])

    def test_downloads_sequentially_in_the_calling_thread(self):
        # Exit-safety is the whole point: no non-daemon worker threads that
        # would block interpreter shutdown when the user quits mid-download.
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap"
            api = make_fake_api()
            threads = []
            resolved = {"count": 0}

            def resolver(repo, cache):
                resolved["count"] += 1
                if resolved["count"] == 1:
                    raise FileNotFoundError("nothing cached")
                return str(snapshot)

            def downloader(**kwargs):
                threads.append(threading.current_thread())
                self.assertEqual(kwargs["revision"], "abc123")
                self.assertEqual(kwargs["cache_dir"], tmp)

            path = download_hf_model(
                "org/name",
                cache_dir=tmp,
                api=api,
                downloader=downloader,
                resolver=resolver,
            )

            self.assertEqual(path, str(snapshot))
            self.assertEqual(threads, [threading.current_thread()] * 2)
            self.assertEqual(api.calls, ["org/name"])

    def test_writes_the_main_ref_so_local_resolution_works(self):
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap"
            api = make_fake_api(sha="deadbeef")

            def resolver(repo, cache):
                ref = Path(tmp) / "models--org--name" / "refs" / "main"
                if not ref.is_file():
                    raise FileNotFoundError("no ref yet")
                return str(snapshot)

            path = download_hf_model(
                "org/name",
                cache_dir=tmp,
                api=api,
                downloader=lambda **kw: None,
                resolver=resolver,
            )

            self.assertEqual(path, str(snapshot))
            ref = Path(tmp) / "models--org--name" / "refs" / "main"
            self.assertEqual(ref.read_text(), "deadbeef")


if __name__ == "__main__":
    unittest.main()
