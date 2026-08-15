"""Model download progress and exit-safe Hugging Face downloads.

The big model downloads (vision locator, whisper ASR, silero TTS) run in
background warm-up threads while the prompt is live, so their progress must
go to the prompt's right-side status line, not to stdout. The warm-up wrapper
opens `reporting_progress(callback)` and the download path calls
`report_progress("↓ 1.2/6.4 GB 19%")`; without an open channel the calls are
no-ops, so library use stays silent.

`download_hf_model` deliberately avoids `snapshot_download` for the download
itself: its internal `thread_map` pool spawns NON-daemon workers, and a /quit
while one of them is mid-gigabyte leaves the interpreter hanging in
`concurrent.futures`' atexit join (observed on first run). Files are instead
fetched sequentially with `hf_hub_download` on the calling thread — the
warm-ups run on daemon threads, so quitting mid-download just works. Byte
progress comes from watching the repo cache directory grow, which stays
correct whichever transport (plain HTTP or hf_xet) does the writing.
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

_local = threading.local()


@contextmanager
def reporting_progress(callback):
    """Route this thread's `report_progress` calls (and its watchers') to `callback`."""
    _local.callback = callback
    try:
        yield
    finally:
        _local.callback = None


def current_reporter():
    """The active callback of this thread, or None; watchers capture it at start."""
    return getattr(_local, "callback", None)


def report_progress(detail: str) -> None:
    """Send one short progress string to the open channel; silent without one."""
    callback = current_reporter()
    if callback is None:
        return
    try:
        callback(detail)
    except Exception:
        logger.exception("Progress callback failed")


def _scale(size: int) -> tuple[float, str, int]:
    if size >= 1_000_000_000:
        return size / 1_000_000_000, "GB", 1
    return size / 1_000_000, "MB", 0


def format_download(done: int, total: int | None) -> str:
    """Compact byte progress for the status line: `↓ 1.2/6.4 GB 19%`."""
    if total:
        _, unit, digits = _scale(total)
        divisor = 1_000_000_000 if unit == "GB" else 1_000_000
        percent = min(100, round(100 * done / total))
        return f"↓ {done / divisor:.{digits}f}/{total / divisor:.{digits}f} {unit} {percent}%"
    value, unit, digits = _scale(done)
    return f"↓ {value:.{digits}f} {unit}"


def directory_size(path: Path | str) -> int:
    """Total bytes under `path`, tolerant of files vanishing mid-walk."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


@contextmanager
def watching_directory_size(path: Path | str, total: int | None, poll_interval: float = 1.0):
    """Report the growing size of `path` through the current progress channel.

    The watcher is a daemon thread that captures the caller's reporter, so it
    keeps reporting while the caller blocks inside a download call.
    """
    callback = current_reporter()
    stop = threading.Event()
    thread = None
    if callback is not None:

        def watch() -> None:
            while not stop.wait(poll_interval):
                try:
                    callback(format_download(directory_size(path), total))
                except Exception:
                    logger.exception("Download progress watcher failed")
                    return

        thread = threading.Thread(target=watch, name="download-progress", daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=poll_interval + 1)


def _repo_cache_dir(repo_id: str, cache_dir) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    return Path(cache_dir or HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")


def _default_resolver(repo_id: str, cache_dir):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True)


def _default_downloader(**kwargs):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _default_api():
    from huggingface_hub import HfApi

    return HfApi()


def download_hf_model(
    repo_id: str,
    cache_dir: Path | str | None = None,
    *,
    required_files: tuple[str, ...] = (),
    poll_interval: float = 1.0,
    api=None,
    downloader=None,
    resolver=None,
) -> str:
    """Local snapshot path for `repo_id`, downloading file-by-file when needed.

    A cached snapshot is trusted only when every name in `required_files`
    exists in it — an interrupted `snapshot_download` writes the ref before
    the weights, so the ref alone proves nothing. The download loop re-fetches
    only missing or incomplete files for the pinned revision.
    """
    cache_dir = str(cache_dir) if cache_dir is not None else None
    resolver = resolver or _default_resolver
    try:
        path = resolver(repo_id, cache_dir)
        if all((Path(path) / name).is_file() for name in required_files):
            return path
    except Exception:
        pass

    api = api or _default_api()
    downloader = downloader or _default_downloader
    info = api.model_info(repo_id, files_metadata=True)
    files = [(sibling.rfilename, sibling.size or 0) for sibling in info.siblings]
    total = sum(size for _, size in files) or None
    repo_dir = _repo_cache_dir(repo_id, cache_dir)
    logger.info("Downloading {} ({} files, {} bytes) to {}", repo_id, len(files), total, repo_dir)
    with watching_directory_size(repo_dir, total, poll_interval=poll_interval):
        for filename, _ in files:
            downloader(repo_id=repo_id, filename=filename, revision=info.sha, cache_dir=cache_dir)

    # snapshot_download writes this ref itself; hf_hub_download with a pinned
    # revision does not, and without it local resolution of "main" fails.
    ref = repo_dir / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(info.sha)
    return resolver(repo_id, cache_dir)
