"""Filesystem operations behind the coding tools.

Pure Python on purpose: no ripgrep or shell dependency, identical behavior
on every OS, and every result capped so a big tree cannot flood the model's
context. Errors are raised as ValueError/OSError with actionable messages —
`tool.py` turns them into strings for the model.
"""

import re
from pathlib import Path

MAX_READ_LINES = 2_000
MAX_LINE_CHARS = 2_000
MAX_GLOB_RESULTS = 200
MAX_GREP_RESULTS = 200
GREP_OUTPUT_MODES = ("content", "files_with_matches", "count")
SKIP_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode", "localwheels"})


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


class FileService:
    """Reads, writes, edits, and searches files for the agent."""

    def read(self, file_path: str, limit: int | None = None, offset: int = 0) -> str:
        if offset < 0:
            raise ValueError("offset must not be negative")
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
        window = limit if limit is not None else MAX_READ_LINES
        segment = lines[offset : offset + window]
        if not segment:
            return "(no lines in this range)" if lines else "(empty file)"

        numbered = [f"{number}\t{line[:MAX_LINE_CHARS]}" for number, line in enumerate(segment, start=offset + 1)]
        remaining = len(lines) - (offset + len(segment))
        if remaining > 0:
            numbered.append(f"… [+{remaining} more line(s); continue with offset={offset + len(segment)}]")
        return "\n".join(numbered)

    def write(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.splitlines())} line(s) ({len(content)} chars) to {path}"

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string was not found in the file — copy it exactly, including whitespace")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string occurs {occurrences} times — add surrounding context to make it unique, "
                "or pass replace_all=True"
            )
        path.write_text(text.replace(old_string, new_string, -1 if replace_all else 1), encoding="utf-8")
        return f"Replaced {occurrences if replace_all else 1} occurrence(s) in {path}"

    def glob(self, pattern: str, path: str = ".") -> str:
        base = Path(path)
        matches = sorted(str(match) for match in base.glob(pattern) if not _skipped(match))
        if not matches:
            return f"No files match {pattern!r} under {base.resolve()}"
        listing = "\n".join(matches[:MAX_GLOB_RESULTS])
        if len(matches) > MAX_GLOB_RESULTS:
            listing += f"\n… [+{len(matches) - MAX_GLOB_RESULTS} more]"
        return listing

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        output_mode: str = "content",
        case_insensitive: bool = False,
        context: int = 0,
    ) -> str:
        if output_mode not in GREP_OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {', '.join(GREP_OUTPUT_MODES)}")
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)

        content_lines: list[str] = []
        matched_files: list[tuple[str, int]] = []
        for candidate in self._candidates(Path(path), glob):
            try:
                data = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in data[:1024]:  # binary
                continue
            lines = data.splitlines()
            hits = [index for index, line in enumerate(lines) if regex.search(line)]
            if not hits:
                continue
            matched_files.append((str(candidate), len(hits)))
            if output_mode == "content":
                for hit in hits:
                    low = max(0, hit - context)
                    high = min(len(lines), hit + context + 1)
                    for index in range(low, high):
                        content_lines.append(f"{candidate}:{index + 1}:{lines[index][:MAX_LINE_CHARS]}")
            if len(matched_files) >= MAX_GREP_RESULTS or len(content_lines) >= MAX_GREP_RESULTS:
                break

        if not matched_files:
            return f"No matches for {pattern!r}"
        if output_mode == "files_with_matches":
            return "\n".join(name for name, _ in matched_files)
        if output_mode == "count":
            return "\n".join(f"{name}:{count}" for name, count in matched_files)
        deduped = list(dict.fromkeys(content_lines))[:MAX_GREP_RESULTS]
        return "\n".join(deduped)

    @staticmethod
    def _candidates(base: Path, file_glob: str | None):
        if base.is_file():
            yield base
            return
        for candidate in base.rglob(file_glob or "*"):
            if candidate.is_file() and not _skipped(candidate):
                yield candidate
