"""Shell command execution behind the bash tool.

Prefers a real bash (Git Bash on Windows, /bin/bash elsewhere) so scripts
behave the same across OSes, and falls back to the system shell when bash
is absent. Output is merged stdout+stderr, capped, and prefixed with the
exit code when the command failed — the model needs the failure, not an
exception.
"""

import shutil
import subprocess

MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT = 30


class CommandRunner:
    """Runs one command per call; no state persists between calls."""

    def __init__(self, *, executable: str | None = None, which=shutil.which) -> None:
        self._executable = executable if executable is not None else which("bash")

    def run(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        try:
            if self._executable:
                completed = subprocess.run(
                    [self._executable, "-c", command], capture_output=True, text=True, timeout=timeout
                )
            else:
                completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"

        output = ((completed.stdout or "") + (completed.stderr or "")).strip() or "(no output)"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"… [+{len(output) - MAX_OUTPUT_CHARS} chars truncated]"
        if completed.returncode != 0:
            output = f"[exit code {completed.returncode}]\n{output}"
        return output
