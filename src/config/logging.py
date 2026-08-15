import sys
import warnings
from pathlib import Path

from loguru import logger


def _warning_to_log(message, category, filename, lineno, file=None, line=None) -> None:
    """Send Python warnings to the log file instead of stderr.

    The terminal belongs to the chat UI; a library warning printed mid-turn
    lands on top of the live progress display.
    """
    logger.warning("{}:{} {}: {}", filename, lineno, category.__name__, message)


class LoggingManager:
    VALID_LEVELS = {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    def __init__(
        self,
        log_file: str = "logs.log",
        level: str = "INFO",
        rotation: str = "1 MB",
        compression: str = "zip",
        log_to_stdout: bool = False,
    ) -> None:
        self.log_file = Path(log_file)
        self.level = self._normalize_level(level)
        self.rotation = rotation
        self.compression = compression
        self.log_to_stdout = log_to_stdout

    def _normalize_level(self, level: str) -> str:
        normalized = level.strip().upper()
        if normalized not in self.VALID_LEVELS:
            raise ValueError(f"Invalid log level: {level}. " f"Expected one of: {', '.join(sorted(self.VALID_LEVELS))}")
        return normalized

    def configure(
        self,
        *,
        level: str | None = None,
        log_file: str | None = None,
        rotation: str | None = None,
        compression: str | None = None,
        log_to_stdout: bool | None = None,
    ) -> None:
        if level is not None:
            self.level = self._normalize_level(level)
        if log_file is not None:
            self.log_file = Path(log_file)
        if rotation is not None:
            self.rotation = rotation
        if compression is not None:
            self.compression = compression
        if log_to_stdout is not None:
            self.log_to_stdout = log_to_stdout

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        warnings.showwarning = _warning_to_log

        logger.remove()

        if self.log_to_stdout:
            logger.add(
                sys.stderr,
                level=self.level,
                colorize=False,
                format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | " "{name}:{function}:{line} - {message}"),
            )

        logger.add(
            str(self.log_file),
            level=self.level,
            rotation=self.rotation,
            compression=self.compression,
            encoding="utf-8",
            format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | " "{name}:{function}:{line} - {message}"),
            backtrace=True,
            diagnose=True,
        )

    def set_level(self, level: str) -> str:
        self.configure(level=level)
        logger.info("Log level changed to [{}]", self.level)
        return self.level

    def get_level(self) -> str:
        return self.level


app_logging = LoggingManager()
app_logging.configure()
