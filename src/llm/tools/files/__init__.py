"""File tools for coding: read, write, edit, glob, and grep.

Per-tool package layout: `service.py` holds `FileService` (the class doing
the real filesystem work, capped and cross-platform), `tool.py` the `@tool`
functions the LLM receives.
"""

from .service import FileService
from .tool import FILE_TOOLS, edit_file, glob_files, grep_files, read_file, write_file

__all__ = ["FILE_TOOLS", "FileService", "edit_file", "glob_files", "grep_files", "read_file", "write_file"]
