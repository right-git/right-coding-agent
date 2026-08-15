"""LangChain tools for reading, writing, editing, and searching files."""

import asyncio

from langchain_core.tools import tool

from .service import FileService

_service = FileService()


@tool(parse_docstring=True, return_direct=False)
async def read_file(file_path: str, limit: int | None = None, offset: int = 0) -> str:
    """Read a file's contents, returned with line numbers as `N<tab>line`.

    Use limit/offset to read large files in chunks — the result tells you
    the offset to continue from when lines remain.

    Args:
        file_path: Path to the file (absolute, or relative to the project).
        limit: Maximum number of lines to return (default 2000).
        offset: Line index to start from, 0-based (default 0).

    Returns:
        The numbered lines, or an error message.
    """
    try:
        return await asyncio.to_thread(_service.read, file_path, limit, offset)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Overwrites the file when it exists — read it first unless you intend a
    full replacement; for partial changes prefer edit_file.

    Args:
        file_path: Path of the file to create or overwrite.
        content: Full new content of the file.

    Returns:
        A confirmation with the written size, or an error message.
    """
    try:
        return await asyncio.to_thread(_service.write, file_path, content)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace exact text in a file.

    old_string must match the file exactly, including whitespace and
    indentation. When it occurs more than once, add surrounding lines to
    make it unique or pass replace_all=True.

    Args:
        file_path: Path of the file to modify.
        old_string: Exact text to replace.
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of requiring a unique match.

    Returns:
        A confirmation with the replacement count, or an error message.
    """
    try:
        return await asyncio.to_thread(_service.edit, file_path, old_string, new_string, replace_all)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern, for example `**/*.py` or `src/**/test_*.py`.
        path: Base directory to search from (default is the project root).

    Returns:
        Matching paths sorted alphabetically, one per line, or an error message.
    """
    try:
        return await asyncio.to_thread(_service.glob, pattern, path)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def grep_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context: int = 0,
) -> str:
    """Search file contents with a regular expression.

    Args:
        pattern: Regex to search for, for example `def \\w+_turn`.
        path: File or directory to search (default is the project root).
        glob: Filename filter such as `*.py` (applies recursively).
        output_mode: `content` for matching lines with `file - line number - text`,
            `files_with_matches` for file paths only, `count` for per-file match counts.
        case_insensitive: Ignore case when matching.
        context: Lines of context to include around each match (content mode).

    Returns:
        Matches in the requested format, or an error message.
    """
    try:
        return await asyncio.to_thread(_service.grep, pattern, path, glob, output_mode, case_insensitive, context)
    except Exception as error:
        return f"Tool call failed, error: {error}"


FILE_TOOLS = [read_file, write_file, edit_file, glob_files, grep_files]
