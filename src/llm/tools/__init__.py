"""The agent's tool layer — one subpackage per concern.

- `meta/` — the single meta tool the model sees (`run_tools`) with its
  in-script `search_tools` / `get_tool` discovery functions, the
  `ToolRegistry` behind them, the default registry, the image-attachment
  channel, and the `sandbox/` interpreter that executes `run_tools`
  scripts;
- `parser/`, `computer/`, `files/`, `shell/` — concrete capabilities, each
  with the fixed shape `tool.py` (the `@tool` functions the LLM receives) +
  `service.py` (the class doing the real work) + optional `utils.py`; new
  tools follow the same pattern and get registered in `meta/defaults.py`.

Instrumentation of the layer (the script tool-call counter) lives in
`src.llm.statistics`.
"""

from .computer.tool import (
    COMPUTER_TOOLS,
    get_computer,
    set_computer,
    warm_up_computer,
)
from .files.tool import FILE_TOOLS, edit_file, glob_files, grep_files, read_file, write_file
from .mcp.manager import get_mcp_manager, set_mcp_manager
from .mcp.tool import MCP_SERVICE_TOOLS, mcp_list_resources, mcp_read_resource
from .meta.attachments import attach_image, collecting_images
from .meta.defaults import default_tools, get_registry, set_registry
from .meta.registry import RESERVED_SCRIPT_NAMES, ToolRegistry
from .meta.tool import (
    MAX_ATTACHED_IMAGES,
    MAX_RESULT_CHARS,
    META_TOOLS,
    get_tool,
    run_tools,
    search_tools,
)
from .parser.tool import web_fetch, web_search
from .shell.tool import bash

__all__ = [
    "COMPUTER_TOOLS",
    "FILE_TOOLS",
    "MAX_ATTACHED_IMAGES",
    "MAX_RESULT_CHARS",
    "MCP_SERVICE_TOOLS",
    "META_TOOLS",
    "RESERVED_SCRIPT_NAMES",
    "ToolRegistry",
    "attach_image",
    "bash",
    "collecting_images",
    "default_tools",
    "edit_file",
    "get_computer",
    "get_mcp_manager",
    "get_registry",
    "get_tool",
    "glob_files",
    "grep_files",
    "mcp_list_resources",
    "mcp_read_resource",
    "read_file",
    "run_tools",
    "search_tools",
    "set_computer",
    "set_mcp_manager",
    "set_registry",
    "warm_up_computer",
    "web_fetch",
    "web_search",
    "write_file",
]
