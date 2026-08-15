"""The meta layer: everything that serves the three meta tools.

`tool.py` — `search_tools` / `get_tool` / `run_tools`, the only tools the
model sees; `registry.py` — the `ToolRegistry` they search, document, and
execute; `defaults.py` — the process-wide default registry with the actual
tool set; `attachments.py` — the channel that carries tool-captured images
out of a run; `sandbox/` — the interpreter that executes `run_tools` scripts.

Deliberately no re-exports here: concrete tool packages import
`meta.attachments` while `defaults.py` imports them back, so an eager
`from .tool import ...` in this __init__ would create a circular import.
Import the submodules directly (the public names are re-exported one level
up, in `src.llm.tools`).
"""
