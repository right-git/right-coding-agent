"""Shell tool for coding: run commands, tests, and builds.

Per-tool package layout: `service.py` holds `CommandRunner` (the class
doing the real work), `tool.py` the `bash` `@tool` the LLM receives.
"""

from .service import CommandRunner
from .tool import bash

__all__ = ["CommandRunner", "bash"]
