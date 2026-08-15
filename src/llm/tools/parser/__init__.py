"""Web-page fetching for the agent.

Per-tool package layout: `service.py` holds `WebParser` (the class that
makes the HTTP requests and converts HTML to Markdown), `utils.py` its
text helpers, and `tool.py` the `@tool`-decorated functions the LLM sees.
"""

from .service import WebParser
from .tool import web_search

__all__ = ["WebParser", "web_search"]
