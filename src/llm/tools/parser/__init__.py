"""Web tools for the agent: fetch pages, search the web.

Per-tool package layout: `service.py` holds `WebParser` (HTTP requests,
HTML-to-Markdown conversion, DuckDuckGo search), `utils.py` its text
helpers, and `tool.py` the `@tool` functions the LLM sees (`web_fetch` for
a URL, `web_search` for a query).
"""

from .service import WebParser
from .tool import web_fetch, web_search

__all__ = ["WebParser", "web_fetch", "web_search"]
