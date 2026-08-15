"""LangChain tools for the web: fetch a page, search the web."""

from langchain_core.tools import tool

from .service import WebParser


@tool(parse_docstring=True, return_direct=False)
async def web_fetch(url: str) -> str:
    """Fetch a web page and return its content as Markdown text.

    Args:
        url: URL of the webpage to fetch.

    Returns:
        The page converted to Markdown, as one plain string — slice it,
        search it, or measure it with len().
    """
    try:
        web_parser = WebParser()
        response = await web_parser.parse_page(url)
        return response
    except Exception as e:
        return f"Tool call failed, error: {e}"


@tool(parse_docstring=True, return_direct=False)
async def web_search(query: str, max_results: int = 8) -> str:
    """Search the web via DuckDuckGo and return the top results.

    Use this to find pages, then read the interesting ones with web_fetch.

    Args:
        query: What to search for, in plain words.
        max_results: How many results to return (default 8).

    Returns:
        Numbered results as `title`, `url`, and a snippet, or an error message.
    """
    try:
        results = await WebParser().search_web(query, max_results=max_results)
        if not results:
            return f"No results for {query!r}"
        lines = []
        for number, result in enumerate(results, start=1):
            title = str(result.get("title") or "").strip()
            url = str(result.get("href") or result.get("url") or "").strip()
            snippet = " ".join(str(result.get("body") or "").split())
            lines.append(f"{number}. {title}\n   {url}\n   {snippet}")
        return "\n".join(lines)
    except Exception as e:
        return f"Tool call failed, error: {e}"
