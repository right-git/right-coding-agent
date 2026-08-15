from langchain_core.tools import tool

from .service import WebParser


@tool(parse_docstring=True, return_direct=False)
async def web_search(url: str) -> str:
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
