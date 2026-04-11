from langchain_core.tools import tool
from src.tools.web_parser import WebParser


@tool(parse_docstring=True, return_direct=False)
async def web_search(url: str) -> str:
    """Parse webpage and get information.

    Args:
        url: URL of the webpage to parse.

    Returns:
        Parsed content of the webpage.
    """
    try:
        web_parser = WebParser()
        response = await web_parser.parse_page(url)
        return response
    except Exception as e:
        return f"Tool call failed, error: {e}"
