"""Three meta tools that replace direct tool wiring: discover, read, run.

Exposing every tool schema to the model bloats the context and forces one
round-trip per call. Instead the agent sees only `search_tools`, `get_tool`,
and `run_tools`: it finds a capability by keyword, reads its full contract,
then drives it — with loops, branching, polling, and fan-out — from one
Python-subset script executed by the sandboxed interpreter in the `sandbox`
subpackage. Intermediate tool results stay inside the script; only what the
script returns or prints enters the conversation.

The `run_tools` docstring is the language contract shown to the model — keep
it in sync with what the interpreter actually allows.
"""

import json
import re

from langchain_core.tools import tool

from src.config.logging import logger

from ...statistics.script_calls import counting_script_calls
from .attachments import collecting_images
from .defaults import get_registry
from .sandbox import Interpreter

MAX_RESULT_CHARS = 40_000
MAX_ATTACHED_IMAGES = 6


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_RESULT_CHARS
    return text[:MAX_RESULT_CHARS] + f" ...[truncated {dropped} characters; " "have the script return or print less]"


def _normalize_tool_names(names: list[str] | str) -> list[str]:
    """A clean, deduplicated name list from either a list or a loose string."""
    if isinstance(names, str):
        names = [part for part in re.split(r"[,\s]+", names) if part]
    cleaned = [name.strip() for name in names if name and name.strip()]
    return list(dict.fromkeys(cleaned))


@tool(parse_docstring=True, return_direct=False)
async def search_tools(query: str) -> str:
    """Find tools for a capability you need but cannot see.

    Only search_tools, get_tool, and run_tools are wired in directly; every
    other capability (web pages, the user's screen, ...) is a registered tool
    that you find here, study with get_tool, and call inside run_tools.

    Args:
        query: A few keywords describing what you want to do, for example
            "click button screen" or "fetch web page".

    Returns:
        One matching tool per line as `signature — summary`. When nothing
        matches, the full catalogue is listed instead.
    """
    try:
        registry = get_registry()
        matches = registry.search(query)
        header = f"Tools matching {query!r}:"
        if not matches:
            matches = registry.all_tools()
            header = f"Nothing matched {query!r}; every registered tool:"
        if not matches:
            return "No tools are registered."
        return "\n".join(
            [
                header,
                *(registry.brief(match) for match in matches),
                "Fetch full contracts with get_tool — it takes several "
                "names at once — before calling them in run_tools.",
            ]
        )
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def get_tool(names: list[str] | str) -> str:
    """Full contracts of one or more tools: description, arguments, usage.

    Read this before the first call to tools found via search_tools, so your
    run_tools script passes the right arguments. Request every tool you plan
    to use in a single call instead of calling once per tool.

    Args:
        names: Exact tool names as returned by search_tools, for example
            ["screen_locate", "screen_click"].

    Returns:
        The signature, documentation, and argument schema of every requested
        tool, in order. Unknown names get the closest registered names
        instead.
    """
    try:
        registry = get_registry()
        requested = _normalize_tool_names(names)
        if not requested:
            return "No tool names given. Find tools with search_tools first."

        sections = []
        for name in requested:
            documentation = registry.document(name)
            if documentation is None:
                suggestions = registry.search(name) or registry.all_tools()
                listed = "\n".join(registry.brief(match) for match in suggestions)
                documentation = f"Unknown tool: {name!r}. Closest matches:\n{listed}"
            sections.append(documentation)
        return "\n\n---\n\n".join(sections)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(
    parse_docstring=True,
    return_direct=False,
    response_format="content_and_artifact",
)
async def run_tools(code: str) -> tuple[str, list[dict]]:
    """Run a Python script, server-side, that calls tools and combines results.

    This is how registered tools get used: find them with search_tools, read
    their contracts with get_tool, then write a short script that calls them
    by bare name. Tool calls look synchronous and are awaited for you, and
    intermediate data never enters the conversation — only what you return
    or print comes back. Prefer one script over many separate tool calls
    whenever results feed each other, need filtering, or can run in parallel.

    The language is a Python subset:
    - statements: assignment, if/elif/else, for, while, break/continue,
      try/except Exception, return (allowed at top level; without it the
      value of the last expression is returned);
    - literals, f-strings, slicing, single-generator list/dict
      comprehensions, common str/list/dict/set methods, and builtins len,
      range, abs, round, min, max, sum, sorted, reversed, enumerate, zip,
      any, all, str, int, float, bool, list, dict, set, tuple, random,
      randint, choice, now;
    - print(...) records a log line that is returned to you;
    - sleep(seconds) waits at zero token cost — poll a slow job with a while
      loop plus sleep instead of checking from the conversation;
    - parallel(tool_a(...), tool_b(...), ...) runs direct tool calls
      concurrently and returns their results as a list, in order. Use it
      only for independent calls, and never for screen_* tools — the
      desktop is one shared device;
    - tools that capture the screen (screen_screenshot, screen_locate with
      return_screen=True) attach the picture to the conversation: you will
      see it right after this tool's result. Base64 text is never visible
      to you, so do not return image data from the script.

    Not available: import, def/lambda/class, generators, for/while-else,
    subscript assignment (build dicts with dict.update({...}) and lists
    with list.append(...)), attribute access beyond whitelisted methods,
    files, network, os. raise ends the whole run. Budgets: 100k interpreter
    ops, 600s total sleep, 900s wall clock; exceeding one aborts the run.

    Args:
        code: Source of the script. It runs once, immediately.

    Returns:
        JSON object with `result` (the returned value), `logs` (print
        output), `error` (null on success, otherwise why the run stopped),
        and `attached_images` when the run captured screenshots for you.
    """
    try:
        registry = get_registry()
        interpreter = Interpreter(registry.callables())
        with collecting_images() as images, counting_script_calls() as calls:
            outcome = await interpreter.run(code)
        if len(images) > MAX_ATTACHED_IMAGES:
            outcome["dropped_images"] = len(images) - MAX_ATTACHED_IMAGES
            del images[:-MAX_ATTACHED_IMAGES]
        if images:
            outcome["attached_images"] = len(images)
        if calls[0]:
            outcome["tool_calls"] = calls[0]
        logger.info(
            "run_tools finished ops [{}] logs [{}] tool_calls [{}] " "images [{}] error [{}]",
            interpreter.ops,
            len(outcome["logs"]),
            calls[0],
            len(images),
            outcome["error"],
        )
        return _clip(json.dumps(outcome, ensure_ascii=False, default=repr)), images
    except Exception as error:
        return f"Tool call failed, error: {error}", []


META_TOOLS = [search_tools, get_tool, run_tools]
