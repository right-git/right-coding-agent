"""Three meta tools that replace direct tool wiring: discover, read, run.

Exposing every tool schema to the model bloats the context and forces one
round-trip per call. Instead the agent sees only `search_tools`, `get_tool`,
and `run_tools`: it finds a capability by keyword, reads its full contract,
then drives it — with loops, branching, polling, and fan-out — from one
Python-subset script executed by the sandboxed interpreter in
`src.tools.base`. Intermediate tool results stay inside the script; only what
the script returns or prints enters the conversation.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool, tool

from src.config.logging import logger
from src.tools.base import Interpreter

from .computer_tools import COMPUTER_TOOLS
from .tools import web_search


SEARCH_LIMIT = 8
MAX_RESULT_CHARS = 40_000

# Names the interpreter resolves before the tool table (its builtins and the
# `parallel` special form). A tool registered under one of these would be
# silently unreachable from scripts, so registration rejects them outright.
RESERVED_SCRIPT_NAMES = frozenset({
    "parallel", "run_functions_in_parallel", "sleep",
    "random", "randint", "choice", "now",
    "len", "range", "abs", "round", "min", "max", "sum",
    "sorted", "reversed", "enumerate", "zip", "any", "all",
    "isinstance", "type_name",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "print", "Exception",
})


def _as_script_callable(
    tool_obj: BaseTool,
) -> Callable[..., Awaitable[Any]]:
    """Adapt a LangChain tool to the plain async callable scripts invoke.

    Scripts call tools like ordinary functions, so positional arguments are
    mapped onto the schema's field order before going through `ainvoke`,
    which keeps the tool's own argument validation in the path.
    """
    field_names = list(tool_obj.args)

    async def call(*args: Any, **kwargs: Any) -> Any:
        if len(args) > len(field_names):
            raise TypeError(
                f"{tool_obj.name}() takes at most {len(field_names)} "
                f"positional argument(s), got {len(args)}"
            )
        payload = dict(zip(field_names, args))
        duplicated = payload.keys() & kwargs.keys()
        if duplicated:
            raise TypeError(
                f"{tool_obj.name}() got multiple values for: "
                + ", ".join(sorted(duplicated))
            )
        payload.update(kwargs)
        return await tool_obj.ainvoke(payload)

    return call


class ToolRegistry:
    """Executable tools plus the metadata needed to find and document them."""

    def __init__(self, tools: Sequence[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool_obj in tools:
            self.register(tool_obj)

    def register(self, tool_obj: BaseTool) -> None:
        if tool_obj.name in RESERVED_SCRIPT_NAMES:
            raise ValueError(
                f"Tool name {tool_obj.name!r} collides with an interpreter "
                "builtin and would be unreachable from scripts"
            )
        if tool_obj.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_obj.name}")
        self._tools[tool_obj.name] = tool_obj

    def all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name.strip())

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[BaseTool]:
        """Keyword search over names and descriptions, best matches first."""
        terms = [term for term in query.casefold().split() if term]
        if not terms:
            return self.all_tools()[:limit]

        scored: list[tuple[int, BaseTool]] = []
        for tool_obj in self._tools.values():
            name = tool_obj.name.casefold()
            description = (tool_obj.description or "").casefold()
            score = 0
            for term in terms:
                if term in name:
                    score += 3
                if term in description:
                    score += 1
            if score:
                scored.append((score, tool_obj))

        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [tool_obj for _, tool_obj in scored[:limit]]

    def signature(self, tool_obj: BaseTool) -> str:
        parts = []
        for field_name, spec in tool_obj.args.items():
            if isinstance(spec, dict) and "default" in spec:
                parts.append(f"{field_name}={spec['default']!r}")
            else:
                parts.append(field_name)
        return f"{tool_obj.name}({', '.join(parts)})"

    def brief(self, tool_obj: BaseTool) -> str:
        description = " ".join((tool_obj.description or "").split())
        head, separator, _ = description.partition(". ")
        summary = f"{head}." if separator else description
        return f"{self.signature(tool_obj)} — {summary}"

    def document(self, name: str) -> str | None:
        """Full contract of one tool, or None when the name is unknown."""
        tool_obj = self.get(name)
        if tool_obj is None:
            return None
        schema = json.dumps(tool_obj.args, ensure_ascii=False, indent=2)
        example = ", ".join(f"{field}=..." for field in tool_obj.args)
        return (
            f"{self.signature(tool_obj)}\n\n"
            f"{(tool_obj.description or '').strip()}\n\n"
            f"Argument schema:\n{schema}\n\n"
            f"Call it inside run_tools code by bare name: "
            f"{tool_obj.name}({example})"
        )

    def callables(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        """The tool table handed to the sandboxed interpreter."""
        return {
            name: _as_script_callable(tool_obj)
            for name, tool_obj in self._tools.items()
        }


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """The process-wide tool registry, created on first use."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry([web_search, *COMPUTER_TOOLS])
    return _registry


def set_registry(registry: ToolRegistry | None) -> None:
    """Replace the shared registry (used by tests and embedders)."""
    global _registry
    _registry = registry


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_RESULT_CHARS
    return (
        text[:MAX_RESULT_CHARS]
        + f" ...[truncated {dropped} characters; "
        "have the script return or print less]"
    )


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
                "Read a tool's full contract with get_tool before calling "
                "it in run_tools.",
            ]
        )
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def get_tool(name: str) -> str:
    """Full contract of one tool: description, arguments, usage.

    Read this before the first call to a tool found via search_tools, so
    your run_tools script passes the right arguments.

    Args:
        name: Exact tool name as returned by search_tools.

    Returns:
        The tool's signature, documentation, and argument schema, or the
        closest registered names when no such tool exists.
    """
    try:
        registry = get_registry()
        documentation = registry.document(name)
        if documentation is not None:
            return documentation
        suggestions = registry.search(name) or registry.all_tools()
        listed = "\n".join(registry.brief(match) for match in suggestions)
        return f"Unknown tool: {name!r}. Closest matches:\n{listed}"
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def run_tools(code: str) -> str:
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
      desktop is one shared device.

    Not available: import, def/lambda/class, generators, for/while-else,
    subscript assignment (build dicts with dict.update({...}) and lists
    with list.append(...)), attribute access beyond whitelisted methods,
    files, network, os. raise ends the whole run. Budgets: 100k interpreter
    ops, 600s total sleep, 900s wall clock; exceeding one aborts the run.

    Args:
        code: Source of the script. It runs once, immediately.

    Returns:
        JSON object with `result` (the returned value), `logs` (print
        output), and `error` (null on success, otherwise why the run
        stopped).
    """
    try:
        registry = get_registry()
        interpreter = Interpreter(registry.callables())
        outcome = await interpreter.run(code)
        logger.info(
            "run_tools finished ops [{}] logs [{}] error [{}]",
            interpreter.ops,
            len(outcome["logs"]),
            outcome["error"],
        )
        return _clip(json.dumps(outcome, ensure_ascii=False, default=repr))
    except Exception as error:
        return f"Tool call failed, error: {error}"


META_TOOLS = [search_tools, get_tool, run_tools]
