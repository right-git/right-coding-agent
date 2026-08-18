"""The tool registry: executable tools plus the metadata to find and document them.

`ToolRegistry` is deliberately free of concrete tools — the process-wide
default registry with its actual tool set lives in `defaults.py`, and the
meta tools that expose a registry to the model live in `tool.py`.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from ...statistics.script_calls import count_script_call

SEARCH_LIMIT = 8
MCP_SOURCE_PREFIX = "mcp:"

# Names the interpreter resolves before the tool table (its builtins and the
# `parallel` special form), plus the meta names run_tools injects into every
# script's tool table (search_tools/get_tool) and run_tools itself. A tool
# registered under one of these would be silently unreachable from scripts,
# so registration rejects them outright.
RESERVED_SCRIPT_NAMES = frozenset(
    {
        "search_tools",
        "get_tool",
        "run_tools",
        "parallel",
        "run_functions_in_parallel",
        "sleep",
        "random",
        "randint",
        "choice",
        "now",
        "len",
        "range",
        "abs",
        "round",
        "min",
        "max",
        "sum",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "any",
        "all",
        "isinstance",
        "type_name",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "print",
        "Exception",
    }
)


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
        count_script_call()
        if len(args) > len(field_names):
            raise TypeError(
                f"{tool_obj.name}() takes at most {len(field_names)} " f"positional argument(s), got {len(args)}"
            )
        payload = dict(zip(field_names, args))
        duplicated = payload.keys() & kwargs.keys()
        if duplicated:
            raise TypeError(f"{tool_obj.name}() got multiple values for: " + ", ".join(sorted(duplicated)))
        payload.update(kwargs)
        return await tool_obj.ainvoke(payload)

    return call


class ToolRegistry:
    """Executable tools plus the metadata needed to find and document them."""

    def __init__(self, tools: Sequence[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._sources: dict[str, str] = {}
        for tool_obj in tools:
            self.register(tool_obj)

    def register(self, tool_obj: BaseTool, source: str | None = None) -> None:
        if tool_obj.name in RESERVED_SCRIPT_NAMES:
            raise ValueError(
                f"Tool name {tool_obj.name!r} collides with an interpreter "
                "builtin and would be unreachable from scripts"
            )
        if tool_obj.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_obj.name}")
        self._tools[tool_obj.name] = tool_obj
        if source:
            self._sources[tool_obj.name] = source

    def unregister(self, name: str) -> bool:
        self._sources.pop(name, None)
        return self._tools.pop(name, None) is not None

    def source_of(self, name: str) -> str | None:
        return self._sources.get(name)

    def all_tools(self, source_prefix: str | None = None) -> list[BaseTool]:
        tools = list(self._tools.values())
        if source_prefix is None:
            return tools
        return [t for t in tools if (self._sources.get(t.name) or "").startswith(source_prefix)]

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name.strip())

    def search(self, query: str, limit: int = SEARCH_LIMIT, source_prefix: str | None = None) -> list[BaseTool]:
        """Keyword search over names and descriptions, best matches first."""
        return self.search_all(query, source_prefix=source_prefix)[:limit]

    def search_all(
        self,
        query: str,
        source_prefix: str | None = None,
        source: str | None = None,
    ) -> list[BaseTool]:
        """Every match, unlimited — callers decide how to truncate or group.

        `source` filters to an exact origin label ("mcp:pw"), `source_prefix`
        to a label family ("mcp:"); an empty query matches everything that
        passes the filters.
        """
        terms = [term for term in query.casefold().split() if term]
        scored: list[tuple[int, BaseTool]] = []
        for tool_obj in self._tools.values():
            tool_source = self._sources.get(tool_obj.name) or ""
            if source_prefix is not None and not tool_source.startswith(source_prefix):
                continue
            if source is not None and tool_source != source:
                continue
            if not terms:
                scored.append((0, tool_obj))
                continue
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
        return [tool_obj for _, tool_obj in scored]

    def mcp_servers(self) -> list[str]:
        """Names of MCP servers with at least one registered tool, sorted."""
        return sorted(
            {
                source[len(MCP_SOURCE_PREFIX) :]
                for source in self._sources.values()
                if source.startswith(MCP_SOURCE_PREFIX)
            }
        )

    def mcp_server_of(self, name: str) -> str | None:
        """The MCP server a tool came from, or None for native tools."""
        source = self._sources.get(name) or ""
        if source.startswith(MCP_SOURCE_PREFIX):
            return source[len(MCP_SOURCE_PREFIX) :]
        return None

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
        line = f"{self.signature(tool_obj)} — {summary}"
        source = self._sources.get(tool_obj.name) or ""
        if source.startswith(MCP_SOURCE_PREFIX):
            line += f" [MCP: {source[len(MCP_SOURCE_PREFIX):]}]"
        return line

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
        return {name: _as_script_callable(tool_obj) for name, tool_obj in self._tools.items()}
