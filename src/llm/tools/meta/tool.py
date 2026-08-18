"""One meta tool that replaces direct tool wiring: run a script that does it all.

Exposing every tool schema to the model bloats the context and forces one
round-trip per call. Instead the agent sees a single tool — `run_tools` —
and drives every registered capability from a Python-subset script executed
by the sandboxed interpreter in the `sandbox` subpackage. Discovery lives
inside the same scripts: `search_tools(query)` finds capabilities by
keyword and `get_tool(names)` fetches their full contracts, which come back
structurally in the run's result JSON (the `contracts` field) so history
compaction can carry them forward verbatim. Intermediate tool results stay
inside the script; only what the script returns or prints enters the
conversation.

The `run_tools` docstring is the whole contract shown to the model — the
script language AND the discovery flow — keep it in sync with what the
interpreter actually allows and with what this module injects into the
script's tool table.
"""

import json
import re

from langchain_core.tools import tool

from src.config.logging import logger

from ...statistics.script_calls import counting_script_calls
from .attachments import collecting_images
from .defaults import get_registry
from .registry import SEARCH_LIMIT, ToolRegistry
from .sandbox import Interpreter

MAX_RESULT_CHARS = 40_000
MAX_ATTACHED_IMAGES = 6

# A log line that is pure decoration ("=====", "-----", "* * *", …) carries
# zero information and still costs tokens on the way back to the model, so
# such lines are stripped mechanically — the prompt alone does not stop
# every model from printing banner art.
DECORATION_LINE_RE = re.compile(r"^[\s=\-*#~_.•⎯—│|+]{4,}$")


def _strip_decoration(logs: list) -> tuple[list, int]:
    """Logs without banner/divider lines, plus how many lines were removed."""
    cleaned: list = []
    removed = 0
    for entry in logs:
        lines = str(entry).splitlines()
        kept = [line for line in lines if not DECORATION_LINE_RE.match(line)]
        removed += len(lines) - len(kept)
        if kept or not lines:
            cleaned.append("\n".join(kept) if lines else entry)
    return cleaned, removed


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


def _lookup_contract(registry: ToolRegistry, name: str) -> tuple[str | None, str | None]:
    """(contract, problem) for one requested name — exactly one is set."""
    documentation = registry.document(name)
    if documentation is not None:
        return documentation, None
    suggestions = registry.search(name) or registry.all_tools()
    listed = "\n".join(registry.brief(match) for match in suggestions)
    return None, f"Unknown tool: {name!r}. Closest matches:\n{listed}"


# A server with more matches than this collapses into one aggregate line —
# with several MCP servers connected a flat listing would drown the core
# tools, and 21 Playwright tools once ate the whole result.
MCP_GROUP_THRESHOLD = 3
# Drill-in listings (server="...") are the model explicitly asking for one
# server's catalogue, so they get room to show all of it.
SERVER_LIST_LIMIT = 50
_SEARCH_GUIDANCE = (
    "Call a tool straight away when its signature is self-explanatory; "
    "fetch full contracts with get_tool([...]) — several names at once — for the rest."
)


def _server_listing(registry: ToolRegistry, query: str, server: str) -> str:
    """The drill-in view: one MCP server's tools, generously sized."""
    known = registry.mcp_servers()
    if server not in known:
        listed = ", ".join(known) if known else "none are connected"
        return f"Unknown MCP server '{server}'. Known servers: {listed}."
    matches = registry.search_all(query, source=f"mcp:{server}")
    if not matches:
        return (
            f"No tools of MCP server '{server}' match {query!r}; "
            f'search_tools("", server="{server}") lists all of them.'
        )
    shown = matches[:SERVER_LIST_LIMIT]
    header = (
        f"Tools of MCP server '{server}' matching {query!r}:"
        if query.split()
        else f"All {len(matches)} tools of MCP server '{server}':"
    )
    lines = [header, *(registry.brief(match) for match in shown)]
    if len(matches) > len(shown):
        lines.append(f"… and {len(matches) - len(shown)} more — refine the query")
    lines.append(_SEARCH_GUIDANCE)
    return "\n".join(lines)


async def search_tools(query: str, only_mcp: bool = False, server: str = "") -> str:
    """Keyword search over the registry; one `signature — summary` per line.

    Callable from inside run_tools scripts (and directly as a library
    function). When nothing matches, the full catalogue is listed instead.
    MCP servers with many matches collapse into one aggregate line; the
    `server` parameter drills into a single server's catalogue.
    """
    registry = get_registry()
    if server:
        return _server_listing(registry, query, server)
    source_prefix = "mcp:" if only_mcp else None
    matches = registry.search_all(query, source_prefix=source_prefix)
    header = f"Tools matching {query!r}:" if query.split() else "Every registered tool:"
    if not matches and query.split():
        matches = registry.search_all("", source_prefix=source_prefix)
        header = f"Nothing matched {query!r}; every registered tool:"
    if not matches:
        return "No MCP tools are registered." if only_mcp else "No tools are registered."

    native = [match for match in matches if registry.mcp_server_of(match.name) is None]
    by_server: dict[str, list] = {}
    for match in matches:
        server_name = registry.mcp_server_of(match.name)
        if server_name is not None:
            by_server.setdefault(server_name, []).append(match)

    lines = [header]
    shown_native = native[:SEARCH_LIMIT]
    lines.extend(registry.brief(match) for match in shown_native)
    hidden_native = len(native) - len(shown_native)
    if hidden_native:
        lines.append(f"… and {hidden_native} more matching tools — refine the query")
    for server_name, server_tools in sorted(by_server.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if len(server_tools) <= MCP_GROUP_THRESHOLD:
            lines.extend(registry.brief(match) for match in server_tools)
        else:
            lines.append(
                f"MCP server '{server_name}': {len(server_tools)} matching tools — "
                f'search_tools({query!r}, server="{server_name}") lists them'
            )
    lines.append(_SEARCH_GUIDANCE)
    return "\n".join(lines)


async def get_tool(names: list[str] | str) -> str:
    """Full contracts of one or more tools, joined; a library-use helper.

    Inside run_tools scripts a wrapped variant of this runs instead: it
    records the contracts into the run's result JSON (`contracts` field)
    and returns only a short confirmation, so contracts are never paid for
    twice in one tool result.
    """
    registry = get_registry()
    requested = _normalize_tool_names(names)
    if not requested:
        return "No tool names given. Find tools with search_tools first."
    sections = []
    for name in requested:
        contract, problem = _lookup_contract(registry, name)
        sections.append(contract or problem)
    return "\n\n---\n\n".join(sections)


def _script_meta_callables(registry: ToolRegistry, contracts: list[str]) -> dict:
    """The in-script discovery functions, with get_tool bound to this run."""

    async def script_get_tool(names: list[str] | str) -> str:
        requested = _normalize_tool_names(names)
        if not requested:
            return "No tool names given. Find tools with search_tools(query) first."
        fetched: list[str] = []
        problems: list[str] = []
        for name in requested:
            contract, problem = _lookup_contract(registry, name)
            if contract is None:
                problems.append(problem)
                continue
            if contract not in contracts:
                contracts.append(contract)
            fetched.append(name)
        parts = []
        if fetched:
            parts.append(
                "Contracts for " + ", ".join(fetched) + " are included in this "
                "result's 'contracts' field — no need to print or return them."
            )
        parts.extend(problems)
        return "\n\n".join(parts)

    return {"search_tools": search_tools, "get_tool": script_get_tool}


@tool(
    parse_docstring=True,
    return_direct=False,
    response_format="content_and_artifact",
)
async def run_tools(code: str) -> tuple[str, list[dict]]:
    """Run a Python script, server-side, that calls tools and combines results.

    This is your only wired-in tool; every capability (the web, files, the
    shell, the user's screen, ...) is a registered tool your script calls
    by bare name. Discovery happens in-script too:
    search_tools("a few keywords", only_mcp=False, server="") returns
    matching tools one per line as `signature — summary`, and
    get_tool(["name", ...]) fetches
    full contracts — they arrive in this result's `contracts` field
    automatically, so call it as a bare statement and never print or return
    its value. Tools provided by connected MCP servers are listed with an
    [MCP: <server>] marker; pass only_mcp=True to browse only those. An MCP
    server with many matches collapses into one summary line with its match
    count — drill in with search_tools("query", server="thatname"), and
    search_tools("", server="thatname") lists that server's whole
    catalogue. A
    listing line's signature is often all you need — call such a tool right
    away, and reach for get_tool only when the arguments need more detail
    than the signature shows. A contract fetched by a script cannot steer
    later lines of that same script (the code is already written), so when
    contracts are needed the usual shape is one discovery script —
    search_tools(...) plus get_tool([...]) in the same script — and the next
    script does the real work; skip discovery entirely for tools whose
    contracts you already see in the conversation. Tool calls look
    synchronous and are awaited for you, and intermediate data never enters
    the conversation — only what you return or print comes back. Batch
    aggressively: a small task belongs in ONE script — create the directory
    tree, write every file, and run the one verification check in the same
    run — because every extra run_tools call re-sends the whole conversation
    at full price. Split only when a later step genuinely depends on output
    you have not seen yet.

    The language is a Python subset:
    - statements: assignment, if/elif/else, for, while, break/continue,
      try/except Exception, return (allowed at top level; without it the
      value of the last expression is returned);
    - literals, f-strings, slicing, single-generator list/dict
      comprehensions, common str/list/dict/set methods, and builtins len,
      range, abs, round, min, max, sum, sorted, reversed, enumerate, zip,
      any, all, str, int, float, bool, list, dict, set, tuple, random,
      randint, choice, now;
    - a bare tool call on its own line logs its result automatically,
      REPL-style — no print() needed around tool calls; assign to a
      variable to keep a result out of the logs;
    - print(...) records a log line that is returned to you — logs are
      read by you, not the user, so print only the data your next step
      needs: never banners, decorated headers, progress reports, or
      celebration output (the user reads your final chat answer instead);
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
        `contracts` (full contracts of every tool the script passed to
        get_tool), and `attached_images` when the run captured screenshots
        for you.
    """
    try:
        registry = get_registry()
        contracts: list[str] = []
        table = dict(registry.callables())
        table.update(_script_meta_callables(registry, contracts))
        interpreter = Interpreter(table)
        with collecting_images() as images, counting_script_calls() as calls:
            outcome = await interpreter.run(code)
        outcome["logs"], removed_decoration = _strip_decoration(outcome["logs"])
        if removed_decoration:
            outcome["removed_decoration"] = (
                f"{removed_decoration} banner/divider line(s) stripped from logs — print data, not decorations"
            )
        if contracts:
            outcome["contracts"] = contracts
        if len(images) > MAX_ATTACHED_IMAGES:
            outcome["dropped_images"] = len(images) - MAX_ATTACHED_IMAGES
            del images[:-MAX_ATTACHED_IMAGES]
        if images:
            outcome["attached_images"] = len(images)
        if calls[0]:
            outcome["tool_calls"] = calls[0]
        logger.info(
            "run_tools finished ops [{}] logs [{}] tool_calls [{}] " "contracts [{}] images [{}] error [{}]",
            interpreter.ops,
            len(outcome["logs"]),
            calls[0],
            len(contracts),
            len(images),
            outcome["error"],
        )
        return _clip(json.dumps(outcome, ensure_ascii=False, default=repr)), images
    except Exception as error:
        return f"Tool call failed, error: {error}", []


META_TOOLS = [run_tools]
