"""`python -m src.main mcp ...` — Claude Code-compatible MCP management.

Grammar note on `add`: the brief's sketch used `args=nargs=argparse.REMAINDER`
for the trailing stdio command/args. REMAINDER turns off argparse's normal
option recognition for everything after the point it starts matching, which
breaks the no-"--" form where a flag (`--header ...`) legitimately follows
the positionals (`argparse.REMAINDER` would swallow it as a plain arg
instead of parsing it). `nargs="*"` does not have that problem: argparse's
ordinary interleaved option/positional matching still applies, and a literal
"--" already tells argparse to stop treating anything after it as an option
— which is exactly the behavior a stdio command's own flags need. Verified
against all five `add` argv shapes the test file exercises (before writing
this module) so this is a drop-in, simpler substitute for REMAINDER here,
not a workaround bolted on afterwards.
"""

import argparse
import asyncio
import json

from rich.console import Console

from .config import (
    SERVER_NAME_RE,
    McpServerConfig,
    add_server,
    config_from_entry,
    load_mcp_servers,
    project_config_path,
    remove_server,
    scopes_containing,
    server_entry_json,
    user_config_path,
)

ERROR_STYLE = "bold red"
SUCCESS_STYLE = "bold green"


def _project_file():
    return project_config_path()


def _user_file():
    return user_config_path()


def _error(console: Console, message: str) -> int:
    console.print(f"error: {message}", style=ERROR_STYLE)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp", description="Manage MCP servers.")
    sub = parser.add_subparsers(dest="command")

    add = sub.add_parser("add", help="Add an MCP server (stdio command or http/sse url).")
    add.add_argument("name")
    add.add_argument("target", nargs="?", help="command (stdio) or url (http/sse)")
    add.add_argument("args", nargs="*", help="stdio command arguments, after --")
    add.add_argument("--transport", choices=["stdio", "http", "sse"])
    add.add_argument("--scope", choices=["project", "user"], default="project")
    add.add_argument("--env", action="append", help="KEY=VALUE, repeatable (stdio only)")
    add.add_argument("--header", action="append", help="'Key: Value', repeatable (http/sse only)")

    add_json = sub.add_parser("add-json", help="Add an MCP server from a raw JSON entry.")
    add_json.add_argument("name")
    add_json.add_argument("json")
    add_json.add_argument("--scope", choices=["project", "user"], default="project")

    sub.add_parser("list", help="List configured MCP servers and connect to them.")

    get = sub.add_parser("get", help="Show one server's configuration.")
    get.add_argument("name")

    remove = sub.add_parser("remove", help="Remove an MCP server.")
    remove.add_argument("name")
    remove.add_argument("--scope", choices=["project", "user"], default=None)

    return parser


def parse_add(args: argparse.Namespace):
    """`McpServerConfig` from a parsed `add` namespace.

    Transport: explicit `--transport` wins; otherwise a target starting with
    `http://`/`https://` implies `http`, everything else implies `stdio`.
    """
    if not SERVER_NAME_RE.match(args.name):
        argparse.ArgumentParser(prog="mcp add").error(f"invalid server name: {args.name!r}")

    target = args.target
    transport = args.transport or ("http" if (target or "").startswith(("http://", "https://")) else "stdio")

    kwargs: dict = {"name": args.name, "transport": transport, "scope": args.scope}
    if transport == "stdio":
        kwargs["command"] = target
        kwargs["args"] = list(args.args or [])
        env = {}
        for item in args.env or []:
            key, _, value = item.partition("=")
            env[key] = value
        kwargs["env"] = env
    else:
        kwargs["url"] = target
        headers = {}
        for item in args.header or []:
            key, _, value = item.partition(":")
            headers[key.strip()] = value.strip()
        kwargs["headers"] = headers

    return McpServerConfig(**kwargs)


def _cmd_add(args: argparse.Namespace) -> int:
    console = Console()
    try:
        config = parse_add(args)
    except ValueError as error:
        return _error(console, str(error))
    file = _user_file() if config.scope == "user" else _project_file()
    try:
        add_server(config, file)
    except ValueError as error:
        return _error(console, str(error))
    console.print(f"Added {config.transport} server '{config.name}' to {config.scope} scope", style=SUCCESS_STYLE)
    return 0


def _cmd_add_json(args: argparse.Namespace) -> int:
    console = Console()
    if not SERVER_NAME_RE.match(args.name):
        return _error(console, f"invalid server name: {args.name!r}")
    try:
        entry = json.loads(args.json)
        if not isinstance(entry, dict):
            raise ValueError("JSON entry must be an object")
        config = config_from_entry(args.name, entry, args.scope)
        file = _user_file() if args.scope == "user" else _project_file()
        add_server(config, file)
    except ValueError as error:
        return _error(console, str(error))
    console.print(f"Added {config.transport} server '{config.name}' to {config.scope} scope", style=SUCCESS_STYLE)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    console = Console()
    servers = load_mcp_servers(project_file=_project_file(), user_file=_user_file())
    if not servers:
        console.print("No MCP servers configured.")
        return 0

    # Imported here, not at module scope: only `list` needs a manager, so
    # this keeps the other subcommands' imports light and scopes the
    # `ToolRegistry` import to where it's built as a private instance below
    # (never the process-default registry). This is NOT resilience against
    # a broken MCP SDK import — `src.llm.tools`'s own `__init__.py` already
    # imports `.mcp.manager` (and transitively the SDK) at module scope, so
    # by the time any code in this file runs, that import has already
    # happened or already failed.
    from src.llm.tools import ToolRegistry

    from .manager import McpManager

    async def _gather():
        manager = McpManager(configs=servers, registry=ToolRegistry())
        await manager.start()
        try:
            return manager.statuses()
        finally:
            await manager.stop()

    statuses = asyncio.run(_gather())
    for status in statuses:
        state = status.state.value
        if state == "connected":
            console.print(f"✓ {status.name}  {status.transport}  {status.scope}  {status.tool_count} tools")
        elif state == "needs auth":
            console.print(f"⚠ {status.name}  {status.transport}  {status.scope}  needs auth")
        else:
            console.print(f"✗ {status.name} — {status.error or state}")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    console = Console()
    scopes = scopes_containing(args.name, _project_file(), _user_file())
    if not scopes:
        return _error(console, f"no MCP server named '{args.name}'")
    servers = load_mcp_servers(project_file=_project_file(), user_file=_user_file(), env={})
    config = servers.get(args.name)
    if config is None:
        return _error(console, f"MCP server '{args.name}' could not be loaded (invalid entry?)")
    console.print(f"{args.name}:")
    console.print(json.dumps(server_entry_json(config), indent=2))
    console.print(f"scope: {'/'.join(scopes)}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    console = Console()
    scopes = scopes_containing(args.name, _project_file(), _user_file())
    if not scopes:
        return _error(console, f"no MCP server named '{args.name}'")

    scope = args.scope
    if scope is None:
        if len(scopes) > 1:
            return _error(console, f"'{args.name}' is configured in both scopes ({', '.join(scopes)}); pass --scope")
        scope = scopes[0]
    elif scope not in scopes:
        return _error(console, f"no MCP server named '{args.name}' in scope '{scope}'")

    file = _user_file() if scope == "user" else _project_file()
    try:
        removed = remove_server(args.name, file)
    except ValueError as error:
        return _error(console, str(error))
    if not removed:
        return _error(console, f"no MCP server named '{args.name}' in scope '{scope}'")
    console.print(f"Removed MCP server '{args.name}' from {scope} scope", style=SUCCESS_STYLE)
    return 0


_HANDLERS = {
    "add": _cmd_add,
    "add-json": _cmd_add_json,
    "list": _cmd_list,
    "get": _cmd_get,
    "remove": _cmd_remove,
}


def run_mcp_cli(argv: list[str]) -> int:
    """Parse and dispatch one `mcp` subcommand; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return _HANDLERS[args.command](args)
