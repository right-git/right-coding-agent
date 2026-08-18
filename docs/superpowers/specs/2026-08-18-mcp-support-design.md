# MCP Support Design

## Goal

Give the agent Model Context Protocol (MCP) support with the same command surface as Claude Code: `mcp add / add-json / list / get / remove` CLI subcommands, Claude Code-compatible `.mcp.json` config files, stdio + streamable HTTP + SSE transports, OAuth for remote servers, and full use of server capabilities — tools, resources, and prompts. MCP tools join the existing `ToolRegistry` behind the meta layer, so connecting any number of servers adds **zero** per-turn context cost: the model discovers them through `search_tools` exactly like built-in tools.

A related REPL feature ships with it: `/tool <name>` pins any registry tool (built-in or MCP) to the next user message, attaching its full contract so the model skips the `search_tools`/`get_tool` round-trip.

Prior art: `/Users/a1/Desktop/Personal/Chattler/backend` contains a production MCP integration (HTTP-only, session-per-call, custom multi-user OAuth). Its pure functions — argument normalization, result serialization, tool-name hashing — and its hard-won lessons (literal `Bearer`, RFC 9728/8414 discovery candidate shapes, the 401 → force-refresh → single-retry pattern) are ported; its Mongo/encryption/FastAPI machinery is not.

## New package

`src/llm/tools/mcp/` following the repo's tool-package pattern:

- `config.py` — config models, file loading/merging, add/remove persistence.
- `manager.py` — `McpManager`: persistent connections, lifecycle, statuses.
- `adapter.py` — MCP tool → LangChain `StructuredTool`; argument normalization; result serialization.
- `oauth.py` — file token storage, localhost callback server, login/logout flows.
- `cli.py` — the `mcp` argparse subcommands.
- `tool.py` — the `mcp_list_resources` / `mcp_read_resource` registry tools.

One new dependency: `mcp` (the official Python SDK). Everything OS- or network-facing takes injectable seams; the unit suite runs with fakes only.

## Config files and scopes

Two scopes, project overriding user on name collision:

- **project**: `.mcp.json` at the repo root (the cwd the REPL starts in), Claude Code's exact format — files copy between the two tools unchanged.
- **user**: `~/.right-agent/mcp.json`, same format.

```json
{
  "mcpServers": {
    "playwright": { "type": "stdio", "command": "npx", "args": ["@playwright/mcp@latest"], "env": {} },
    "context7":   { "type": "http", "url": "https://mcp.context7.com/mcp", "headers": { "X-API-Key": "${CTX7_KEY}" } },
    "legacy":     { "type": "sse",  "url": "https://old.example/sse" }
  }
}
```

- `type` is optional: an entry with `command` and no `type` reads as stdio (Claude Code behavior). Valid types: `stdio`, `http`, `sse`.
- `${VAR}` and `${VAR:-default}` expand from the environment in `command`, `args`, `env` values, `url`, and `headers` values — at load time only; the file keeps the placeholder.
- `McpServerConfig` (pydantic): `name`, `transport`, `command`/`args`/`env` (stdio), `url`/`headers` (http/sse), `scope`. Server names validate as `[A-Za-z0-9_-]+` at add time.
- `load_mcp_servers()` merges both files; `add_server()`/`remove_server()` rewrite only the `mcpServers` entry they own, preserving any other JSON keys in the file. File paths are constructor/function parameters so tests use temp dirs.

## CLI subcommands

`src/main.py` dispatches `sys.argv[1] == "mcp"` to `run_mcp_cli(argv)` before entering the REPL path. Syntax mirrors `claude mcp`:

```
uv run python -m src.main mcp add [--transport stdio|http|sse] [--scope project|user] \
    [--env KEY=VAL]... [--header "K: V"]... <name> <commandOrUrl> [args...]
uv run python -m src.main mcp add pw -- npx @playwright/mcp@latest
uv run python -m src.main mcp add-json <name> '<json>' [--scope project|user]
uv run python -m src.main mcp list
uv run python -m src.main mcp get <name>
uv run python -m src.main mcp remove <name> [--scope project|user]
```

- Default scope: `project`. Default transport: `stdio` (a URL argument with no `--transport` implies `http`, matching Claude Code).
- `--` separates the server command from our flags.
- `mcp list` actually connects to each server (bounded by the connect timeout) and prints ✓ connected (with tool count) / ✗ failed (with the error) / needs auth.
- `mcp remove` without `--scope`: removes when the name lives in exactly one scope; demands `--scope` when it lives in both.
- Output uses the same rich console styling as the REPL. `mcp serve` and `add-from-claude-desktop` are out of scope.

## Connection manager

`McpManager` owns one `McpConnection` per configured server. Connections are **persistent**: an asyncio task per server enters the transport context (`stdio_client` / `streamablehttp_client` / `sse_client`) and a `ClientSession`, calls `initialize()`, lists tools/prompts/resources, registers the adapted tools into the process registry, then parks on a stop event; on shutdown it unregisters and unwinds the contexts. The task owns the context managers because anyio requires entering and exiting them in the same task. A stdio server's process therefore survives between calls — Playwright keeps its browser open.

- **Startup**: `main()` launches `manager.start()` as a background task; servers connect concurrently; progress reports through the existing `ChatUI.set_model_status` right-prompt channel (`⏳ mcp:playwright` → `✓`). The REPL never blocks on MCP. Tools that register after a turn already started simply appear in the next `search_tools` — the meta layer reads the registry live.
- **States**: `connecting`, `connected`, `failed` (with the error), `needs_auth`, `disconnected`.
- **Timeouts** (new settings with defaults): `mcp_connect_timeout` = 30 s, `mcp_tool_timeout` = 60 s (passed per call), `mcp_oauth_port` = 43110.
- **Auto-reconnect**: when a call fails because the session died (remote idle timeout, crashed stdio process), the manager reconnects once and retries the call once; a second failure returns as an error string. `/mcp reconnect <name>` forces it manually.
- **Shutdown**: REPL exit stops every connection task; stdio subprocesses terminate.
- The manager is injectable (`set_mcp_manager()` mirror of `set_computer()`/`set_registry()`); the session factory is a constructor seam so tests connect fake sessions.

## Tool adaptation and registry integration

`adapter.py` builds a LangChain `StructuredTool` per remote tool:

- **Name**: `mcp__<server>__<tool>`, both parts sanitized to identifier characters; names over 64 chars truncate and append an 8-char sha256 suffix (Chattler's scheme) so they stay unique and model-safe. The `mcp__` prefix keeps them clear of `RESERVED_SCRIPT_NAMES`.
- **Schema**: the raw MCP `inputSchema` JSON-schema dict becomes `args_schema` — LangChain accepts dicts, and `ToolRegistry.callables()`/`signature()`/`document()` keep working off `tool.args` unchanged.
- **Arguments**: before the call, the ported `normalize_mcp_tool_arguments` coerces model-sent strings to schema types (booleans, integers, numbers, arrays, objects) — weak models routinely send `"true"` for a boolean.
- **Results**: ported `serialize_mcp_call_result`, adapted: text content returns as plain text; `structuredContent` as JSON; **image content goes through `attach_image()`** so the model actually sees it via the existing `AttachedImagesMiddleware` channel (falling back to the stub note outside a collection channel); resource / resource_link items become compact JSON summaries. `isError` results return as error strings — MCP tools never raise, matching the repo's tool contract.
- **Annotations**: `readOnlyHint` / `destructiveHint` append `[read-only]` / `[DESTRUCTIVE]` to the description so discovery surfaces risk.

`ToolRegistry` changes, kept generic for future sources (skills):

- `register(tool, source=None)` stores an origin label per tool (e.g. `mcp:playwright`); `unregister(name)` removes one (reconnect unregisters a server's tools before re-registering).
- `search(query, source_prefix=None)` filters by origin; `brief()` output appends the origin marker (`… [MCP: playwright]`).
- The in-script `search_tools` gains `only_mcp=False` (filter to MCP-sourced tools; an `only_skills` twin arrives with the future skills feature). The `run_tools` docstring — the language contract shown to the model — documents both the parameter and the origin markers.

## Resources

Two ordinary registry tools, registered whenever at least one MCP server is configured:

- `mcp_list_resources(server=None)` — resources (and resource templates) across connected servers or one server.
- `mcp_read_resource(server, uri)` — reads one resource; text returns as text, blobs report mime type and size, images go through `attach_image()`.

The model reaches them through normal `search_tools` discovery inside `run_tools` scripts. No @-mention syntax.

## Prompts as slash commands

Server prompts surface as dynamic REPL commands, Claude Code style: `/mcp__<server>__<prompt> [args...]`, arguments mapped positionally onto the prompt's declared arguments. `CommandHandler.handle` grows a structured return contract (today it returns `"clear" | None`): a command may now also return a *send-this-as-user-message* outcome or an *await-this-async-action* outcome. The main loop feeds a prompt result (`get_prompt` messages flattened to one user text) into a normal turn; async actions (`reconnect`, `login`, `logout`) are awaited by the loop, keeping `handle` itself synchronous.

`CommandCompleter` reads live command names (prompts appear after their server connects) and completes `/mcp` subcommands and `/tool` tool names.

## /mcp status command

- `/mcp` — a rich table: name, transport, scope, state, tool/prompt/resource counts, last error.
- `/mcp reconnect <name>` — drop and re-establish the connection.
- `/mcp login <name>` / `/mcp logout <name>` — OAuth (below).
- `/help` lists `/mcp` and `/tool`.

## OAuth for remote servers

Built on the SDK's `OAuthClientProvider`, which handles RFC 9728/8414 metadata discovery, dynamic client registration, PKCE, token exchange, and automatic refresh (and sends the literal `Bearer` scheme — the Chattler lesson the SDK already encodes). Ours:

- `FileTokenStorage` implements the SDK `TokenStorage` protocol: tokens and client registrations in `~/.right-agent/mcp-tokens.json`, keyed by server name + URL, file mode 0600.
- A minimal localhost HTTP callback server on `127.0.0.1:<mcp_oauth_port>` (fixed port → stable redirect URI across sessions), showing a "you can close this tab" page and handing the code to the waiting flow.
- `/mcp login <name>`: opens the browser (`webbrowser.open`), waits for the callback, lets the SDK exchange and store tokens, reconnects the server with auth attached. `/mcp logout <name>`: deletes the server's tokens and reconnects.
- An http/sse server answering 401 with no stored tokens gets state `needs_auth` ("run /mcp login <name>") instead of an error. With stored tokens, a 401 triggers force-refresh and exactly one retry, distinguishing "refresh failed — login again" from "server rejected a fresh token — login will not help" (Chattler's two-stage pattern).
- The transports receive the provider through their `auth` parameter (httpx auth); stdio servers never see OAuth.

## /tool pinning

`/tool <name>` pins any registry tool for the next message; multiple pins accumulate; `/tool` alone shows pins; `/tool none` clears; unknown names answer with closest substring matches (like `/model`). On the next user message the pinned tools' full contracts (`registry.document(name)`) are appended to the **user message** as a directive block — "the user explicitly asks you to use this tool; its contract follows" — never to the system prompt, which heads the provider prompt-cache prefix and must stay byte-stable. Pins are consumed by one message.

## Error handling

- No MCP failure ever breaks REPL startup, a turn, or the usage footer: connection failures land in `/mcp` state + `logs.log`; tool-call failures return `[mcp error] …` strings.
- Every lifecycle stage (connect, initialize, register, reconnect, oauth steps, shutdown) logs to `logs.log` — "why didn't my server connect" is answered there and by `/mcp`.
- A server vanishing mid-session flips its tools' calls to error strings; the registry entries stay until reconnect replaces them.

## Testing

Unit tests (unittest, no network, no subprocesses, no browser — consistent with the existing suite):

- config: scope merging, `${VAR}`/`${VAR:-default}` expansion, add/remove round-trips in temp dirs, preservation of foreign JSON keys.
- cli: argument parsing for every form (`--`, `--env`, `--header`, `add-json`), remove-scope resolution.
- adapter: name sanitization and hash truncation, argument normalization table, result serialization incl. image → `attach_image` and isError paths.
- registry: source labels, `unregister`, `search` source filter, `only_mcp` through the in-script `search_tools`.
- manager: lifecycle against a fake session factory — connect, register, reconnect re-registration, dead-session auto-retry, shutdown.
- ui: `/mcp` rendering, prompt slash routing, the new `handle` return contract, `/tool` pin/consume, completer entries.
- oauth: `FileTokenStorage` round-trip and permissions, callback request parsing.

Opt-in integration (`RUN_MCP_TESTS=1`): an in-process FastMCP server exercises a real session end-to-end.

## Out of scope

`mcp serve`, `add-from-claude-desktop`, @-mention resource syntax, MCP sampling/elicitation requests from servers, the `only_skills` filter (arrives with the skills feature), and project-scope `.mcp.json` trust prompts (single-user tool).
