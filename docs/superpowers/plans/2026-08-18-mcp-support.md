# MCP Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code-compatible MCP support: `mcp add/add-json/list/get/remove` CLI, `.mcp.json` configs (project + user scopes), persistent stdio/HTTP/SSE sessions, server tools/resources/prompts surfaced through the existing meta-tool registry, OAuth for remote servers, and `/mcp` + `/tool` REPL commands.

**Architecture:** A new `src/llm/tools/mcp/` package (patterned after the voice layer) built directly on the official `mcp` Python SDK. An `McpManager` owns one persistent connection task per server; adapted tools register into the process-wide `ToolRegistry` with a `source` label so the model discovers them via `search_tools` at zero extra context cost. Pure helpers (argument normalization, result serialization, name hashing) are ported from the proven implementation in `/Users/a1/Desktop/Personal/Chattler/backend`.

**Tech Stack:** Python 3.12, `mcp` SDK (new dependency), LangChain `StructuredTool` with raw JSON-schema dicts, pydantic v2, argparse, rich, prompt-toolkit, unittest.

**Spec:** `docs/superpowers/specs/2026-08-18-mcp-support-design.md`

## Global Constraints

- Environment: `uv` manages everything; run tests as `uv run python -m unittest tests.test_x`; a `.env` with `ENV`, `LLM_API_KEY`, `LLM_API_BASE` must exist at the repo root (it does on this machine).
- pytest is NOT installed. unittest only. Every test file prepends the repo root to `sys.path` (copy the header from `tests/test_meta_tools.py`).
- Lint: `bash lint.sh` (black 120 cols, then flake8 over `src/`, `evaluation/`, `tests/`) must pass before the final commit of every task.
- Tools never raise to the model: failures return error strings (`[mcp error] …`).
- Unit tests touch no network, no subprocesses, no browser, no real desktop. Everything OS/network-facing gets an injectable seam.
- No changes to `Prompts.coding_system` / `session_context` — the system prompt must stay byte-stable (prompt-cache prefix).
- Tool names must be valid Python identifiers (scripts call them by bare name) and must not collide with `RESERVED_SCRIPT_NAMES` (the `mcp__` prefix guarantees this).
- Config formats copied from Claude Code exactly: `.mcp.json` `{"mcpServers": {...}}`; user scope at `~/.right-agent/mcp.json`; tokens at `~/.right-agent/mcp-tokens.json`.
- New settings (all with defaults, in `src/config/settings.py`): `mcp_connect_timeout: float = 30.0`, `mcp_tool_timeout: float = 60.0`, `mcp_oauth_port: int = 43110`.
- Commit after every task with the repo's `feat:`/`docs:` style and the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Add the `mcp` dependency and pin down the SDK surface

**Files:**
- Modify: `pyproject.toml` (dependency added via `uv add`)
- Create: `tests/test_mcp_sdk_surface.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the installed `mcp` package; a test asserting every SDK name later tasks import actually exists, so an SDK upgrade that renames something fails loudly here, not deep inside the manager.

- [ ] **Step 1: Add the dependency**

```bash
uv add mcp
```

- [ ] **Step 2: Write the surface test**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestMcpSdkSurface(unittest.TestCase):
    """Every SDK name the mcp layer builds on, in one place.

    If an mcp upgrade moves or renames one of these, this test points at the
    exact break before any runtime code misbehaves.
    """

    def test_client_entry_points_exist(self):
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        from mcp.client.stdio import stdio_client, get_default_environment  # noqa: F401
        from mcp.client.streamable_http import streamablehttp_client  # noqa: F401
        from mcp.client.sse import sse_client  # noqa: F401

    def test_session_methods_exist(self):
        from mcp import ClientSession

        for method in (
            "initialize",
            "list_tools",
            "call_tool",
            "list_prompts",
            "get_prompt",
            "list_resources",
            "read_resource",
        ):
            self.assertTrue(callable(getattr(ClientSession, method)), method)

    def test_oauth_names_exist(self):
        from mcp.client.auth import OAuthClientProvider, TokenStorage  # noqa: F401
        from mcp.shared.auth import (  # noqa: F401
            OAuthClientInformationFull,
            OAuthClientMetadata,
            OAuthToken,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it**

Run: `uv run python -m unittest tests.test_mcp_sdk_surface -v`
Expected: PASS. If any import fails, inspect the installed SDK (`uv run python -c "import mcp, inspect; print(mcp.__file__)"`), find the moved name, and update BOTH this test and the plan notes for the task that uses that name. Do not proceed with guessed imports.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/test_mcp_sdk_surface.py
git commit -m "feat(mcp): add mcp SDK dependency and surface test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Config models, loading, env expansion

**Files:**
- Create: `src/llm/tools/mcp/__init__.py` (empty for now, mirroring `meta/__init__.py`'s no-re-export note)
- Create: `src/llm/tools/mcp/config.py`
- Create: `tests/test_mcp_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by manager, CLI, oauth):

```python
class McpServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    scope: Literal["project", "user"] = "project"

SERVER_NAME_RE: re.Pattern  # ^[A-Za-z0-9_-]+$
def project_config_path(root: Path | None = None) -> Path      # <root or cwd>/.mcp.json
def user_config_path(home: Path | None = None) -> Path         # <home or ~>/.right-agent/mcp.json
def expand_env(value: str, env: Mapping[str, str]) -> str      # ${VAR} and ${VAR:-default}
def load_mcp_servers(project_file: Path | None = None, user_file: Path | None = None,
                     env: Mapping[str, str] | None = None) -> dict[str, McpServerConfig]
```

- [ ] **Step 1: Write the failing tests**

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.config import (
    McpServerConfig,
    expand_env,
    load_mcp_servers,
    project_config_path,
    user_config_path,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestExpandEnv(unittest.TestCase):
    def test_plain_var(self):
        self.assertEqual(expand_env("Bearer ${TOK}", {"TOK": "abc"}), "Bearer abc")

    def test_default_used_when_missing(self):
        self.assertEqual(expand_env("${TOK:-fallback}", {}), "fallback")

    def test_missing_var_without_default_becomes_empty(self):
        self.assertEqual(expand_env("x${TOK}y", {}), "xy")

    def test_no_placeholder_passes_through(self):
        self.assertEqual(expand_env("plain", {"TOK": "abc"}), "plain")


class TestLoadMcpServers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / ".mcp.json"
        self.user = self.root / "user-mcp.json"

    def test_stdio_entry_without_type_reads_as_stdio(self):
        write_json(self.project, {"mcpServers": {"pw": {"command": "npx", "args": ["@playwright/mcp@latest"]}}})
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertEqual(servers["pw"].transport, "stdio")
        self.assertEqual(servers["pw"].command, "npx")
        self.assertEqual(servers["pw"].scope, "project")

    def test_http_entry_with_header_expansion(self):
        write_json(
            self.project,
            {"mcpServers": {"ctx": {"type": "http", "url": "https://x/mcp", "headers": {"X-Key": "${K}"}}}},
        )
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={"K": "sekret"})
        self.assertEqual(servers["ctx"].transport, "http")
        self.assertEqual(servers["ctx"].headers["X-Key"], "sekret")

    def test_project_overrides_user_on_collision(self):
        write_json(self.user, {"mcpServers": {"a": {"type": "http", "url": "https://user/"}}})
        write_json(self.project, {"mcpServers": {"a": {"type": "http", "url": "https://project/"}}})
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertEqual(servers["a"].url, "https://project/")
        self.assertEqual(servers["a"].scope, "project")

    def test_missing_and_broken_files_yield_what_is_readable(self):
        self.project.write_text("{not json", encoding="utf-8")
        write_json(self.user, {"mcpServers": {"u": {"type": "sse", "url": "https://u/sse"}}})
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertEqual(list(servers), ["u"])

    def test_invalid_entry_is_skipped_not_fatal(self):
        write_json(self.project, {"mcpServers": {"bad": {"type": "http"}, "ok": {"command": "x"}}})
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertEqual(list(servers), ["ok"])


class TestPaths(unittest.TestCase):
    def test_default_paths(self):
        self.assertEqual(project_config_path(Path("/repo")), Path("/repo/.mcp.json"))
        self.assertEqual(user_config_path(Path("/home/u")), Path("/home/u/.right-agent/mcp.json"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.tools.mcp'`.

- [ ] **Step 3: Implement `config.py`**

`src/llm/tools/mcp/__init__.py` gets only a comment (`# Deliberately empty: names are imported from concrete modules; see meta/__init__.py for why.`). `config.py`:

```python
"""MCP server configuration: Claude Code-compatible files, two scopes.

Project scope is `.mcp.json` at the repo root, user scope is
`~/.right-agent/mcp.json`; both hold `{"mcpServers": {name: entry}}` exactly
as Claude Code writes them, so files copy between the two tools unchanged.
`${VAR}` / `${VAR:-default}` placeholders expand from the environment at load
time only — the file keeps the placeholder.
"""

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.config.logging import logger

SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class McpServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    scope: Literal["project", "user"] = "project"

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "McpServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio server needs a command")
        if self.transport in ("http", "sse") and not self.url:
            raise ValueError(f"{self.transport} server needs a url")
        return self


def project_config_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / ".mcp.json"


def user_config_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".right-agent" / "mcp.json"


def expand_env(value: str, env: Mapping[str, str]) -> str:
    return _ENV_RE.sub(lambda m: env.get(m.group(1), m.group(2) or ""), value)


def _expand_entry(entry: dict, env: Mapping[str, str]) -> dict:
    def walk(node):
        if isinstance(node, str):
            return expand_env(node, env)
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        return node

    return walk(entry)


def read_raw_entries(file: Path) -> dict[str, dict]:
    """The `mcpServers` mapping of one file; {} when missing or unreadable."""
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("Unreadable MCP config file [{}]", file)
        return {}
    servers = payload.get("mcpServers")
    return servers if isinstance(servers, dict) else {}


def _parse_entry(name: str, entry: dict, scope: str, env: Mapping[str, str]) -> McpServerConfig | None:
    expanded = _expand_entry(dict(entry), env)
    transport = expanded.pop("type", None) or ("stdio" if expanded.get("command") else "http")
    try:
        return McpServerConfig(name=name, transport=transport, scope=scope, **expanded)
    except Exception as error:
        logger.warning("Skipping invalid MCP server entry [{}]: {}", name, error)
        return None


def load_mcp_servers(
    project_file: Path | None = None,
    user_file: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, McpServerConfig]:
    """Merged configs from both scopes; project wins on a name collision."""
    env = os.environ if env is None else env
    servers: dict[str, McpServerConfig] = {}
    for file, scope in (
        (user_file or user_config_path(), "user"),
        (project_file or project_config_path(), "project"),
    ):
        for name, entry in read_raw_entries(file).items():
            if not isinstance(entry, dict):
                continue
            parsed = _parse_entry(name, entry, scope, env)
            if parsed is not None:
                servers[name] = parsed
    return servers
```

Note the entry parser pops `type` and passes the REST as kwargs — unknown keys in an entry would raise; pydantic's default is to error on extras, so add `model_config = ConfigDict(extra="ignore")` on `McpServerConfig` (import `ConfigDict` from pydantic) so foreign keys in someone's Claude Code file don't kill the entry.

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_config -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/tools/mcp/ tests/test_mcp_config.py
git commit -m "feat(mcp): config models, scope merging, env expansion

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Config persistence — add/remove

**Files:**
- Modify: `src/llm/tools/mcp/config.py`
- Modify: `tests/test_mcp_config.py`

**Interfaces:**
- Consumes: `read_raw_entries`, `McpServerConfig`, `SERVER_NAME_RE` from Task 2.
- Produces (used by the CLI):

```python
def add_server(config: McpServerConfig, file: Path) -> None      # writes/creates file, keeps foreign JSON keys
def remove_server(name: str, file: Path) -> bool                 # True when the entry existed
def scopes_containing(name: str, project_file: Path | None = None,
                      user_file: Path | None = None) -> list[str]  # subset of ["project", "user"]
def server_entry_json(config: McpServerConfig) -> dict           # the on-disk entry shape (no name/scope)
```

- [ ] **Step 1: Add failing tests to `tests/test_mcp_config.py`**

```python
from src.llm.tools.mcp.config import add_server, remove_server, scopes_containing, server_entry_json


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / ".mcp.json"

    def test_add_creates_file_and_round_trips(self):
        config = McpServerConfig(name="pw", command="npx", args=["@playwright/mcp@latest"])
        add_server(config, self.file)
        loaded = load_mcp_servers(project_file=self.file, user_file=self.file.with_name("none.json"), env={})
        self.assertEqual(loaded["pw"].command, "npx")

    def test_add_preserves_foreign_json_keys(self):
        write_json(self.file, {"otherTool": {"keep": True}, "mcpServers": {"old": {"command": "x"}}})
        add_server(McpServerConfig(name="new", transport="http", url="https://n/"), self.file)
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(payload["otherTool"], {"keep": True})
        self.assertIn("old", payload["mcpServers"])
        self.assertIn("new", payload["mcpServers"])

    def test_stdio_entry_omits_url_fields(self):
        entry = server_entry_json(McpServerConfig(name="pw", command="npx"))
        self.assertEqual(entry, {"type": "stdio", "command": "npx"})

    def test_http_entry_shape(self):
        entry = server_entry_json(
            McpServerConfig(name="c", transport="http", url="https://c/", headers={"A": "B"})
        )
        self.assertEqual(entry, {"type": "http", "url": "https://c/", "headers": {"A": "B"}})

    def test_remove_true_then_false(self):
        add_server(McpServerConfig(name="pw", command="npx"), self.file)
        self.assertTrue(remove_server("pw", self.file))
        self.assertFalse(remove_server("pw", self.file))

    def test_scopes_containing(self):
        user = Path(self.tmp.name) / "user.json"
        add_server(McpServerConfig(name="both", command="x"), self.file)
        add_server(McpServerConfig(name="both", command="x"), user)
        add_server(McpServerConfig(name="only-user", command="x"), user)
        self.assertEqual(scopes_containing("both", project_file=self.file, user_file=user), ["project", "user"])
        self.assertEqual(scopes_containing("only-user", project_file=self.file, user_file=user), ["user"])
        self.assertEqual(scopes_containing("nope", project_file=self.file, user_file=user), [])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_config -v`
Expected: FAIL — ImportError on `add_server`.

- [ ] **Step 3: Implement in `config.py`**

```python
def server_entry_json(config: McpServerConfig) -> dict:
    """The on-disk entry: always an explicit `type`, empty fields omitted."""
    entry: dict = {"type": config.transport}
    if config.transport == "stdio":
        entry["command"] = config.command
        if config.args:
            entry["args"] = list(config.args)
        if config.env:
            entry["env"] = dict(config.env)
    else:
        entry["url"] = config.url
        if config.headers:
            entry["headers"] = dict(config.headers)
    return entry


def _read_payload(file: Path) -> dict:
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_payload(file: Path, payload: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_server(config: McpServerConfig, file: Path) -> None:
    payload = _read_payload(file)
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[config.name] = server_entry_json(config)
    payload["mcpServers"] = servers
    _write_payload(file, payload)


def remove_server(name: str, file: Path) -> bool:
    payload = _read_payload(file)
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        return False
    del servers[name]
    _write_payload(file, payload)
    return True


def scopes_containing(
    name: str, project_file: Path | None = None, user_file: Path | None = None
) -> list[str]:
    found = []
    if name in read_raw_entries(project_file or project_config_path()):
        found.append("project")
    if name in read_raw_entries(user_file or user_config_path()):
        found.append("user")
    return found
```

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run python -m unittest tests.test_mcp_config -v` → PASS.
Run: `uv run python -m unittest discover -s tests` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/llm/tools/mcp/config.py tests/test_mcp_config.py
git commit -m "feat(mcp): config add/remove persistence with scope resolution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Registry source labels, unregister, and `search_tools(only_mcp=...)`

**Files:**
- Modify: `src/llm/tools/meta/registry.py`
- Modify: `src/llm/tools/meta/tool.py` (library + in-script `search_tools`, `run_tools` docstring)
- Create: `tests/test_mcp_registry.py`

**Interfaces:**
- Consumes: existing `ToolRegistry`.
- Produces (used by manager, resources tools, completer, `/tool`):

```python
MCP_SOURCE_PREFIX = "mcp:"                                   # in registry.py
ToolRegistry.register(tool_obj, source: str | None = None)   # source e.g. "mcp:playwright"
ToolRegistry.unregister(name: str) -> bool
ToolRegistry.source_of(name: str) -> str | None
ToolRegistry.search(query, limit=SEARCH_LIMIT, source_prefix: str | None = None)
ToolRegistry.brief(tool_obj)                                  # appends " [MCP: <server>]" for mcp-sourced tools
ToolRegistry.all_tools(source_prefix: str | None = None)
# meta/tool.py — both the library function and the in-script variant:
async def search_tools(query: str, only_mcp: bool = False) -> str
```

- [ ] **Step 1: Write failing tests**

```python
import asyncio
import sys
import unittest
from pathlib import Path

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry, search_tools, set_registry


@tool(parse_docstring=True)
async def native_probe(x: str) -> str:
    """A native probe tool.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


@tool(parse_docstring=True)
async def mcp__srv__probe(x: str) -> str:
    """A remote probe tool.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


class TestRegistrySources(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(native_probe)
        self.registry.register(mcp__srv__probe, source="mcp:srv")

    def test_source_of(self):
        self.assertIsNone(self.registry.source_of("native_probe"))
        self.assertEqual(self.registry.source_of("mcp__srv__probe"), "mcp:srv")

    def test_brief_marks_mcp_tools(self):
        self.assertIn("[MCP: srv]", self.registry.brief(mcp__srv__probe))
        self.assertNotIn("[MCP", self.registry.brief(native_probe))

    def test_search_source_filter(self):
        names = [t.name for t in self.registry.search("probe", source_prefix="mcp:")]
        self.assertEqual(names, ["mcp__srv__probe"])

    def test_unregister(self):
        self.assertTrue(self.registry.unregister("mcp__srv__probe"))
        self.assertFalse(self.registry.unregister("mcp__srv__probe"))
        self.assertIsNone(self.registry.get("mcp__srv__probe"))
        self.assertIsNone(self.registry.source_of("mcp__srv__probe"))


class TestSearchToolsOnlyMcp(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry()
        registry.register(native_probe)
        registry.register(mcp__srv__probe, source="mcp:srv")
        set_registry(registry)
        self.addCleanup(set_registry, None)

    def test_only_mcp_filters(self):
        listing = asyncio.run(search_tools("probe", only_mcp=True))
        self.assertIn("mcp__srv__probe", listing)
        self.assertNotIn("native_probe", listing)

    def test_default_lists_both_with_marker(self):
        listing = asyncio.run(search_tools("probe"))
        self.assertIn("native_probe", listing)
        self.assertIn("[MCP: srv]", listing)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_registry -v`
Expected: FAIL — `register() got an unexpected keyword argument 'source'`.

- [ ] **Step 3: Implement registry changes**

In `registry.py`: add `MCP_SOURCE_PREFIX = "mcp:"`; `self._sources: dict[str, str] = {}` in `__init__`; extend:

```python
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
```

`search()` gains `source_prefix: str | None = None` and skips non-matching tools inside its loop (same startswith check). `brief()` appends the marker:

```python
    def brief(self, tool_obj: BaseTool) -> str:
        description = " ".join((tool_obj.description or "").split())
        head, separator, _ = description.partition(". ")
        summary = f"{head}." if separator else description
        line = f"{self.signature(tool_obj)} — {summary}"
        source = self._sources.get(tool_obj.name) or ""
        if source.startswith(MCP_SOURCE_PREFIX):
            line += f" [MCP: {source[len(MCP_SOURCE_PREFIX):]}]"
        return line
```

In `meta/tool.py`, thread the flag through `search_tools` (both no-match fallback lists must respect it):

```python
async def search_tools(query: str, only_mcp: bool = False) -> str:
    registry = get_registry()
    source_prefix = "mcp:" if only_mcp else None
    matches = registry.search(query, source_prefix=source_prefix)
    header = f"Tools matching {query!r}:"
    if not matches:
        matches = registry.all_tools(source_prefix=source_prefix)
        header = f"Nothing matched {query!r}; every registered tool:"
    if not matches:
        return "No MCP tools are registered." if only_mcp else "No tools are registered."
    ...  # unchanged tail
```

Update the `run_tools` docstring (the model-facing contract): in the sentence describing `search_tools`, change to `search_tools("a few keywords", only_mcp=False)` and append: `Tools provided by connected MCP servers are listed with an [MCP: <server>] marker; pass only_mcp=True to browse only those.`

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_registry tests.test_meta_tools -v`
Expected: PASS (test_meta_tools guards the existing behavior you just touched).

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run python -m unittest discover -s tests && bash lint.sh
git add src/llm/tools/meta/registry.py src/llm/tools/meta/tool.py tests/test_mcp_registry.py
git commit -m "feat(mcp): registry source labels, unregister, search_tools only_mcp

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Adapter — tool names and argument normalization

**Files:**
- Create: `src/llm/tools/mcp/adapter.py`
- Create: `tests/test_mcp_adapter.py`

**Interfaces:**
- Consumes: nothing runtime; ports pure functions from `/Users/a1/Desktop/Personal/Chattler/backend/common/service/extensions/providers/mcp/runtime.py` (read it before writing).
- Produces:

```python
MAX_TOOL_NAME_LENGTH = 64
def build_tool_name(server: str, remote_tool: str) -> str        # mcp__<server>__<tool>, hash-truncated
def build_prompt_command(server: str, prompt: str) -> str        # /mcp__<server>__<prompt>, same sanitizer
def normalize_tool_arguments(arguments: dict, input_schema: dict | None) -> dict
```

- [ ] **Step 1: Write failing tests**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.adapter import (
    MAX_TOOL_NAME_LENGTH,
    build_prompt_command,
    build_tool_name,
    normalize_tool_arguments,
)


class TestToolNames(unittest.TestCase):
    def test_simple_name(self):
        self.assertEqual(build_tool_name("playwright", "click"), "mcp__playwright__click")

    def test_sanitizes_non_identifier_chars(self):
        name = build_tool_name("my-server", "do.thing")
        self.assertEqual(name, "mcp__my_server__do_thing")
        self.assertTrue(name.isidentifier())

    def test_long_name_truncates_with_stable_hash(self):
        long_tool = "extremely_" * 12
        first = build_tool_name("srv", long_tool)
        second = build_tool_name("srv", long_tool)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), MAX_TOOL_NAME_LENGTH)
        self.assertTrue(first.isidentifier())

    def test_long_names_differing_only_in_tail_stay_distinct(self):
        a = build_tool_name("srv", "x" * 80 + "a")
        b = build_tool_name("srv", "x" * 80 + "b")
        self.assertNotEqual(a, b)

    def test_prompt_command(self):
        self.assertEqual(build_prompt_command("srv", "code-review"), "/mcp__srv__code_review")


class TestNormalizeArguments(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "items": {"type": "array"},
            "config": {"type": "object"},
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    def test_string_booleans_and_numbers_coerce(self):
        result = normalize_tool_arguments(
            {"flag": "true", "count": "3", "ratio": "0.5", "text": "x"}, self.SCHEMA
        )
        self.assertEqual(result, {"flag": True, "count": 3, "ratio": 0.5, "text": "x"})

    def test_json_string_array_and_object_coerce(self):
        result = normalize_tool_arguments({"items": '["a", 1]', "config": '{"k": 2}', "text": "x"}, self.SCHEMA)
        self.assertEqual(result["items"], ["a", 1])
        self.assertEqual(result["config"], {"k": 2})

    def test_comma_string_becomes_array(self):
        result = normalize_tool_arguments({"items": "a, b; c", "text": "x"}, self.SCHEMA)
        self.assertEqual(result["items"], ["a", "b", "c"])

    def test_empty_optional_values_dropped_required_kept(self):
        result = normalize_tool_arguments({"count": "", "text": ""}, self.SCHEMA)
        self.assertNotIn("count", result)
        self.assertIn("text", result)

    def test_no_schema_passes_through(self):
        self.assertEqual(normalize_tool_arguments({"a": "1"}, None), {"a": 1})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_adapter -v`
Expected: FAIL — no module `adapter`.

- [ ] **Step 3: Implement**

Port from Chattler `runtime.py` (keep its logic; rename to our conventions). Name builder differs from Chattler's on purpose — Claude Code's `mcp__server__tool` shape, hash only when needed:

```python
"""MCP tool adaptation: names, argument coercion, result serialization.

The pure pieces are ported from the Chattler backend's proven MCP runtime
(`common/service/extensions/providers/mcp/runtime.py` there): models — the
weak ones especially — routinely send `"true"` for a boolean or a JSON
string for an array, and the coercion table below is what made that safe in
production.
"""

import hashlib
import json
import re

_HASH_LENGTH = 8
MAX_TOOL_NAME_LENGTH = 64
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_part(value: str) -> str:
    normalized = _SAFE_PART_RE.sub("_", str(value or "").strip()).strip("_")
    return normalized or "x"


def _identifier(server: str, tool: str) -> str:
    readable = f"mcp__{_safe_part(server)}__{_safe_part(tool)}"
    if len(readable) <= MAX_TOOL_NAME_LENGTH:
        return readable
    digest = hashlib.sha256(f"{server}\x1f{tool}".encode()).hexdigest()[:_HASH_LENGTH]
    suffix = f"_{digest}"
    return readable[: MAX_TOOL_NAME_LENGTH - len(suffix)].rstrip("_") + suffix


def build_tool_name(server: str, remote_tool: str) -> str:
    return _identifier(server, remote_tool)


def build_prompt_command(server: str, prompt: str) -> str:
    return "/" + _identifier(server, prompt)
```

Then port `_schema_properties`, `_schema_required`, `_schema_type`, `_parse_json_string`, `_normalize_boolean`, `_normalize_array`, `_normalize_object`, `_normalize_scalar`, and `normalize_mcp_tool_arguments` (rename to `normalize_tool_arguments`) verbatim from Chattler's `runtime.py` lines 71–206 — same behavior, our docstring style.

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_adapter -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/tools/mcp/adapter.py tests/test_mcp_adapter.py
git commit -m "feat(mcp): tool name building and argument normalization (ported from Chattler)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Adapter — result serialization and LangChain tool building

**Files:**
- Modify: `src/llm/tools/mcp/adapter.py`
- Modify: `tests/test_mcp_adapter.py`

**Interfaces:**
- Consumes: `attach_image` from `src/llm/tools/meta/attachments.py` (`attach_image(base64_data, mime_type, label) -> bool`); `normalize_tool_arguments`, `build_tool_name` from Task 5.
- Produces (used by the manager):

```python
def serialize_call_result(result: Any, *, server: str, tool_name: str) -> str
# CallToolResult -> text for the model; images -> attach_image side channel; isError -> "[mcp error] ..."
def build_mcp_tool(server: str, remote_tool: Any,
                   call: Callable[[str, dict], Awaitable[Any]]) -> StructuredTool
# remote_tool: SDK Tool (has .name/.description/.inputSchema/.annotations)
# call(remote_tool_name, normalized_args) -> CallToolResult; wrapper never raises
```

- [ ] **Step 1: Add failing tests**

```python
import asyncio
from types import SimpleNamespace

from src.llm.tools.mcp.adapter import build_mcp_tool, serialize_call_result
from src.llm.tools.meta.attachments import collecting_images


def call_result(*content, structured=None, is_error=False):
    return SimpleNamespace(content=list(content), structuredContent=structured, isError=is_error)


def text_item(text):
    return SimpleNamespace(type="text", text=text)


def image_item(data="aGk=", mime="image/png"):
    return SimpleNamespace(type="image", data=data, mimeType=mime)


class TestSerializeCallResult(unittest.TestCase):
    def test_single_text_returns_plain(self):
        out = serialize_call_result(call_result(text_item("hello")), server="s", tool_name="t")
        self.assertEqual(out, "hello")

    def test_error_flag_prefixes(self):
        out = serialize_call_result(call_result(text_item("boom"), is_error=True), server="s", tool_name="t")
        self.assertIn("[mcp error]", out)
        self.assertIn("boom", out)

    def test_structured_content_serialized(self):
        out = serialize_call_result(call_result(structured={"k": 1}), server="s", tool_name="t")
        self.assertIn('"k": 1', out)

    def test_image_goes_to_attachment_channel(self):
        with collecting_images() as images:
            out = serialize_call_result(call_result(image_item()), server="s", tool_name="t")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["mime_type"], "image/png")
        self.assertIn("image attached", out)

    def test_image_without_channel_reports_stub(self):
        out = serialize_call_result(call_result(image_item()), server="s", tool_name="t")
        self.assertIn("image", out.lower())

    def test_resource_link_summarized(self):
        link = SimpleNamespace(
            type="resource_link", name="doc", title=None, uri="res://x", description=None, mimeType=None, size=None
        )
        out = serialize_call_result(call_result(link), server="s", tool_name="t")
        self.assertIn("res://x", out)


class TestBuildMcpTool(unittest.TestCase):
    def remote_tool(self, annotations=None):
        return SimpleNamespace(
            name="click",
            description="Click an element.",
            inputSchema={
                "type": "object",
                "properties": {"selector": {"type": "string"}, "force": {"type": "boolean"}},
                "required": ["selector"],
            },
            annotations=annotations,
        )

    def build(self, annotations=None):
        self.calls = []

        async def call(name, args):
            self.calls.append((name, args))
            return call_result(text_item("ok"))

        return build_mcp_tool("playwright", self.remote_tool(annotations), call)

    def test_name_description_and_args(self):
        tool_obj = self.build()
        self.assertEqual(tool_obj.name, "mcp__playwright__click")
        self.assertIn("Click an element.", tool_obj.description)
        self.assertEqual(list(tool_obj.args), ["selector", "force"])

    def test_invoke_normalizes_and_calls(self):
        tool_obj = self.build()
        out = asyncio.run(tool_obj.ainvoke({"selector": "#a", "force": "true"}))
        self.assertEqual(out, "ok")
        self.assertEqual(self.calls, [("click", {"selector": "#a", "force": True})])

    def test_call_failure_returns_error_string(self):
        async def call(name, args):
            raise RuntimeError("gone")

        tool_obj = build_mcp_tool("playwright", self.remote_tool(), call)
        out = asyncio.run(tool_obj.ainvoke({"selector": "#a"}))
        self.assertIn("[mcp error]", out)
        self.assertIn("gone", out)

    def test_destructive_annotation_marks_description(self):
        tool_obj = self.build(annotations=SimpleNamespace(readOnlyHint=None, destructiveHint=True))
        self.assertIn("[DESTRUCTIVE]", tool_obj.description)

    def test_registry_integration_with_dict_schema(self):
        from src.llm.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(self.build(), source="mcp:playwright")
        signature = registry.signature(registry.get("mcp__playwright__click"))
        self.assertIn("selector", signature)
        self.assertIn("mcp__playwright__click", registry.document("mcp__playwright__click"))
        table = registry.callables()
        out = asyncio.run(table["mcp__playwright__click"]("#a"))
        self.assertEqual(out, "ok")
```

The last test is load-bearing: it proves a dict `args_schema` satisfies `registry.signature/document/callables` including positional-arg mapping. If `StructuredTool` rejects a dict schema in the installed LangChain version, STOP and check `langchain_core.tools.StructuredTool` — the fallback is generating a pydantic model from the schema with `create_model` limited to top-level properties (type map: string→str, integer→int, number→float, boolean→bool, array→list, object→dict, default `Any`); implement that inside `build_mcp_tool` only if the dict path genuinely fails.

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_adapter -v`
Expected: FAIL — ImportError on `serialize_call_result`.

- [ ] **Step 3: Implement**

```python
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import StructuredTool

from src.config.logging import logger

from ..meta.attachments import attach_image


def _read_field(value: Any, field_name: str, default=None):
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _serialize_content_item(item: Any, *, server: str, tool_name: str, parts: list[str]) -> None:
    item_type = _read_field(item, "type")
    if item_type == "text":
        parts.append(str(_read_field(item, "text", "")))
        return
    if item_type == "image":
        data = _read_field(item, "data", "") or ""
        mime = _read_field(item, "mimeType") or "image/png"
        if attach_image(data, mime, label=f"{server}:{tool_name}"):
            parts.append("[image attached — you will see it right after this result]")
        else:
            parts.append(f"[image result ({mime}, {len(data)} base64 chars) — no attachment channel open]")
        return
    if item_type == "audio":
        parts.append(f"[audio result ({_read_field(item, 'mimeType')}) — not supported]")
        return
    if item_type == "resource":
        resource = _read_field(item, "resource")
        summary = {"type": "resource", "uri": str(_read_field(resource, "uri", ""))}
        text = _read_field(resource, "text")
        if text is not None:
            summary["text"] = text
        blob = _read_field(resource, "blob")
        if blob is not None:
            summary["blob_chars"] = len(blob) if isinstance(blob, str) else None
        parts.append(json.dumps(summary, ensure_ascii=False))
        return
    if item_type == "resource_link":
        parts.append(
            json.dumps(
                {
                    "type": "resource_link",
                    "name": _read_field(item, "name"),
                    "uri": str(_read_field(item, "uri", "")),
                    "description": _read_field(item, "description"),
                },
                ensure_ascii=False,
            )
        )
        return
    parts.append(repr(item))


def serialize_call_result(result: Any, *, server: str, tool_name: str) -> str:
    parts: list[str] = []
    for item in _read_field(result, "content", []) or []:
        _serialize_content_item(item, server=server, tool_name=tool_name, parts=parts)
    structured = _read_field(result, "structuredContent")
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, default=repr))
    text = "\n".join(part for part in parts if part) or "(empty result)"
    if _read_field(result, "isError", False):
        return f"[mcp error] {server}: {text}"
    return text


def _annotation_marker(remote_tool: Any) -> str:
    annotations = _read_field(remote_tool, "annotations")
    if annotations is None:
        return ""
    if _read_field(annotations, "destructiveHint"):
        return " [DESTRUCTIVE]"
    if _read_field(annotations, "readOnlyHint"):
        return " [read-only]"
    return ""


def build_mcp_tool(
    server: str,
    remote_tool: Any,
    call: Callable[[str, dict], Awaitable[Any]],
) -> StructuredTool:
    remote_name = str(_read_field(remote_tool, "name") or "")
    tool_name = build_tool_name(server, remote_name)
    input_schema = _read_field(remote_tool, "inputSchema") or {"type": "object", "properties": {}}
    description = (_read_field(remote_tool, "description") or remote_name).strip()
    description = f"{description}{_annotation_marker(remote_tool)} (MCP server: {server})"

    async def run(**kwargs: Any) -> str:
        try:
            arguments = normalize_tool_arguments(kwargs, input_schema)
            result = await call(remote_name, arguments)
            return serialize_call_result(result, server=server, tool_name=remote_name)
        except Exception as error:
            logger.exception("MCP tool failed server [{}] tool [{}]", server, remote_name)
            return f"[mcp error] {server}.{remote_name}: {error}"

    return StructuredTool(
        name=tool_name,
        description=description,
        args_schema=input_schema,
        coroutine=run,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_adapter -v`
Expected: PASS (including the registry-integration test).

- [ ] **Step 5: Lint, commit**

```bash
bash lint.sh
git add src/llm/tools/mcp/adapter.py tests/test_mcp_adapter.py
git commit -m "feat(mcp): result serialization with image attachments and LangChain tool building

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Manager — connection lifecycle with an injectable session factory

**Files:**
- Modify: `src/config/settings.py` (three new fields)
- Create: `src/llm/tools/mcp/manager.py`
- Create: `tests/test_mcp_manager.py`

**Interfaces:**
- Consumes: `McpServerConfig`, `load_mcp_servers` (Task 2); `build_mcp_tool` (Task 6); `ToolRegistry.register/unregister` (Task 4).
- Produces (used by CLI, REPL commands, resources tools, oauth):

```python
class ServerState(str, Enum):
    CONNECTING = "connecting"; CONNECTED = "connected"; FAILED = "failed"
    NEEDS_AUTH = "needs auth"; DISCONNECTED = "disconnected"

@dataclass
class ServerStatus:
    name: str; transport: str; scope: str; state: ServerState
    error: str | None; tool_count: int; prompt_count: int; resource_count: int

class McpManager:
    def __init__(self, configs: dict[str, McpServerConfig] | None = None,
                 session_factory=None,        # Callable[[McpServerConfig, Any | None], AsyncContextManager[session]]
                 registry=None,               # defaults to get_registry() at connect time
                 on_status: Callable[[str, ServerState], None] | None = None) -> None
    async def start(self) -> None             # connect all configured servers concurrently; never raises
    async def stop(self) -> None
    async def reconnect(self, name: str) -> ServerStatus
    async def call_tool(self, server: str, tool: str, arguments: dict) -> Any   # dead session -> one reconnect+retry
    async def list_resources(self, server: str | None = None) -> list[dict]
    async def read_resource(self, server: str, uri: str) -> Any                 # raw SDK result
    async def get_prompt(self, server: str, prompt: str, arguments: dict) -> str  # flattened message text
    def statuses(self) -> list[ServerStatus]
    def prompt_commands(self) -> list[tuple[str, str]]    # ("/mcp__srv__name", "description")
    def find_prompt(self, command: str) -> tuple[str, Any] | None   # (server, SDK Prompt) for "/mcp__srv__name"

def get_mcp_manager() -> McpManager           # process-wide singleton, built from load_mcp_servers()
def set_mcp_manager(manager: McpManager | None) -> None   # test seam, mirrors set_registry/set_computer
```

Settings additions in `src/config/settings.py` (same Field style as neighbors):

```python
    mcp_connect_timeout: float = Field(default=30.0, description="Seconds to wait for an MCP server to connect and initialize.")
    mcp_tool_timeout: float = Field(default=60.0, description="Seconds to wait for a single MCP tool call.")
    mcp_oauth_port: int = Field(default=43110, description="Localhost port for the MCP OAuth redirect callback.")
```

- [ ] **Step 1: Write failing tests**

The fake session factory is the heart of the test file; it needs no SDK types:

```python
import asyncio
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry
from src.llm.tools.mcp.config import McpServerConfig
from src.llm.tools.mcp.manager import McpManager, ServerState


def remote_tool(name="click"):
    return SimpleNamespace(
        name=name,
        description=f"{name} tool.",
        inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        annotations=None,
    )


def text_result(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], structuredContent=None, isError=False)


class FakeSession:
    def __init__(self, tools=(), prompts=(), resources=(), fail_calls=0):
        self.tools, self.prompts, self.resources = list(tools), list(prompts), list(resources)
        self.fail_calls = fail_calls
        self.calls = []

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def list_resources(self):
        return SimpleNamespace(resources=self.resources)

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise ConnectionError("stream closed")
        self.calls.append((name, arguments))
        return text_result("done")

    async def get_prompt(self, name, arguments=None):
        message = SimpleNamespace(role="user", content=SimpleNamespace(type="text", text=f"prompt:{name}"))
        return SimpleNamespace(messages=[message])

    async def read_resource(self, uri):
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, text="resource body", mimeType="text/plain")])


class ManagerHarness:
    """One fake server; factory counts connections and can fail first."""

    def __init__(self, session=None, connect_failures=0):
        self.session = session or FakeSession(tools=[remote_tool()])
        self.connect_failures = connect_failures
        self.connections = 0
        self.registry = ToolRegistry()
        config = McpServerConfig(name="srv", command="fake")
        self.manager = McpManager(
            configs={"srv": config}, session_factory=self.factory, registry=self.registry
        )

    @asynccontextmanager
    async def factory(self, config, auth=None):
        self.connections += 1
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise ConnectionError("refused")
        yield self.session


class TestManagerLifecycle(unittest.TestCase):
    def test_start_registers_tools_and_reports_connected(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            try:
                status = harness.manager.statuses()[0]
                self.assertEqual(status.state, ServerState.CONNECTED)
                self.assertEqual(status.tool_count, 1)
                self.assertIsNotNone(harness.registry.get("mcp__srv__click"))
                self.assertEqual(harness.registry.source_of("mcp__srv__click"), "mcp:srv")
            finally:
                await harness.manager.stop()

        asyncio.run(scenario())

    def test_stop_unregisters(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            await harness.manager.stop()
            self.assertIsNone(harness.registry.get("mcp__srv__click"))
            self.assertEqual(harness.manager.statuses()[0].state, ServerState.DISCONNECTED)

        asyncio.run(scenario())

    def test_connect_failure_is_contained(self):
        harness = ManagerHarness(connect_failures=1)

        async def scenario():
            await harness.manager.start()
            status = harness.manager.statuses()[0]
            self.assertEqual(status.state, ServerState.FAILED)
            self.assertIn("refused", status.error)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_reconnect_after_failure_recovers(self):
        harness = ManagerHarness(connect_failures=1)

        async def scenario():
            await harness.manager.start()
            status = await harness.manager.reconnect("srv")
            self.assertEqual(status.state, ServerState.CONNECTED)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_reconnect_replaces_registrations_without_duplicate_error(self):
        harness = ManagerHarness()

        async def scenario():
            await harness.manager.start()
            await harness.manager.reconnect("srv")
            self.assertIsNotNone(harness.registry.get("mcp__srv__click"))
            self.assertEqual(harness.connections, 2)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_dead_session_call_reconnects_once_and_retries(self):
        harness = ManagerHarness(session=FakeSession(tools=[remote_tool()], fail_calls=1))

        async def scenario():
            await harness.manager.start()
            result = await harness.manager.call_tool("srv", "click", {"x": "1"})
            self.assertEqual(harness.session.calls, [("click", {"x": "1"})])
            self.assertEqual(harness.connections, 2)
            self.assertFalse(result.isError)
            await harness.manager.stop()

        asyncio.run(scenario())

    def test_get_prompt_flattens_text(self):
        session = FakeSession(tools=[], prompts=[SimpleNamespace(name="review", description="d", arguments=[])])
        harness = ManagerHarness(session=session)

        async def scenario():
            await harness.manager.start()
            self.assertEqual(harness.manager.prompt_commands(), [("/mcp__srv__review", "d")])
            server, prompt = harness.manager.find_prompt("/mcp__srv__review")
            self.assertEqual(server, "srv")
            text = await harness.manager.get_prompt("srv", "review", {})
            self.assertEqual(text, "prompt:review")
            await harness.manager.stop()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_manager -v`
Expected: FAIL — no module `manager`.

- [ ] **Step 3: Implement `manager.py`**

Core shape (the connection task owns the context manager — anyio contexts must enter and exit in one task):

```python
"""Persistent MCP connections: one background task per configured server.

The task enters the transport + session context, initializes, registers the
adapted tools into the shared ToolRegistry, then parks on a stop event; the
context managers unwind in the same task that entered them (anyio requires
it). A stdio server's subprocess therefore lives for the whole REPL session.
Every failure is contained: statuses record it, callers get error strings,
and nothing here ever breaks startup or a turn.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from src.config.logging import logger
from src.config.settings import settings

from ..meta.defaults import get_registry
from .adapter import build_mcp_tool, build_prompt_command
from .config import McpServerConfig, load_mcp_servers


class ServerState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    NEEDS_AUTH = "needs auth"
    DISCONNECTED = "disconnected"


@dataclass
class ServerStatus:
    name: str
    transport: str
    scope: str
    state: ServerState
    error: str | None = None
    tool_count: int = 0
    prompt_count: int = 0
    resource_count: int = 0


@dataclass
class _Connection:
    config: McpServerConfig
    state: ServerState = ServerState.DISCONNECTED
    error: str | None = None
    session: object | None = None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None
    ready: asyncio.Event | None = None
    registered: list[str] = field(default_factory=list)
    tools: list = field(default_factory=list)
    prompts: list = field(default_factory=list)
    resources: list = field(default_factory=list)
```

Key methods (write them fully; the logic that matters):

- `__init__` stores `configs or load_mcp_servers()`, `session_factory or default_session_factory` (Task 8 provides the real one; until then default to a factory that raises `RuntimeError("no session factory")` — tests always inject), `registry` (may be None → resolved via `get_registry()` at use), `on_status`, and builds `self._connections = {name: _Connection(config)}`.
- `_set_state(conn, state, error=None)` updates fields, calls `on_status(conn.config.name, state)` in a try/except, logs.
- `async def _run_connection(self, conn)` — the owning task:

```python
    async def _run_connection(self, conn: _Connection) -> None:
        conn.stop_event = asyncio.Event()
        try:
            async with self._session_factory(conn.config, self._auth_for(conn.config)) as session:
                await asyncio.wait_for(session.initialize(), timeout=settings.mcp_connect_timeout)
                conn.session = session
                await self._load_inventory(conn, session)
                self._register_tools(conn)
                self._set_state(conn, ServerState.CONNECTED)
                conn.ready.set()
                await conn.stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._set_state(conn, self._failure_state(conn.config, error), str(error))
        finally:
            self._unregister_tools(conn)
            conn.session = None
            conn.ready.set()
            if conn.state == ServerState.CONNECTED:
                self._set_state(conn, ServerState.DISCONNECTED)
```

  `_auth_for` returns None in this task (oauth wires in at Task 14). `_failure_state` returns `ServerState.FAILED` here; Task 14 teaches it `NEEDS_AUTH`. `_load_inventory` fills `conn.tools/prompts/resources` from `list_tools()` and, inside `try/except Exception` each (servers legitimately lack the capability and answer with an error), `list_prompts()` / `list_resources()`.
- `_register_tools(conn)`: for each remote tool build `build_mcp_tool(name, tool, call=lambda t, a, _n=conn.config.name: self.call_tool(_n, t, a, _raw=True))` — actually pass a bound coroutine `functools.partial(self._call_via_session, conn.config.name)`; unregister any leftover name first (`registry.unregister(tool_name)`) so reconnect never hits the duplicate-name ValueError; record names in `conn.registered`.
- `_connect(conn)`: create `conn.ready = asyncio.Event()`, set CONNECTING, spawn `conn.task = asyncio.create_task(self._run_connection(conn))`, `await conn.ready.wait()`.
- `start()`: `await asyncio.gather(*(self._connect(c) for c in self._connections.values()), return_exceptions=True)` — plus guard: no configs → return quietly.
- `_disconnect(conn)`: set stop_event, await task with `asyncio.wait_for(..., 5)`, cancel on timeout.
- `stop()`: disconnect all.
- `reconnect(name)`: `_disconnect` then `_connect`, return the status.
- `call_tool(server, tool, arguments)`:

```python
    async def call_tool(self, server: str, tool: str, arguments: dict):
        conn = self._require(server)
        try:
            return await self._call_via_session(server, tool, arguments)
        except Exception as first_error:
            logger.warning("MCP call failed, reconnecting once server [{}] tool [{}]: {}", server, tool, first_error)
            await self.reconnect(server)
            if conn.state != ServerState.CONNECTED:
                raise ConnectionError(f"server '{server}' is {conn.state.value}: {conn.error}") from first_error
            return await self._call_via_session(server, tool, arguments)
```

  `_call_via_session` raises `ConnectionError` when `conn.session is None`, else `await asyncio.wait_for(session.call_tool(tool, arguments, read_timeout_seconds=timedelta(seconds=settings.mcp_tool_timeout)), timeout=settings.mcp_tool_timeout + 5)`. NOTE: pass `read_timeout_seconds` as a `datetime.timedelta` — that is the SDK's parameter type; if the surface test (Task 1) revealed a different signature, adapt here only.
  IMPORTANT — the tool wrapper built in `_register_tools` must reach `call_tool` (the reconnect-retry wrapper), NOT `_call_via_session`, so a dead session inside a script call heals itself.
- `get_prompt(server, prompt, arguments)`: call session `get_prompt`, flatten: for each message take `content.text` when present (else `str(content)`), join with `"\n\n"`.
- `list_resources(server=None)`: for the named (or every connected) connection return dicts `{"server", "uri", "name", "description", "mime_type"}` from cached `conn.resources`; `read_resource(server, uri)` goes to the live session.
- `statuses()`, `prompt_commands()` (uses `build_prompt_command(conn.config.name, prompt.name)` + first sentence of prompt description), `find_prompt(command)` (rebuild each candidate's command string and compare — the sanitizer is not reversible).
- Singleton tail, mirroring `defaults.py`:

```python
_manager: McpManager | None = None


def get_mcp_manager() -> McpManager:
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager


def set_mcp_manager(manager: McpManager | None) -> None:
    global _manager
    _manager = manager
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_manager -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run python -m unittest discover -s tests && bash lint.sh
git add src/config/settings.py src/llm/tools/mcp/manager.py tests/test_mcp_manager.py
git commit -m "feat(mcp): persistent connection manager with registry integration

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Real session factory — stdio/http/sse transports

**Files:**
- Create: `src/llm/tools/mcp/transports.py`
- Modify: `src/llm/tools/mcp/manager.py` (default factory import)
- Modify: `tests/test_mcp_manager.py`

**Interfaces:**
- Consumes: `McpServerConfig`; SDK names verified in Task 1.
- Produces:

```python
@asynccontextmanager
async def default_session_factory(config: McpServerConfig, auth: Any | None = None)
# yields an initialized-capable ClientSession for any of the three transports
```

- [ ] **Step 1: Write failing tests (transport selection only — no I/O)**

Patch the SDK entry points and record what the factory passes them:

```python
from unittest.mock import patch

from src.llm.tools.mcp import transports


class _FakeStreams:
    def __init__(self, count):
        self.count = count

    async def __aenter__(self):
        return tuple(object() for _ in range(self.count))

    async def __aexit__(self, *exc):
        return False


class TestDefaultSessionFactory(unittest.TestCase):
    def run_factory(self, config, patches):
        async def scenario():
            async with transports.default_session_factory(config) as session:
                self.assertIsNotNone(session)

        with patch.object(transports, "ClientSession") as session_cls:
            session_cls.return_value.__aenter__ = lambda s: asyncio.sleep(0, result=object())
            session_cls.return_value.__aexit__ = lambda s, *e: asyncio.sleep(0, result=False)
            with patches:
                asyncio.run(scenario())

    def test_stdio_builds_server_params(self):
        config = McpServerConfig(name="pw", command="npx", args=["-y", "x"], env={"A": "1"})
        with patch.object(transports, "stdio_client", return_value=_FakeStreams(2)) as client:
            self.run_factory(config, patch("builtins.id", side_effect=id))  # no-op ctx
        params = client.call_args.args[0]
        self.assertEqual(params.command, "npx")
        self.assertEqual(params.args, ["-y", "x"])
        self.assertEqual(params.env.get("A"), "1")

    def test_http_passes_url_headers_auth(self):
        config = McpServerConfig(name="c", transport="http", url="https://c/mcp", headers={"K": "V"})
        with patch.object(transports, "streamablehttp_client", return_value=_FakeStreams(3)) as client:
            self.run_factory(config, patch("builtins.id", side_effect=id))
        self.assertEqual(client.call_args.args[0], "https://c/mcp")
        self.assertEqual(client.call_args.kwargs["headers"], {"K": "V"})

    def test_sse_selected_for_sse_transport(self):
        config = McpServerConfig(name="l", transport="sse", url="https://l/sse")
        with patch.object(transports, "sse_client", return_value=_FakeStreams(2)) as client:
            self.run_factory(config, patch("builtins.id", side_effect=id))
        self.assertEqual(client.call_args.args[0], "https://l/sse")
```

(The `patch("builtins.id", ...)` no-op is just a stand-in context manager for `patches`; simplify to `contextlib.nullcontext()` — import it — when writing the real file.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_manager -v`
Expected: FAIL — no module `transports`.

- [ ] **Step 3: Implement `transports.py`**

```python
"""The one module that touches SDK transport entry points directly."""

from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import McpServerConfig


@asynccontextmanager
async def default_session_factory(config: McpServerConfig, auth: Any | None = None):
    if config.transport == "stdio":
        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env={**get_default_environment(), **config.env},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
        return
    if config.transport == "http":
        async with streamablehttp_client(config.url, headers=config.headers or None, auth=auth) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                yield session
        return
    if config.transport == "sse":
        async with sse_client(config.url, headers=config.headers or None, auth=auth) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
        return
    raise ValueError(f"Unsupported MCP transport: {config.transport}")
```

In `manager.py`, replace the raising placeholder default: `from .transports import default_session_factory` and use it when `session_factory is None`. If Task 1's surface test showed `sse_client` lacks an `auth` parameter in the installed version, drop `auth=` for SSE here and note it in the docstring (SSE is legacy; header auth still works).

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_manager -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke test (real server, optional but do it once)**

```bash
uv run python - <<'EOF'
import asyncio
from src.llm.tools import ToolRegistry
from src.llm.tools.mcp.config import McpServerConfig
from src.llm.tools.mcp.manager import McpManager

async def main():
    registry = ToolRegistry()
    manager = McpManager(
        configs={"everything": McpServerConfig(name="everything", command="npx", args=["-y", "@modelcontextprotocol/server-everything"])},
        registry=registry,
    )
    await manager.start()
    print([s for s in manager.statuses()])
    print([t.name for t in registry.all_tools()][:5])
    result = await manager.call_tool("everything", "echo", {"message": "hi"})
    print(result)
    await manager.stop()

asyncio.run(main())
EOF
```

Expected: CONNECTED status, `mcp__everything__echo` among tools, echo result. Needs node/npx; skip if offline and note it in the commit message.

- [ ] **Step 6: Lint, commit**

```bash
bash lint.sh
git add src/llm/tools/mcp/transports.py src/llm/tools/mcp/manager.py tests/test_mcp_manager.py
git commit -m "feat(mcp): stdio/http/sse session factory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Resources tools in the registry

**Files:**
- Create: `src/llm/tools/mcp/tool.py`
- Modify: `src/llm/tools/meta/defaults.py`
- Modify: `src/llm/tools/__init__.py` (export the new names alongside the existing ones)
- Create: `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: `get_mcp_manager()` (Task 7); `attach_image` (existing).
- Produces:

```python
# src/llm/tools/mcp/tool.py
MCP_SERVICE_TOOLS: list  # [mcp_list_resources, mcp_read_resource]
@tool(parse_docstring=True) async def mcp_list_resources(server: str = "") -> str
@tool(parse_docstring=True) async def mcp_read_resource(server: str, uri: str) -> str
# meta/defaults.py: default_tools() appends MCP_SERVICE_TOOLS when load_mcp_servers() is non-empty
```

- [ ] **Step 1: Write failing tests**

```python
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp import tool as mcp_tool
from src.llm.tools.mcp.manager import set_mcp_manager


class FakeManager:
    def __init__(self):
        self.read_requests = []

    async def list_resources(self, server=None):
        return [{"server": "srv", "uri": "res://a", "name": "A", "description": "", "mime_type": "text/plain"}]

    async def read_resource(self, server, uri):
        self.read_requests.append((server, uri))
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, text="the body", mimeType="text/plain")])


class TestResourceTools(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)

    def test_list_resources(self):
        out = asyncio.run(mcp_tool.mcp_list_resources.ainvoke({}))
        self.assertIn("res://a", out)

    def test_read_resource_text(self):
        out = asyncio.run(mcp_tool.mcp_read_resource.ainvoke({"server": "srv", "uri": "res://a"}))
        self.assertIn("the body", out)
        self.assertEqual(self.manager.read_requests, [("srv", "res://a")])

    def test_errors_return_strings(self):
        async def boom(server, uri):
            raise RuntimeError("nope")

        self.manager.read_resource = boom
        out = asyncio.run(mcp_tool.mcp_read_resource.ainvoke({"server": "srv", "uri": "res://a"}))
        self.assertIn("[mcp error]", out)


class TestDefaultRegistryGating(unittest.TestCase):
    def test_service_tools_registered_only_with_servers_configured(self):
        from unittest.mock import patch

        from src.llm.tools.meta import defaults

        with patch.object(defaults, "_mcp_servers_configured", return_value=False):
            names = {t.name for t in defaults.default_tools()}
            self.assertNotIn("mcp_list_resources", names)
        with patch.object(defaults, "_mcp_servers_configured", return_value=True):
            names = {t.name for t in defaults.default_tools()}
            self.assertIn("mcp_list_resources", names)
            self.assertIn("mcp_read_resource", names)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_tools -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`src/llm/tools/mcp/tool.py` — docstrings are the model-facing contract; follow the `parser/tool.py` voice. Both tools import the manager lazily (`from .manager import get_mcp_manager` inside the function) and wrap everything in try/except returning `[mcp error] …`. `mcp_read_resource` serializes contents: text items appended as text; blob items with an image mime go through `attach_image`; other blobs report `[binary resource (mime, N chars)]`.

In `meta/defaults.py`:

```python
def _mcp_servers_configured() -> bool:
    from ..mcp.config import load_mcp_servers

    try:
        return bool(load_mcp_servers())
    except Exception:
        return False


def default_tools() -> list:
    """Every default tool, minus the ones whose heavy models are disabled.

    (Existing docstring stays; only the MCP tail below is new.)
    """
    # Imported here so `import src.llm.tools` keeps working without a .env.
    from src.config.settings import settings

    tools = [web_fetch, web_search, *FILE_TOOLS, bash, *COMPUTER_TOOLS]
    if not settings.enable_vision_model:
        vision_names = {tool.name for tool in VISION_TOOLS}
        tools = [tool for tool in tools if tool.name not in vision_names]
    if _mcp_servers_configured():
        from ..mcp.tool import MCP_SERVICE_TOOLS

        tools = [*tools, *MCP_SERVICE_TOOLS]
    return tools
```

In `src/llm/tools/__init__.py` add `mcp_list_resources`, `mcp_read_resource`, `get_mcp_manager`, `set_mcp_manager` to the re-exports (check the existing file for its style first).

- [ ] **Step 4: Run tests, full suite, lint, commit**

```bash
uv run python -m unittest tests.test_mcp_tools -v
uv run python -m unittest discover -s tests && bash lint.sh
git add src/llm/tools/mcp/tool.py src/llm/tools/meta/defaults.py src/llm/tools/__init__.py tests/test_mcp_tools.py
git commit -m "feat(mcp): resource tools in the default registry

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: CLI subcommands

**Files:**
- Create: `src/llm/tools/mcp/cli.py`
- Modify: `src/main.py` (dispatch at the bottom)
- Create: `tests/test_mcp_cli.py`

**Interfaces:**
- Consumes: config API (Tasks 2–3); `McpManager` with real factory (Task 8) for `list`.
- Produces:

```python
def run_mcp_cli(argv: list[str]) -> int          # argv AFTER the "mcp" token; returns exit code
def build_parser() -> argparse.ArgumentParser
def parse_add(args) -> McpServerConfig           # url/transport inference, --env/--header parsing
```

`src/main.py` gains, replacing the current `if __name__ == "__main__":` block:

```python
def cli_main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from src.llm.tools.mcp.cli import run_mcp_cli

        raise SystemExit(run_mcp_cli(sys.argv[2:]))
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 1: Write failing tests**

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp import cli
from src.llm.tools.mcp.config import load_mcp_servers


class TestParseAdd(unittest.TestCase):
    def parse(self, *argv):
        return cli.parse_add(cli.build_parser().parse_args(["add", *argv]))

    def test_stdio_with_double_dash(self):
        config = self.parse("pw", "--", "npx", "@playwright/mcp@latest", "--isolated")
        self.assertEqual(config.transport, "stdio")
        self.assertEqual(config.command, "npx")
        self.assertEqual(config.args, ["@playwright/mcp@latest", "--isolated"])

    def test_url_implies_http(self):
        config = self.parse("ctx", "https://mcp.context7.com/mcp")
        self.assertEqual(config.transport, "http")
        self.assertEqual(config.url, "https://mcp.context7.com/mcp")

    def test_explicit_sse_with_header(self):
        config = self.parse("--transport", "sse", "legacy", "https://old/sse", "--header", "X-Key: abc")
        self.assertEqual(config.transport, "sse")
        self.assertEqual(config.headers, {"X-Key": "abc"})

    def test_env_flags(self):
        config = self.parse("srv", "--env", "A=1", "--env", "B=2", "--", "cmd")
        self.assertEqual(config.env, {"A": "1", "B": "2"})

    def test_bad_name_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("bad name!", "--", "cmd")


class TestCliCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.project = root / ".mcp.json"
        self.user = root / "user.json"
        self.patch = unittest.mock.patch.multiple(
            cli, _project_file=lambda: self.project, _user_file=lambda: self.user
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_add_then_get_then_remove(self):
        self.assertEqual(cli.run_mcp_cli(["add", "pw", "--", "npx", "x"]), 0)
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertIn("pw", servers)
        self.assertEqual(cli.run_mcp_cli(["get", "pw"]), 0)
        self.assertEqual(cli.run_mcp_cli(["remove", "pw"]), 0)
        self.assertEqual(load_mcp_servers(project_file=self.project, user_file=self.user, env={}), {})

    def test_add_user_scope(self):
        cli.run_mcp_cli(["add", "--scope", "user", "g", "https://g/mcp"])
        self.assertIn("g", json.loads(self.user.read_text())["mcpServers"])

    def test_add_json(self):
        cli.run_mcp_cli(["add-json", "pw", '{"command": "npx", "args": ["x"]}'])
        servers = load_mcp_servers(project_file=self.project, user_file=self.user, env={})
        self.assertEqual(servers["pw"].command, "npx")

    def test_remove_ambiguous_scope_fails(self):
        cli.run_mcp_cli(["add", "b", "--", "cmd"])
        cli.run_mcp_cli(["add", "--scope", "user", "b", "--", "cmd"])
        self.assertEqual(cli.run_mcp_cli(["remove", "b"]), 1)
        self.assertEqual(cli.run_mcp_cli(["remove", "b", "--scope", "user"]), 0)

    def test_remove_unknown_fails(self):
        self.assertEqual(cli.run_mcp_cli(["remove", "ghost"]), 1)
```

Add `import unittest.mock` at the top (or `from unittest import mock`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_cli -v`
Expected: FAIL.

- [ ] **Step 3: Implement `cli.py`**

Structure:

```python
"""`python -m src.main mcp ...` — Claude Code-compatible MCP management."""

import argparse
import asyncio
import json
import sys

from rich.console import Console

from .config import (
    McpServerConfig,
    SERVER_NAME_RE,
    add_server,
    load_mcp_servers,
    project_config_path,
    remove_server,
    scopes_containing,
    server_entry_json,
    user_config_path,
)


def _project_file():
    return project_config_path()


def _user_file():
    return user_config_path()
```

(`_project_file`/`_user_file` exist so tests patch them in one place.)

- `build_parser()`: subparsers `add`, `add-json`, `list`, `get`, `remove`. `add` takes `name`, `target` (command-or-url), `args` (`nargs=argparse.REMAINDER` — argparse hands everything after `--` through), `--transport`, `--scope` (default `project`), `--env` (`action="append"`), `--header` (`action="append"`).
- `parse_add(args)`: validate `SERVER_NAME_RE.match(args.name)` (parser.error → SystemExit on failure); transport inference: explicit `--transport` wins, else `http` when target starts with `http://`/`https://`, else `stdio`; `--env A=1` split on first `=`; `--header "K: V"` split on first `:`, both sides stripped; build `McpServerConfig`.
- `run_mcp_cli(argv)`: parse, dispatch to `_cmd_add/_cmd_add_json/_cmd_list/_cmd_get/_cmd_remove`, each printing via `Console()` and returning 0/1. `_cmd_add`/`_cmd_add_json` route scope → file via `_project_file()/_user_file()`. `_cmd_remove`: `scopes = scopes_containing(name, _project_file(), _user_file())`; empty → print error, 1; two without `--scope` → print "in both scopes, pass --scope", 1; else remove from the right file.
- `_cmd_list`: load servers; when any exist, build an `McpManager` with those configs and a private registry (`ToolRegistry()` — do NOT touch the process default), `asyncio.run` a start/statuses/stop scenario, print one line per server: `✓ name  transport  scope  N tools` / `✗ name — error` / `needs auth`. Import `McpManager` lazily inside `_cmd_list` so `mcp add` works even if SDK import breaks.
- `_cmd_get`: print the entry (`server_entry_json`) plus scope.
- `add-json` parses the JSON entry, feeds it through the same `_parse_entry` shape as config loading: reuse `McpServerConfig(name=..., transport=entry.pop("type", ...), **entry)` — factor a tiny helper `config_from_entry(name, entry, scope)` into `config.py` if duplication appears.

Then update `src/main.py` exactly as in Interfaces.

- [ ] **Step 4: Run tests + manual dispatch check**

Run: `uv run python -m unittest tests.test_mcp_cli -v` → PASS.
Run: `cd /tmp && uv run --project /Users/a1/Desktop/Personal/ismail python -m src.main mcp list` → prints "no MCP servers configured" (or the project's servers) and exits without starting the REPL. (cd back after.)

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run python -m unittest discover -s tests && bash lint.sh
git add src/llm/tools/mcp/cli.py src/main.py tests/test_mcp_cli.py
git commit -m "feat(mcp): claude-style mcp CLI subcommands

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: `/mcp` REPL command, command-outcome contract, main-loop wiring

**Files:**
- Modify: `src/ui/commands.py`
- Modify: `src/ui/chat.py` (only the `handle_command` docstring if needed — the method already just delegates)
- Modify: `src/main.py`
- Create: `tests/test_mcp_commands.py`
- Modify: `tests/test_main_turns.py` or `tests/test_main_startup.py` only if their assertions break (keep green, don't rewrite)

**Interfaces:**
- Consumes: `get_mcp_manager` / `set_mcp_manager`, `ServerStatus`, `ServerState` (Task 7).
- Produces:

```python
# commands.py
@dataclass(frozen=True)
class McpAction:
    kind: str        # "reconnect" | "login" | "logout" | "prompt"
    argument: str    # server name, or the full "/mcp__srv__prompt arg..." text

CommandHandler.handle(text) -> str | McpAction | None    # "clear" stays a plain string

async def run_mcp_action(action: McpAction, manager, console) -> str | None
# executes the action; returns text to send as a user message (prompt kind), else None
```

Main-loop change in `src/main.py` (inside the command branch):

```python
            if user_content.startswith("/"):
                result = ui.handle_command(user_content)
                if result == "clear":
                    messages = []
                model = ui.model
                if isinstance(result, McpAction):
                    prompt_text = await run_mcp_action(result, mcp_manager, ui.console)
                    if prompt_text is None:
                        continue
                    user_content = prompt_text  # fall through into the turn below
                else:
                    continue
```

and startup/shutdown wiring in `main()` (after the vision preload thread):

```python
    from src.llm.tools.mcp.manager import get_mcp_manager

    mcp_manager = get_mcp_manager()
    mcp_manager.on_status = lambda name, state: ui.set_model_status(f"mcp:{name}", _status_word(state))
    mcp_task = asyncio.create_task(mcp_manager.start())
```

`_status_word` maps ServerState → the set_model_status vocabulary ("loading" for CONNECTING, "ready" for CONNECTED, "failed" for FAILED/NEEDS_AUTH — check `ChatUI.set_model_status` for the exact accepted states first). In the outer `finally`: `mcp_task.cancel()` then `await mcp_manager.stop()` wrapped in try/except.

- [ ] **Step 1: Write failing tests**

```python
import asyncio
import io
import sys
import unittest
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.manager import ServerState, ServerStatus, set_mcp_manager
from src.ui.commands import CommandHandler, McpAction, run_mcp_action


class FakeManager:
    def __init__(self):
        self.reconnected = []

    def statuses(self):
        return [
            ServerStatus(name="pw", transport="stdio", scope="project", state=ServerState.CONNECTED, tool_count=3),
            ServerStatus(name="ctx", transport="http", scope="user", state=ServerState.FAILED, error="401"),
        ]

    def prompt_commands(self):
        return [("/mcp__pw__review", "Review code")]

    def find_prompt(self, command):
        if command == "/mcp__pw__review":
            return ("pw", type("P", (), {"name": "review", "arguments": []})())
        return None

    async def reconnect(self, name):
        self.reconnected.append(name)
        return self.statuses()[0]

    async def get_prompt(self, server, prompt, arguments):
        return f"PROMPT {server}/{prompt} {arguments}"


class FakeUI:
    def __init__(self):
        self.console = Console(file=io.StringIO(), width=100)


class TestMcpCommand(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)
        self.ui = FakeUI()
        self.handler = CommandHandler(self.ui)

    def output(self):
        return self.ui.console.file.getvalue()

    def test_mcp_renders_status_table(self):
        self.assertIsNone(self.handler.handle("/mcp"))
        out = self.output()
        self.assertIn("pw", out)
        self.assertIn("connected", out)
        self.assertIn("401", out)

    def test_mcp_reconnect_returns_action(self):
        result = self.handler.handle("/mcp reconnect pw")
        self.assertEqual(result, McpAction("reconnect", "pw"))

    def test_prompt_command_returns_action(self):
        result = self.handler.handle("/mcp__pw__review please")
        self.assertEqual(result, McpAction("prompt", "/mcp__pw__review please"))

    def test_unknown_prompt_command_errors(self):
        self.assertIsNone(self.handler.handle("/mcp__nope__x"))
        self.assertIn("unknown", self.output().lower())


class TestRunMcpAction(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.console = Console(file=io.StringIO(), width=100)

    def test_reconnect_awaits_manager(self):
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertEqual(self.manager.reconnected, ["pw"])

    def test_prompt_fetches_text(self):
        action = McpAction("prompt", "/mcp__pw__review please fix")
        out = asyncio.run(run_mcp_action(action, self.manager, self.console))
        self.assertIn("PROMPT pw/review", out)

    def test_failures_print_not_raise(self):
        async def boom(name):
            raise RuntimeError("down")

        self.manager.reconnect = boom
        out = asyncio.run(run_mcp_action(McpAction("reconnect", "pw"), self.manager, self.console))
        self.assertIsNone(out)
        self.assertIn("down", self.console.file.getvalue())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_commands -v`
Expected: FAIL — ImportError on `McpAction`.

- [ ] **Step 3: Implement in `commands.py`**

- Add the `McpAction` dataclass and `run_mcp_action` (async, module level). `run_mcp_action` dispatch: `reconnect` → `await manager.reconnect(argument)` + print resulting state; `prompt` → split `argument` into command + rest, `manager.find_prompt(command)`, map whitespace-split words positionally onto `prompt.arguments` (each SDK PromptArgument has `.name`; extra words join into the last argument), `await manager.get_prompt(server, prompt.name, mapping)`, return the text; `login`/`logout` → print "not supported yet" until Task 14 replaces the branch. Wrap the whole body in try/except printing the error (style `"error"`), returning None.
- In `handle()`: before the `unknown command` fallback add:

```python
        if command == "/mcp":
            return self._mcp(argument)
        if command.startswith("/mcp__"):
            return self._mcp_prompt(stripped)
        if command == "/tool":
            return self._tool_pin(argument)   # Task 13 implements; stub prints "coming soon" for now
```

- `_mcp(argument)`: no argument → render the table from `get_mcp_manager().statuses()` (import lazily inside the method) using plain `console.print` lines in the `/models` style: state glyph `●`/`○`/`✗`, name, transport, scope, `N tools`, error tail, then the prompt-command list from `prompt_commands()`, then a hint line (`/mcp reconnect <name>, /mcp login <name>; manage servers with: uv run python -m src.main mcp add ...`). `reconnect <name>`/`login <name>`/`logout <name>` → validate the name exists in statuses, return `McpAction(kind, name)`; unknown subcommand → error print.
- `_mcp_prompt(text)`: command word = first whitespace-token; `find_prompt` → `McpAction("prompt", text)`; None → print `unknown MCP prompt command` + the available list.
- `_print_help()` gains `("/mcp", "MCP servers: status, reconnect <name>, login/logout <name>")` and `("/tool <name>", "pin a tool for the next message (its contract goes along)")`.
- Wire `src/main.py` exactly as shown in Interfaces (import `McpAction`, `run_mcp_action` from `src.ui.commands`). Check `ChatUI.set_model_status` (`src/ui/chat.py:78`) for accepted `state` strings before writing `_status_word`.

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_commands tests.test_model_commands tests.test_main_turns tests.test_main_startup -v`
Expected: PASS — the pre-existing files prove the contract change broke nothing.

- [ ] **Step 5: Full suite, lint, manual REPL check, commit**

Manual: `uv run python -m src.main`, type `/mcp` (expect "no MCP servers configured" or the table), `/help` (expect the two new rows), `/quit`.

```bash
uv run python -m unittest discover -s tests && bash lint.sh
git add src/ui/commands.py src/main.py tests/test_mcp_commands.py
git commit -m "feat(mcp): /mcp command, McpAction outcome contract, REPL manager wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Prompt slash-commands in the completer

**Files:**
- Modify: `src/ui/completer.py`
- Modify: `tests/test_mcp_commands.py`

**Interfaces:**
- Consumes: `get_mcp_manager().prompt_commands()` (Task 7), `McpAction` routing (Task 11).
- Produces: completions for `/mcp` subcommands, `/mcp__…` prompt commands with descriptions, and `/mcp reconnect|login|logout <name>` server names.

- [ ] **Step 1: Add failing tests**

```python
from prompt_toolkit.document import Document

from src.ui.completer import CommandCompleter


class TestCompleterMcp(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)

    def completions(self, text):
        ui = FakeUI()
        ui.available_models = []
        ui.model_catalog = {}
        completer = CommandCompleter(ui)
        return [c.text for c in completer.get_completions(Document(text, len(text)), None)]

    def test_prompt_commands_offered(self):
        self.assertIn("/mcp__pw__review", self.completions("/mcp_"))

    def test_mcp_offered(self):
        self.assertIn("/mcp", self.completions("/mc"))

    def test_mcp_subcommands(self):
        self.assertIn("reconnect", self.completions("/mcp rec"))

    def test_server_names_after_reconnect(self):
        self.assertIn("pw", self.completions("/mcp reconnect p"))
```

(`FakeUI` in this file needs the extra attributes the completer touches — `available_models`, `model_catalog`, and `commands = CommandHandler(self)`; extend the existing FakeUI from Task 11's tests.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_commands -v`
Expected: FAIL on the new tests.

- [ ] **Step 3: Implement in `completer.py`**

- Add `"/mcp": "MCP servers: status / reconnect / login / logout"` to `COMMANDS`.
- In `get_completions`, the no-space branch additionally offers dynamic prompt commands:

```python
        if " " not in text:
            needle = text.lower()
            for command, description in COMMANDS.items():
                if command.startswith(needle):
                    yield Completion(command, start_position=-len(text), display_meta=description)
            for command, description in self._mcp_prompt_commands():
                if command.startswith(needle):
                    yield Completion(command, start_position=-len(text), display_meta=description)
            return
```

with

```python
    @staticmethod
    def _mcp_prompt_commands() -> list[tuple[str, str]]:
        try:
            from src.llm.tools.mcp.manager import get_mcp_manager

            return get_mcp_manager().prompt_commands()
        except Exception:
            return []

    @staticmethod
    def _mcp_server_names() -> list[str]:
        try:
            from src.llm.tools.mcp.manager import get_mcp_manager

            return [status.name for status in get_mcp_manager().statuses()]
        except Exception:
            return []
```

- In `_argument_completions`, add:

```python
        if command == "/mcp":
            first, _, rest = argument.partition(" ")
            if not rest and " " not in argument:
                yield from self._option_completions(word, ("reconnect", "login", "logout"))
                return
            yield from self._option_completions(word, tuple(self._mcp_server_names()))
            return
```

- [ ] **Step 4: Run tests, lint, commit**

```bash
uv run python -m unittest tests.test_mcp_commands -v
uv run python -m unittest discover -s tests && bash lint.sh
git add src/ui/completer.py tests/test_mcp_commands.py
git commit -m "feat(mcp): completer support for /mcp and prompt slash-commands

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: `/tool` pinning

**Files:**
- Modify: `src/ui/chat.py` (`pinned_tools` state + `take_user_content`)
- Modify: `src/ui/commands.py` (`_tool_pin` real implementation)
- Modify: `src/ui/completer.py` (tool-name completions for `/tool`)
- Create: `tests/test_tool_pin.py`

**Interfaces:**
- Consumes: `get_registry()` (`document`, `search`, `get`), `set_registry` seam.
- Produces:

```python
ChatUI.pinned_tools: list[str]                 # initialized in __init__
ChatUI.take_user_content(text)                 # prepends nothing; APPENDS the directive block, then images as before
CommandHandler._tool_pin(argument) -> None     # "" -> show pins; "none" -> clear; name -> pin or suggest
TOOL_DIRECTIVE_HEADER = (
    "[Tool directive: use the tool(s) below for this request — the user "
    "picked them explicitly. Contracts follow; no need for search_tools/get_tool.]"
)
```

- [ ] **Step 1: Write failing tests**

```python
import sys
import unittest
from pathlib import Path

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry, set_registry
from src.ui.chat import ChatUI


@tool(parse_docstring=True)
async def demo_probe(x: str) -> str:
    """Probe something.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


class TestToolPin(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry()
        registry.register(demo_probe)
        set_registry(registry)
        self.addCleanup(set_registry, None)
        self.ui = ChatUI(model="test/model")

    def handle(self, text):
        return self.ui.handle_command(text)

    def test_pin_and_consume(self):
        self.handle("/tool demo_probe")
        content = self.ui.take_user_content("do the thing")
        self.assertIn("do the thing", content)
        self.assertIn("Tool directive", content)
        self.assertIn("demo_probe(x)", content)
        self.assertEqual(self.ui.pinned_tools, [])
        self.assertEqual(self.ui.take_user_content("next"), "next")

    def test_unknown_name_suggests_and_does_not_pin(self):
        self.handle("/tool demo_pro")
        self.assertEqual(self.ui.pinned_tools, [])

    def test_none_clears(self):
        self.handle("/tool demo_probe")
        self.handle("/tool none")
        self.assertEqual(self.ui.pinned_tools, [])

    def test_pin_with_pending_image_lands_in_text_block(self):
        self.handle("/tool demo_probe")
        self.ui.pending_images.append({"data_uri": "data:image/jpeg;base64,x", "width": 1, "height": 1})
        content = self.ui.take_user_content("look")
        text_blocks = [b for b in content if b.get("type") == "text"]
        self.assertTrue(any("Tool directive" in b.get("text", "") for b in text_blocks))


if __name__ == "__main__":
    unittest.main()
```

Before writing the image test, read `ChatUI.take_user_content` (`src/ui/chat.py:184-198`) and `attach_clipboard_image` to copy the exact pending-image dict shape — adjust the fake dict keys to what the real code reads. Constructing `ChatUI` in a test must not require a terminal; `tests/test_model_commands.py` already constructs it — copy its setup pattern.

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_tool_pin -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`chat.py`: add `self.pinned_tools: list[str] = []` in `__init__`. At the top of `take_user_content`:

```python
        text = self._apply_tool_pins(text)
```

and the helper:

```python
    def _apply_tool_pins(self, text: str) -> str:
        """Append pinned tools' contracts to the outgoing message, consuming the pins."""
        if not self.pinned_tools:
            return text
        from src.llm.tools import get_registry

        registry = get_registry()
        contracts = [registry.document(name) for name in self.pinned_tools]
        self.pinned_tools = []
        blocks = "\n\n---\n\n".join(c for c in contracts if c)
        if not blocks:
            return text
        from src.ui.commands import TOOL_DIRECTIVE_HEADER

        return f"{text}\n\n{TOOL_DIRECTIVE_HEADER}\n\n{blocks}"
```

`commands.py`: `TOOL_DIRECTIVE_HEADER` constant; `_tool_pin`:

```python
    def _tool_pin(self, argument: str) -> None:
        from src.llm.tools import get_registry

        registry = get_registry()
        if not argument:
            pinned = ", ".join(self.ui.pinned_tools) or "nothing"
            self.console.print(f"  pinned for the next message: {pinned} (/tool <name>, /tool none)", style="info")
            return None
        if argument.lower() in CLEAR_WORDS:
            self.ui.pinned_tools.clear()
            self.console.print("  tool pins cleared", style="success")
            return None
        name = argument.strip()
        if registry.get(name) is None:
            matches = [t.name for t in registry.all_tools() if name.lower() in t.name.lower()]
            if len(matches) == 1:
                name = matches[0]
            else:
                listed = ", ".join(matches[:MAX_LISTED_MATCHES]) or "no similar names"
                self.console.print(f"  unknown tool: {name} — {listed}", style="error")
                return None
        if name not in self.ui.pinned_tools:
            self.ui.pinned_tools.append(name)
        self.console.print(
            f"  pinned {name} — its contract goes with your next message", style="success"
        )
        return None
```

Replace the Task 11 stub routing (`/tool` already routes here). `completer.py`: in `_argument_completions` add:

```python
        if command == "/tool":
            yield from self._tool_name_completions(word)
            return
```

with

```python
    @staticmethod
    def _tool_name_completions(word: str):
        try:
            from src.llm.tools import get_registry

            registry = get_registry()
        except Exception:
            return
        needle = word.lower()
        for tool_obj in registry.all_tools():
            if needle in tool_obj.name.lower():
                yield Completion(tool_obj.name, start_position=-len(word))
```

Add `"/tool": "pin a tool for the next message"` to `COMMANDS`.

- [ ] **Step 4: Run tests, full suite, lint, commit**

```bash
uv run python -m unittest tests.test_tool_pin -v
uv run python -m unittest discover -s tests && bash lint.sh
git add src/ui/chat.py src/ui/commands.py src/ui/completer.py tests/test_tool_pin.py
git commit -m "feat: /tool pins a registry tool's contract onto the next message

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14: OAuth — token storage, callback server, login/logout, needs-auth

**Files:**
- Create: `src/llm/tools/mcp/oauth.py`
- Modify: `src/llm/tools/mcp/manager.py` (`_auth_for`, `_failure_state`, `login`, `logout`)
- Modify: `src/ui/commands.py` (`run_mcp_action` login/logout branches)
- Create: `tests/test_mcp_oauth.py`

**Interfaces:**
- Consumes: SDK `OAuthClientProvider`, `TokenStorage`, `OAuthClientMetadata`, `OAuthToken`, `OAuthClientInformationFull` (verified Task 1); `settings.mcp_oauth_port`.
- Produces:

```python
def default_token_path(home: Path | None = None) -> Path   # ~/.right-agent/mcp-tokens.json
class FileTokenStorage:                                     # implements the SDK TokenStorage protocol
    def __init__(self, server_name: str, server_url: str, path: Path | None = None)
    async def get_tokens() -> OAuthToken | None
    async def set_tokens(tokens: OAuthToken) -> None
    async def get_client_info() -> OAuthClientInformationFull | None
    async def set_client_info(info: OAuthClientInformationFull) -> None
def has_stored_tokens(server_name: str, server_url: str, path: Path | None = None) -> bool
def clear_tokens(server_name: str, server_url: str, path: Path | None = None) -> bool
class CallbackServer:                                       # 127.0.0.1:<port>, one-shot
    def __init__(self, port: int)
    def redirect_uri() -> str                               # http://127.0.0.1:<port>/callback
    async def wait_for_code(timeout: float = 300.0) -> tuple[str, str | None]   # (code, state)
    def start() / stop()
def build_oauth_provider(config, *, interactive: bool, storage: FileTokenStorage | None = None,
                         port: int | None = None, opener=webbrowser.open) -> Any  # httpx.Auth
# interactive=False -> redirect_handler raises NeedsInteractiveAuth (manager maps to NEEDS_AUTH)
class NeedsInteractiveAuth(Exception): ...
# manager additions:
McpManager.login(name: str) -> ServerStatus      # async; interactive provider, browser flow, reconnect
McpManager.logout(name: str) -> ServerStatus     # async; clear tokens, reconnect
```

- [ ] **Step 1: Write failing tests**

```python
import asyncio
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from src.llm.tools.mcp.oauth import CallbackServer, FileTokenStorage, clear_tokens, has_stored_tokens


class TestFileTokenStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tokens.json"

    def storage(self, name="srv", url="https://s/mcp"):
        return FileTokenStorage(name, url, path=self.path)

    def test_round_trip_tokens(self):
        storage = self.storage()
        token = OAuthToken(access_token="abc", token_type="Bearer", refresh_token="r1")
        asyncio.run(storage.set_tokens(token))
        loaded = asyncio.run(storage.get_tokens())
        self.assertEqual(loaded.access_token, "abc")
        self.assertEqual(loaded.refresh_token, "r1")

    def test_missing_returns_none(self):
        self.assertIsNone(asyncio.run(self.storage().get_tokens()))
        self.assertIsNone(asyncio.run(self.storage().get_client_info()))

    def test_servers_are_isolated(self):
        asyncio.run(self.storage("a", "https://a/").set_tokens(OAuthToken(access_token="ta", token_type="Bearer")))
        self.assertIsNone(asyncio.run(self.storage("b", "https://b/").get_tokens()))

    def test_client_info_round_trip(self):
        info = OAuthClientInformationFull(client_id="cid", redirect_uris=["http://127.0.0.1:43110/callback"])
        storage = self.storage()
        asyncio.run(storage.set_client_info(info))
        self.assertEqual(asyncio.run(storage.get_client_info()).client_id, "cid")

    def test_file_permissions_are_owner_only(self):
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        mode = os.stat(self.path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_helpers(self):
        self.assertFalse(has_stored_tokens("srv", "https://s/mcp", path=self.path))
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        self.assertTrue(has_stored_tokens("srv", "https://s/mcp", path=self.path))
        self.assertTrue(clear_tokens("srv", "https://s/mcp", path=self.path))
        self.assertFalse(has_stored_tokens("srv", "https://s/mcp", path=self.path))


class TestCallbackServer(unittest.TestCase):
    def test_receives_code_from_local_request(self):
        server = CallbackServer(port=0)  # port 0 -> OS-assigned; redirect_uri() reports the real one
        server.start()
        self.addCleanup(server.stop)

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_code(timeout=5))
            await asyncio.sleep(0.05)
            url = server.redirect_uri() + "?code=xyz&state=st"
            await asyncio.to_thread(urllib.request.urlopen, url)
            return await waiter

        code, state = asyncio.run(scenario())
        self.assertEqual(code, "xyz")
        self.assertEqual(state, "st")


if __name__ == "__main__":
    unittest.main()
```

Plus manager-level tests appended to `tests/test_mcp_manager.py`:

```python
class TestNeedsAuth(unittest.TestCase):
    def test_http_401_without_tokens_reports_needs_auth(self):
        from src.llm.tools.mcp.oauth import NeedsInteractiveAuth

        config = McpServerConfig(name="remote", transport="http", url="https://r/mcp")

        @asynccontextmanager
        async def factory(config, auth=None):
            raise NeedsInteractiveAuth("authorization required")
            yield  # pragma: no cover

        manager = McpManager(configs={"remote": config}, session_factory=factory, registry=ToolRegistry())

        async def scenario():
            await manager.start()
            status = manager.statuses()[0]
            self.assertEqual(status.state, ServerState.NEEDS_AUTH)
            await manager.stop()

        asyncio.run(scenario())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest tests.test_mcp_oauth tests.test_mcp_manager -v`
Expected: FAIL.

- [ ] **Step 3: Implement `oauth.py`**

- `FileTokenStorage`: one JSON file `{key: {"tokens": {...}, "client_info": {...}}}` with key `f"{server_name}|{server_url}"`; read tolerant of missing/corrupt file; write via `json.dumps(..., indent=2)`, then `os.chmod(path, 0o600)` (wrap chmod in try/except for Windows, where it partially applies — the test asserting 0o600 must be skipped on win32 with `@unittest.skipIf(sys.platform == "win32", ...)` — actually this repo's suite runs on macOS/Windows; add the skip decorator now). Serialize pydantic objects with `.model_dump(mode="json", exclude_none=True)`, rebuild with `OAuthToken.model_validate` / `OAuthClientInformationFull.model_validate`.
- `CallbackServer`: `http.server.HTTPServer(("127.0.0.1", port), handler)` on a daemon thread; the handler parses `code`/`state`/`error` query params into a `queue.Queue`, answers 200 with a tiny HTML "Authorized — you can close this tab." (or the error). `wait_for_code` polls the queue via `asyncio.to_thread(queue.get, timeout)`; `error` param → raise `RuntimeError(error)`. `redirect_uri()` uses the bound port (`server.server_address[1]`) so `port=0` works in tests.
- `NeedsInteractiveAuth(Exception)`.
- `build_oauth_provider`:

```python
def build_oauth_provider(config, *, interactive, storage=None, port=None, opener=None):
    import webbrowser

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    port = port or settings.mcp_oauth_port
    storage = storage or FileTokenStorage(config.name, config.url)
    opener = opener or webbrowser.open
    callback = CallbackServer(port=port) if interactive else None

    async def redirect_handler(authorization_url: str) -> None:
        if not interactive:
            raise NeedsInteractiveAuth(f"run /mcp login {config.name}")
        opener(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        if callback is None:
            raise NeedsInteractiveAuth(f"run /mcp login {config.name}")
        return await callback.wait_for_code()

    provider = OAuthClientProvider(
        server_url=config.url,
        client_metadata=OAuthClientMetadata(
            client_name="right-coding-agent",
            redirect_uris=[f"http://127.0.0.1:{port}/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    return provider, callback
```

  NOTE: check the exact `OAuthClientProvider` constructor signature in the installed SDK (`uv run python -c "import inspect; from mcp.client.auth import OAuthClientProvider; print(inspect.signature(OAuthClientProvider.__init__))"`) and match it; the fields above are the documented shape. `OAuthClientMetadata` may require `AnyUrl` values — pass plain strings and let pydantic coerce; if validation demands more fields, supply the minimal valid set and record it in the module docstring.
- Manager wiring:
  - `_auth_for(config)`: transports stdio → None; http/sse → when `has_stored_tokens(config.name, config.url)`, build a non-interactive provider (`build_oauth_provider(config, interactive=False)[0]`) so refresh happens silently; else None.
  - `_failure_state(config, error)`: `NEEDS_AUTH` when the error is `NeedsInteractiveAuth` OR (transport http/sse AND "401" in `str(error)` AND no stored tokens); else `FAILED`. Store a `needs auth — run /mcp login <name>` error text for the first case.
  - `login(name)`: `_disconnect`; `provider, callback = build_oauth_provider(config, interactive=True)`; `callback.start()`; temporarily stash the provider on the connection (`conn.override_auth = provider` — add the field to `_Connection`; `_auth_for` returns it once and clears it); `_connect(conn)` — the SDK flow runs during the transport's first request, opening the browser; finally `callback.stop()`; return status.
  - `logout(name)`: `_disconnect`, `clear_tokens(config.name, config.url)`, `_connect`, return status.
- `run_mcp_action` in `commands.py`: replace the login/logout stubs with `await manager.login(action.argument)` / `await manager.logout(action.argument)`, printing the resulting state; keep the try/except-print wrapper.

- [ ] **Step 4: Run tests**

Run: `uv run python -m unittest tests.test_mcp_oauth tests.test_mcp_manager tests.test_mcp_commands -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run python -m unittest discover -s tests && bash lint.sh
git add src/llm/tools/mcp/oauth.py src/llm/tools/mcp/manager.py src/ui/commands.py tests/test_mcp_oauth.py tests/test_mcp_manager.py
git commit -m "feat(mcp): OAuth login flow with file token storage and needs-auth states

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 15: Opt-in integration test, CLAUDE.md, final sweep

**Files:**
- Create: `tests/test_mcp_integration.py`
- Modify: `CLAUDE.md`
- Modify: `README.md` (only if it lists commands — check first)

**Interfaces:** consumes everything; produces documentation and the live proof.

- [ ] **Step 1: Write the opt-in integration test**

Gated like `tests/test_vision_integration.py` (read its gating pattern first):

```python
import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN = os.environ.get("RUN_MCP_TESTS") == "1"


@unittest.skipUnless(RUN, "set RUN_MCP_TESTS=1 to run against a live npx MCP server")
class TestLiveStdioServer(unittest.TestCase):
    def test_everything_server_end_to_end(self):
        from src.llm.tools import ToolRegistry
        from src.llm.tools.mcp.config import McpServerConfig
        from src.llm.tools.mcp.manager import McpManager

        async def scenario():
            registry = ToolRegistry()
            manager = McpManager(
                configs={
                    "everything": McpServerConfig(
                        name="everything",
                        command="npx",
                        args=["-y", "@modelcontextprotocol/server-everything"],
                    )
                },
                registry=registry,
            )
            await manager.start()
            try:
                names = {t.name for t in registry.all_tools(source_prefix="mcp:")}
                self.assertIn("mcp__everything__echo", names)
                echo = registry.get("mcp__everything__echo")
                out = await echo.ainvoke({"message": "round-trip"})
                self.assertIn("round-trip", out)
            finally:
                await manager.stop()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
```

Run once locally: `RUN_MCP_TESTS=1 uv run python -m unittest tests.test_mcp_integration -v` → PASS (requires node). Without the env var it must skip.

- [ ] **Step 2: End-to-end REPL smoke test**

```bash
uv run python -m src.main mcp add everything -- npx -y @modelcontextprotocol/server-everything
uv run python -m src.main mcp list        # expect ✓ everything ... N tools
uv run python -m src.main                 # then in the REPL:
#   /mcp                       -> table shows everything connected
#   ask: "use the mcp echo tool to echo 'hello'"  -> agent finds it via search_tools and calls it
#   /tool mcp__everything__echo  then "echo hi"    -> agent uses the pinned tool directly
#   /quit
uv run python -m src.main mcp remove everything
```

Verify in `logs.log` that the turn's `run_tools` script called `mcp__everything__echo`. Fix whatever this surfaces before proceeding.

- [ ] **Step 3: Document in CLAUDE.md**

Add an `### MCP servers` subsection under "Meta tools" (match the file's dense prose style), covering: the package layout (`config/manager/adapter/transports/oauth/cli/tool`), Claude Code-compatible `.mcp.json` + `~/.right-agent/mcp.json` with `${VAR}` expansion, the CLI (`uv run python -m src.main mcp add|add-json|list|get|remove`), persistent per-server connection tasks owning their anyio contexts, `mcp__server__tool` naming with hash truncation, source labels + `search_tools(only_mcp=...)`, image results through `attach_image`, prompts as `/mcp__…` commands via the `McpAction` return contract, `/tool` pinning (and WHY the directive goes into the user message, not the system prompt), OAuth (SDK provider, `~/.right-agent/mcp-tokens.json`, fixed callback port), and the test seams (`set_mcp_manager`, injectable `session_factory`, config file paths). Also update the "To add a tool" sentence to mention MCP servers as the no-code path.

- [ ] **Step 4: Final full sweep**

```bash
uv run python -m unittest discover -s tests
bash lint.sh
```

Expected: everything green, lint clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mcp_integration.py CLAUDE.md README.md
git commit -m "docs(mcp): document the MCP layer; add opt-in live integration test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes

- **Spec coverage**: config/scopes (T2–3), CLI (T10), transports (T8), persistent manager + reconnect (T7), registry marking + `only_mcp` (T4), adaptation incl. normalization/serialization/images/annotations (T5–6), resources tools (T9), prompts as slash commands (T11–12), `/mcp` (T11), OAuth incl. needs-auth and login/logout (T14), `/tool` (T13), error containment (throughout; asserted in T7/T9/T11 tests), settings (T7), docs + live proof (T15). The spec's "401 with stored tokens → force-refresh → one retry" is covered by the SDK's own refresh plus the manager's reconnect-retry (T7) — the distinct user-facing messages land in `_failure_state` (T14).
- **Deliberate deviations**: none from the spec. The spec's `search(query, source_prefix=None)` naming is kept; `only_mcp` maps onto it.
- **Type consistency**: `McpAction(kind, argument)` is produced in T11 and consumed in T11/T12/T14; `build_mcp_tool(server, remote_tool, call)` signature identical in T6 (definition) and T7 (use); `ServerStatus` fields match between T7 (definition) and T11 (rendering); `FileTokenStorage(server_name, server_url, path)` identical in T14 tests and implementation.
