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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.logging import logger

SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

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
