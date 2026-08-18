import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.config import (
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
