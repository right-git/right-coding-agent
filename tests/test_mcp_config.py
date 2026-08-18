import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.config import (
    add_server,
    expand_env,
    load_mcp_servers,
    project_config_path,
    remove_server,
    scopes_containing,
    server_entry_json,
    user_config_path,
    McpServerConfig,
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
        entry = server_entry_json(McpServerConfig(name="c", transport="http", url="https://c/", headers={"A": "B"}))
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

    def test_add_to_unreadable_file_raises_and_preserves_content(self):
        bad_json = "{not json"
        self.file.write_text(bad_json, encoding="utf-8")
        with self.assertRaises(ValueError):
            add_server(McpServerConfig(name="new", command="x"), self.file)
        self.assertEqual(self.file.read_text(encoding="utf-8"), bad_json)

    def test_remove_from_unreadable_file_raises(self):
        bad_json = "{not json"
        self.file.write_text(bad_json, encoding="utf-8")
        with self.assertRaises(ValueError):
            remove_server("any", self.file)

    def test_remove_preserves_foreign_keys(self):
        write_json(
            self.file,
            {"otherTool": {"keep": True}, "mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}},
        )
        remove_server("a", self.file)
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(payload["otherTool"], {"keep": True})
        self.assertIn("b", payload["mcpServers"])
        self.assertNotIn("a", payload["mcpServers"])


if __name__ == "__main__":
    unittest.main()
