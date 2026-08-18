import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
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
        self.patch = unittest.mock.patch.multiple(cli, _project_file=lambda: self.project, _user_file=lambda: self.user)
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

    def test_add_to_unreadable_file_fails_cleanly(self):
        bad_json = "{not json"
        self.project.write_text(bad_json, encoding="utf-8")
        self.assertEqual(cli.run_mcp_cli(["add", "x", "--", "cmd"]), 1)
        self.assertEqual(self.project.read_text(encoding="utf-8"), bad_json)

    def test_add_json_rejects_reserved_transport_key(self):
        # Users naturally write "transport" (the CLI flag's name) instead of
        # the on-disk "type" field; this must raise ValueError -> exit 1
        # with a hint, not a raw TypeError from a duplicate keyword arg.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.run_mcp_cli(["add-json", "srv", '{"transport": "http", "url": "https://x/"}'])
        self.assertEqual(code, 1)
        printed = buf.getvalue()
        self.assertIn("transport", printed)
        self.assertIn('"type"', printed)
        self.assertEqual(load_mcp_servers(project_file=self.project, user_file=self.user, env={}), {})

    def test_add_json_rejects_reserved_name_key(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.run_mcp_cli(["add-json", "srv", '{"name": "other", "command": "npx"}'])
        self.assertEqual(code, 1)
        self.assertIn("name", buf.getvalue())
        self.assertEqual(load_mcp_servers(project_file=self.project, user_file=self.user, env={}), {})


if __name__ == "__main__":
    unittest.main()
