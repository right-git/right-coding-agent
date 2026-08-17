import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.files.service import FileService
from src.llm.tools.files.tool import edit_file, read_file
from src.llm.tools.parser.tool import web_search
from src.llm.tools.shell.service import CommandRunner


class FileServiceTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.service = FileService()

    def make(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class ReadTests(FileServiceTestCase):
    def test_lines_come_numbered_from_one(self):
        path = self.make("a.py", "first\nsecond\n")

        self.assertEqual(self.service.read(str(path)), "1\tfirst\n2\tsecond")

    def test_offset_and_limit_window_the_file(self):
        path = self.make("a.txt", "\n".join(f"line{i}" for i in range(10)))

        result = self.service.read(str(path), limit=2, offset=3)

        self.assertIn("4\tline3", result)
        self.assertIn("5\tline4", result)
        self.assertIn("continue with offset=5", result)

    def test_empty_file_and_missing_file(self):
        path = self.make("empty.txt", "")

        self.assertEqual(self.service.read(str(path)), "(empty file)")
        with self.assertRaises(OSError):
            self.service.read(str(self.root / "nope.txt"))


class WriteAndEditTests(FileServiceTestCase):
    def test_write_creates_parent_directories(self):
        target = self.root / "deep" / "nested" / "new.txt"

        report = self.service.write(str(target), "hello\nworld\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "hello\nworld\n")
        self.assertIn("2 line(s)", report)

    def test_edit_replaces_a_unique_match(self):
        path = self.make("a.py", "value = 1\nother = 2\n")

        self.service.edit(str(path), "value = 1", "value = 42")

        self.assertIn("value = 42", path.read_text(encoding="utf-8"))

    def test_edit_rejects_missing_and_ambiguous_matches(self):
        path = self.make("a.py", "x = 1\nx = 1\n")

        with self.assertRaisesRegex(ValueError, "not found"):
            self.service.edit(str(path), "y = 3", "y = 4")
        with self.assertRaisesRegex(ValueError, "occurs 2 times"):
            self.service.edit(str(path), "x = 1", "x = 2")

    def test_edit_replace_all(self):
        path = self.make("a.py", "x = 1\nx = 1\n")

        report = self.service.edit(str(path), "x = 1", "x = 2", replace_all=True)

        self.assertIn("2 occurrence(s)", report)
        self.assertEqual(path.read_text(encoding="utf-8"), "x = 2\nx = 2\n")


class GlobAndGrepTests(FileServiceTestCase):
    def test_glob_finds_matching_files_recursively(self):
        self.make("src/a.py", "")
        self.make("src/deep/b.py", "")
        self.make("src/readme.md", "")

        result = self.service.glob("**/*.py", str(self.root))

        self.assertIn("src/a.py", result)
        self.assertIn("src/deep/b.py", result)
        self.assertNotIn("readme.md", result)

    def test_glob_supports_custom_max_results(self):
        self.make("f1.py", "")
        self.make("f2.py", "")
        self.make("f3.py", "")

        result = self.service.glob("*.py", str(self.root), max_results=2)

        self.assertIn("+1 more", result)

    def test_glob_skips_ignored_directories(self):
        self.make(".venv/lib.py", "")
        self.make("src/ok.py", "")

        result = self.service.glob("**/*.py", str(self.root))

        self.assertNotIn(".venv", result)
        self.assertIn("ok.py", result)

    def test_grep_content_mode_reports_file_line_and_text(self):
        self.make("src/a.py", "def alpha():\n    return 1\n")
        self.make("src/b.py", "def beta():\n    return 2\n")

        result = self.service.grep(r"def \w+", str(self.root))

        self.assertIn("a.py:1:def alpha():", result)
        self.assertIn("b.py:1:def beta():", result)

    def test_grep_files_and_count_modes(self):
        self.make("a.txt", "cat\ncat\n")
        self.make("b.txt", "dog\n")

        files = self.service.grep("cat", str(self.root), output_mode="files_with_matches")
        counts = self.service.grep("cat", str(self.root), output_mode="count")

        self.assertIn("a.txt", files)
        self.assertNotIn("b.txt", files)
        self.assertIn("a.txt:2", counts)

    def test_grep_case_insensitive_and_glob_filter(self):
        self.make("a.py", "TODO fix\n")
        self.make("b.md", "todo later\n")

        result = self.service.grep("todo", str(self.root), glob="*.py", case_insensitive=True)

        self.assertIn("a.py", result)
        self.assertNotIn("b.md", result)

    def test_grep_no_matches(self):
        self.make("a.txt", "nothing here\n")

        self.assertIn("No matches", self.service.grep("unicorn", str(self.root)))


class HomeExpansionTests(FileServiceTestCase):
    """`~` paths reach the real home directory, never a literal `./~` dir.

    The unexpanded form once made write() report success into `./~/...`
    while the shell (which does expand `~`) saw nothing — the agent rebuilt
    the same files three times before finding out.
    """

    def home(self):
        # Path.expanduser reads HOME on POSIX and USERPROFILE on Windows.
        return patch.dict("os.environ", {"HOME": str(self.root), "USERPROFILE": str(self.root)})

    def test_write_expands_tilde_and_reports_the_resolved_path(self):
        with self.home():
            report = self.service.write("~/blog/index.html", "<html>\n")

        target = self.root / "blog" / "index.html"
        self.assertEqual(target.read_text(encoding="utf-8"), "<html>\n")
        self.assertIn(str(target.resolve()), report)
        self.assertNotIn("~", report)

    def test_read_edit_glob_and_grep_expand_tilde(self):
        self.make("notes/todo.txt", "alpha\n")

        with self.home():
            self.assertIn("alpha", self.service.read("~/notes/todo.txt"))
            edit_report = self.service.edit("~/notes/todo.txt", "alpha", "beta")
            self.assertIn("todo.txt", self.service.glob("**/todo.txt", "~"))
            self.assertIn("beta", self.service.grep("beta", "~"))

        self.assertIn("1 occurrence", edit_report)
        self.assertNotIn("~", edit_report)


class CommandRunnerTests(unittest.TestCase):
    def test_stdout_comes_back(self):
        runner = CommandRunner()

        self.assertEqual(runner.run("echo hello"), "hello")

    def test_nonzero_exit_code_is_prefixed(self):
        runner = CommandRunner()

        result = runner.run("exit 3")

        self.assertIn("[exit code 3]", result)

    def test_timeout_is_reported(self):
        runner = CommandRunner()

        result = runner.run("sleep 5", timeout=1)

        self.assertIn("timed out after 1s", result)


class CodingToolWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_tool_returns_errors_as_strings(self):
        result = await read_file.ainvoke({"file_path": "definitely/missing.txt"})

        self.assertTrue(result.startswith("Tool call failed"))

    async def test_edit_tool_reports_ambiguity_as_string(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "a.py"
        path.write_text("x = 1\nx = 1\n", encoding="utf-8")

        result = await edit_file.ainvoke({"file_path": str(path), "old_string": "x = 1", "new_string": "x = 2"})

        self.assertIn("occurs 2 times", result)


class WebSearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_results_are_numbered_with_title_url_snippet(self):
        results = [
            {"title": "Chattler", "href": "https://chattler.ai", "body": "AI agent builder"},
            {"title": "Docs", "href": "https://chattler.ai/docs", "body": "How it works"},
        ]
        with patch("src.llm.tools.parser.tool.WebParser") as parser_class:
            parser_class.return_value.search_web = AsyncMock(return_value=results)

            rendered = await web_search.ainvoke({"query": "chattler"})

        self.assertIn("1. Chattler", rendered)
        self.assertIn("https://chattler.ai", rendered)
        self.assertIn("AI agent builder", rendered)
        self.assertIn("2. Docs", rendered)

    async def test_empty_results_are_reported(self):
        with patch("src.llm.tools.parser.tool.WebParser") as parser_class:
            parser_class.return_value.search_web = AsyncMock(return_value=[])

            rendered = await web_search.ainvoke({"query": "nothing"})

        self.assertIn("No results", rendered)


if __name__ == "__main__":
    unittest.main()
