import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.skills.cli import run_skills_cli  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestSkillsCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "ours"
        self.home = base / "home"
        (self.home / ".claude" / "skills").mkdir(parents=True)
        make_skill_dir(self.home / ".claude" / "skills", "found", "---\ndescription: foreign\n---\nbody\n")
        self.user_dir.mkdir()
        make_skill_dir(self.user_dir, "mine", "---\ndescription: local\n---\nbody\n")

    def run_cli(self, argv) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_skills_cli(argv, user_dir=self.user_dir, project_root=None, home=self.home)
        return code, buffer.getvalue()

    def test_list(self):
        code, output = self.run_cli(["list"])
        self.assertEqual(code, 0)
        self.assertIn("mine", output)

    def test_import_listing(self):
        code, output = self.run_cli(["import"])
        self.assertEqual(code, 0)
        self.assertIn("found", output)

    def test_import_all(self):
        code, output = self.run_cli(["import", "--all"])
        self.assertEqual(code, 0)
        self.assertTrue((self.user_dir / "found" / "SKILL.md").is_file())

    def test_import_by_name_unknown(self):
        code, output = self.run_cli(["import", "nosuch"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
