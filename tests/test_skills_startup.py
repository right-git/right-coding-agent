import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.store import SkillStore, skills_startup_report  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestStartupReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "ours"
        self.user_dir.mkdir()
        self.home = base / "home"
        (self.home / ".codex" / "skills").mkdir(parents=True)
        make_skill_dir(self.home / ".codex" / "skills", "foreign", "---\ndescription: f\n---\nbody\n")
        self.store = SkillStore(user_dir=self.user_dir, project_dirs=[], registry=ToolRegistry())
        self.store.scan()

    def test_hint_when_foreign_found(self):
        report = skills_startup_report(self.store, auto_import=False, home=self.home, repo_root=None)
        self.assertIn("/skills import", report)

    def test_auto_import_copies_and_reports(self):
        report = skills_startup_report(self.store, auto_import=True, home=self.home, repo_root=None)
        self.assertIn("foreign", report)
        self.assertTrue((self.user_dir / "foreign" / "SKILL.md").is_file())
        self.assertIsNotNone(self.store.get("foreign"))  # rescanned after the copy

    def test_silent_when_nothing_foreign(self):
        empty_home = Path(self.tmp.name) / "empty"
        empty_home.mkdir()
        self.assertIsNone(skills_startup_report(self.store, auto_import=False, home=empty_home, repo_root=None))

    def test_hint_finds_project_skills_under_repo_root(self):
        empty_home = Path(self.tmp.name) / "empty_home"
        empty_home.mkdir()
        repo_root = Path(self.tmp.name) / "repo"
        make_skill_dir(repo_root / ".claude" / "skills", "proj", "---\ndescription: p\n---\nbody\n")
        # Without repo_root, the project-scoped source is never consulted.
        self.assertIsNone(skills_startup_report(self.store, auto_import=False, home=empty_home, repo_root=None))
        report = skills_startup_report(self.store, auto_import=False, home=empty_home, repo_root=repo_root)
        self.assertIn("/skills import", report)


if __name__ == "__main__":
    unittest.main()
