import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.channel import collecting_skill_bodies  # noqa: E402
from src.llm.tools.skills.store import SkillStore, project_skills_dirs  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402

BODY = "---\ndescription: d{n}\n---\nbody {n}\n"


class TestProjectDirs(unittest.TestCase):
    def test_walk_up_to_git_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            nested = root / "apps" / "web"
            nested.mkdir(parents=True)
            # Create only one of the three candidates; all should be returned.
            (root / ".agents" / "skills").mkdir(parents=True)
            # (root / "apps" / ".agents" / "skills").mkdir(parents=True)  # Intentionally NOT created
            # (nested / ".agents" / "skills").mkdir(parents=True)  # Intentionally NOT created
            dirs = project_skills_dirs(nested)
            # All candidates are returned, nearest first, regardless of existence.
            # Walking from nested up to root includes nested, apps, and root levels.
            expected = [
                nested / ".agents" / "skills",  # nearest
                root / "apps" / ".agents" / "skills",  # intermediate
                root / ".agents" / "skills",  # git root
            ]
            self.assertEqual(dirs, expected)
            # Verify ordering is nearest first (deeper paths come before shallower).
            for i in range(len(dirs) - 1):
                # Each subsequent dir should have fewer parts (is a parent).
                self.assertGreater(len(dirs[i].parts), len(dirs[i + 1].parts))


class TestSkillStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "user"
        self.project_dir = base / "project"
        self.user_dir.mkdir()
        self.project_dir.mkdir()
        self.registry = ToolRegistry()

    def store(self) -> SkillStore:
        return SkillStore(user_dir=self.user_dir, project_dirs=[self.project_dir], registry=self.registry)

    def test_scan_registers_with_source(self):
        make_skill_dir(self.user_dir, "alpha", BODY.format(n=1))
        store = self.store()
        store.scan()
        self.assertIsNotNone(self.registry.get("skill__alpha"))
        self.assertEqual(self.registry.source_of("skill__alpha"), "skill:alpha")

    def test_project_beats_user(self):
        make_skill_dir(self.user_dir, "dup", "---\ndescription: user one\n---\nuser body\n")
        make_skill_dir(self.project_dir, "dup", "---\ndescription: project one\n---\nproject body\n")
        store = self.store()
        store.scan()
        self.assertEqual(store.get("dup").scope, "project")

    def test_disable_model_invocation_not_registered(self):
        make_skill_dir(self.user_dir, "manual", "---\ndescription: d\ndisable-model-invocation: true\n---\nbody\n")
        store = self.store()
        store.scan()
        self.assertIsNone(self.registry.get("skill__manual"))
        self.assertIsNotNone(store.get("manual"))  # still user-invocable

    def test_refresh_detects_add_and_remove(self):
        store = self.store()
        store.scan()
        self.assertFalse(store.refresh())
        directory = make_skill_dir(self.user_dir, "late", BODY.format(n=2))
        self.assertTrue(store.refresh())
        self.assertIsNotNone(self.registry.get("skill__late"))
        (directory / "SKILL.md").unlink()
        directory.rmdir()
        self.assertTrue(store.refresh())
        self.assertIsNone(self.registry.get("skill__late"))

    def test_refresh_detects_frontmatter_change(self):
        directory = make_skill_dir(self.user_dir, "mut", BODY.format(n=3))
        store = self.store()
        store.scan()
        import os
        import time

        content = "---\ndescription: changed\nuser-invocable: false\n---\nbody\n"
        (directory / "SKILL.md").write_text(content, encoding="utf-8")
        os.utime(directory / "SKILL.md", (time.time() + 5, time.time() + 5))
        self.assertTrue(store.refresh())
        self.assertFalse(store.get("mut").user_invocable)

    def test_render_for_user_marks_seen(self):
        make_skill_dir(self.user_dir, "task", "---\ndescription: d\n---\ndo $ARGUMENTS\n")
        store = self.store()
        store.scan()
        rendered = store.render_for_user("task", 'fix "the bug"')
        self.assertEqual(rendered, "do fix the bug\n")
        self.assertIn("task", store.seen_hashes)

    def test_user_commands_lists_only_user_invocable(self):
        make_skill_dir(self.user_dir, "visible", BODY.format(n=4))
        make_skill_dir(self.user_dir, "hidden", "---\ndescription: d\nuser-invocable: false\n---\nbody\n")
        store = self.store()
        store.scan()
        slugs = [slug for slug, _, _ in store.user_commands()]
        self.assertIn("visible", slugs)
        self.assertNotIn("hidden", slugs)

    def test_project_dir_created_after_init_is_discovered(self):
        """Regression test: a project dir that doesn't exist at init should be
        discovered when created later, without restarting the store."""
        # Create a store with a project_dir that doesn't exist yet.
        nonexistent_project = Path(self.tmp.name) / "future" / "project"
        store = SkillStore(user_dir=self.user_dir, project_dirs=[nonexistent_project], registry=self.registry)
        store.scan()
        self.assertEqual(len(store.skills), 0)
        # Create the project dir and add a skill.
        nonexistent_project.mkdir(parents=True)
        make_skill_dir(nonexistent_project, "late", BODY.format(n=5))
        # Scan should now find the skill without restarting the store.
        store.scan()
        self.assertIsNotNone(store.get("late"))
        self.assertIsNotNone(self.registry.get("skill__late"))


class _FailOnceRegistry(ToolRegistry):
    """Test double: raises ValueError on the next register() call when armed."""

    def __init__(self):
        super().__init__()
        self.fail_next = False

    def register(self, tool_obj, source=None):
        if self.fail_next:
            self.fail_next = False
            raise ValueError("simulated collision")
        super().register(tool_obj, source=source)


class TestResetSession(unittest.IsolatedAsyncioTestCase):
    """reset_session() must forget delivered-body dedupe (seen_hashes) so a
    skill re-delivers its body after /clear wipes the conversation history
    that held it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = Path(self.tmp.name) / "user"
        self.user_dir.mkdir()
        make_skill_dir(self.user_dir, "task", "---\ndescription: d\n---\nbody\n")
        self.registry = ToolRegistry()
        self.store = SkillStore(user_dir=self.user_dir, project_dirs=[], registry=self.registry)
        self.store.scan()
        self.tool = self.registry.get("skill__task")

    async def test_reset_session_allows_body_to_be_redelivered(self):
        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"arguments": "", "force": False})
        self.assertIn("loaded", confirmation)
        self.assertNotIn("already loaded", confirmation)
        self.assertIn("task", bucket)
        self.assertIn("task", self.store.seen_hashes)

        self.store.reset_session()
        self.assertEqual(self.store.seen_hashes, {})

        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"arguments": "", "force": False})
        self.assertIn("loaded", confirmation)
        self.assertNotIn("already loaded", confirmation)
        self.assertIn("task", bucket)


class TestSkillStoreRegistrationFailure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "user"
        self.user_dir.mkdir()
        self.registry = _FailOnceRegistry()

    def test_failed_reregister_clears_bookkeeping_and_self_heals(self):
        make_skill_dir(self.user_dir, "flaky", "---\ndescription: d\n---\nbody\n")
        store = SkillStore(user_dir=self.user_dir, project_dirs=[], registry=self.registry)
        store.scan()
        self.assertIn("flaky", store._registered)
        self.assertIsNotNone(self.registry.get("skill__flaky"))

        # Force the re-registration attempt (unregister-then-register on an
        # unchanged skill) to fail, simulating a name collision.
        self.registry.fail_next = True
        store.scan()
        self.assertNotIn("flaky", store._registered)
        self.assertIsNone(self.registry.get("skill__flaky"))

        # Collision cleared: the next scan retries from a clean state.
        store.scan()
        self.assertIn("flaky", store._registered)
        self.assertIsNotNone(self.registry.get("skill__flaky"))


if __name__ == "__main__":
    unittest.main()
