import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.naming import hashed_identifier  # noqa: E402
from src.llm.tools.skills.channel import collecting_skill_bodies, record_skill_body  # noqa: E402
from src.llm.tools.skills.skill import parse_skill  # noqa: E402
from src.llm.tools.skills.tool import build_skill_tool, build_skill_tool_name  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestNaming(unittest.TestCase):
    def test_short_name_untouched(self):
        self.assertEqual(build_skill_tool_name("k8s-debug"), "skill__k8s-debug")

    def test_long_name_hash_truncated(self):
        name = build_skill_tool_name("s" * 100)
        self.assertLessEqual(len(name), 64)
        self.assertRegex(name, r"_[0-9a-f]{8}$")

    def test_hashed_identifier_stable(self):
        self.assertEqual(hashed_identifier("x" * 100, "key"), hashed_identifier("x" * 100, "key"))

    def test_mcp_naming_still_works(self):
        from src.llm.tools.mcp.adapter import build_tool_name

        self.assertEqual(build_tool_name("pw", "browser_click"), "mcp__pw__browser_click")
        self.assertLessEqual(len(build_tool_name("s" * 50, "t" * 50)), 64)


class TestChannel(unittest.TestCase):
    def test_record_without_channel_returns_false(self):
        self.assertFalse(record_skill_body("a", "body"))

    def test_record_inside_channel(self):
        with collecting_skill_bodies() as bucket:
            self.assertTrue(record_skill_body("a", "body"))
        self.assertEqual(bucket, {"a": "body"})


class TestSkillTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = make_skill_dir(
            Path(self.tmp.name), "demo", "---\ndescription: demo skill\narguments: [target]\n---\nUse $target now.\n"
        )
        self.skill = parse_skill(directory, "user")
        self.seen: dict[str, str] = {}
        self.tool = build_skill_tool(self.skill, self.seen)

    async def test_first_call_records_body(self):
        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"target": "the file", "force": False})
        self.assertIn("loaded", confirmation)
        self.assertEqual(bucket["demo"], "Use the file now.\n")
        self.assertIn("demo", self.seen)

    async def test_second_identical_call_dedupes(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"target": "x", "force": False})
        self.assertIn("already loaded", confirmation)
        self.assertEqual(bucket, {})

    async def test_force_resends(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            await self.tool.ainvoke({"target": "x", "force": True})
        self.assertIn("demo", bucket)

    async def test_changed_arguments_resend(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            await self.tool.ainvoke({"target": "y", "force": False})
        self.assertEqual(bucket["demo"], "Use y now.\n")

    async def test_no_channel_returns_body_inline(self):
        confirmation = await self.tool.ainvoke({"target": "z", "force": False})
        self.assertIn("Use z now.", confirmation)

    async def test_missing_file_returns_error_string(self):
        (self.skill.directory / "SKILL.md").unlink()
        with collecting_skill_bodies():
            result = await self.tool.ainvoke({"target": "x", "force": False})
        self.assertTrue(result.startswith("[skill error]"))

    def test_positional_field_order(self):
        # Interpreter maps positional args onto schema field order: declared
        # names first, force last — skill__demo("x") must bind target="x".
        self.assertEqual(list(self.tool.args), ["target", "force"])


if __name__ == "__main__":
    unittest.main()
