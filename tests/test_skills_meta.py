import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.defaults import set_registry  # noqa: E402
from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.meta.tool import run_tools, search_tools  # noqa: E402
from src.llm.tools.skills.skill import parse_skill  # noqa: E402
from src.llm.tools.skills.tool import build_skill_tool  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestSkillsMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = make_skill_dir(
            Path(self.tmp.name), "guide", "---\ndescription: coding guide for widgets\n---\nAlways use widgets.\n"
        )
        self.skill = parse_skill(directory, "user")
        self.registry = ToolRegistry()
        self.registry.register(build_skill_tool(self.skill, {}), source="skill:guide")
        set_registry(self.registry)
        self.addCleanup(set_registry, None)

    def test_brief_tags_skill(self):
        line = self.registry.brief(self.registry.get("skill__guide"))
        self.assertIn("[Skill]", line)

    def test_only_skills_filter(self):
        listing = asyncio.run(search_tools("widgets", only_skills=True))
        self.assertIn("skill__guide", listing)

    def test_only_both_filters_rejected(self):
        listing = asyncio.run(search_tools("x", only_mcp=True, only_skills=True))
        self.assertIn("only one", listing)

    def test_run_tools_ships_skills_field(self):
        content = asyncio.run(run_tools.ainvoke({"code": 'skill__guide()\nreturn "done"'}))
        payload = json.loads(content)
        self.assertEqual(payload["skills"]["guide"], "Always use widgets.\n")
        self.assertEqual(payload["result"], "done")

    def test_skills_field_survives_generic_clip(self):
        # A script that prints far past MAX_RESULT_CHARS: the clipped JSON is
        # rebuilt so the skills field stays intact and the payload stays JSON.
        code = 'skill__guide(force=True)\nfor i in range(60):\n    print("y" * 1000)\nreturn "ok"'
        content = asyncio.run(run_tools.ainvoke({"code": code}))
        payload = json.loads(content)
        self.assertEqual(payload["skills"]["guide"], "Always use widgets.\n")


if __name__ == "__main__":
    unittest.main()
