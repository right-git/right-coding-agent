import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.llm.history import RECAP_SKILLS_TOTAL_CHARS, compact_finished_turn  # noqa: E402


def turn_with_skills(skills: dict) -> list:
    payload = json.dumps({"result": "ok", "logs": [], "error": None, "skills": skills})
    return [
        HumanMessage(content="do it", id="u1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "run_tools", "args": {"code": "x"}, "id": "c1"}]),
        ToolMessage(content=payload, tool_call_id="c1", name="run_tools", id="t1"),
        AIMessage(content="done", id="a2"),
    ]


class TestSkillRecap(unittest.TestCase):
    def test_body_carried_verbatim(self):
        result = compact_finished_turn(turn_with_skills({"guide": "Always use widgets."}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertIn("skill instructions (kept):", recap.content)
        self.assertIn("Always use widgets.", recap.content)

    def test_skills_not_in_result_slices(self):
        result = compact_finished_turn(turn_with_skills({"guide": "SECRETBODY"}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertEqual(recap.content.count("SECRETBODY"), 1)  # once in the skills section only

    def test_total_budget_drops_oldest_with_note(self):
        skills = {"old": "o" * 9000, "new": "n" * 9000}  # 16k total: newest fits, oldest partially
        result = compact_finished_turn(turn_with_skills(skills))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertIn("n" * 100, recap.content)
        self.assertLessEqual(recap.content.count("o"), RECAP_SKILLS_TOTAL_CHARS)
        self.assertIn("force=True", recap.content)  # per-skill clip or drop note teaches recovery

    def test_no_skills_no_section(self):
        result = compact_finished_turn(turn_with_skills({}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertNotIn("skill instructions", recap.content)


if __name__ == "__main__":
    unittest.main()
