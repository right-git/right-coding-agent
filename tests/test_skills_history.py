import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.llm.history import (  # noqa: E402
    RECAP_SKILLS_TOTAL_CHARS,
    SKILL_DROP_NOTE_CHARS,
    compact_finished_turn,
)


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

    def test_many_dropped_skills_note_stays_bounded(self):
        # keep1/keep2 alone consume the whole 16k total budget; the 30 small
        # "extra" skills that follow must all be fully dropped once it's exhausted.
        skills = {f"extra{i}": "z" * 100 for i in range(30)}
        skills["keep2"] = "b" * 8000
        skills["keep1"] = "a" * 8000  # last-inserted = processed first ("newest")
        result = compact_finished_turn(turn_with_skills(skills))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        section = recap.content.split("skill instructions (kept):", 1)[1]

        # Bounded by the tracked per-skill budget plus at most one combined drop
        # note — not one note per dropped skill. A little slack covers the
        # "### skill: <name>" headers and block separators, which are formatting
        # overhead the per-skill budget never counted against (true before this
        # fix too) rather than something either budget is meant to bound.
        formatting_slack = 200
        self.assertLessEqual(len(section), RECAP_SKILLS_TOTAL_CHARS + SKILL_DROP_NOTE_CHARS + formatting_slack)
        self.assertEqual(section.count("force=True"), 1)  # one combined note, not thirty


if __name__ == "__main__":
    unittest.main()
