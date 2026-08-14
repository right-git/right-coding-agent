import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.usage import (
    SessionUsage,
    TurnUsage,
    collect_message_ids,
    format_money,
    turn_usage_from_messages,
)


def ai(identifier, input_tokens, output_tokens):
    return AIMessage(
        content="step",
        id=identifier,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class TurnUsageTests(unittest.TestCase):
    def test_sums_all_calls_and_takes_context_from_the_last(self):
        usage = turn_usage_from_messages(
            [HumanMessage("hi"), ai("a", 100, 20), ai("b", 130, 30)]
        )

        self.assertEqual(usage.input_tokens, 230)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 280)
        self.assertEqual(usage.context_tokens, 160)
        self.assertEqual(usage.calls, 2)

    def test_history_messages_are_excluded_by_id(self):
        history = [ai("old", 500, 50)]
        response = [*history, HumanMessage("hi"), ai("new", 100, 10)]

        usage = turn_usage_from_messages(
            response, collect_message_ids(history)
        )

        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.calls, 1)

    def test_messages_without_usage_metadata_are_skipped(self):
        usage = turn_usage_from_messages(
            [
                AIMessage(content="no usage"),
                ToolMessage(content="x", tool_call_id="1"),
            ]
        )

        self.assertEqual(usage, TurnUsage())

    def test_collect_message_ids_skips_missing_ids(self):
        ids = collect_message_ids([HumanMessage("hi"), ai("a", 1, 1)])

        self.assertEqual(ids, frozenset({"a"}))


class SessionUsageTests(unittest.TestCase):
    def test_accumulates_tokens_and_priced_costs(self):
        session = SessionUsage()

        session.add(TurnUsage(100, 10, 110, calls=1), 0.002)
        session.add(TurnUsage(200, 20, 220, calls=2), None)

        self.assertEqual(session.total_tokens, 330)
        self.assertEqual(session.turns, 2)
        self.assertEqual(session.unpriced_turns, 1)
        self.assertAlmostEqual(session.cost, 0.002)

    def test_empty_turns_are_ignored(self):
        session = SessionUsage()

        session.add(TurnUsage(), None)

        self.assertEqual(session.turns, 0)
        self.assertEqual(session.unpriced_turns, 0)


class FormatMoneyTests(unittest.TestCase):
    def test_scales_precision_with_the_amount(self):
        self.assertEqual(format_money(0), "$0.00")
        self.assertEqual(format_money(2.5), "$2.50")
        self.assertEqual(format_money(0.25), "$0.25")
        self.assertEqual(format_money(0.0012345), "$0.0012")
        self.assertEqual(format_money(0.000035), "$0.000035")


if __name__ == "__main__":
    unittest.main()
