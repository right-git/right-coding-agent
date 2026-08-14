import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.usage import (
    SessionUsage,
    TurnUsage,
    collect_message_ids,
    format_duration,
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

    def test_zero_usage_final_call_keeps_the_previous_context(self):
        aborted = AIMessage(
            content="",
            id="aborted",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

        usage = turn_usage_from_messages([ai("a", 100, 20), aborted])

        self.assertEqual(usage.context_tokens, 120)
        self.assertEqual(usage.calls, 2)

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

    def test_direct_tool_calls_are_counted_even_without_usage(self):
        usage = turn_usage_from_messages(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "run_tools", "args": {}, "id": "c1", "type": "tool_call"},
                        {"name": "search_tools", "args": {}, "id": "c2", "type": "tool_call"},
                    ],
                )
            ]
        )

        self.assertEqual(usage.tool_calls, 2)
        self.assertEqual(usage.calls, 0)

    def test_script_tool_calls_come_from_run_tools_json(self):
        usage = turn_usage_from_messages(
            [
                ToolMessage(
                    content='{"result": "ok", "tool_calls": 5}',
                    tool_call_id="c1",
                ),
                ToolMessage(content="plain text result", tool_call_id="c2"),
                ToolMessage(content='{"result": "ok"}', tool_call_id="c3"),
            ]
        )

        self.assertEqual(usage.script_tool_calls, 5)

    def test_history_tool_messages_are_excluded_by_id(self):
        old = ToolMessage(
            content='{"tool_calls": 9}', tool_call_id="c0", id="old"
        )

        usage = turn_usage_from_messages(
            [old, ToolMessage(content='{"tool_calls": 2}', tool_call_id="c1")],
            frozenset({"old"}),
        )

        self.assertEqual(usage.script_tool_calls, 2)


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

        session.add(TurnUsage(), None, 100.0)

        self.assertEqual(session.turns, 0)
        self.assertEqual(session.unpriced_turns, 0)
        self.assertEqual(session.duration, 0.0)

    def test_accumulates_processing_time(self):
        session = SessionUsage()

        session.add(TurnUsage(100, 10, 110, calls=1), 0.001, 2.5)
        session.add(TurnUsage(100, 10, 110, calls=1), 0.001, 3.0)

        self.assertAlmostEqual(session.duration, 5.5)


class FormatDurationTests(unittest.TestCase):
    def test_scales_units_with_length(self):
        self.assertEqual(format_duration(0.83), "0.8s")
        self.assertEqual(format_duration(12.4), "12s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(3712), "1h 01m")
        self.assertEqual(format_duration(-5), "0.0s")


class FormatMoneyTests(unittest.TestCase):
    def test_scales_precision_with_the_amount(self):
        self.assertEqual(format_money(0), "$0.00")
        self.assertEqual(format_money(2.5), "$2.50")
        self.assertEqual(format_money(0.25), "$0.25")
        self.assertEqual(format_money(0.0012345), "$0.0012")
        self.assertEqual(format_money(0.000035), "$0.000035")


if __name__ == "__main__":
    unittest.main()
