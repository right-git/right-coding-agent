"""Token accounting for chat turns.

Providers attach `usage_metadata` to every AIMessage. A turn may contain
several model calls (tool loops), so turn totals sum all calls, while the
context figure comes from the last call alone — its input already contains
the whole history, so input plus output of that call is what will occupy the
window at the start of the next turn.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, ToolMessage


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    calls: int = 0
    tool_calls: int = 0
    script_tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def collect_message_ids(messages: Sequence) -> frozenset[str]:
    """Ids of the messages already present before the turn ran."""
    return frozenset(
        message.id for message in messages if getattr(message, "id", None)
    )


def _embedded_tool_calls(content) -> int:
    """The `tool_calls` count a run_tools result reports for its script."""
    if not isinstance(content, str) or not content.startswith("{"):
        return 0
    try:
        parsed = json.loads(content)
    except ValueError:
        return 0
    value = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    return value if isinstance(value, int) and value > 0 else 0


def turn_usage_from_messages(
    messages: Iterable,
    previous_ids: frozenset[str] = frozenset(),
) -> TurnUsage:
    """Sum usage over the messages this turn produced.

    `previous_ids` excludes history that came back in the response — old
    messages keep their usage_metadata and would be double-counted.
    `tool_calls` counts the model's direct calls; `script_tool_calls` sums
    the registry-tool invocations run_tools results report for their scripts.
    """
    input_total = 0
    output_total = 0
    context = 0
    calls = 0
    tool_calls = 0
    script_tool_calls = 0

    for message in messages:
        identifier = getattr(message, "id", None)
        if identifier and identifier in previous_ids:
            continue
        if isinstance(message, ToolMessage):
            script_tool_calls += _embedded_tool_calls(message.content)
            continue
        if not isinstance(message, AIMessage):
            continue
        tool_calls += len(message.tool_calls or [])
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        input_total += input_tokens
        output_total += output_tokens
        calls += 1
        # An aborted/empty completion reports zero usage; keep the last real
        # call's figure instead of showing an impossible "ctx 0".
        if input_tokens + output_tokens > 0:
            context = input_tokens + output_tokens

    return TurnUsage(
        input_tokens=input_total,
        output_tokens=output_total,
        context_tokens=context,
        calls=calls,
        tool_calls=tool_calls,
        script_tool_calls=script_tool_calls,
    )


class SessionUsage:
    """Running totals across the whole chat session."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.turns = 0
        self.unpriced_turns = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, turn: TurnUsage, cost: float | None) -> None:
        if turn.calls == 0:
            return
        self.turns += 1
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens
        if cost is None:
            self.unpriced_turns += 1
        else:
            self.cost += cost


def format_money(amount: float) -> str:
    """Dollar amounts with enough digits to keep tiny per-turn costs visible."""
    if amount == 0:
        return "$0.00"
    if amount >= 0.1:
        return f"${amount:.2f}"
    if amount >= 0.001:
        return f"${amount:.4f}"
    return f"${amount:.6f}"
