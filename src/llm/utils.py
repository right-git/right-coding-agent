"""Client-side helpers for the chat loop.

Message-history hygiene (detecting empty final completions, trimming
dangling tool-call blocks), small extractors used by usage accounting, and
human-readable formatting of durations and dollar amounts.
"""

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def is_empty_final_response(
    messages: list[HumanMessage | AIMessage | ToolMessage],
) -> bool:
    """True when the conversation ends on an assistant message that says nothing.

    Providers occasionally return a completion with no text, no tool calls,
    and zeroed usage (seen with Gemini via OpenRouter after image inputs);
    the agent loop reads "no tool calls" as "final answer" and stops. This
    detects that case so the caller can nudge the model to continue.
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return False

    content = last.content
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                return False
            if not isinstance(block, dict):
                continue
            if str(block.get("text") or "").strip():
                return False
            for summary in block.get("summary") or []:
                if isinstance(summary, dict) and str(summary.get("text") or "").strip():
                    return False
        return True
    return False


def trim_incomplete_tool_calls(
    messages: list[HumanMessage | AIMessage | ToolMessage],
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """Drop a trailing assistant tool-call block if it has missing tool outputs."""
    trimmed: list[HumanMessage | AIMessage | ToolMessage] = []
    pending_tool_calls: set[str] = set()
    pending_start_index: int | None = None

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            if pending_tool_calls and pending_start_index is not None:
                trimmed = trimmed[:pending_start_index]
            pending_start_index = len(trimmed)
            pending_tool_calls = {
                tool_call["id"] for tool_call in message.tool_calls if tool_call.get("id")
            }  # type: ignore
            trimmed.append(message)
            continue

        if isinstance(message, ToolMessage):
            if message.tool_call_id in pending_tool_calls:
                pending_tool_calls.remove(message.tool_call_id)
            trimmed.append(message)
            continue

        if pending_tool_calls and pending_start_index is not None:
            trimmed = trimmed[:pending_start_index]
            pending_tool_calls = set()
            pending_start_index = None

        trimmed.append(message)

    if pending_tool_calls and pending_start_index is not None:
        trimmed = trimmed[:pending_start_index]

    return trimmed


def collect_message_ids(messages: Sequence) -> frozenset[str]:
    """Ids of the messages already present before the turn ran."""
    return frozenset(message.id for message in messages if getattr(message, "id", None))


def embedded_tool_calls(content) -> int:
    """The `tool_calls` count a run_tools result reports for its script."""
    if not isinstance(content, str) or not content.startswith("{"):
        return 0
    try:
        parsed = json.loads(content)
    except ValueError:
        return 0
    value = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    return value if isinstance(value, int) and value > 0 else 0


def format_duration(seconds: float) -> str:
    """Wall-clock spans the way people read them, from 0.8s to 1h 05m."""
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_money(amount: float) -> str:
    """Dollar amounts with enough digits to keep tiny per-turn costs visible."""
    if amount == 0:
        return "$0.00"
    if amount >= 0.1:
        return f"${amount:.2f}"
    if amount >= 0.001:
        return f"${amount:.4f}"
    return f"${amount:.6f}"
