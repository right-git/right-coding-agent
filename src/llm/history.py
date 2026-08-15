"""Compacting a finished turn's tool history.

While a turn is running the model must see every tool call and result in
full — it works with that data. The moment the turn ends with a plain
assistant answer, the detail turns into dead weight: the conclusions live
in the answer text, while raw scripts, tool listings, and results would
keep riding along with every future model call.

`compact_finished_turn` rewrites the just-finished turn's tool tail into
one synthetic `run_tools` pair — the merged scripts as the call, counts
plus trimmed result heads as the output — cutting a heavy tool turn from
thousands of tokens to a few hundred. `search_tools`/`get_tool` results are
pure discovery noise and survive only as counters. Earlier turns are never
rewritten, so provider prompt caching keeps its stable prefix; attached
screenshot messages are passed through untouched (image pruning is a
separate concern).
"""

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.middlewares.attachments import ATTACHMENT_MARKER
from src.llm.middlewares.message_log import scrub_text

RECAP_CODE_CHARS = 2_000
RECAP_RESULT_CHARS = 1_500
RESULT_SLICE_CHARS = 300
SCRIPT_SEPARATOR = "\n# --- next script ---\n"
NOISE_TOOLS = frozenset({"search_tools", "get_tool"})
RECAP_MARKER = "tool_recap"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def _tail_bounds(messages: list) -> tuple[int, int] | None:
    """(tail_start, final_start) of the finished turn, or None when absent.

    The final assistant block is the trailing run of AI messages without
    tool calls; the tail is everything directly before it that belongs to
    this turn's tool work. The walk stops at the user's message (or any
    other foreign message), and at a previous recap — that one belongs to
    an already-compacted earlier turn.
    """
    final_start = len(messages)
    while final_start > 0:
        candidate = messages[final_start - 1]
        if isinstance(candidate, AIMessage) and not candidate.tool_calls:
            final_start -= 1
            continue
        break
    if final_start == len(messages):  # the turn produced no final answer
        return None

    tail_start = final_start
    while tail_start > 0:
        candidate = messages[tail_start - 1]
        if isinstance(candidate, AIMessage) and candidate.tool_calls:
            if candidate.additional_kwargs.get(RECAP_MARKER):
                break
            tail_start -= 1
            continue
        if isinstance(candidate, ToolMessage):
            tail_start -= 1
            continue
        if isinstance(candidate, HumanMessage) and candidate.additional_kwargs.get(ATTACHMENT_MARKER):
            tail_start -= 1
            continue
        break
    return tail_start, final_start


def compact_finished_turn(messages: list) -> list:
    """History with the last turn's tool tail collapsed into one recap pair."""
    bounds = _tail_bounds(messages)
    if bounds is None:
        return messages
    tail_start, final_start = bounds
    tail = messages[tail_start:final_start]

    call_names: list[str] = []
    scripts: list[str] = []
    call_kinds: dict[str, str] = {}  # tool_call_id -> tool name
    image_messages: list[HumanMessage] = []
    result_slices: list[str] = []

    for message in tail:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.get("name") or "unknown"
                call_names.append(name)
                if tool_call.get("id"):
                    call_kinds[tool_call["id"]] = name
                if name == "run_tools":
                    code = str((tool_call.get("args") or {}).get("code") or "")
                    if code.strip():
                        scripts.append(code.strip())
        elif isinstance(message, ToolMessage):
            name = message.name or call_kinds.get(message.tool_call_id, "")
            if name in NOISE_TOOLS:
                continue
            preview = scrub_text(" ".join(str(message.content).split()), RESULT_SLICE_CHARS)
            if preview:
                result_slices.append(f"- {name or 'tool'}: {preview}")
        elif isinstance(message, HumanMessage):
            image_messages.append(message)

    if not call_names:  # nothing to compact
        return messages

    counts: dict[str, int] = {}
    for name in call_names:
        counts[name] = counts.get(name, 0) + 1
    counts_line = ", ".join(f"{count}×{name}" for name, count in counts.items())

    merged_code = _clip(SCRIPT_SEPARATOR.join(scripts), RECAP_CODE_CHARS)
    if not merged_code:
        merged_code = "# (no run_tools scripts this turn)"
    digest = f"recap of {len(call_names)} tool call(s): {counts_line}"
    if result_slices:
        digest += "\nresults (trimmed):\n" + _clip("\n".join(result_slices), RECAP_RESULT_CHARS)

    recap_id = f"recap_{uuid4().hex[:12]}"
    recap_call = AIMessage(
        content="",
        id=f"msg_{recap_id}",
        tool_calls=[{"name": "run_tools", "args": {"code": merged_code}, "id": recap_id, "type": "tool_call"}],
        additional_kwargs={RECAP_MARKER: True},
    )
    recap_result = ToolMessage(
        content=digest,
        tool_call_id=recap_id,
        name="run_tools",
        id=f"msg_{recap_id}_result",
        additional_kwargs={RECAP_MARKER: True},
    )

    return [
        *messages[:tail_start],
        recap_call,
        recap_result,
        *image_messages,
        *messages[final_start:],
    ]
