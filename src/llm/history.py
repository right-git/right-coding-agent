"""Compacting a finished turn's tool history.

While a turn is running the model must see every tool call and result in
full — it works with that data. The moment the turn ends with a plain
assistant answer, the detail turns into dead weight: the conclusions live
in the answer text, while raw scripts, tool listings, and results would
keep riding along with every future model call.

`compact_finished_turn` rewrites the just-finished turn's tool tail into
one synthetic `run_tools` pair — the merged scripts as the call, counts
plus trimmed result heads as the output — cutting a heavy tool turn from
thousands of tokens to a few hundred. Tool contracts fetched by in-script
`get_tool` calls arrive structurally in the run_tools result JSON (its
`contracts` field) and are carried in the recap verbatim (capped): they are
durable knowledge, and dropping them makes the model re-discover the same
tools next turn — two extra model round-trips at full context, far dearer
than the ~1k tokens the contracts cost. This self-regulates: once contracts
are visible in history the model stops calling `get_tool`, and later recaps
add none. Earlier turns are never rewritten by compaction, so provider
prompt caching keeps its stable prefix.

`prune_images` is the second half: every image in history is re-sent with
every model call, and a screenshot goes stale the moment its turn ends — so
only the newest tool screenshot survives (continuity of "what's on screen")
and the newest few user-pasted images (the user may keep asking about
them); older image blocks become short text stubs. Pruning does rewrite an
old message once, when its image ages out of the keep-window — a one-time
prompt-cache break per expiring image, far cheaper than re-sending the
image forever. `finalize_turn_history` applies both steps; that is what the
chat loop calls when a turn ends.
"""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.middlewares.attachments import ATTACHMENT_MARKER
from src.llm.middlewares.message_log import scrub_text

RECAP_CODE_CHARS = 2_000
RECAP_RESULT_CHARS = 1_500
RECAP_CONTRACT_CHARS = 6_000
RECAP_SKILL_CHARS = 8_000
RECAP_SKILLS_TOTAL_CHARS = 16_000
SKILL_DROP_NOTE = "[skill '{slug}' body dropped from history — re-invoke skill__{slug}(force=True) to reload]"
RESULT_SLICE_CHARS = 300
SCRIPT_SEPARATOR = "\n# --- next script ---\n"
CONTRACT_SEPARATOR = "\n\n---\n\n"
RECAP_MARKER = "tool_recap"

KEEP_TOOL_IMAGES = 1
KEEP_USER_IMAGES = 3
TOOL_IMAGE_STUB = "[screenshot removed to save context]"
USER_IMAGE_STUB = "[image removed to save context]"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def _extract_contracts(content: str) -> tuple[str, list[str]]:
    """A run_tools result without its `contracts` field, plus the contracts.

    In-script `get_tool` calls deliver contracts structurally in the result
    JSON; they are pulled out here so the recap carries them verbatim while
    the rest of the result is trimmed like any other. Non-JSON content (a
    clipped result, a tool-level failure string) passes through untouched.
    """
    try:
        payload = json.loads(content)
    except ValueError:
        return content, []
    if not isinstance(payload, dict) or not isinstance(payload.get("contracts"), list):
        return content, []
    contracts = [str(item).strip() for item in payload.pop("contracts") if str(item).strip()]
    return json.dumps(payload, ensure_ascii=False), contracts


def _extract_skills(content: str) -> tuple[str, dict[str, str]]:
    """A run_tools result without its `skills` field, plus the skill bodies.

    Skill bodies are durable instructions (the analog of contracts): the
    recap carries them verbatim under budgets instead of trimming them into
    the generic result slice."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), dict):
        return content, {}
    skills = {str(slug): str(body) for slug, body in payload.pop("skills").items() if str(body).strip()}
    return json.dumps(payload, ensure_ascii=False), skills


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
    contracts: list[str] = []
    skill_bodies: dict[str, str] = {}

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
            content = str(message.content)
            if name == "run_tools":
                content, extracted = _extract_contracts(content)
                for contract in extracted:
                    if contract not in contracts:
                        contracts.append(contract)
                content, extracted_skills = _extract_skills(content)
                for slug, body in extracted_skills.items():
                    skill_bodies.pop(slug, None)  # re-delivery moves it to newest
                    skill_bodies[slug] = body
            preview = scrub_text(" ".join(content.split()), RESULT_SLICE_CHARS)
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
    if contracts:
        digest += "\ntool contracts (kept — call get_tool only for tools not listed here):\n" + _clip(
            CONTRACT_SEPARATOR.join(contracts), RECAP_CONTRACT_CHARS
        )
    if skill_bodies:
        blocks: list[str] = []
        remaining = RECAP_SKILLS_TOTAL_CHARS
        for slug, body in reversed(list(skill_bodies.items())):  # newest first
            if remaining <= 0:
                blocks.append(SKILL_DROP_NOTE.format(slug=slug))
                continue
            kept = _clip(body, min(RECAP_SKILL_CHARS, remaining))
            if len(kept) < len(body):
                kept += "\n" + SKILL_DROP_NOTE.format(slug=slug).replace("dropped from", "truncated in")
            remaining -= len(kept)
            blocks.append(f"### skill: {slug}\n{kept}")
        digest += "\nskill instructions (kept):\n" + "\n\n".join(blocks)
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


def _is_image_block(block) -> bool:
    return isinstance(block, dict) and block.get("type") == "image_url"


def prune_images(
    messages: list,
    *,
    keep_tool: int = KEEP_TOOL_IMAGES,
    keep_user: int = KEEP_USER_IMAGES,
) -> list:
    """History with all but the newest images replaced by text stubs.

    Tool screenshots and user-pasted images have separate budgets, both
    spent newest-first. Messages keep their ids, so usage accounting and
    stream dedup are unaffected; already-stubbed blocks are plain text and
    never counted again, which makes pruning idempotent.
    """
    tool_budget = keep_tool
    user_budget = keep_user
    result = list(messages)

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
            continue
        is_tool = bool(message.additional_kwargs.get(ATTACHMENT_MARKER))
        stub = TOOL_IMAGE_STUB if is_tool else USER_IMAGE_STUB

        rebuilt_reversed = []
        changed = False
        for block in reversed(message.content):
            if not _is_image_block(block):
                rebuilt_reversed.append(block)
                continue
            if is_tool and tool_budget > 0:
                tool_budget -= 1
                rebuilt_reversed.append(block)
            elif not is_tool and user_budget > 0:
                user_budget -= 1
                rebuilt_reversed.append(block)
            else:
                rebuilt_reversed.append({"type": "text", "text": stub})
                changed = True

        if changed:
            result[index] = HumanMessage(
                content=list(reversed(rebuilt_reversed)),
                id=message.id,
                additional_kwargs=dict(message.additional_kwargs),
            )
    return result


def finalize_turn_history(messages: list) -> list:
    """Everything that happens to history when a turn ends: compact, then prune."""
    return prune_images(compact_finished_turn(messages))
