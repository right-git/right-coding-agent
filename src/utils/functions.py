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
                tool_call["id"]
                for tool_call in message.tool_calls
                if tool_call.get("id")
            } # type: ignore
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