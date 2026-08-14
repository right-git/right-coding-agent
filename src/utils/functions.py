from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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