"""Keep the reasoning text that `langchain_openai` throws away.

`ChatOpenAI` targets the official OpenAI spec and says so in its own module
docstring: non-standard fields added by third-party providers "are not
extracted or preserved". Its `_convert_delta_to_message_chunk` copies only
`function_call` and `tool_calls` out of a streaming delta, so OpenRouter's
`reasoning` field is dropped inside the SDK — before LangGraph, before the
agent, before the UI's token callback. That is why the turn ticker could sit
on "thinking" for twenty seconds with nothing to show.

Verified on the wire (2026-08-19, `moonshotai/kimi-k2.7-code` via
OpenRouter): one short completion carried 46 `content` deltas and 38
`reasoning` deltas. The openai SDK keeps unknown fields on its chunk models
and `model_dump()` includes them, so the raw dict handed to
`_convert_chunk_to_generation_chunk` still has the reasoning — this subclass
overrides that one method and re-attaches it.

It lands in `additional_kwargs["reasoning"]`, where LangChain's chunk merging
concatenates it like any other string. That key is deliberate: on the way
back out, `_convert_message_to_dict` copies only `name`/`tool_calls`/
`function_call`/`audio` from `additional_kwargs`, so reasoning kept there is
never echoed into a later request.
"""

from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI

# Flat spellings, in order: OpenRouter's `reasoning`, then the
# DeepSeek/Qwen-style `reasoning_content` that other gateways emit.
REASONING_KEYS = ("reasoning", "reasoning_content")


def reasoning_delta(chunk: Mapping[str, Any]) -> str:
    """The reasoning text of one raw streaming chunk, or "" if it carries none."""
    choices = chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}

    for key in REASONING_KEYS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value

    # Structured form: [{"type": "reasoning.text", "text": "...", "index": 0}].
    parts = []
    for detail in delta.get("reasoning_details") or []:
        if not isinstance(detail, Mapping):
            continue
        text = detail.get("text") or detail.get("summary")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class ReasoningChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` that keeps provider reasoning deltas instead of dropping them."""

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if generation_chunk is None:
            return None
        text = reasoning_delta(chunk)
        if text:
            generation_chunk.message.additional_kwargs["reasoning"] = text
        return generation_chunk
