"""MCP tool adaptation: names, argument coercion, result serialization.

The pure pieces are ported from the Chattler backend's proven MCP runtime
(`common/service/extensions/providers/mcp/runtime.py` there): models — the
weak ones especially — routinely send `"true"` for a boolean or a JSON
string for an array, and the coercion table below is what made that safe in
production.
"""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import StructuredTool

from src.config.logging import logger

from ..meta.attachments import attach_image
from .utils import read_field as _read_field

_HASH_LENGTH = 8
MAX_TOOL_NAME_LENGTH = 64
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_part(value: str) -> str:
    """Normalize a component to alphanumeric + underscores; fallback to 'x'."""
    normalized = _SAFE_PART_RE.sub("_", str(value or "").strip()).strip("_")
    return normalized or "x"


def _identifier(server: str, tool: str) -> str:
    """Build an identifier from server and tool names; hash if too long."""
    readable = f"mcp__{_safe_part(server)}__{_safe_part(tool)}"
    if len(readable) <= MAX_TOOL_NAME_LENGTH:
        return readable
    digest = hashlib.sha256(f"{server}\x1f{tool}".encode()).hexdigest()[:_HASH_LENGTH]
    suffix = f"_{digest}"
    return readable[: MAX_TOOL_NAME_LENGTH - len(suffix)].rstrip("_") + suffix


def build_tool_name(server: str, remote_tool: str) -> str:
    """Build a tool name in the form `mcp__server__tool`.

    Non-identifier chars are replaced with underscores. If the result
    exceeds MAX_TOOL_NAME_LENGTH, it is truncated and a hash suffix is
    appended for disambiguation.
    """
    return _identifier(server, remote_tool)


def build_prompt_command(server: str, prompt: str) -> str:
    """Build a prompt command in the form `/mcp__server__prompt`.

    Same sanitization as build_tool_name; result is always a valid
    slash command for the REPL.
    """
    return "/" + _identifier(server, prompt)


def _schema_properties(input_schema: dict | None) -> dict:
    """Extract the `properties` object from an input schema."""
    if not isinstance(input_schema, dict):
        return {}
    properties = input_schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _schema_required(input_schema: dict | None) -> set[str]:
    """Extract the `required` array from an input schema as a set."""
    if not isinstance(input_schema, dict):
        return set()
    required = input_schema.get("required")
    if not isinstance(required, list):
        return set()
    return {str(item) for item in required}


def _schema_type(schema: dict) -> str | None:
    """Determine the JSON schema type of a property.

    Handles type arrays (filtering out 'null'), explicit type fields,
    and inferred types from `properties` or `items`.
    """
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        for item in raw_type:
            if item != "null":
                return str(item)
        return None
    if raw_type is not None:
        return str(raw_type)
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return None


def _parse_json_string(value: str) -> object:
    """Parse a JSON string, or return it unchanged if not valid JSON."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_boolean(value: object) -> object:
    """Coerce a value to a boolean.

    Handles booleans, and string forms: true/1/yes/y/on → True,
    false/0/no/n/off → False. Returns unchanged if not recognized.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return value


def _normalize_array(value: object) -> object:
    """Coerce a value to an array.

    Handles lists, JSON strings representing arrays, and comma/semicolon/
    newline-separated strings.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = _parse_json_string(value)
        if isinstance(parsed, list):
            return parsed
        return [part.strip() for part in re.split(r"[;\n,]", value) if part.strip()]
    return value


def _normalize_object(value: object) -> object:
    """Coerce a value to a dictionary.

    Handles dicts and JSON strings representing objects.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _parse_json_string(value)
        if isinstance(parsed, dict):
            return parsed
    return value


def _normalize_scalar(value: object, schema_type: str | None) -> object:
    """Normalize a scalar value according to its schema type.

    Applies type-specific coercion (booleans, integers, floats, arrays,
    objects) and falls back to JSON-parsing strings.
    """
    if value is None:
        return None
    if schema_type == "boolean":
        return _normalize_boolean(value)
    if schema_type == "integer" and isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    if schema_type == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if schema_type == "array":
        return _normalize_array(value)
    if schema_type == "object":
        return _normalize_object(value)
    if isinstance(value, str):
        return _parse_json_string(value)
    return value


def normalize_tool_arguments(arguments: dict, input_schema: dict | None) -> dict:
    """Normalize MCP tool arguments according to their input schema.

    - Empty strings are dropped unless the field is required.
    - Values are coerced to their schema types (booleans, numbers, arrays, etc.).
    - Without a schema, values pass through after attempting JSON parsing of strings.
    """
    properties = _schema_properties(input_schema)
    required = _schema_required(input_schema)
    normalized: dict = {}
    for field_name, value in arguments.items():
        if value in (None, "") and field_name not in required:
            continue
        schema = properties.get(field_name) or {}
        schema_type = _schema_type(schema)
        normalized[field_name] = _normalize_scalar(value, schema_type)
    return normalized


def _serialize_content_item(item: Any, *, server: str, tool_name: str, parts: list[str]) -> None:
    """Render one MCP content block into `parts`, routing images out of band."""
    item_type = _read_field(item, "type")
    if item_type == "text":
        parts.append(str(_read_field(item, "text", "")))
        return
    if item_type == "image":
        data = _read_field(item, "data", "") or ""
        mime = _read_field(item, "mimeType") or "image/png"
        if attach_image(data, mime, label=f"{server}:{tool_name}"):
            parts.append("[image attached — you will see it right after this result]")
        else:
            parts.append(f"[image result ({mime}, {len(data)} base64 chars) — no attachment channel open]")
        return
    if item_type == "audio":
        parts.append(f"[audio result ({_read_field(item, 'mimeType')}) — not supported]")
        return
    if item_type == "resource":
        resource = _read_field(item, "resource")
        summary = {"type": "resource", "uri": str(_read_field(resource, "uri", ""))}
        text = _read_field(resource, "text")
        if text is not None:
            summary["text"] = text
        blob = _read_field(resource, "blob")
        if blob is not None:
            summary["blob_chars"] = len(blob) if isinstance(blob, str) else None
        parts.append(json.dumps(summary, ensure_ascii=False))
        return
    if item_type == "resource_link":
        parts.append(
            json.dumps(
                {
                    "type": "resource_link",
                    "name": _read_field(item, "name"),
                    "uri": str(_read_field(item, "uri", "")),
                    "description": _read_field(item, "description"),
                },
                ensure_ascii=False,
            )
        )
        return
    parts.append(repr(item))


def serialize_call_result(result: Any, *, server: str, tool_name: str) -> str:
    """Render a `CallToolResult` into text for the model.

    Text content passes through, images are routed to the attachment side
    channel (`attach_image`) and replaced with a stub, structured content is
    appended as JSON, and an `isError` result is prefixed for visibility.
    """
    parts: list[str] = []
    for item in _read_field(result, "content", []) or []:
        _serialize_content_item(item, server=server, tool_name=tool_name, parts=parts)
    structured = _read_field(result, "structuredContent")
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, default=repr))
    text = "\n".join(part for part in parts if part) or "(empty result)"
    if _read_field(result, "isError", False):
        return f"[mcp error] {server}: {text}"
    return text


def _annotation_marker(remote_tool: Any) -> str:
    """Prefix hint from MCP tool annotations, if the tool has any."""
    annotations = _read_field(remote_tool, "annotations")
    if annotations is None:
        return ""
    if _read_field(annotations, "destructiveHint"):
        return " [DESTRUCTIVE]"
    if _read_field(annotations, "readOnlyHint"):
        return " [read-only]"
    return ""


def build_mcp_tool(
    server: str,
    remote_tool: Any,
    call: Callable[[str, dict], Awaitable[Any]],
) -> StructuredTool:
    """Adapt a remote MCP tool (SDK `Tool`) into a LangChain `StructuredTool`.

    `call(remote_tool_name, normalized_args)` performs the actual RPC and
    returns a `CallToolResult`; the wrapper never raises — failures (bad
    arguments, transport errors, remote exceptions) come back as an
    "[mcp error] ..." string so a flaky server tool degrades a script step
    rather than blowing up the whole run.
    """
    remote_name = str(_read_field(remote_tool, "name") or "")
    tool_name = build_tool_name(server, remote_name)
    input_schema = _read_field(remote_tool, "inputSchema") or {"type": "object", "properties": {}}
    description = (_read_field(remote_tool, "description") or remote_name).strip()
    description = f"{description}{_annotation_marker(remote_tool)} (MCP server: {server})"

    async def run(**kwargs: Any) -> str:
        try:
            arguments = normalize_tool_arguments(kwargs, input_schema)
            result = await call(remote_name, arguments)
            return serialize_call_result(result, server=server, tool_name=remote_name)
        except Exception as error:
            logger.exception("MCP tool failed server [{}] tool [{}]", server, remote_name)
            return f"[mcp error] {server}.{remote_name}: {error}"

    return StructuredTool(
        name=tool_name,
        description=description,
        args_schema=input_schema,
        coroutine=run,
    )
