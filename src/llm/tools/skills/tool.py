"""Skill -> registry pseudo-tool: calling it delivers the rendered body."""

import hashlib

from langchain_core.tools import StructuredTool

from ..naming import hashed_identifier
from .channel import record_skill_body
from .render import render_body
from .skill import Skill

SKILL_TOOL_PREFIX = "skill__"
SKILL_TOOL_NOTE = (
    "Calling this tool loads the skill's instructions — they arrive in this "
    "result's `skills` field; follow them for the current task."
)


def build_skill_tool_name(slug: str) -> str:
    return hashed_identifier(f"{SKILL_TOOL_PREFIX}{slug}", f"skill:{slug}")


def _args_schema(skill: Skill) -> dict:
    properties: dict = {}
    for name in skill.argument_names or ["arguments"]:
        properties[name] = {"type": "string", "description": f"value for ${name} in the skill body"}
    properties["force"] = {"type": "boolean", "default": False, "description": "resend even if already loaded"}
    return {"type": "object", "properties": properties}


def build_skill_tool(skill: Skill, seen_hashes: dict) -> StructuredTool:
    async def call(**kwargs) -> str:
        try:
            force = bool(kwargs.pop("force", False))
            body = skill.load_body()
            if skill.argument_names:
                positional = [str(kwargs.get(name) or "") for name in skill.argument_names]
            else:
                free = str(kwargs.get("arguments") or "")
                positional = [free] if free else []
            rendered = render_body(body, positional, skill.argument_names, skill.directory)
            digest = hashlib.sha256(rendered.encode()).hexdigest()
            if not force and seen_hashes.get(skill.slug) == digest:
                return f"skill '{skill.slug}' already loaded earlier this session — pass force=True to resend"
            if not record_skill_body(skill.slug, rendered):
                return rendered  # library use outside run_tools
            seen_hashes[skill.slug] = digest
            return f"skill '{skill.slug}' loaded — instructions arrive with this result"
        except Exception as error:
            return f"[skill error] {error}"

    return StructuredTool(
        name=build_skill_tool_name(skill.slug),
        description=f"{skill.description}\n\n{SKILL_TOOL_NOTE}",
        args_schema=_args_schema(skill),
        coroutine=call,
    )
