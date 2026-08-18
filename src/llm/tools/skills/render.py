"""Substitutions applied to a skill body at invocation time.

One regex pass (re.sub never rescans its own output): $1..$n, $ARGUMENTS,
declared $name tokens, ${SKILL_DIR} (with ${CLAUDE_SKILL_DIR} accepted as an
alias so imported Claude Code skills work unmodified). A backslash directly
before a substitutable token escapes it; missing arguments render empty.
"""

import re
from pathlib import Path


def render_body(body: str, arguments: list[str], names: list[str], skill_dir: Path) -> str:
    values = {name: (arguments[i] if i < len(arguments) else "") for i, name in enumerate(names)}
    name_alt = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    token = r"ARGUMENTS(?![A-Za-z0-9_])|\d+"
    if name_alt:
        token += rf"|(?:{name_alt})(?![A-Za-z0-9_])"
    pattern = re.compile(rf"(\\?)\$({token})|\$\{{(SKILL_DIR|CLAUDE_SKILL_DIR)\}}")

    def replace(match: re.Match) -> str:
        if match.group(3):
            return str(skill_dir)
        if match.group(1):  # escaped: drop the backslash, keep the token
            return "$" + match.group(2)
        found = match.group(2)
        if found == "ARGUMENTS":
            return " ".join(arguments)
        if found.isdigit():
            index = int(found) - 1
            return arguments[index] if 0 <= index < len(arguments) else ""
        return values.get(found, "")

    return pattern.sub(replace, body)
