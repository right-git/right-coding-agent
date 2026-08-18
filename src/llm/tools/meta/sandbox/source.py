"""Wrapping a script for parsing without rewriting its string literals.

The interpreter parses a script as the body of a synthetic function so that a
top-level `return` is legal, which means the source has to be re-indented
first. Doing that line by line as plain text re-indents the *inside* of every
multi-line string literal too — and those literals are exactly how a script
hands a file body to `write_file`/`edit_file`. The agent's Python then landed
on disk with its body indented deeper than its `def`, its tabs turned into
spaces and its blank lines emptied, and died with an IndentationError the
moment anything ran it.

So the rewriting here is token-aware: `tokenize` marks every line whose
column 0 falls inside a multi-line string, and those lines are copied through
byte for byte. Such a line is always the continuation of a logical line that
started earlier, so leaving it alone can never change what Python parses —
only what the string contains.
"""

import io
import os
import tokenize

BODY_INDENT = "    "
TAB_WIDTH = 4

# FSTRING_MIDDLE carries the literal text of an f-string in 3.12+; the START
# and END tokens are just the quotes and cannot span lines, but they cost
# nothing to include and keep this correct if that ever changes.
_STRING_TOKENS = frozenset(
    getattr(tokenize, name)
    for name in ("STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
    if hasattr(tokenize, name)
)


def _string_interior_lines(code: str) -> set[int]:
    """1-based line numbers whose column 0 sits inside a string literal.

    A tokenizer failure (an unterminated string, say) means the script cannot
    parse either, so nothing is protected and the resulting SyntaxError is
    what the model sees — mangling text that is already broken costs nothing.
    """
    protected: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type in _STRING_TOKENS and token.end[0] > token.start[0]:
                protected.update(range(token.start[0] + 1, token.end[0] + 1))
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return set()
    return protected


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def wrap_as_function(code: str, name: str) -> str:
    """`code` as the body of `def name():`, string literals left untouched.

    Indentation of the script's own lines is normalized on the way — tabs
    expanded, a shared prefix removed — so a wholly indented script (models
    copy them out of markdown blocks) still parses.
    """
    lines = code.splitlines()
    protected = _string_interior_lines(code)

    normalized: list[str] = []
    for number, line in enumerate(lines, start=1):
        if number in protected:
            normalized.append(line)
            continue
        indent = _leading_whitespace(line)
        normalized.append(indent.expandtabs(TAB_WIDTH) + line[len(indent) :])

    shared = os.path.commonprefix(
        [
            _leading_whitespace(line)
            for number, line in enumerate(normalized, start=1)
            if number not in protected and line.strip()
        ]
    )

    body: list[str] = []
    for number, line in enumerate(normalized, start=1):
        if number in protected:
            body.append(line)
        elif not line.strip():
            body.append("")
        else:
            body.append(BODY_INDENT + line[len(shared) :])

    return f"def {name}():\n" + "\n".join(body) + "\n"
