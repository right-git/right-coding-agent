"""Size guards for operations that can amplify data.

The periodic memory audit in the interpreter only sees allocations *between*
ticks, so any single operation able to produce arbitrarily large output gets
a guard here that checks the PROJECTED size before allocating.
"""

import sys

from .errors import SandboxError
from .limits import MAX_RANGE, MAX_SEQ_LEN


def deep_size(roots: list, limit: int) -> int:
    """Approximate recursive size of all live sandbox data. Short-circuits
    as soon as the limit is exceeded, so cost is bounded."""
    seen: set[int] = set()
    stack = list(roots)
    total = 0
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        if total > limit:
            return total
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
    return total


def safe_range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise SandboxError(f"range() longer than {MAX_RANGE}")
    return r


def _g_zfill(s):
    def zfill(width):
        if not isinstance(width, int) or width > MAX_SEQ_LEN:
            raise SandboxError(f"zfill width exceeds {MAX_SEQ_LEN}")
        return s.zfill(width)

    return zfill


def _g_replace(s):
    def replace(old, new, *rest):
        old, new = str(old), str(new)
        n = s.count(old) if old else 0
        projected = len(s) + n * (len(new) - len(old))
        if projected > MAX_SEQ_LEN:
            raise SandboxError(f"replace() would produce > {MAX_SEQ_LEN} chars")
        return s.replace(old, new, *rest)

    return replace


def _g_join(s):
    def join(iterable):
        items = list(iterable)  # may be large but bounded by existing data
        if len(items) > MAX_SEQ_LEN:
            raise SandboxError(f"join() over > {MAX_SEQ_LEN} items")
        total = sum(len(str(x)) for x in items) + len(s) * max(len(items) - 1, 0)
        if total > MAX_SEQ_LEN:
            raise SandboxError(f"join() would produce > {MAX_SEQ_LEN} chars")
        return s.join(items)

    return join


def _g_extend(lst):
    def extend(other):
        other = list(other)
        if len(lst) + len(other) > MAX_SEQ_LEN:
            raise SandboxError(f"extend() would exceed {MAX_SEQ_LEN} items")
        return lst.extend(other)

    return extend


# method name -> factory that binds `obj` and returns a guarded callable
GUARDED_METHODS = {
    (str, "zfill"): _g_zfill,
    (str, "replace"): _g_replace,
    (str, "join"): _g_join,
    (list, "extend"): _g_extend,
}
